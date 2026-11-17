"""Swimmer (MuJoCo, gymnasium Swimmer-v5) with a linear policy: the case study used by both GIBO
(Swimmer-v1, 16 parameters) and NeST-BO (Swimmer-v5, 16 parameters, box [-10, 10]^16).

Composite version ("sparse controller"):
    min_x  F(x) = f(x) + lam * ||x||_1   over [-10, 10]^16,   f(x) = -(episode return of policy x)
The policy is action = W s with W = x.reshape(2, 8) (same parametrisation as NeST-BO's RL_fun_Swimmer);
lam > 0 asks for a controller that uses few sensor -> motor gains. lam = 0 is the original benchmark.

The episode is deterministic (env.reset(seed=0), as in NeST-BO's benchmark), so evaluations are noise
free. F* is unknown: Fstar = 0 is used, so the logged "gap" is simply F(x_t), and support/F1 are
meaningless here (support is set to all-False; report nnz instead).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .prox import BoxL1


class SwimmerL1:
    d, lo, hi = 16, -10.0, 10.0

    def __init__(self, lam: float = 0.0, env_seed: int = 0, max_step: int = 1000):
        self.lam, self.env_seed, self.max_step = lam, env_seed, max_step
        self.h = BoxL1(lo=self.lo, hi=self.hi, lam=lam)
        self.Fstar = 0.0
        self.support = np.zeros(self.d, bool)
        self._env = None
        self.n_episodes = 0

    def _ret(self, x):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make("Swimmer-v5")
        W = np.asarray(x, float).reshape(2, 8)
        o, _ = self._env.reset(seed=self.env_seed)
        tot = 0.0
        for _ in range(self.max_step):
            o, r, te, tr, _ = self._env.step(W @ o)
            tot += r
            if te or tr:
                break
        self.n_episodes += 1
        return tot

    def reward(self, X):
        X = np.atleast_2d(X)
        return np.array([self._ret(x) for x in X])

    def f(self, X):
        out = -self.reward(X)
        return out if np.ndim(X) > 1 else out[0]

    def F(self, X):
        X = np.asarray(X, float)
        return self.f(X) + self.lam * np.abs(X).sum(-1)


class SwimmerF:
    """Black-box F for the NeST-BO loop: (n, d) tensor in [lb, ub] -> (n,) values."""

    def __init__(self, lam):
        self.p = SwimmerL1(lam=lam)

    def __call__(self, X):
        x = X.detach().cpu().numpy().reshape(-1, self.p.d)
        return torch.as_tensor(self.p.F(x), dtype=torch.float64)


def run_nestbo_swimmer(lam: float, x0: np.ndarray, budget: int, seed: int, M=16, N_init=10,
                       delta=0.2, raw_samples=256) -> dict:
    """NeST-BO repo loop, unchanged, with its own Swimmer settings (M = 16, N_init = 10, delta = 0.2)."""
    import sys
    from omegaconf import OmegaConf
    from .nest_adapter import NEST_ROOT
    for path in (str(NEST_ROOT), str(NEST_ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import optimization_loop_NeSTBO as nest  # noqa: E402

    p = SwimmerL1(lam=lam)
    cfg = OmegaConf.create({
        "seed": seed, "device": "cpu", "gd_lengthscale_scaling": False, "raw_samples": raw_samples,
        "benchmark": {
            "name": "Swimmer", "n_tot": budget - N_init, "N_init": N_init, "delta": delta, "dim": p.d,
            "M": M, "lb": p.lo, "ub": p.hi, "start_point": "init", "init_point": [float(v) for v in x0],
            "fn": {"_target_": "prox_bo.swimmer.SwimmerF", "lam": lam},
        },
    })
    t0 = time.time()
    X, _, _ = nest.main(cfg).exec_alg()
    elapsed = time.time() - t0
    U = X.detach().cpu().numpy()[::-1]
    Xc = p.lo + (p.hi - p.lo) * U
    idx = list(range(N_init, len(Xc), M + 1))
    iters = Xc[idx]
    return {"evals": np.array(idx) + 1, "X": iters, "gap": p.F(iters), "f1_exact": np.zeros(len(idx)),
            "eval_Z": Xc, "eval_F": p.F(Xc),
            "n_settled": np.zeros(len(idx), int), "batch": np.full(len(idx), M),
            "time": np.linspace(0, elapsed, len(idx)), "total_evals": len(Xc)}
