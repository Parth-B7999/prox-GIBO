"""Run the NeST-BO repo's own loop (baselines/nest_bo, unchanged) on F = f + h as a black box.

The repo works in the unit box and maps u -> lb + (ub - lb) u; it projects iterates onto the box.
Its loop returns all evaluated inputs newest-first; the iterates are recovered from their positions:
chronological index N_init is x_0, and each outer iteration appends M acquisition points followed
by the new iterate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .metrics import support_f1
from .problems import ProblemA

NEST_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "nest_bo"


class NoisyF:
    """Callable for the NeST loop: (n, d) tensor in [lb, ub] -> (n,) noisy F values."""

    def __init__(self, d, s, lam, cond, seed, noise, noise_seed):
        self.p = ProblemA(d=d, s=s, lam=lam, cond=cond, seed=seed)
        self.noise = noise
        self.rng = np.random.default_rng(noise_seed)

    def __call__(self, X):
        x = X.detach().cpu().numpy().reshape(-1, self.p.d)
        y = self.p.F(x) + self.noise * self.rng.normal(size=len(x))
        return torch.as_tensor(y, dtype=torch.float64)


def run_nestbo(prob_kw: dict, x0: np.ndarray, budget: int, noise_std: float, seed: int,
               M=None, N_init: int = 10, delta: float = 0.2, raw_samples: int = 256) -> dict:
    for path in (str(NEST_ROOT), str(NEST_ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import optimization_loop_NeSTBO as nest  # noqa: E402  (repo module, unchanged)

    d = prob_kw["d"]
    M = d if M is None else M
    cfg = OmegaConf.create({
        "seed": seed, "device": "cpu", "gd_lengthscale_scaling": False, "raw_samples": raw_samples,
        "benchmark": {
            "name": "ProblemA", "n_tot": budget - N_init, "N_init": N_init, "delta": delta, "dim": d,
            "M": M, "lb": -1.0, "ub": 1.0, "start_point": "init", "init_point": [float(v) for v in x0],
            "fn": {"_target_": "prox_bo.nest_adapter.NoisyF", **prob_kw, "noise": noise_std,
                   "noise_seed": 10_000 + seed},
        },
    })
    t0 = time.time()
    alg = nest.main(cfg)
    X, Y, _ = alg.exec_alg()
    elapsed = time.time() - t0
    U = X.detach().cpu().numpy()[::-1]  # chronological, unit box
    Xc = -1.0 + 2.0 * U
    p = ProblemA(**prob_kw)
    idx = list(range(N_init, len(Xc), M + 1))
    iters = Xc[idx]
    gap = p.F(iters) - p.Fstar
    return {
        "evals": np.array(idx) + 1, "X": iters, "gap": gap,
        "f1_exact": np.array([support_f1(x != 0, p.support) for x in iters]),
        "n_settled": np.zeros(len(idx), int), "batch": np.full(len(idx), M),
        "time": np.linspace(0, elapsed, len(idx)), "total_evals": len(Xc),
    }
