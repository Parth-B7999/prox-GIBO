"""GIBO-family local BO loops for composite problems  F = f + h.

One shared loop, configured per method (see PRESETS):

    gibo             F as black box; GIBO's own rule (1 eval at iterate, up to d acquisition
                     points, stop when gain < eps), normalised step, window 5d. [GIBO defaults]
    proj_gibo        F as black box; LCBO's reduced case (no black-box constraints): b1 repeated
                     evaluations at the iterate + b2 acquisition points, normalised step,
                     projection onto the box.  [LCBO paper Table 1 defaults; OUR reimplementation]
    prox_gibo        GP on f only (h known); proximal step with backtracking on the GP mean.
    margin_prox_gibo same, but the acquisition ignores coordinates confidently settled at zero.

Ablation chain: proj_gibo -> prox_gibo (known h + prox step) -> margin_prox_gibo (weights only).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

from .gp import GradGP
from .metrics import support_f1
from .prox import BoxL1, prox_l1_box


@dataclass(frozen=True)
class MethodCfg:
    name: str
    model: str = "F"             # 'F': GP on f+h as black box;  'f': GP on f only, h known
    step: str = "normalized"     # 'normalized' (GIBO) or 'prox'
    b1: int = 2                  # repeated evaluations at the iterate
    b2: int = 5                  # acquisition points per iteration (cap if adaptive)
    adaptive: bool = False       # GIBO rule: stop adding points when gain < eps
    eps: Optional[float] = 0.1   # None: no early stop (GIBO paper config)
    delta_unit: float = 0.1      # acquisition box half-width as a fraction of the domain width
    lr: float = 0.5              # step length in lengthscale units
    step_decay: bool = False     # lr_k = lr / sqrt(k)
    window_mult: float = 2.0     # GP window = window_mult * d most recent points
    hyper_every: int = 5         # refit hyperparameters every k iterations
    use_h_in_step: bool = True   # False: step uses the box only (black-box baselines)
    margin: bool = False
    kappa: float = 2.0
    oracle_settle: bool = False  # DIAGNOSTIC ONLY: settled = (x_i == 0) & (x*_i == 0), uses the truth
    hess_floor: float = 1e-2     # prox_newton: eigenvalues of the GP-mean Hessian floored at
                                 # hess_floor * (largest eigenvalue)  -> positive definite metric
    oracle_hess: bool = False    # DIAGNOSTIC ONLY: prox_newton uses the true Hessian of f (gradient still GP)


PRESETS = {
    "gibo": MethodCfg("gibo", model="F", step="normalized", b1=1, b2=-1, adaptive=True, eps=0.1,
                      delta_unit=0.2, lr=0.25, window_mult=5.0, hyper_every=1),
    # GIBO paper's synthetic-experiment config (configs/synthetic_experiment/100621/gibo.yaml):
    # d acquisition points per iteration, no early stop (epsilon_diff_acq_value empty)
    "gibo_paper": MethodCfg("gibo_paper", model="F", step="normalized", b1=1, b2=-1, adaptive=True,
                            eps=None, delta_unit=0.2, lr=0.25, window_mult=5.0, hyper_every=1),
    "proj_gibo": MethodCfg("proj_gibo", model="F", step="normalized"),
    "proj_gibo_decay": MethodCfg("proj_gibo_decay", model="F", step="normalized", step_decay=True),
    # same black-box GP of F as proj_gibo, but with the proximal methods' backtracking step
    # (box projection only) -> isolates the benefit of exploiting known h from the step rule
    "proj_gibo_ls": MethodCfg("proj_gibo_ls", model="F", step="prox", use_h_in_step=False),
    "prox_gibo": MethodCfg("prox_gibo", model="f", step="prox"),
    "margin_prox_gibo": MethodCfg("margin_prox_gibo", model="f", step="prox", margin=True),
    # Diagnostic upper bound for the weighting idea (not a method): the settled set is the true
    # zero set (once the iterate is at zero there), instead of the |mu| + kappa*sigma < lam test.
    "oracle_margin_prox_gibo": MethodCfg("oracle_margin_prox_gibo", model="f", step="prox", margin=True,
                                         oracle_settle=True),
    # Step-1 Newton variants: identical to prox_gibo / margin_prox_gibo (same acquisition, same caps)
    # except the step uses the GP-mean Hessian as metric (proximal Newton, see prox_newton_subproblem).
    "prox_newton_gibo": MethodCfg("prox_newton_gibo", model="f", step="prox_newton"),
    "margin_prox_newton_gibo": MethodCfg("margin_prox_newton_gibo", model="f", step="prox_newton", margin=True),
    # GIBO's own settings and fixed-length step, followed by the prox (per-coordinate step lr*ls_i/||g||,
    # so the prox is a weighted soft-threshold, still exact). With lam = 0 this IS gibo. Added 2026-11-16
    # for Swimmer, where the GP-mean line search of prox_gibo stalls (steps rejected, x frozen).
    "prox_gibo_ns": MethodCfg("prox_gibo_ns", model="f", step="prox_normalized", b1=1, b2=-1, adaptive=True,
                              eps=0.1, delta_unit=0.2, lr=0.25, window_mult=5.0, hyper_every=1),
    "margin_prox_gibo_ns": MethodCfg("margin_prox_gibo_ns", model="f", step="prox_normalized", b1=1, b2=-1,
                                     adaptive=True, eps=0.1, delta_unit=0.2, lr=0.25, window_mult=5.0,
                                     hyper_every=1, margin=True),
    # Diagnostic upper bound (not a method): would Newton help if the Hessian were learned exactly?
    "oracle_hess_margin_prox_newton_gibo": MethodCfg("oracle_hess_margin_prox_newton_gibo", model="f",
                                                     step="prox_newton", margin=True, oracle_hess=True),
}


def prox_newton_subproblem(x, g, A, h, z0=None, iters=500, tol=1e-10):
    """argmin_z  g^T (z - x) + 0.5 (z - x)^T A (z - x) + h(z)   (A symmetric positive definite),
    solved by FISTA with step 1/lambda_max(A). Exact zeros come from h.prox, as in the prox step."""
    L = float(np.linalg.eigvalsh(A)[-1])
    z = x.copy() if z0 is None else z0.copy()
    y, tk = z.copy(), 1.0
    for _ in range(iters):
        zn = h.prox(y - (g + A @ (y - x)) / L, 1.0 / L)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        y = zn + (tk - 1) / tn * (zn - z)
        done = np.linalg.norm(zn - z) <= tol * (1 + np.linalg.norm(z))
        z, tk = zn, tn
        if done:
            break
    return z


class Oracle:
    """Noisy black box with an evaluation counter and a hard budget."""

    def __init__(self, fn, noise_std: float, rng: np.random.Generator, budget: int):
        self.fn, self.noise, self.rng, self.budget, self.n = fn, noise_std, rng, budget, 0
        self.hist_Z, self.hist_y = [], []  # every evaluation, in order (for best-so-far metrics)

    @property
    def left(self):
        return self.budget - self.n

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)[: max(self.left, 0)]
        y = np.atleast_1d(self.fn(Z)) + self.noise * self.rng.normal(size=len(Z))
        self.n += len(Z)
        self.hist_Z.append(Z)
        self.hist_y.append(y)
        return Z, y


def run(problem, cfg: MethodCfg, x0: np.ndarray, budget: int, noise_std: float, seed: int,
        verbose: bool = False) -> dict:
    """Run one method on a composite problem; returns per-iteration logs (numpy arrays)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    d, lo, hi = problem.d, problem.lo, problem.hi
    width = hi - lo
    h = problem.h if cfg.use_h_in_step else BoxL1(lo=lo, hi=hi, lam=0.0)
    lam = problem.h.l1_weight
    fn = problem.F if cfg.model == "F" else problem.f
    oracle = Oracle(fn, noise_std, rng, budget)
    gp = GradGP(d, width=width, window=int(cfg.window_mult * d))
    delta = cfg.delta_unit * width
    max_b = d if cfg.adaptive else cfg.b2

    x = np.clip(np.asarray(x0, float), lo, hi)
    log = dict(evals=[0], X=[x.copy()], gap=[float(problem.F(x) - problem.Fstar)], n_settled=[0],
               batch=[0], t=[np.nan], ls_med=[np.nan], time=[0.0])
    if cfg.step == "prox_newton":
        log["hess_err"] = [np.nan]  # ||H_gp - H_true||_F / ||H_true||_F (diagnostic, truth not used)
    t0, k, t_prev = time.time(), 0, None

    while oracle.left > 0:
        k += 1
        # -- evaluations at the iterate (b1 repeats) -------------------------------------------
        Z1, y1 = oracle(np.repeat(x[None], cfg.b1, 0))
        gp.add(Z1, y1)
        if oracle.left <= 0:
            break
        # -- hyperparameters / caches ------------------------------------------------------------
        gp.fit() if (k - 1) % cfg.hyper_every == 0 else gp.refresh()
        mean_g, var_g = gp.post_grad(x)
        # -- weights (margin-aware) ----------------------------------------------------------------
        if cfg.margin:
            if cfg.oracle_settle:
                settled = (x == 0) & ~problem.support
            else:
                settled = (x == 0) & (np.abs(mean_g) + cfg.kappa * np.sqrt(var_g) < lam)
            w = (~settled).astype(float)
            if w.sum() == 0:
                w[:] = 1.0
        else:
            settled, w = np.zeros(d, bool), np.ones(d)
        # -- acquisition batch ---------------------------------------------------------------------
        nb = min(max_b, oracle.left)
        Z2, _ = gp.select_batch(x, w, np.maximum(x - delta, lo), np.minimum(x + delta, hi), nb,
                                eps=cfg.eps if cfg.adaptive else None, rng=rng)
        Z2, y2 = oracle(Z2)
        gp.add(Z2, y2)
        gp.refresh()
        # -- step ------------------------------------------------------------------------------------
        g, _ = gp.post_grad(x)
        ls = gp.ls.numpy()
        lr = cfg.lr / np.sqrt(k) if cfg.step_decay else cfg.lr
        t_acc = np.nan
        if cfg.step == "normalized":
            gn = np.linalg.norm(g)
            if gn > 0:
                x = np.clip(x - lr * ls * g / gn, lo, hi)
        elif cfg.step == "prox_normalized":
            gn = np.linalg.norm(g)
            if gn > 0:
                tau = lr * ls / gn  # per-coordinate step = GIBO's step lr * ls_i * g_i / ||g||
                x = prox_l1_box(x - tau * g, tau * h.l1_weight, h.lo, h.hi)
        elif cfg.step == "prox_newton":
            # Proximal Newton on the GP mean: z_t = argmin g^T dz + dz^T Hp dz / (2t) + h(z), with Hp the
            # GP-mean Hessian, eigenvalues floored (PD). Backtracking halves t (t <= 1; t = 1 is the
            # full Newton step) until the same caps as the prox step hold and the quadratic model
            # majorises the GP mean at z_t (=> GP-mean + h decreases). Every trial is a prox
            # solution, so exact zeros are kept (no damping x + s(z - x) that would destroy them).
            Hg = gp.post_hess(x)
            if hasattr(problem, "hess_f"):
                Ht = problem.hess_f(x)
                log["hess_err"].append(float(np.linalg.norm(Hg - Ht) / np.linalg.norm(Ht)))
            if cfg.oracle_hess:
                Hg = problem.hess_f(x)
            ev, V = np.linalg.eigh(Hg)
            ev = np.maximum(ev, cfg.hess_floor * max(ev[-1], 1e-12))
            Hp = (V * ev) @ V.T
            if np.linalg.norm(g) > 0:
                t = 1.0 if t_prev is None else min(2.0 * t_prev, 1.0)
                mu_x = gp.post_mean(x)[0]
                z = None
                for _ in range(40):
                    z = prox_newton_subproblem(x, g, Hp / t, h, z0=z)
                    dx = z - x
                    ok_cap = (np.linalg.norm(dx / ls) <= lr * (1 + 1e-9)) and (np.abs(dx).max() <= delta * (1 + 1e-9))
                    ok_dec = gp.post_mean(z)[0] <= mu_x + g @ dx + dx @ Hp @ dx / (2 * t) + 1e-12
                    if ok_cap and ok_dec:
                        x, t_prev, t_acc = z, t, t
                        break
                    t *= 0.5
        else:
            gl = np.linalg.norm(g / ls)
            if gl > 0:
                # Step caps (same for every line-search method):
                #  * GIBO-style length cap in lengthscale units: ||dx / ls|| <= lr
                #  * trust region = the box just sampled by the acquisition: ||dx||_inf <= delta.
                #    Needed because ARD lengthscales of rarely-sampled coordinates drift upward,
                #    which makes the lengthscale cap meaningless and the GP mean flat (observed:
                #    margin-aware run collapsed to x = 0 with t ~ 1e2 before this cap).
                #  * prox threshold within the trust region: t * lam <= delta.
                t_cap = lr / gl
                if h.l1_weight > 0:
                    t_cap = min(t_cap, delta / h.l1_weight)
                t = t_cap if t_prev is None else min(2.0 * t_prev, t_cap)
                mu_x = gp.post_mean(x)[0]
                for _ in range(40):
                    xp = h.prox(x - t * g, t)
                    dx = xp - x
                    ok_cap = (np.linalg.norm(dx / ls) <= lr * (1 + 1e-9)) and (np.abs(dx).max() <= delta * (1 + 1e-9))
                    ok_dec = gp.post_mean(xp)[0] <= mu_x + g @ dx + dx @ dx / (2 * t) + 1e-12
                    if ok_cap and ok_dec:
                        x, t_prev, t_acc = xp, t, t
                        break
                    t *= 0.5
        # -- log ----------------------------------------------------------------------------------------
        log["evals"].append(oracle.n)
        log["X"].append(x.copy())
        log["gap"].append(float(problem.F(x) - problem.Fstar))
        log["n_settled"].append(int(settled.sum()))
        log["batch"].append(len(Z2))
        log["t"].append(t_acc)
        log["ls_med"].append(float(np.median(ls)))
        log["time"].append(time.time() - t0)
        if verbose:
            print(f"[{cfg.name}] it={k:3d} evals={oracle.n:5d} gap={log['gap'][-1]:.4g} "
                  f"nnz={(x != 0).sum()} settled={log['n_settled'][-1]} b={len(Z2)} t={t_acc:.3g} "
                  f"{log['time'][-1]:.0f}s", flush=True)

    out = {k_: np.array(v) for k_, v in log.items()}
    out["f1_exact"] = np.array([support_f1(xx != 0, problem.support) for xx in out["X"]])
    # every evaluated point and the true objective there (noise-free part of h added for model "f")
    Zh = np.vstack(oracle.hist_Z) if oracle.hist_Z else np.empty((0, d))
    yh = np.concatenate(oracle.hist_y) if oracle.hist_y else np.empty(0)
    out["eval_Z"] = Zh
    out["eval_F"] = yh + (lam * np.abs(Zh).sum(1) if cfg.model == "f" else 0.0)
    return out
