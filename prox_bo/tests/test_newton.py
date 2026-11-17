"""Step-1 Newton variants: GP-mean Hessian and the proximal-Newton subproblem."""
import numpy as np

from prox_bo.gp import GradGP
from prox_bo.optimizers import PRESETS, prox_newton_subproblem, run
from prox_bo.problems import ProblemA
from prox_bo.prox import BoxL1


def _gp(d=6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    p = ProblemA(d=d, s=2, lam=1.0, cond=10.0, seed=seed)
    X = rng.uniform(-0.5, 0.5, (n, d))
    gp = GradGP(d)
    gp.add(X, p.f(X) + 0.01 * rng.normal(size=n))
    gp.fit()
    return gp, p, rng


def test_post_hess_matches_finite_differences_of_post_grad():
    gp, _, rng = _gp()
    x = rng.uniform(-0.3, 0.3, gp.d)
    H = gp.post_hess(x)
    e = 1e-5
    Hfd = np.stack([(gp.post_grad(x + e * u)[0] - gp.post_grad(x - e * u)[0]) / (2 * e) for u in np.eye(gp.d)])
    assert np.allclose(H, H.T)
    assert np.allclose(H, Hfd, atol=1e-5 * (1 + np.abs(H).max()))


def test_subproblem_with_scaled_identity_is_the_prox_step():
    rng = np.random.default_rng(1)
    h = BoxL1(lo=-1.0, hi=1.0, lam=0.7)
    x, g, t = rng.uniform(-1, 1, 8), rng.normal(size=8), 0.4
    z = prox_newton_subproblem(x, g, np.eye(8) / t, h)
    assert np.allclose(z, h.prox(x - t * g, t), atol=1e-9)


def test_subproblem_optimality_general_metric():
    rng = np.random.default_rng(2)
    d = 8
    M = rng.normal(size=(d, d))
    A = M @ M.T + 0.5 * np.eye(d)
    h = BoxL1(lo=-1.0, hi=1.0, lam=0.5)
    x, g = rng.uniform(-1, 1, d), rng.normal(size=d)
    z = prox_newton_subproblem(x, g, A, h, iters=20000, tol=1e-14)
    q = lambda w: g @ (w - x) + 0.5 * (w - x) @ A @ (w - x) + h.value(w)
    for _ in range(200):  # no feasible random perturbation does better
        w = np.clip(z + 1e-3 * rng.normal(size=d), -1, 1)
        assert q(w) >= q(z) - 1e-10


def test_prox_newton_runs_and_logs_hessian_error():
    p = ProblemA(d=10, s=3, lam=1.0, cond=10.0, seed=0)
    x0 = np.random.default_rng(1000).uniform(-0.5, 0.5, 10)
    out = run(p, PRESETS["margin_prox_newton_gibo"], x0, budget=120, noise_std=0.01, seed=0)
    assert len(out["hess_err"]) == len(out["gap"])
    assert out["gap"][-1] < out["gap"][0]
