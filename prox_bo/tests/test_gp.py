import numpy as np
import torch

from prox_bo.gp import DT, GradGP


def make_gp(d=5, n=20, seed=0, noise=1e-3):
    rng = np.random.default_rng(seed)
    gp = GradGP(d, width=2.0)
    X = rng.uniform(-1, 1, (n, d))
    y = np.sin(X @ rng.normal(size=d)) + 0.3 * (X ** 2).sum(1)
    gp.add(X, y)
    gp._standardize()
    gp.ls = torch.tensor(rng.uniform(0.6, 1.5, d), dtype=DT)
    gp.os = torch.tensor(1.7, dtype=DT)
    gp.noise = torch.tensor(noise, dtype=DT)
    gp.refresh()
    return gp, rng


def test_grad_mean_matches_finite_difference():
    gp, rng = make_gp()
    x = rng.uniform(-1, 1, 5)
    g, _ = gp.post_grad(x)
    e = 1e-5
    fd = np.array([(gp.post_mean(x + e * np.eye(5)[i])[0] - gp.post_mean(x - e * np.eye(5)[i])[0]) / (2 * e)
                   for i in range(5)])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_grad_var_matches_autograd_joint_gp():
    """Var[g_i(x)] = d^2k/(dx_i dx'_i) - k_{dX} K^-1 k_{Xd}, using autograd on the kernel."""
    gp, rng = make_gp()
    x = rng.uniform(-1, 1, 5)
    _, var = gp.post_grad(x)
    xt = torch.tensor(x, dtype=DT)
    X = gp._Xt
    ref = []
    for i in range(5):
        x1 = xt.clone().requires_grad_(True)
        x2 = xt.clone().requires_grad_(True)
        k = gp.kern(x1[None], x2[None])[0, 0]
        g1 = torch.autograd.grad(k, x1, create_graph=True)[0][i]
        prior = torch.autograd.grad(g1, x2)[0][i].item()
        x1 = xt.clone().requires_grad_(True)
        kv = gp.kern(x1[None], X)[0]
        cross = torch.stack([torch.autograd.grad(kv[j], x1, retain_graph=True)[0][i] for j in range(X.shape[0])])
        ref.append(gp.y_std ** 2 * (prior - cross @ gp.Kinv @ cross).item())
    np.testing.assert_allclose(var, ref, rtol=1e-7)


def test_selected_gain_equals_variance_reduction():
    """The acquisition's gain for the chosen points == actual drop in weighted gradient variance."""
    gp, rng = make_gp()
    x = rng.uniform(-0.5, 0.5, 5)
    w = np.array([1, 0, 1, 1, 0.0])
    _, var0 = gp.post_grad(x)
    Z, gains = gp.select_batch(x, w, np.clip(x - 0.3, -1, 1), np.clip(x + 0.3, -1, 1), b=3, rng=rng)
    assert Z.shape == (3, 5)
    assert np.all(Z >= x - 0.3 - 1e-9) and np.all(Z <= x + 0.3 + 1e-9)
    # refit-free: add the points (values irrelevant for the variance) and recompute from scratch
    gp2 = GradGP(5, width=2.0)
    gp2.X, gp2.y = np.vstack([gp.X, Z]), np.concatenate([gp.y, np.zeros(3)])
    gp2.y_mean, gp2.y_std = gp.y_mean, gp.y_std
    gp2.ls, gp2.os, gp2.noise = gp.ls, gp.os, gp.noise
    gp2.refresh()
    _, var1 = gp2.post_grad(x)
    drop_std_units = (w * (var0 - var1)).sum() / gp.y_std ** 2
    np.testing.assert_allclose(gains.sum(), drop_std_units, rtol=1e-6)
    assert np.all(gains > 0)


def test_zero_weight_coordinates_are_ignored():
    gp, rng = make_gp(seed=3)
    x = rng.uniform(-0.5, 0.5, 5)
    lo, hi = np.clip(x - 0.3, -1, 1), np.clip(x + 0.3, -1, 1)
    w = np.array([1.0, 0, 0, 0, 0])
    _, g1 = gp.select_batch(x, w, lo, hi, b=2, rng=np.random.default_rng(0))
    _, g2 = gp.select_batch(x, np.ones(5), lo, hi, b=2, rng=np.random.default_rng(0))
    assert g2.sum() >= g1.sum() - 1e-12  # more weighted coordinates -> at least as much total gain


def test_fit_runs_and_is_sane():
    d = 6
    rng = np.random.default_rng(0)
    gp = GradGP(d, width=2.0)
    X = rng.uniform(-1, 1, (60, d))
    gp.add(X, (X ** 2).sum(1) + 0.01 * rng.normal(size=60))
    gp.fit()
    assert gp.fit_ok
    g, _ = gp.post_grad(np.full(d, 0.4))
    assert np.all(g > 0)  # true gradient 2x > 0


def test_single_point_fit_does_not_collapse_outputscale():
    """Regression (2026-11-15): GIBO (1 eval per iterate) fitted hypers on a single unstandardised
    point, the outputscale collapsed to ~1e-319 and the gradient stayed exactly 0 forever."""
    d = 20
    gp = GradGP(d, width=2.0)
    gp.add(np.zeros((1, d)), np.array([84.0]))
    gp.fit()
    assert float(gp.os) > 1e-3
    rng = np.random.default_rng(0)
    Z = rng.uniform(-0.2, 0.2, (5, d))
    gp.add(Z, 84.0 + Z.sum(1))
    gp.fit()
    g, _ = gp.post_grad(np.zeros(d))
    assert float(gp.os) > 1e-3 and np.linalg.norm(g) > 0


def test_gibo_moves_on_the_instance_that_got_stuck():
    from prox_bo.optimizers import PRESETS, run
    from prox_bo.problems import ProblemA
    d, seed = 200, 4
    p = ProblemA(d=d, s=10, lam=1.0, cond=10.0, seed=seed)
    x0 = np.random.default_rng(1000 + seed).uniform(-0.5, 0.5, d)
    out = run(p, PRESETS["gibo"], x0, 12, 0.01, seed)
    assert np.abs(out["X"][-1] - x0).max() > 0
