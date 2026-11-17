import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from prox_bo.prox import (BoxGroupLasso, BoxL1, clip_box, group_soft_threshold,
                          prox_l1_box, soft_threshold)


def brute_1d(v, tau, lo, hi):
    """Brute-force argmin_z 0.5(z-v)^2 + tau|z| on [lo,hi]: dense grid + golden-section on both
    sides of the kink (the objective is convex, but minimize_scalar can miss the kink)."""
    if hi == lo:  # degenerate box: the only feasible point
        return lo
    best = None
    for a, b in [(lo, 0.0), (0.0, hi)]:
        if b - a <= 0:
            continue
        r = minimize_scalar(lambda z: 0.5 * (z - v) ** 2 + tau * abs(z), bounds=(a, b),
                            method="bounded", options={"xatol": 1e-12})
        val = 0.5 * (r.x - v) ** 2 + tau * abs(r.x)
        if best is None or val < best[1] - 1e-15:
            best = (r.x, val)
    return best[0]


def test_soft_threshold_basic():
    v = np.array([-2.0, -0.3, 0.0, 0.3, 2.0])
    np.testing.assert_allclose(soft_threshold(v, 0.5), [-1.5, 0, 0, 0, 1.5])
    # per-coordinate tau
    np.testing.assert_allclose(soft_threshold(v, np.array([1, 1, 1, 0.1, 0.1])), [-1, 0, 0, 0.2, 1.9])


def test_clip_box():
    np.testing.assert_allclose(clip_box(np.array([-3.0, 0.2, 5.0]), -1, 1), [-1, 0.2, 1])


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-0.5, 2.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0)])
def test_l1_box_matches_bruteforce(lo, hi):
    rng = np.random.default_rng(0)
    for tau in [0.0, 0.05, 0.3, 1.5]:
        for v in rng.normal(0, 1.5, size=40):
            got = prox_l1_box(np.array([v]), tau, lo, hi)[0]
            ref = brute_1d(v, tau, lo, hi)
            assert abs(got - ref) < 1e-6, (v, tau, lo, hi, got, ref)


def test_boxl1_prox_and_value():
    h = BoxL1(lo=-1, hi=1, lam=0.1)
    x = np.array([0.5, -0.2, 0.0])
    assert h.value(x) == pytest.approx(0.1 * 0.7)
    assert np.isinf(h.value(np.array([1.5, 0, 0])))
    np.testing.assert_allclose(h.prox(np.array([3.0, -0.05, 0.4]), 0.5), [1.0, 0.0, 0.35])


def test_group_soft_threshold():
    groups = [np.array([0, 1]), np.array([2, 3])]
    v = np.array([3.0, 4.0, 0.3, 0.4])  # norms 5 and 0.5
    out = group_soft_threshold(v, 1.0, groups)
    np.testing.assert_allclose(out[:2], v[:2] * (1 - 1 / 5))
    np.testing.assert_allclose(out[2:], 0.0)


def test_group_lasso_box_matches_numeric():
    rng = np.random.default_rng(1)
    groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    h = BoxGroupLasso(groups, lam=1.0, lo=-1.0, hi=1.0)
    for _ in range(25):
        v = rng.normal(0, 1.5, size=6)
        t = rng.uniform(0.05, 1.0)
        got = h.prox(v, t)

        def obj(z):  # smoothed norm only to let L-BFGS-B work; tolerance accounts for it
            return sum(t * np.sqrt(np.sum(z[g] ** 2) + 1e-14) for g in groups) + 0.5 * np.sum((z - v) ** 2)

        ref = minimize(obj, np.clip(v, -1, 1), bounds=[(-1, 1)] * 6, method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000}).x
        # compare objective values (robust where the solution is exactly 0)
        f = lambda z: sum(t * np.linalg.norm(z[g]) for g in groups) + 0.5 * np.sum((z - v) ** 2)
        assert f(got) <= f(ref) + 1e-7, (f(got), f(ref))
        assert np.all(got >= -1 - 1e-12) and np.all(got <= 1 + 1e-12)
