import numpy as np
import pytest

from prox_bo.problems import ProblemA


@pytest.mark.parametrize("d,seed", [(20, 0), (50, 1), (100, 2), (200, 3)])
def test_problem_a_xstar_is_optimum(d, seed):
    p = ProblemA(d=d, s=10, lam=0.1, seed=seed)
    v = p.verify()
    assert v["grad_mapping_at_xstar"] < 1e-9
    assert v["fista_dist_to_xstar"] < 1e-5
    assert v["support_match"]
    assert v["min_offsupport_margin"] >= 0.2 * p.lam - 1e-12
    # x* is no worse than random feasible perturbations
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = np.clip(p.xstar + rng.normal(0, 0.05, d), -1, 1)
        assert p.F(y) >= p.Fstar - 1e-12


def test_problem_a_gradient_fd():
    p = ProblemA(d=15, seed=4)
    x = np.random.default_rng(0).uniform(-1, 1, 15)
    g = p.grad_f(x)
    e = 1e-6
    fd = np.array([(p.f(x + e * np.eye(15)[i]) - p.f(x - e * np.eye(15)[i])) / (2 * e) for i in range(15)])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
