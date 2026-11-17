"""Test problems with known composite optimum.

Problem A (convex, controlled margin):
    phi(x) = 0.5 x^T B x + sum_i log cosh(x_i),   f(x) = phi(x) - b^T x,
    b = grad phi(x*) + lam * z,   z_i = sign(x*_i) on the support, z_i ~ U(-0.8, 0.8) off it.
Then grad f(x*) = -lam z, so 0 in grad f(x*) + lam * d||x*||_1 and x* is the (unique, since
B >= I makes f strongly convex) minimiser of F = f + lam ||.||_1 over the box. Off-support
|grad f_i(x*)| <= 0.8 lam: a controlled margin of at least 0.2 lam to the L1 threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prox import BoxL1, Regularizer


def logcosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2 * ax)) - np.log(2.0)


@dataclass
class ProblemA:
    d: int
    s: int = 10
    lam: float = 0.1
    cond: float = 100.0
    seed: int = 0
    lo: float = -1.0
    hi: float = 1.0
    xstar_range: tuple = (0.2, 0.8)  # |x*_i| on the support (kept away from 0 on purpose)
    B: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    xstar: np.ndarray = field(init=False, repr=False)
    h: Regularizer = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        d = self.d
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        eig = np.exp(np.linspace(0.0, np.log(self.cond), d))  # spectrum in [1, cond]
        self.B = (Q * eig) @ Q.T
        self.B = 0.5 * (self.B + self.B.T)
        supp = rng.choice(d, size=self.s, replace=False)
        xs = np.zeros(d)
        xs[supp] = rng.choice([-1.0, 1.0], size=self.s) * rng.uniform(*self.xstar_range, size=self.s)
        self.xstar = xs
        z = rng.uniform(-0.8, 0.8, size=d)
        z[supp] = np.sign(xs[supp])
        self.b = self._grad_phi(xs) + self.lam * z
        self.h = BoxL1(lo=self.lo, hi=self.hi, lam=self.lam)
        self.L = float(eig.max() + 1.0)  # Lipschitz const of grad f (log cosh'' <= 1)

    # ------------------------------------------------------------------ f, h, F, grad
    def _grad_phi(self, x):
        return x @ self.B + np.tanh(x)

    def f(self, x):
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * np.einsum("...i,ij,...j->...", x, self.B, x) + logcosh(x).sum(-1) - x @ self.b

    def grad_f(self, x):
        return self._grad_phi(np.asarray(x, dtype=np.float64)) - self.b

    def hess_f(self, x):
        """True Hessian of f (used only to log the GP Hessian error, never by a method)."""
        return self.B + np.diag(1.0 - np.tanh(np.asarray(x, dtype=np.float64)) ** 2)

    def F(self, x):
        x = np.asarray(x, dtype=np.float64)
        return self.f(x) + self.lam * np.abs(x).sum(-1)

    @property
    def Fstar(self):
        return float(self.F(self.xstar))

    @property
    def support(self):
        return self.xstar != 0

    # ------------------------------------------------------------------ verification
    def fista(self, x0=None, iters=20000, tol=1e-12):
        """Exact-gradient FISTA on F; returns (x, prox-gradient-mapping norm)."""
        x = np.zeros(self.d) if x0 is None else np.array(x0, dtype=np.float64)
        y, t, step = x.copy(), 1.0, 1.0 / self.L
        for _ in range(iters):
            xn = self.h.prox(y - step * self.grad_f(y), step)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = xn + (t - 1) / tn * (xn - x)
            if np.linalg.norm(xn - x) < tol:
                x = xn
                break
            x, t = xn, tn
        gm = np.linalg.norm(x - self.h.prox(x - step * self.grad_f(x), step)) / step
        return x, gm

    def verify(self):
        """Check x* numerically: FISTA from 0 must land on x* and the gradient mapping at x* vanish."""
        x, _ = self.fista()
        gm_star = np.linalg.norm(self.xstar - self.h.prox(self.xstar - self.grad_f(self.xstar) / self.L, 1 / self.L)) * self.L
        return {
            "fista_dist_to_xstar": float(np.linalg.norm(x - self.xstar)),
            "fista_F_gap": float(self.F(x) - self.Fstar),
            "grad_mapping_at_xstar": float(gm_star),
            "support_match": bool(np.array_equal(x != 0, self.support)),
            "min_offsupport_margin": float(self.lam - np.abs(self.grad_f(self.xstar)[~self.support]).max()),
        }
