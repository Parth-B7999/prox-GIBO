"""Local derivative GP (SE-ARD kernel) with weighted gradient-information batch selection.

* Hyperparameters are fitted with the NeST-BO repo's `ExactGPSEModel` (imported unchanged from
  baselines/nest_bo/src/model.py) and BoTorch's `fit_gpytorch_mll`.
* Everything derived from the hyperparameters (posterior gradient mean / variance, posterior
  mean, the greedy acquisition) is computed here in closed form, because the repo model only
  exposes the posterior *mean* gradient and we also need per-coordinate gradient variances.

All inputs live in the original x-space (a box of width `width`); lengthscale bounds and the
initial lengthscale are the NeST-BO unit-box values scaled by `width`.

Acquisition ("weighted GI"): greedily choose z to maximise the reduction of
    sum_i w_i Var_post[ g_i(x) ],
which with w = 1 is exactly GIBO's trace-of-gradient-covariance criterion. For one extra point z
(rank-one update of K^{-1}), with a = k(X, z), s = k(z,z) + sigma_n^2 - a^T K^{-1} a,
u = dk(x,z)/dx - (G K^{-1}) a   (G = dk(x,X)/dx), the reduction is  sum_i w_i u_i^2 / s.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import botorch
import gpytorch
import numpy as np
import torch
from gpytorch.priors import GammaPrior, NormalPrior
from scipy.optimize import minimize

_NEST_MODEL = Path(__file__).resolve().parents[1] / "baselines" / "nest_bo" / "src" / "model.py"
_spec = importlib.util.spec_from_file_location("nestbo_model", _NEST_MODEL)
_nest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nest)
ExactGPSEModel = _nest.ExactGPSEModel

DT = torch.float64


class GradGP:
    def __init__(self, d: int, width: float = 2.0, window: Optional[int] = None, use_priors: bool = True):
        # use_priors: Gamma(3,3) on the lengthscale (in unit-box units) and N(2,1) on the outputscale,
        # the priors the LCBO paper uses for its real-world benchmarks (Appendix E).
        self.d, self.width, self.window, self.use_priors = d, width, window, use_priors
        self.X = np.empty((0, d))
        self.y = np.empty(0)
        self.ls = torch.full((d,), 0.6931 * width, dtype=DT)
        self.os = torch.tensor(1.0, dtype=DT)
        self.noise = torch.tensor(1e-2, dtype=DT)
        self.y_mean, self.y_std = 0.0, 1.0
        self.fit_ok = True

    # ------------------------------------------------------------------ data
    def add(self, X: np.ndarray, y: np.ndarray):
        self.X = np.vstack([self.X, np.atleast_2d(X)])
        self.y = np.concatenate([self.y, np.atleast_1d(y)])
        if self.window is not None and len(self.y) > self.window:
            self.X, self.y = self.X[-self.window:], self.y[-self.window:]

    def _standardize(self):
        if len(self.y) > 1:
            self.y_mean = float(self.y.mean())
            self.y_std = max(float(self.y.std()), 1e-12)
        else:
            self.y_mean, self.y_std = 0.0, 1.0

    # ------------------------------------------------------------------ fitting
    def fit(self):
        """Refit hyperparameters (warm-started) by marginal likelihood, then refresh caches."""
        self._standardize()
        if len(self.y) < 2:
            # One point cannot be standardised: MAP then explains y (~F(x0), large) as pure noise,
            # the outputscale collapses to ~1e-319 and warm starts never recover (posterior gradient
            # exactly 0 -> GIBO never moved; found 2026-11-15). Keep the initial hypers instead.
            self.refresh()
            return
        Xt = torch.as_tensor(self.X, dtype=DT)
        yt = torch.as_tensor((self.y - self.y_mean) / self.y_std, dtype=DT).unsqueeze(-1)
        lo, hi = 0.005 * self.width, 10.0 * self.width
        kw = {}
        if self.use_priors:
            kw = dict(lengthscale_prior=GammaPrior(3.0, 3.0 / self.width),
                      outputscale_hyperprior=NormalPrior(2.0, 1.0))
        model = ExactGPSEModel(
            Xt, yt, lengthscale_constraint=gpytorch.constraints.Interval(lo, hi), ard_num_dims=self.d, **kw
        ).double()
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = self.ls.clamp(lo * 1.01, hi * 0.99).view(1, -1)
            model.covar_module.outputscale = self.os
            model.likelihood.noise = self.noise.clamp(min=2e-4)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        try:
            botorch.fit.fit_gpytorch_mll(mll)
            self.fit_ok = True
        except Exception:  # keep previous hypers (NeST-BO does the same)
            self.fit_ok = False
        with torch.no_grad():
            self.ls = model.covar_module.base_kernel.lengthscale.detach().view(-1).clone()
            self.os = model.covar_module.outputscale.detach().clone()
            self.noise = model.likelihood.noise.detach().view(()).clone()
        self.refresh()

    def refresh(self):
        """Recompute K^{-1}, alpha from current data and hypers (no hyperparameter fit)."""
        self._Xt = torch.as_tensor(self.X, dtype=DT)
        ys = torch.as_tensor((self.y - self.y_mean) / self.y_std, dtype=DT)
        K = self.kern(self._Xt, self._Xt) + self.noise * torch.eye(len(ys), dtype=DT)
        L = torch.linalg.cholesky(K)
        self.Kinv = torch.cholesky_inverse(L)
        self.alpha = self.Kinv @ ys

    # ------------------------------------------------------------------ kernel & posterior
    def kern(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a, b = A / self.ls, B / self.ls
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return self.os * torch.exp(-0.5 * d2.clamp_min(0.0))

    def _G(self, x: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """G[i, j] = d k(x, X_j) / d x_i  -> (d, N)."""
        k = self.kern(x[None], X)  # (1, N)
        return ((X - x) / self.ls ** 2).T * k

    def post_mean(self, Xq: np.ndarray) -> np.ndarray:
        Xq = torch.as_tensor(np.atleast_2d(Xq), dtype=DT)
        return self.y_mean + self.y_std * (self.kern(Xq, self._Xt) @ self.alpha).numpy()

    def post_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance (per coordinate) of grad f(x), in the units of f."""
        xt = torch.as_tensor(x, dtype=DT)
        G = self._G(xt, self._Xt)
        mean = self.y_std * (G @ self.alpha)
        var = self.y_std ** 2 * (self.os / self.ls ** 2 - ((G @ self.Kinv) * G).sum(1))
        return mean.numpy(), var.clamp_min(1e-18).numpy()

    def post_hess(self, x: np.ndarray) -> np.ndarray:
        """Hessian of the posterior mean at x, (d, d), in the units of f.

        For the SE kernel, d2 k(x, X_j)/dx dx^T = k_j (r_j r_j^T - diag(1/ls^2)), r_j = (X_j - x)/ls^2.
        """
        xt = torch.as_tensor(x, dtype=DT)
        ka = self.kern(xt[None], self._Xt)[0] * self.alpha  # (N,)  k_j * alpha_j
        R = (self._Xt - xt) / self.ls ** 2                    # (N, d)
        H = (R.T * ka) @ R - torch.diag(ka.sum() / self.ls ** 2)
        H = self.y_std * 0.5 * (H + H.T)
        return H.numpy()

    # ------------------------------------------------------------------ acquisition
    def select_batch(self, x: np.ndarray, w: np.ndarray, lo: np.ndarray, hi: np.ndarray, b: int,
                     eps: Optional[float] = None, n_raw: int = 64, n_restarts: int = 5,
                     rng: Optional[np.random.Generator] = None):
        """Greedily pick up to `b` points in [lo, hi] maximising the weighted gradient-variance
        reduction at x. Stops early after a point whose gain < eps (GIBO's rule; the point is
        still returned, as in GIBO). Returns (Z, gains) with gains in standardised y-units.
        """
        rng = rng or np.random.default_rng()
        d = self.d
        xt = torch.as_tensor(x, dtype=DT)
        wt = torch.as_tensor(w, dtype=DT)
        Xa = self._Xt
        Kinv = self.Kinv
        R = self._G(xt, Xa) @ Kinv  # (d, N)
        kzz = self.os + self.noise
        Z_out, gains = [], []

        def gain(Z):  # Z: (m, d) tensor
            a = self.kern(Z, Xa)  # (m, N)
            gz = ((Z - xt) / self.ls ** 2) * self.kern(xt[None], Z).T  # (m, d)
            u = gz - a @ R.T
            s = kzz - ((a @ Kinv) * a).sum(1)
            return (wt * u ** 2).sum(1) / s.clamp_min(1e-12)

        lo_t, hi_t = torch.as_tensor(lo, dtype=DT), torch.as_tensor(hi, dtype=DT)
        for i in range(b):
            Z0 = lo + (hi - lo) * rng.random((n_raw, d))
            with torch.no_grad():
                g0 = gain(torch.as_tensor(Z0, dtype=DT)).numpy()
            starts = Z0[np.argsort(-g0)[:n_restarts]]
            best_z, best_g = starts[0], g0.max()
            for z0 in starts:
                def fun(z):
                    zt = torch.as_tensor(z, dtype=DT).requires_grad_(True)
                    val = -gain(zt[None])[0]
                    val.backward()
                    return val.item(), zt.grad.numpy().copy()
                try:
                    r = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                                 options={"maxiter": 100})
                    if -r.fun > best_g:
                        best_z, best_g = r.x, -r.fun
                except Exception:
                    pass
            z = torch.as_tensor(np.clip(best_z, lo, hi), dtype=DT)
            # rank-one update of K^{-1} and R = G K^{-1}
            a = self.kern(z[None], Xa)[0]
            v = Kinv @ a
            s = float(kzz - a @ v)
            gz = ((z - xt) / self.ls ** 2) * self.kern(xt[None], z[None])[0, 0]
            u = gz - R @ a
            n = Xa.shape[0]
            Kn = torch.empty(n + 1, n + 1, dtype=DT)
            Kn[:n, :n] = Kinv + torch.outer(v, v) / s
            Kn[:n, n] = -v / s
            Kn[n, :n] = -v / s
            Kn[n, n] = 1.0 / s
            Rn = torch.empty(self.d, n + 1, dtype=DT)
            Rn[:, :n] = R - torch.outer(u, v) / s
            Rn[:, n] = u / s
            Kinv, R, Xa = Kn, Rn, torch.cat([Xa, z[None]])
            Z_out.append(z.numpy())
            g_i = float((wt * u ** 2).sum() / s)
            gains.append(g_i)
            if eps is not None and i >= 1 and g_i < eps:
                break
        return np.array(Z_out), np.array(gains)
