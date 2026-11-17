"""Proximal operators and composite regularizers h(x).

Convention: prox_{t h}(v) = argmin_z  h(z) + ||z - v||^2 / (2 t).
All functions work on numpy arrays (float64) and are vectorised over the
last axis, so a batch of points (n, d) can be passed directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def soft_threshold(v: np.ndarray, tau) -> np.ndarray:
    """prox of tau*||.||_1 : sign(v) * max(|v| - tau, 0). `tau` may be per-coordinate."""
    v = np.asarray(v, dtype=np.float64)
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def group_soft_threshold(v: np.ndarray, tau, groups: Sequence[np.ndarray]) -> np.ndarray:
    """prox of tau * sum_g ||v_g||_2 (non-overlapping groups; ungrouped coords are left alone).

    `tau` is a scalar or one value per group.
    """
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64), (len(groups),))
    for g, tg in zip(groups, taus):
        g = np.asarray(g)
        norm = np.linalg.norm(v[..., g], axis=-1, keepdims=True)
        scale = np.maximum(1.0 - tg / np.maximum(norm, 1e-300), 0.0)
        out[..., g] = v[..., g] * scale
    return out


def clip_box(v: np.ndarray, lo, hi) -> np.ndarray:
    """prox of the indicator of the box [lo, hi] = Euclidean projection."""
    return np.clip(np.asarray(v, dtype=np.float64), lo, hi)


def prox_l1_box(v: np.ndarray, tau, lo, hi) -> np.ndarray:
    """prox of tau*||.||_1 + indicator([lo, hi]).

    Exact for this separable case when lo <= 0 <= hi: clip(soft_threshold(v, tau)).
    (Per coordinate the objective is convex piecewise-quadratic with its kink at 0, and 0 lies
    in the box, so the unconstrained minimiser clipped to the box is the constrained one.)
    """
    return clip_box(soft_threshold(v, tau), lo, hi)


# ---------------------------------------------------------------- regularizer objects
@dataclass
class Regularizer:
    """Known convex term h with value() and prox(). `lam` is the L1 weight (0 if none)."""

    lo: float = -1.0
    hi: float = 1.0
    lam: float = 0.0

    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def prox(self, v: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    # threshold below which a coordinate counts as "exactly zero" for the black-box baselines
    @property
    def l1_weight(self) -> float:
        return self.lam


class BoxL1(Regularizer):
    """h(x) = lam*||x||_1 + indicator(box). With lam = 0 this is the plain box indicator."""

    def value(self, x):
        x = np.asarray(x)
        if np.any(x < self.lo - 1e-12) or np.any(x > self.hi + 1e-12):
            return np.inf
        return self.lam * float(np.abs(x).sum())

    def prox(self, v, t):
        return prox_l1_box(v, t * self.lam, self.lo, self.hi)


class BoxGroupLasso(Regularizer):
    """h(x) = lam * sum_g ||x_g||_2 + indicator(box), non-overlapping groups.

    The prox is solved per group: for a nonzero solution z, KKT gives
    z_i = clip(v_i * r / (r + tau), lo, hi) with r = ||z||, and r is found by bisection.
    If ||v_g|| <= tau the group is set to 0. Checked against a numerical solve in the tests.
    """

    def __init__(self, groups: Sequence[np.ndarray], lam: float, lo: float = -1.0, hi: float = 1.0):
        super().__init__(lo=lo, hi=hi, lam=lam)
        self.groups = [np.asarray(g) for g in groups]

    def value(self, x):
        x = np.asarray(x)
        if np.any(x < self.lo - 1e-12) or np.any(x > self.hi + 1e-12):
            return np.inf
        return float(self.lam * sum(np.linalg.norm(x[g]) for g in self.groups))

    def prox(self, v, t):
        out = np.asarray(v, dtype=np.float64).copy()
        tau = t * self.lam
        for g in self.groups:
            vg = out[g].copy()
            if np.linalg.norm(vg) <= tau:  # then z = 0 is optimal (box contains 0)
                out[g] = 0.0
                continue
            f = lambda r: r - np.linalg.norm(np.clip(vg * (r / (r + tau)), self.lo, self.hi))
            lo_r, hi_r = 0.0, np.linalg.norm(vg)
            for _ in range(200):
                mid = 0.5 * (lo_r + hi_r)
                (hi_r, lo_r) = (mid, lo_r) if f(mid) > 0 else (hi_r, mid)
            r = 0.5 * (lo_r + hi_r)
            out[g] = np.clip(vg * (r / (r + tau)), self.lo, self.hi)
        return out
