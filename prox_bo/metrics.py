import numpy as np


def support_f1(est: np.ndarray, true: np.ndarray) -> float:
    est, true = np.asarray(est, bool), np.asarray(true, bool)
    tp = float((est & true).sum())
    if est.sum() + true.sum() == 0:
        return 1.0
    return 2.0 * tp / (est.sum() + true.sum())


def f1_curve(X: np.ndarray, true: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Support F1 for each iterate (rows of X); a coordinate is 'nonzero' if |x_i| > thr."""
    return np.array([support_f1(np.abs(x) > thr, true) for x in X])


def stable_iteration(X: np.ndarray, thr: float = 0.0) -> int:
    """First iteration index after which the estimated support never changes (len(X)-1 if it
    changes at the very last step)."""
    S = np.abs(X) > thr
    changes = np.where((S[1:] != S[:-1]).any(1))[0]
    return int(changes[-1] + 1) if len(changes) else 0


def evals_to_reach(evals: np.ndarray, gap: np.ndarray, tol: float) -> float:
    """First evaluation count at which F - F* <= tol (nan if never)."""
    idx = np.where(gap <= tol)[0]
    return float(evals[idx[0]]) if len(idx) else float("nan")
