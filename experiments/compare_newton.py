"""Step 1 of the Newton follow-up: proximal Newton (GP-mean Hessian, and true-Hessian diagnostic)
vs the existing gradient-step prox methods, paired by seed, d = 50.

    python experiments/compare_newton.py   -> results/newton_summary.md, results/newton_curves.png (+ .json)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_demo import INK, MUTED, band  # noqa: E402

D, SEEDS = 50, range(5)
SETS = {10: {"ref": "results/demo", "new": "results/newton_cond10"},
        100: {"ref": "results/stress_cond100", "new": "results/newton_cond100"}}
METHODS = [  # (method, which dir, label, color, linestyle)
    ("prox_gibo", "ref", "Prox GIBO (gradient step)", "#eb6834", "-"),
    ("prox_newton_gibo", "new", "Prox Newton GIBO (GP-mean Hessian)", "#eb6834", "--"),
    ("margin_prox_gibo", "ref", "Margin-aware prox GIBO (gradient step)", "#2a78d6", "-"),
    ("margin_prox_newton_gibo", "new", "Margin-aware prox Newton (GP-mean Hessian)", "#2a78d6", "--"),
    ("oracle_hess_margin_prox_newton_gibo", "new", "Margin-aware prox Newton, TRUE Hessian (diagnostic)",
     "#6b6a63", "-."),
]
TOLS = ["0.1", "0.01", "0.001"]


def load(path, m, s):
    stem = ROOT / path / f"d{D}" / f"{m}_seed{s}"
    return json.loads(stem.with_suffix(".json").read_text()), dict(np.load(stem.with_suffix(".npz")))


def main():
    lines = ["| cond | method | final gap (median [min, max]) | F1 (median) | "
             + " | ".join(f"evals to ≤{t} (median; #seeds)" for t in TOLS)
             + " | Hessian rel. error (median over run) |", "|---" * (5 + len(TOLS)) + "|"]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    grid = np.arange(0, 1001, 5)
    for ax, (cond, dirs) in zip(axes, SETS.items()):
        for m, which, lab, col, ls in METHODS:
            R = [load(dirs[which], m, s) for s in SEEDS]
            g = np.array([j["final_gap"] for j, _ in R])
            f1 = np.median([j["final_f1_exact"] for j, _ in R])
            cells = []
            for t in TOLS:
                v = [j["evals_to_tol"][t] for j, _ in R]
                hit = [x for x in v if x is not None]
                cells.append(f"{np.median(hit):.0f}; {len(hit)}/{len(v)}" if hit else f"—; 0/{len(v)}")
            he = (f"{np.median([np.nanmedian(r['hess_err']) for _, r in R]):.2f}"
                  if "hess_err" in R[0][1] and "oracle" not in m else "—")
            lines.append(f"| {cond} | {lab} | {np.median(g):.3g} [{g.min():.3g}, {g.max():.3g}] | {f1:.2f} | "
                         + " | ".join(cells) + f" | {he} |")
            out.setdefault(str(cond), {})[m] = {"final_gap": g.tolist(),
                                                "evals_to_tol": [j["evals_to_tol"] for j, _ in R]}
            med, lo, hi = band(R, "gap", grid)
            ax.plot(grid, med, color=col, ls=ls, label=lab)
            ax.fill_between(grid, lo, hi, color=col, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_title(f"Optimality gap, d = {D}, cond = {cond}", loc="left", color=INK)
    axes[0].set_ylabel("F(x_t) − F*  (median over 5 seeds, 25–75% band)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(ROOT / "results/newton_curves.png", dpi=160, bbox_inches="tight")
    (ROOT / "results/newton_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results/newton_summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
