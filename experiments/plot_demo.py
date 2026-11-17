"""Figures + summary table for the demo runs.

    python experiments/plot_demo.py results/demo

Curves: per run, the value at evaluation e is that of the latest iterate reached with <= e
evaluations (step function); lines are the median over seeds, bands the 25-75% range.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# fixed entity -> style mapping (reference categorical palette slots 1-6, fixed order);
# line style is a second channel so identity never depends on color alone
STYLE = {
    "oracle_margin_prox_gibo": ("Margin-aware, oracle zero set (diagnostic)", "#6b6a63", "-.", "*"),
    "margin_prox_gibo": ("Margin-aware prox GIBO (ours)", "#2a78d6", "-", "o"),
    "prox_gibo": ("Prox GIBO (ours)", "#eb6834", "-", "s"),
    "proj_gibo_ls": ("Projected GIBO + line search (black box)", "#1baf7a", "--", "^"),
    "proj_gibo": ("Projected GIBO / LCBO-style (black box)", "#eda100", "--", "v"),
    "gibo": ("GIBO, repo default ε=0.1 (black box)", "#e87ba4", ":", "D"),
    "gibo_paper": ("GIBO, paper config (black box)", "#6250d6", ":", "X"),
    "nestbo": ("NeST-BO (black box)", "#008300", ":", "P"),
}
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6a63", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(root: Path):
    runs = {}
    for js in sorted(root.glob("d*/*.json")):
        meta = json.loads(js.read_text())
        npz = np.load(js.with_suffix(".npz"))
        runs.setdefault((meta["d"], meta["method"]), []).append((meta, dict(npz)))
    return runs


def on_grid(evals, vals, grid):
    idx = np.searchsorted(evals, grid, side="right") - 1
    out = np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)
    return out


def band(runs, key, grid):
    Y = np.array([on_grid(r["evals"], r[key], grid) for _, r in runs])
    return np.nanmedian(Y, 0), np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)


def main(root):
    root = Path(root)
    runs = load(root)
    figs = root / "figs"
    figs.mkdir(exist_ok=True)
    dims = sorted({d for d, _ in runs})
    budget = max(m["budget"] for v in runs.values() for m, _ in v)
    grid = np.arange(0, budget + 1, 5)
    methods = [m for m in STYLE if any((d, m) in runs for d in dims)]

    for d in dims:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for m in methods:
            if (d, m) not in runs:
                continue
            lab, col, ls, mk = STYLE[m]
            for ax, key in zip(axes[:2], ["gap", "f1_exact"]):
                med, lo, hi = band(runs[(d, m)], key, grid)
                ax.plot(grid, med, color=col, ls=ls, label=lab, marker=mk, markevery=len(grid) // 6, ms=6)
                ax.fill_between(grid, lo, hi, color=col, alpha=0.12, lw=0)
            if m == "margin_prox_gibo":
                med, lo, hi = band(runs[(d, m)], "n_settled", grid)
                axes[2].plot(grid, med, color=col, label="settled (acquisition ignores)")
                axes[2].fill_between(grid, lo, hi, color=col, alpha=0.12, lw=0)
            if m in ("margin_prox_gibo", "prox_gibo"):
                Z = np.array([on_grid(r["evals"], (r["X"] == 0).sum(1), grid) for _, r in runs[(d, m)]])
                axes[2].plot(grid, np.nanmedian(Z, 0), color=col, ls="--", lw=1.5,
                             label=f"exact zeros in x ({STYLE[m][0].split(' (')[0]})")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("F(x_t) − F*  (median, 25–75% band)")
        axes[1].set_ylabel("support F1 (exact zeros)")
        axes[1].set_ylim(0, 1.05)
        axes[2].axhline(d - 10, color=MUTED, lw=1, ls=":")
        axes[2].text(grid[-1], d - 10, f" true zeros = {d - 10}", color=MUTED, va="bottom", ha="right")
        axes[2].set_ylabel("number of coordinates")
        for ax in axes:
            ax.set_xlabel("function evaluations")
        axes[0].set_title(f"Optimality gap, d = {d}", loc="left", color=INK)
        axes[1].set_title(f"Support recovery, d = {d}", loc="left", color=INK)
        axes[2].set_title("Zeros and settled coordinates", loc="left", color=INK)
        axes[2].legend(loc="lower right", fontsize=8.5)
        h, l = axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(figs / f"curves_d{d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # evaluations to reach tolerance vs d
    tols = sorted({float(t) for v in runs.values() for m, _ in v for t in m["evals_to_tol"]}, reverse=True)
    ptols = [t for t in tols if t >= 0.01]  # 1e-3 is reached by too few runs to plot (it is in the table)
    fig, axes = plt.subplots(1, len(ptols), figsize=(6 * len(ptols), 4.2), sharey=True)
    rows = []
    for ax, tol in zip(np.atleast_1d(axes), ptols):
        notes = []
        for m in methods:
            lab, col, ls, mk = STYLE[m]
            xs, ys, fr = [], [], []
            for d in dims:
                if (d, m) not in runs:
                    continue
                v = [r[0]["evals_to_tol"][str(tol)] for r in runs[(d, m)]]
                hit = [x for x in v if x is not None]
                xs.append(d)
                ys.append(np.median(hit) if len(hit) > len(v) / 2 else np.nan)  # median only if most seeds hit
                fr.append(f"{len(hit)}/{len(v)}")
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=8, label=lab)
            for x, y, f in zip(xs, ys, fr):
                if not np.isfinite(y):
                    notes.append(f"{lab.split(' (')[0]}, d={x}: {f} seeds")
            for x, y, f in zip(xs, ys, fr):
                if np.isfinite(y):
                    ax.annotate(f, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
        if notes:
            ax.text(0.02, 0.98, "not reached by most seeds:\n" + "\n".join(notes), transform=ax.transAxes,
                    va="top", fontsize=8, color=MUTED)
        ax.set_title(f"Evaluations to reach F − F* ≤ {tol:g}", loc="left", color=INK)
        ax.set_xlabel("dimension d")
        ax.set_xticks(dims)
    np.atleast_1d(axes)[0].set_ylabel("median evaluations (seeds reaching it)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(figs / "evals_to_tol_vs_d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # summary table (markdown + json)
    lines = ["| d | method | seeds | final gap (median [min, max]) | final support F1, exact zeros (median) | "
             "F1 counting \\|x\\|>0.01 as nonzero | "
             + " | ".join(f"evals to ≤{t:g} (median; #seeds)" for t in tols) + " | runtime/run (s) |",
             "|---" * (7 + len(tols)) + "|"]  # 6 fixed columns + tolerances + runtime column
    for d in dims:
        for m in methods:
            if (d, m) not in runs:
                continue
            R = runs[(d, m)]
            g = np.array([r[0]["final_gap"] for r in R])
            f1 = np.median([r[0]["final_f1_exact"] for r in R])
            from prox_bo.metrics import support_f1
            from prox_bo.problems import ProblemA
            f1t = np.median([support_f1(np.abs(r[1]["X"][-1]) > 0.01,
                                        ProblemA(**r[0]["problem"]).support) for r in R])
            cells = []
            for t in tols:
                v = [r[0]["evals_to_tol"][str(t)] for r in R]
                hit = [x for x in v if x is not None]
                cells.append(f"{np.median(hit):.0f}; {len(hit)}/{len(v)}" if hit else f"—; 0/{len(v)}")
            rt = np.median([r[0]["runtime_s"] for r in R])
            lines.append(f"| {d} | {STYLE[m][0]} | {len(R)} | {np.median(g):.3g} [{g.min():.3g}, {g.max():.3g}] | "
                         f"{f1:.2f} | {f1t:.2f} | " + " | ".join(cells) + f" | {rt:.0f} |")
            rows.append({"d": d, "method": m, "final_gap": g.tolist(), "f1_median": float(f1)})
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/demo")
