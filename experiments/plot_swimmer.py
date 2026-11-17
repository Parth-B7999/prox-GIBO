"""Figure + table for the Swimmer case study (results/swimmer_v2).

    python experiments/plot_swimmer.py  -> results/swimmer_v2/figs/swimmer.png, results/swimmer_v2/summary.{md,json}

Two metrics are reported for every method:
  * best found: the lowest F among ALL evaluated points so far. This is the metric of the NeST-BO paper
    ("final best-found objective", median (IQR) over 10 replicates).
  * current iterate: F (and reward, number of nonzero gains) of the point the method currently proposes.
Curves: median over seeds, 25-75% bands.
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
from plot_demo import INK, on_grid  # noqa: E402

OUT = ROOT / "results/swimmer_v2"
STYLE = {  # same colours as the demo figures for the same entities
    "margin_prox_gibo_ns": ("Margin-aware prox-GIBO (ours)", "#2a78d6", "-", "o"),
    "prox_gibo_ns": ("Prox-GIBO (ours)", "#eb6834", "-", "s"),
    "gibo": ("GIBO", "#e87ba4", ":", "D"),
    "proj_gibo": ("LCBO-style projected GIBO", "#eda100", "--", "v"),
    "nestbo": ("NeST-BO", "#008300", ":", "P"),
}
STUCK = 100.0  # final-iterate reward below this = the known Swimmer local optimum (~40-50)


def load(lam):
    runs = {}
    for js in sorted((OUT / f"lam{lam:g}").glob("*.json")):
        meta = json.loads(js.read_text())
        runs.setdefault(meta["method"], []).append((meta, dict(np.load(js.with_suffix(".npz")))))
    return runs


def band(Y):
    return np.nanmedian(Y, 0), np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)


def med_iqr(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.1f} ({np.percentile(v, 75) - np.percentile(v, 25):.1f})"


def main():
    lams = sorted(float(p.name[3:]) for p in OUT.glob("lam*") if any(p.glob("*.json")))
    figs = OUT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    lines = ["| λ | method | seeds | **best found** F, median (IQR) | reward at best found | nonzero gains at best found | "
             "final-iterate F, median (IQR) | final-iterate reward | final-iterate nonzero gains | stuck (final reward < 100) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summ = []
    fig, axes = plt.subplots(len(lams), 3, figsize=(15, 4.0 * len(lams)), squeeze=False)
    for row, lam in zip(axes, lams):
        runs = load(lam)
        budget = max(m["budget"] for v in runs.values() for m, _ in v)
        grid = np.arange(1, budget + 1)
        for m, (lab, col, ls, mk) in STYLE.items():
            if m not in runs:
                continue
            R = runs[m]
            curves = [
                np.array([on_grid(np.arange(1, len(r["bsf_F"]) + 1), r["bsf_F"], grid) for _, r in R]),
                np.array([on_grid(r["evals"], r["gap"], grid) for _, r in R]),
                np.array([on_grid(r["evals"], r["nnz"].astype(float), grid) for _, r in R]),
            ]
            for ax, Y in zip(row, curves):
                med, lo, hi = band(Y)
                ax.plot(grid, med, color=col, ls=ls, marker=mk, markevery=len(grid) // 6, ms=6, label=lab)
                ax.fill_between(grid, lo, hi, color=col, alpha=0.12, lw=0)
            bF = [j["best_F_observed"] for j, _ in R]
            rb = [j["reward_at_best"] for j, _ in R]
            nb = [j["nnz_at_best"] for j, _ in R]
            fF = [j["final_F"] for j, _ in R]
            rw = [j["final_reward"] for j, _ in R]
            nz = [j["final_nnz"] for j, _ in R]
            lines.append(f"| {lam:g} | {lab} | {len(R)} | {med_iqr(bF)} | {np.median(rb):.1f} | {np.median(nb):.0f} | "
                         f"{med_iqr(fF)} | {np.median(rw):.1f} | {np.median(nz):.0f} | {int((np.array(rw) < STUCK).sum())}/{len(R)} |")
            summ.append({"lam": lam, "method": m, "best_F_observed": bF, "reward_at_best": rb, "nnz_at_best": nb,
                         "final_F": fF, "final_reward": rw, "final_nnz": nz})
        name = "original benchmark (λ = 0)" if lam == 0 else f"sparse controller (λ = {lam:g})"
        row[0].set_title(f"Best F found so far — {name}", loc="left", color=INK, fontsize=10)
        row[1].set_title("F at the current iterate", loc="left", color=INK, fontsize=10)
        row[2].set_title("Nonzero gains of the current iterate (of 16)", loc="left", color=INK, fontsize=10)
        row[0].set_ylabel("F = −reward + λ‖x‖₁  (lower is better)")
        row[1].set_ylabel("F (lower is better)")
        row[2].set_ylabel("nonzero gains")
        row[2].set_ylim(0, 17)
        for ax in row:
            ax.set_xlabel("episodes (function evaluations)")
    h, l = axes[-1, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(figs / "swimmer.png", dpi=160, bbox_inches="tight")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
