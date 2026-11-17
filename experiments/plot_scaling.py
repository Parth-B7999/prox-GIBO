"""Scaling with d of the proximal variants, incl. the oracle-zero-set diagnostic.

    python experiments/plot_scaling.py   -> results/demo/figs/scaling_prox_variants.png (+ .json)

Per seed: evaluations to reach F - F* <= tol (seeds that never reach it within the budget are
drawn at the top as open markers, labelled "not reached"). Line = median over seeds that reached it,
only when most seeds reached it.
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
from plot_demo import INK, MUTED, STYLE  # noqa: E402  (same entity -> style mapping and rcParams)

SRC = {"oracle_margin_prox_gibo": "results/diag_oracle", "margin_prox_gibo": "results/demo",
       "prox_gibo": "results/demo"}
DIMS, SEEDS, TOLS = [50, 100, 200], range(5), ["0.1", "0.01"]


def main():
    data = {m: {d: [json.loads((ROOT / r / f"d{d}" / f"{m}_seed{s}.json").read_text()) for s in SEEDS]
                for d in DIMS} for m, r in SRC.items()}
    budget = data["prox_gibo"][50][0]["budget"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    offs = {"oracle_margin_prox_gibo": -6, "margin_prox_gibo": 0, "prox_gibo": 6}
    out = {}
    for ax, tol in zip(axes, TOLS):
        notes = []
        for m in SRC:
            lab, col, ls, mk = STYLE[m]
            med = []
            for d in DIMS:
                v = [j["evals_to_tol"][tol] for j in data[m][d]]
                hit = [x for x in v if x is not None]
                xs = d + offs[m] * d / 100
                ax.scatter([xs] * len(hit), hit, color=col, marker=mk, s=22, alpha=0.55, lw=0)
                nmiss = len(v) - len(hit)
                if nmiss:
                    ax.scatter([xs], [budget * 1.04], facecolors="none", edgecolors=col, marker=mk, s=40)
                    notes.append(f"{lab.split(' (')[0].replace('Margin-aware, oracle zero set', 'Oracle')}, "
                                 f"d = {d}: {nmiss}/{len(v)}")
                med.append(np.median(hit) if len(hit) > len(v) / 2 else np.nan)
                out.setdefault(m, {}).setdefault(tol, {})[d] = v
            ax.plot(DIMS, med, color=col, ls=ls, marker=mk, ms=8, label=lab)
        if notes:
            ax.text(0.98, 0.04, "open markers above budget = seeds not reaching it:\n" + "\n".join(notes),
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=8, color=MUTED)
        ax.axhline(budget, color=MUTED, lw=1, ls=":")
        ax.text(70, budget * 0.985, "budget", color=MUTED, va="top", ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(DIMS)
        ax.set_xticklabels([str(d) for d in DIMS])
        ax.minorticks_off()
        ax.set_xlabel("dimension d (s = 10 nonzeros throughout)")
        ax.set_title(f"Evaluations to reach F − F* ≤ {tol}", loc="left", color=INK)
    axes[0].set_ylabel("evaluations (dots = seeds, line = median)")
    axes[0].set_ylim(0, budget * 1.15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    figs = ROOT / "results/demo/figs"
    fig.savefig(figs / "scaling_prox_variants.png", dpi=160, bbox_inches="tight")
    (figs / "scaling_prox_variants.json").write_text(json.dumps(out, indent=1))
    print("wrote", figs / "scaling_prox_variants.png")


if __name__ == "__main__":
    main()
