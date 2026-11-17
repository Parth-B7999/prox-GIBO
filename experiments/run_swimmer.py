"""Swimmer case study (GIBO + NeST-BO benchmark) with and without an L1 term.

    python experiments/run_swimmer.py --lams 0 2 --seeds 0 1 2 3 4 --budget 300 --workers 10

One NPZ + JSON per run in results/swimmer_v2/lam{lam}/{method}_seed{s} (or --out). Existing runs are skipped.
All methods start at x0 = 0 (NeST-BO's Swimmer config: start_point center). The episode is
deterministic, so seeds only change the algorithms' own randomness (acquisition starts, Sobol init).
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = ROOT / "results" / "swimmer_v2"  # v2: every evaluation logged (best-found metric)


def job(args):
    method, lam, seed, budget, out_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from prox_bo.optimizers import PRESETS, run
    from prox_bo.swimmer import SwimmerL1, run_nestbo_swimmer

    d = Path(out_dir) / f"lam{lam:g}"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"{method}_seed{seed}"
    if stem.with_suffix(".json").exists():
        return f"skip {stem.name}"
    p = SwimmerL1(lam=lam)
    x0 = np.zeros(p.d)
    t0 = time.time()
    try:
        if method == "nestbo":
            out = run_nestbo_swimmer(lam, x0, budget, seed)
        else:
            out = run(p, PRESETS[method], x0, budget, 0.0, seed)
    except Exception:
        stem.with_suffix(".err").write_text(traceback.format_exc())
        return f"FAIL {stem.name}"
    rt = time.time() - t0
    X = out["X"]
    rew = SwimmerL1(lam=0.0).reward(X)  # episode return of every iterate (deterministic)
    out["reward"] = rew
    out["nnz"] = (X != 0).sum(1)
    # best point found among ALL evaluated points (the usual BO metric, used e.g. by NeST-BO)
    eF, eZ = out["eval_F"], out["eval_Z"]
    out["bsf_F"] = np.minimum.accumulate(eF)  # best-so-far F after 1, 2, ... evaluations
    ib = int(np.argmin(eF))
    zb = eZ[ib]
    np.savez_compressed(stem.with_suffix(".npz"), **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    js = {"method": method, "lam": lam, "seed": seed, "budget": budget, "runtime_s": rt,
          "final_F": float(out["gap"][-1]), "final_reward": float(rew[-1]), "final_nnz": int(out["nnz"][-1]),
          "final_l1": float(np.abs(X[-1]).sum()), "best_F_iterate": float(np.min(out["gap"])),
          "n_iterations": int(len(X) - 1), "x_final": X[-1].tolist(),
          "best_F_observed": float(eF[ib]), "reward_at_best": float(-(eF[ib] - lam * np.abs(zb).sum())),
          "nnz_at_best": int((zb != 0).sum()), "n_evals": int(len(eF)), "x_best": zb.tolist()}
    stem.with_suffix(".json").write_text(json.dumps(js, indent=1))
    return f"done {stem.parent.name}/{stem.name} {rt:.0f}s F={js['final_F']:.1f} R={js['final_reward']:.1f} nnz={js['final_nnz']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--methods", nargs="+",
                    default=["gibo", "proj_gibo", "proj_gibo_ls", "prox_gibo", "margin_prox_gibo", "nestbo"])
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    jobs = [(m, l, s, a.budget, a.out) for m in sorted(a.methods, key=lambda m: m != "nestbo")  # slow ones first
            for l in a.lams for s in a.seeds]
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(job, jobs):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
