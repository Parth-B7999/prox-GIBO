"""Run the demo grid (methods x dims x seeds) in parallel; one NPZ + JSON per run.

    python experiments/run_demo.py experiments/config_demo.yaml [--only nestbo] [--dims 50]

Existing result files are skipped, so the script can be re-run to resume.
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
import yaml  # noqa: E402


def job(args):
    cfg, method, d, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from prox_bo.metrics import evals_to_reach, stable_iteration
    from prox_bo.optimizers import PRESETS, run
    from prox_bo.problems import ProblemA

    out_dir = ROOT / cfg["out_dir"] / f"d{d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{method}_seed{seed}"
    if stem.with_suffix(".json").exists():
        return f"skip {stem.name}"
    pc = cfg["problem"]
    prob_kw = dict(d=d, s=pc["s"], lam=pc["lam"], cond=pc["cond"], seed=seed)
    p = ProblemA(**prob_kw, lo=pc["lo"], hi=pc["hi"])
    x0 = np.random.default_rng(1000 + seed).uniform(-cfg["x0_halfwidth"], cfg["x0_halfwidth"], d)
    t0 = time.time()
    try:
        if method == "nestbo":
            from prox_bo.nest_adapter import run_nestbo
            nc = cfg["nestbo"]
            out = run_nestbo(prob_kw, x0, cfg["budget"], cfg["noise_std"], seed, M=nc["M"],
                             N_init=nc["N_init"], delta=nc["delta"], raw_samples=nc["raw_samples"])
        else:
            out = run(p, PRESETS[method], x0, cfg["budget"], cfg["noise_std"], seed)
    except Exception:
        (stem.with_suffix(".err")).write_text(traceback.format_exc())
        return f"FAIL {stem.name}"
    runtime = time.time() - t0
    np.savez_compressed(stem.with_suffix(".npz"), **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    summary = {
        "method": method, "d": d, "seed": seed, "runtime_s": runtime,
        "problem": prob_kw, "noise_std": cfg["noise_std"], "budget": cfg["budget"],
        "F0_gap": float(out["gap"][0]), "final_gap": float(out["gap"][-1]),
        "best_gap_of_iterates": float(np.min(out["gap"])),
        "final_f1_exact": float(out["f1_exact"][-1]),
        "final_nnz": int((out["X"][-1] != 0).sum()),
        "support_stable_from_iter": stable_iteration(out["X"]),
        "n_iterations": int(len(out["gap"]) - 1),
        "evals_to_tol": {str(t): (None if np.isnan(v) else v) for t in cfg["tolerances"]
                         for v in [evals_to_reach(out["evals"], out["gap"], t)]},
    }
    if method != "nestbo":
        summary["method_cfg"] = PRESETS[method].__dict__
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return f"done {stem.name} {runtime:.0f}s gap={summary['final_gap']:.3g} f1={summary['final_f1_exact']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dims", nargs="*", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    methods = a.only or cfg["methods"]
    dims = a.dims or cfg["dims"]
    # fast GP-family jobs first (results within minutes); slow NeST-BO jobs last
    jobs = [(cfg, m, d, s) for m in sorted(methods, key=lambda m: m == "nestbo")
            for d in sorted(dims, reverse=True) for s in cfg["seeds"]]
    (ROOT / cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["out_dir"] / f"config_used_{Path(a.config).stem}.yaml").write_text(yaml.safe_dump(cfg))
    with Pool(a.workers or cfg["workers"]) as pool:
        for msg in pool.imap_unordered(job, jobs):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
