"""Sanity check of the black-box baselines on the SMOOTH problem (lam = 0, so F = f):
if they converge here, their failure on lam = 1 comes from the L1 kink, not from a bug."""
import json, sys
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np

def job(args):
    m, lam = args
    import torch; torch.set_num_threads(1)
    from prox_bo.optimizers import PRESETS, run
    from prox_bo.problems import ProblemA
    p = ProblemA(d=50, s=10, lam=lam, cond=10.0, seed=0)
    x0 = np.random.default_rng(1000).uniform(-.5, .5, 50)
    out = run(p, PRESETS[m], x0, 1000, 0.01, seed=0)
    np.savez_compressed(ROOT / f"results/sanity/{m}_lam{lam}.npz", **out)
    g = out["gap"]; ev = out["evals"]
    return {"method": m, "lam": lam, "gap_at_500": float(g[ev <= 500][-1]), "final_gap": float(g[-1])}

if __name__ == "__main__":
    with Pool(3) as pool:
        res = pool.map(job, [(m, 0.0) for m in ["gibo", "proj_gibo", "proj_gibo_ls"]])
    (ROOT / "results/sanity/smooth_lam0.json").write_text(json.dumps(res, indent=1))
    for r in res: print(r)
