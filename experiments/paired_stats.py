"""Paired per-seed statistics for prox vs margin-aware prox GIBO (and support stabilisation)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prox_bo.problems import ProblemA
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo")
for d in sorted(int(p.name[1:]) for p in root.glob("d*") if p.is_dir()):
    print(f"--- d={d}")
    for m in ["prox_gibo", "margin_prox_gibo"]:
        rows = []
        for js in sorted((root / f"d{d}").glob(f"{m}_seed*.json")):
            j = json.loads(js.read_text()); z = np.load(js.with_suffix(".npz"))
            it = j["support_stable_from_iter"]; ev_stable = int(z["evals"][it])
            correct = bool(np.array_equal(z["X"][-1] != 0, ProblemA(**j["problem"]).support))
            rows.append((j["seed"], j["evals_to_tol"]["0.01"], j["evals_to_tol"]["0.1"], ev_stable, correct, j["final_gap"]))
        for r in rows:
            print(f"{m:17s} seed{r[0]}: evals->1e-2={r[1]}  ->1e-1={r[2]}  support fixed from eval {r[3]} (final support exact: {r[4]})  final gap {r[5]:.3g}")
    a = {json.loads(p.read_text())["seed"]: json.loads(p.read_text())["evals_to_tol"]["0.01"] for p in (root / f"d{d}").glob("prox_gibo_seed*.json")}
    b = {json.loads(p.read_text())["seed"]: json.loads(p.read_text())["evals_to_tol"]["0.01"] for p in (root / f"d{d}").glob("margin_prox_gibo_seed*.json")}
    ratios = [(a[s] if a[s] is not None else np.inf) / b[s] for s in sorted(b) if s in a and b[s] is not None]
    print("ratio prox/margin evals to 1e-2 per seed (inf = prox never reached within budget):", [round(x, 2) for x in ratios])
