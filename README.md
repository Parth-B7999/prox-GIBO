# prox-GIBO: proximal local Bayesian optimization for composite black-box problems

This repo is for problems of the form

```
min_x  F(x) = f(x) + h(x)
```

- `f` is smooth but expensive. It can only be queried through noisy function values.
- `h` is a known convex term, possibly non-smooth: `λ‖x‖₁`, group lasso, or a box constraint.

**Prox-GIBO** combines two pieces:
- a Gaussian-process estimate of `∇f`, chosen as in GIBO (Müller et al., NeurIPS 2021) to minimize the posterior uncertainty of the gradient;
- the proximal step of `h`, so `x⁺ = prox_{t h}(x − t ĝ)`, i.e. ISTA with an estimated gradient.

**Margin-aware prox-GIBO** also stops spending evaluations on coordinates that the optimality condition of `h` already certifies as settled at zero (`x_i = 0` and `|μ_g,i| + κ σ_g,i < λ`).

The technical note [`report/ProxBO_note_v2.pdf`](report/ProxBO_note_v2.pdf) covers:
- the proximal-algorithm background;
- extensions to other non-smooth terms (sums, `h(Dx)`, nonconvex penalties, non-smooth `f`);
- related work;
- all results.

## Results

| Setting | Margin-aware prox-GIBO | Best black-box baseline (GIBO, LCBO-style, NeST-BO, projected GIBO + line search) |
|---|---|---|
| Synthetic, d = 50 / 100 / 200 (known `x*`, s = 10) | final gap 0.0029 / 0.0021 / 0.020, exact support | 0.83 / 2.52 / 7.02, never sparse |
| Swimmer (GIBO/NeST-BO benchmark), `F = −return + 2‖x‖₁`, 10 replicates | best-found F −318 (median); final controller uses 11–12 of 16 gains | best-found F: GIBO −271, NeST-BO −243 (untuned, 300 episodes); all 16 gains |

Swimmer has two versions:
- **λ = 0, the original benchmark:** prox-GIBO is the same algorithm as GIBO. Both reach a best-found reward of about 337.
- **λ = 2, the sparse controller:** the λ = 2 row in the table above.

Swimmer results are reported with two metrics:
- the best point found, which is the metric of the NeST-BO paper;
- the final iterate.

The note discusses how our NeST-BO numbers relate to those in its paper.

The open question is the cost scaling in d. With the current settling test, the cost still grows linearly in d. An oracle with the true zero set is nearly flat in d. The details are in the note.

## Layout

```
prox_bo/            library code
  prox.py           proximal operators (L1+box, group lasso+box)
  problems.py       synthetic problem A with known solution (verified by FISTA)
  gp.py             derivative GP (gradient mean/covariance, GP-mean Hessian), weighted gradient-information acquisition
  optimizers.py     one loop, presets: gibo, gibo_paper, proj_gibo (LCBO-style), proj_gibo_ls, prox_gibo,
                    margin_prox_gibo, prox_gibo_ns, margin_prox_gibo_ns, proximal-Newton variants, oracle diagnostics
  nest_adapter.py   runs the unmodified NeST-BO loop on problem A
  swimmer.py        Swimmer-v5 linear policy with an L1 term, plus a NeST-BO runner
  tests/            pytest suite
experiments/        run scripts, YAML configs, plotting scripts
results/            per-run JSON summaries, summary tables and figures (raw NPZ arrays are not tracked)
report/             technical note (LaTeX source, bibliography, PDF)
LITERATURE.md       related-work survey notes
REPO_NOTES.md       notes on the baseline repositories and their defaults
```

## Setup

```bash
conda create -n nestbo python=3.11
conda activate nestbo
pip install -r requirements.txt

# baselines (not redistributed here), cloned next to the code:
git clone https://github.com/PaulsonLab/NeST-BO baselines/nest_bo && git -C baselines/nest_bo checkout 1ab626c
git clone https://github.com/sarmueller/gibo baselines/gibo        # reference only, not imported
```

`prox_bo/gp.py` imports the NeST-BO repository's `ExactGPSEModel`, and `prox_bo/nest_adapter.py` runs the NeST-BO loop unmodified. So `baselines/nest_bo` is required even for the tests.

## Reproducing

```bash
python -m pytest -q prox_bo/tests

# synthetic problem (d = 50, 100; 5 seeds; all methods incl. NeST-BO, which takes about 24 h per run at d = 50)
python experiments/run_demo.py experiments/config_demo.yaml
python experiments/run_demo.py experiments/config_d200.yaml           # d = 200
# also: config_noise0, config_stress_cond100, config_gibo_paper, config_diag_oracle,
#       config_newton_cond10, config_newton_cond100
python experiments/plot_demo.py results/demo
python experiments/plot_scaling.py
python experiments/compare_newton.py

# Swimmer (results/swimmer_v2)
python experiments/run_swimmer.py --lams 0 2 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --methods nestbo gibo proj_gibo prox_gibo_ns margin_prox_gibo_ns --workers 11
python experiments/plot_swimmer.py

# report
cd report && pdflatex ProxBO_note_v2 && bibtex ProxBO_note_v2 && pdflatex ProxBO_note_v2 && pdflatex ProxBO_note_v2
```

- All seeds are fixed: problem instance, starting point (`1000 + seed`), noise, and method.
- `run_demo.py` and `run_swimmer.py` skip runs whose results already exist, so they can be restarted.
- Runs use the CPU.
