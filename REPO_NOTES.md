# Baseline repositories: where things live and their defaults

Last checked 2026-11-14. Paths are relative to `baselines/`.

## Repos and versions

- **GIBO**, original: `gibo/`, from github.com/sarmueller/gibo, MIT license, HEAD `4671302` (2022-10-27).
  - The copy at github.com/Data-Science-in-Mechanical-Engineering/gibo was at the same commit with an identical tree, so neither is newer. I deleted it.
- **NeST-BO**: `nest_bo/`, from github.com/PaulsonLab/NeST-BO, HEAD `1ab626c` (2026-10-09).
  - **It contains no GIBO implementation.** A grep for "gibo" finds nothing, and the only algorithms are `nestbo` and `nestbo_sub`. So the planned check of "NeST-BO's GIBO vs original GIBO" doesn't exist as a comparison.
  - Instead, we wrote a GIBO on the BoTorch stack in `prox_bo/` (presets `gibo` and `gibo_paper`).
- **LCBO**: no public code. No link in the arXiv abstract page or the HTML paper, and a web search found none. It's reimplemented in `prox_bo/optimizers.py`; see `LCBO_NOTES.md`.

## GIBO, original (`gibo/`)

| What | Where |
|---|---|
| GP model | `src/model.py`: `DerivativeExactGPSEModel`, an ExactGP with SE kernel. `append_train_data` keeps the newest N_max points and standardizes y. `posterior_derivative` gives the gradient mean and covariance. |
| GI acquisition | `src/acquisition_function.py`: `GradientInformation`, the explained trace of the gradient covariance after adding θ (one-step Cholesky update). In its loop, a gain below ε stops the batch. |
| Acquisition optimization | `optimize_acqf_custom_bo`: BoTorch `optimize_acqf`, q = 1, 5 restarts, 64 raw samples, batch_limit 5, within [x_t − δ, x_t + δ]. |
| Step | `src/optimizers.py` `BayesianGradientAscent.step`: gradient normalized by its L2 norm, times the lengthscales, then a torch SGD step (lr). No line search. |
| Box projection | none. The search box is only for the acquisition. |
| Hyperparameters | fit by marginal likelihood only if `optimize_hyperparameters` is set and N ≥ N_max. The synthetic experiments use fixed true hyperparameters. |

Defaults:

| Setting | Synthetic paper config (`configs/synthetic_experiment/100621/gibo.yaml`) | `gibo_default.yaml` | RL default (`configs/rl_experiment/gibo_default.yaml`) |
|---|---|---|---|
| Batch size (`max_samples_per_iteration`) | d | d | 16 |
| Early-stop ε | none (no early stop) | 0.1 | 0.01 |
| lr | 0.25 | 0.25 | 0.5 |
| N_max (window) | 5d | 5d | 32 |
| δ | 0.2 | 0.2 | 0.1 |
| Hyperparameters | fixed to the true values (`set_hypers: True`) | | |
| Kernel | SE-ARD, prior mean 0 | | |
| Objective calls | 300 | | |

**Environment.**
- The repo pins Python 3.8.5, torch 1.6, botorch 0.3.1, gpytorch 1.2, gym 0.9.3 and mujoco-py 0.5.7. Torch 1.6 has no native macOS arm64 wheel.
- For the synthetic experiments only (`generate_data_synthetic_functions.py`, `run_synthetic_experiment.py`), a Python 3.8 environment with torch about 1.12 plus small compatibility patches, or an x86 (Rosetta) conda environment, should work. **Not done yet.**
- **MuJoCo (RL experiments)** would need mujoco-py 0.5.7 with MuJoCo Pro 1.31: an old binary, x86 Linux or macOS-x86, a license key file and glfw. The practical route is an x86 Linux Docker or CHTC container image. Gymnasium with the new `mujoco` bindings would not reproduce the original exactly.

## NeST-BO (`nest_bo/`)

| What | Where |
|---|---|
| GP model | `src/model.py`: `ExactGPSEModel` (zero mean, ScaleKernel(RBF-ARD), lengthscale initialized to 0.6931) and `DerivativeExactGPSEModel` with `posterior_derivative` (mean only) and `posterior_hessian`. |
| NeST acquisition | `src/Acquisition_NeSTBO.py`: `NewtonInformation`, the reduction of trace(gradient covariance) plus the summed Hessian variances. It builds (D, D, N) tensors per candidate, so the cost is O(D²N). |
| Acquisition optimization | `optimize_acqf_custom_bo`: BoTorch `optimize_acqf`, q = 1, 5 restarts, `raw_samples` (256 by default), batch_limit 5, maxiter 300, within [x_t ± δ] clipped to [0,1]. |
| GP fitting | `src/optimization_loop_NeSTBO.py`: a fresh GP each iteration on all data. Lengthscale constrained to Interval(0.005, 10), default noise floor 1e-4, y standardized. `fit_gpytorch_mll` runs before and after the inner loop. |
| Step / line search | `move_Newton`: if the Hessian is positive definite, a Newton step with Armijo backtracking on the GP mean (σ = 0.1, s₀ = 1, β = 0.8, s > 0.05). Otherwise `move_GD`: a normalized gradient step (optionally times the lengthscales) with Armijo (s₀ = 0.5, β = 0.5). |
| Box projection | clamp to [0,1] in unit-box coordinates after each step. |
| NeST-BO-sub | `src/optimization_loop_NeSTBO_sub.py`, a random nested subspace (BAxUS-style). Not read in detail. |

Defaults (from `configs/benchmark/*.yaml` and `configs/default.yaml`):
- M = d points per iteration (e.g. 20 for 20-d Ackley).
- N_init = 10 Sobol points.
- δ = 0.2 in the unit box.
- `start_point: random`, `device: cuda` (use `device=cpu` on a Mac), `raw_samples: 256`.
- The README says `main.py` does not save results; our adapter (`prox_bo/nest_adapter.py`) recovers the iterates from the returned data.

## Our GIBO on the BoTorch stack, compared with the original

`prox_bo/gp.py` and `prox_bo/optimizers.py` (presets `gibo`, `gibo_paper`):
- **Same as the original:** GI criterion (trace of gradient covariance, exact rank-one update), greedy q = 1 selection inside ±δ, the original's ε stop rule, normalized step lr·ℓ⊙ĝ/‖ĝ‖, window 5d.
- **Different from the original:**
  - Hyperparameters are fit by MAP (LCBO's Gamma(3,3) and N(2,1) priors) every iteration, not fixed to true values.
  - Iterates are clipped to the domain box.
  - The acquisition is optimized with scipy L-BFGS-B from the best 5 of 64 random starts, not BoTorch's `optimize_acqf`. The heuristic is the same.
  - x-space is [-1,1]^d, and δ and the lengthscale bounds are scaled by the box width. This is equivalent to the unit box.
- **Not yet done:** a numerical cross-check against the original repo on its synthetic functions, which needs the old environment.
