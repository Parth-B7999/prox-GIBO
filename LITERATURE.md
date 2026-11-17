# Related work: proximal local BO for composite problems

Survey date: 2026-11-14. This was a quick, search-based survey, not a systematic review.

**Verification levels**
- **[V]**: abstract or paper text was read.
- **[S]**: seen only in search snippets. Check these before citing.
- I spot-checked the titles, authors and abstracts of LCBO, Mueller 2017, BO-ADMM, SEBO and Ghadimi–Lan–Zhang myself against the arXiv API.

## Is the idea new?

Our claim was: *"LCBO only handles the box; nobody has done general non-smooth h with support identification."*

**The LCBO part is mostly correct, with one correction.** LCBO (Jing, Fan, Ng & Hu, ICML 2026, arXiv 2603.07965) [V]:
- It handles a smooth black-box f with smooth black-box equality constraints.
- Constraints enter through the quadratic penalty f + (ρ/2)‖c‖², plus Euclidean projection onto a convex compact domain.
- So LCBO is more than "box only". It also handles smooth constraints by penalty.
- It does not mention proximal operators, L1, lasso, sparsity, non-smooth terms or support identification.

**"Nobody has done it" is too strong. Soften it.** No paper does what we propose:
- local GIBO-type GP gradients;
- a prox step for a known non-smooth h;
- support identification;
- margin-aware sampling.

But three lines of prior work come close:
1. **GP + proximal gradient + L1, offline, not active BO.** Mueller, Reshef, Du & Jaakkola, "Learning Optimal Interventions", AISTATS 2017, arXiv 1606.05027 [V].
   - Fits a GP to observational data, then optimizes a posterior-quantile objective minus a weighted L1 penalty by proximal gradient.
   - It shares a first author with GIBO.
2. **GP-BO with ADMM, where a known h is handled by closed-form prox steps.** Zhao et al., "On the Design of Black-box Adversarial Examples by Leveraging Gradient-free Optimization and Operator Splitting Method", ICCV 2019, arXiv 1907.11684 [V].
   - The method is called BO-ADMM.
   - h can be box, L0, L1, or L1 + L2².
3. **Zeroth-order proximal methods, which use random or finite-difference gradients and no GP.**
   - Ghadimi, Lan & Zhang, Math. Programming 155:267–305 (2016), arXiv 1308.6594 [V abstract]. Composite smooth + convex non-smooth, with a gradient-free variant. Our earlier citation was correct.
   - ZO-ProxSVRG / ZO-ProxSAGA, Huang et al., AAAI 2019, arXiv 1902.06158 [V].
   - ZORO, Cai, McKenzie, Yin & Zhang, SIAM J. Optim. 2022, arXiv 2003.13001 [V]. Its query cost scales with the gradient's sparsity s, giving O(s log d). This is the closest analogue of our "cost scales with s" claim. Its sparsity is in the gradient, not in the iterate's support.
   - TRFD, Davar & Grapiglia, arXiv 2410.09165 [V]. Includes an L1 case.

**Suggested wording:**
> To our knowledge, no prior local GP-gradient (GIBO-type) method handles a known non-smooth convex h through proximal steps or provides support identification. Related work combines GPs with proximal steps offline (Mueller et al., 2017) or within ADMM-based global BO (Zhao et al., 2019), puts known sparsity penalties into BO acquisition functions without proximal steps (SEBO; Liu et al., 2023), or uses proximal steps with random or finite-difference gradient estimators (Ghadimi et al., 2016; Huang et al., 2019; Cai et al., 2022).

## Closest papers, ranked by overlap

| # | Paper | What it does | Overlap / difference |
|---|---|---|---|
| 1 | Mueller et al., AISTATS 2017 [V] | GP posterior + proximal gradient with L1 | Offline data, not local or active BO; no identification analysis |
| 2 | Zhao et al. (BO-ADMM), ICCV 2019 [V] | GP-BO subproblem + prox for L0/L1/box | Global BO inside ADMM; no local gradients |
| 3 | Liu et al. (SEBO), AISTATS 2023, arXiv 2203.01900 [V] | Global BO with a known L0/L1/group sparsity penalty in the acquisition | No prox, not local. **Likely baseline** |
| 4 | Cai et al. (ZORO), SIOPT 2022 [V] | ZO inexact prox-gradient, O(s log d) queries | Gradient sparsity, not iterate support; no GP |
| 5 | Davar & Grapiglia (TRFD), arXiv 2410.09165 [V] | DFO trust region for f + non-smooth h, L1 case | Finite differences; no GP |
| 6 | Huang et al. (ZO-ProxSVRG/SAGA), AAAI 2019 [V] | ZO prox with variance reduction | No GP |
| 7 | Ghadimi, Lan & Zhang, Math. Prog. 2016 [V] | Stochastic composite, includes a ZO variant | Random-smoothing gradients |
| 8 | LCBO, ICML 2026 [V] | GIBO + penalty + projection | Smooth constraints only; our "projected GIBO" baseline |
| 9 | GIBO (NeurIPS 2021) [S]; MPD (NeurIPS 2022) [S]; Wu et al., NeurIPS 2023 [V]; NeST-BO, arXiv 2510.05516 [V]; GRAPE, arXiv 2608.25116 [V]; Marchildon & Zingg, arXiv 2506.00648 [V] | Local BO family | All smooth; none handles non-smooth h |

## Support identification (theory to build on)

- Nutini, Schmidt & Hare, "Active-set complexity of proximal gradient", arXiv 1712.03577 [V].
- Sun, Jeong, Nutini & Schmidt, "Are we there yet? Manifold identification of gradient-related proximal methods", AISTATS 2019 [V].
- Poon, Liang & Schönlieb, ICML 2018 (SAGA / Prox-SVRG identification) [S].
- Dai & Robinson, arXiv 2211.02214 (inexact prox-gradient with support identification) [S].
- Hare & Lewis, J. Convex Anal. 2004 (partial smoothness) [S].
- **Safe screening** is the deterministic analogue of our margin rule |ĝ_i| + κσ_i < λ:
  - Fercoq, Gramfort & Salmon, ICML 2015 [S];
  - Ndiaye et al., JMLR 2017 [S].

**Negative result:** the search found no zeroth-order or GP-gradient method with support-identification guarantees. That is a possible contribution.

## Caveats

- Most [V] entries rest on the abstract only. Full text was read only for Mueller 2017, SEBO and BO-ADMM.
- The search covered about 22 queries. Very recent work, workshop papers and non-indexed work could have been missed.
- Not checked: TuRBO / GTBO in detail, and the full LCBO appendix.
