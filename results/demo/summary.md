| d | method | seeds | final gap (median [min, max]) | final support F1, exact zeros (median) | F1 counting \|x\|>0.01 as nonzero | evals to ≤0.1 (median; #seeds) | evals to ≤0.01 (median; #seeds) | evals to ≤0.001 (median; #seeds) | runtime/run (s) |
|---|---|---|---|---|---|---|---|---|---|
| 50 | Margin-aware prox GIBO (ours) | 5 | 0.00291 [0.00123, 0.00327] | 1.00 | 1.00 | 189; 5/5 | 287; 5/5 | 497; 1/5 | 125 |
| 50 | Prox GIBO (ours) | 5 | 0.00783 [0.00635, 0.00982] | 1.00 | 1.00 | 252; 5/5 | 770; 5/5 | —; 0/5 | 138 |
| 50 | Projected GIBO + line search (black box) | 5 | 0.825 [0.336, 0.966] | 0.33 | 0.47 | —; 0/5 | —; 0/5 | —; 0/5 | 129 |
| 50 | Projected GIBO / LCBO-style (black box) | 5 | 2.77 [2.05, 3.15] | 0.33 | 0.37 | —; 0/5 | —; 0/5 | —; 0/5 | 96 |
| 50 | GIBO, repo default ε=0.1 (black box) | 5 | 1.33 [0.849, 1.73] | 0.33 | 0.37 | —; 0/5 | —; 0/5 | —; 0/5 | 405 |
| 50 | GIBO, paper config (black box) | 5 | 1.34 [0.967, 1.47] | 0.33 | 0.39 | —; 0/5 | —; 0/5 | —; 0/5 | 284 |
| 50 | NeST-BO (black box) | 5 | 3.87 [2.64, 4.8] | 0.33 | 0.34 | —; 0/5 | —; 0/5 | —; 0/5 | 86796 |
| 100 | Margin-aware prox GIBO (ours) | 5 | 0.00211 [0.00108, 0.00793] | 1.00 | 1.00 | 385; 5/5 | 553; 5/5 | 728; 2/5 | 161 |
| 100 | Prox GIBO (ours) | 5 | 0.0151 [0.00736, 0.0208] | 0.95 | 1.00 | 406; 5/5 | 987; 2/5 | —; 0/5 | 162 |
| 100 | Projected GIBO + line search (black box) | 5 | 2.52 [2.42, 3.22] | 0.18 | 0.25 | —; 0/5 | —; 0/5 | —; 0/5 | 124 |
| 100 | Projected GIBO / LCBO-style (black box) | 5 | 7.03 [5.77, 9.78] | 0.18 | 0.21 | —; 0/5 | —; 0/5 | —; 0/5 | 110 |
| 100 | GIBO, repo default ε=0.1 (black box) | 5 | 2.76 [2.5, 3.78] | 0.18 | 0.21 | —; 0/5 | —; 0/5 | —; 0/5 | 1154 |
| 100 | GIBO, paper config (black box) | 5 | 3.65 [3.26, 4.41] | 0.18 | 0.21 | —; 0/5 | —; 0/5 | —; 0/5 | 671 |
| 200 | Margin-aware prox GIBO (ours) | 5 | 0.0196 [0.00503, 0.0275] | 1.00 | 1.00 | 742; 5/5 | 948; 2/5 | —; 0/5 | 616 |
| 200 | Prox GIBO (ours) | 5 | 0.0699 [0.0178, 0.0721] | 0.83 | 0.87 | 798; 5/5 | —; 0/5 | —; 0/5 | 652 |
| 200 | Projected GIBO + line search (black box) | 5 | 7.02 [6.73, 8.76] | 0.10 | 0.11 | —; 0/5 | —; 0/5 | —; 0/5 | 487 |
| 200 | Projected GIBO / LCBO-style (black box) | 5 | 19.2 [14.1, 23.7] | 0.10 | 0.11 | —; 0/5 | —; 0/5 | —; 0/5 | 414 |
| 200 | GIBO, repo default ε=0.1 (black box) | 5 | 9.56 [8.89, 10.3] | 0.10 | 0.11 | —; 0/5 | —; 0/5 | —; 0/5 | 1265 |
