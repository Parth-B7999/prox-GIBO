| d | method | seeds | final gap (median [min, max]) | final support F1, exact zeros (median) | F1 counting \|x\|>0.01 as nonzero | evals to ≤0.1 (median; #seeds) | evals to ≤0.01 (median; #seeds) | evals to ≤0.001 (median; #seeds) | runtime/run (s) |
|---|---|---|---|---|---|---|---|---|---|
| 50 | Margin-aware prox GIBO (ours) | 5 | 9.9e-05 [2.93e-05, 0.000143] | 1.00 | 1.00 | 203; 5/5 | 252; 5/5 | 308; 5/5 | 505 |
| 50 | Prox GIBO (ours) | 5 | 0.00141 [0.000984, 0.00498] | 1.00 | 1.00 | 196; 5/5 | 301; 5/5 | 609; 5/5 | 521 |
| 50 | Projected GIBO + line search (black box) | 5 | 0.671 [0.579, 1.28] | 0.33 | 0.42 | —; 0/5 | —; 0/5 | —; 0/5 | 426 |
| 50 | Projected GIBO / LCBO-style (black box) | 5 | 3.09 [2.21, 3.25] | 0.33 | 0.40 | —; 0/5 | —; 0/5 | —; 0/5 | 353 |
| 50 | GIBO, repo default ε=0.1 (black box) | 5 | 1.54 [0.953, 2.13] | 0.33 | 0.38 | —; 0/5 | —; 0/5 | —; 0/5 | 296 |
| 100 | Margin-aware prox GIBO (ours) | 5 | 0.000343 [0.00032, 0.0526] | 1.00 | 1.00 | 406; 5/5 | 476; 5/5 | 525; 5/5 | 634 |
| 100 | Prox GIBO (ours) | 5 | 0.00937 [0.000951, 0.077] | 0.91 | 1.00 | 336; 5/5 | 532; 5/5 | 819; 2/5 | 548 |
| 100 | Projected GIBO + line search (black box) | 5 | 2.23 [2.08, 2.5] | 0.18 | 0.26 | —; 0/5 | —; 0/5 | —; 0/5 | 520 |
| 100 | Projected GIBO / LCBO-style (black box) | 5 | 9.03 [6.07, 14.7] | 0.18 | 0.21 | —; 0/5 | —; 0/5 | —; 0/5 | 407 |
| 100 | GIBO, repo default ε=0.1 (black box) | 5 | 3.21 [2.89, 3.74] | 0.18 | 0.21 | —; 0/5 | —; 0/5 | —; 0/5 | 719 |
