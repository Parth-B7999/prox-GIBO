| d | method | seeds | final gap (median [min, max]) | final support F1, exact zeros (median) | F1 counting \|x\|>0.01 as nonzero | evals to ≤0.1 (median; #seeds) | evals to ≤0.01 (median; #seeds) | evals to ≤0.001 (median; #seeds) | runtime/run (s) |
|---|---|---|---|---|---|---|---|---|---|
| 50 | Margin-aware prox GIBO (ours) | 5 | 0.00298 [0.0004, 0.0301] | 1.00 | 1.00 | 651; 5/5 | 735; 5/5 | 973; 1/5 | 306 |
| 50 | Prox GIBO (ours) | 5 | 0.0156 [0.0044, 0.125] | 0.87 | 1.00 | 609; 5/5 | 906; 2/5 | —; 0/5 | 337 |
| 50 | Projected GIBO + line search (black box) | 5 | 1.93 [1.17, 2.43] | 0.33 | 0.39 | —; 0/5 | —; 0/5 | —; 0/5 | 312 |
| 50 | Projected GIBO / LCBO-style (black box) | 5 | 7.22 [3.31, 9.98] | 0.33 | 0.36 | —; 0/5 | —; 0/5 | —; 0/5 | 369 |
| 100 | Margin-aware prox GIBO (ours) | 5 | 0.143 [0.043, 0.191] | 0.80 | 0.91 | 924; 2/5 | —; 0/5 | —; 0/5 | 378 |
| 100 | Prox GIBO (ours) | 5 | 0.159 [0.0871, 0.549] | 0.65 | 0.87 | 910; 2/5 | —; 0/5 | —; 0/5 | 455 |
| 100 | Projected GIBO + line search (black box) | 5 | 6.51 [5.41, 9.25] | 0.18 | 0.20 | —; 0/5 | —; 0/5 | —; 0/5 | 429 |
| 100 | Projected GIBO / LCBO-style (black box) | 5 | 19.2 [15.5, 26.6] | 0.18 | 0.19 | —; 0/5 | —; 0/5 | —; 0/5 | 426 |
