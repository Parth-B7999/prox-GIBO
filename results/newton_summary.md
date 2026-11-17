| cond | method | final gap (median [min, max]) | F1 (median) | evals to ≤0.1 (median; #seeds) | evals to ≤0.01 (median; #seeds) | evals to ≤0.001 (median; #seeds) | Hessian rel. error (median over run) |
|---|---|---|---|---|---|---|---|
| 10 | Prox GIBO (gradient step) | 0.00783 [0.00635, 0.00982] | 1.00 | 252; 5/5 | 770; 5/5 | —; 0/5 | — |
| 10 | Prox Newton GIBO (GP-mean Hessian) | 0.32 [0.146, 0.38] | 0.87 | 238; 5/5 | —; 0/5 | —; 0/5 | 1.01 |
| 10 | Margin-aware prox GIBO (gradient step) | 0.00291 [0.00123, 0.00327] | 1.00 | 189; 5/5 | 287; 5/5 | 497; 1/5 | — |
| 10 | Margin-aware prox Newton (GP-mean Hessian) | 0.0049 [0.0047, 0.115] | 1.00 | 238; 5/5 | 483; 5/5 | —; 0/5 | 0.94 |
| 10 | Margin-aware prox Newton, TRUE Hessian (diagnostic) | 0.00248 [0.00148, 0.00438] | 1.00 | 203; 5/5 | 308; 5/5 | 861; 3/5 | — |
| 100 | Prox GIBO (gradient step) | 0.0156 [0.0044, 0.125] | 0.87 | 609; 5/5 | 906; 2/5 | —; 0/5 | — |
| 100 | Prox Newton GIBO (GP-mean Hessian) | 0.47 [0.155, 0.809] | 0.83 | 728; 3/5 | —; 0/5 | —; 0/5 | 1.04 |
| 100 | Margin-aware prox GIBO (gradient step) | 0.00298 [0.0004, 0.0301] | 1.00 | 651; 5/5 | 735; 5/5 | 973; 1/5 | — |
| 100 | Margin-aware prox Newton (GP-mean Hessian) | 0.0852 [0.0327, 0.424] | 1.00 | 651; 5/5 | 819; 3/5 | —; 0/5 | 1.00 |
| 100 | Margin-aware prox Newton, TRUE Hessian (diagnostic) | 0.00302 [0.00242, 0.024] | 1.00 | 462; 5/5 | 654; 4/5 | 826; 3/5 | — |
