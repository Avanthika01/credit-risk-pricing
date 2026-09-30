# Test-period (holdout) performance

| model | PD | auc | gini | ks | brier | log_loss | mean_pd | observed_default_rate |
|---|---|---|---|---|---|---|---|---|
| WoE scorecard | uncalibrated | 0.6496 | 0.2992 | 0.2144 | 0.1228 | 0.4048 | 0.1270 | 0.1489 |
| LightGBM | uncalibrated | 0.6658 | 0.3316 | 0.2392 | 0.1216 | 0.4000 | 0.1301 | 0.1489 |
| WoE scorecard | calibrated | 0.6495 | 0.2990 | 0.2139 | 0.1225 | 0.4037 | 0.1328 | 0.1489 |
| LightGBM | calibrated | 0.6655 | 0.3311 | 0.2391 | 0.1214 | 0.3991 | 0.1355 | 0.1489 |

**Validation (uncalibrated; used for champion selection)**

| model | auc | gini | ks |
|---|---|---|---|
| WoE scorecard | 0.6485 | 0.2969 | 0.2158 |
| LightGBM | 0.6637 | 0.3274 | 0.2360 |
