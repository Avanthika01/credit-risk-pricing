# Adverse impact ratio by 3-digit ZIP (test period)

Lending Club data has no protected attributes. This is a geographic (redlining-style) proxy review by state and 3-digit ZIP, not a full fair-lending analysis. Approval rates are hypothetical model decisions over loans Lending Club already booked.

Groups with >= 500 loans: 162; flagged (AIR < 0.8): 1

| zip3 | n | approval_rate | mean_pd | observed_default_rate | air | flagged |
|---|---|---|---|---|---|---|
| 482 | 557 | 67.86% | 16.62% | 22.08% | 0.765 | yes |
| 338 | 607 | 73.64% | 15.42% | 16.97% | 0.830 | no |
| 104 | 1996 | 73.65% | 15.52% | 20.24% | 0.830 | no |
| 441 | 1037 | 74.06% | 14.92% | 15.43% | 0.835 | no |
| 325 | 588 | 75.17% | 14.39% | 14.12% | 0.847 | no |
| 785 | 729 | 75.45% | 15.02% | 17.28% | 0.851 | no |
| 347 | 761 | 75.56% | 14.78% | 17.35% | 0.852 | no |
| 296 | 634 | 75.71% | 14.62% | 10.25% | 0.853 | no |
| 381 | 541 | 75.79% | 14.62% | 14.42% | 0.854 | no |
| 967 | 934 | 76.45% | 14.77% | 13.17% | 0.862 | no |
| 339 | 725 | 76.55% | 14.76% | 15.31% | 0.863 | no |
| 328 | 1041 | 76.85% | 14.40% | 14.89% | 0.866 | no |
| 462 | 742 | 76.95% | 14.72% | 17.12% | 0.868 | no |
| 907 | 729 | 76.95% | 14.09% | 18.11% | 0.868 | no |
| 191 | 986 | 76.98% | 14.57% | 15.82% | 0.868 | no |

(Showing the 15 lowest-AIR ZIP3 groups plus any flagged groups.)
