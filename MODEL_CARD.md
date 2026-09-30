# Model card: 36-month unsecured personal-loan PD model

All figures come from [`reports/metrics.json`](reports/metrics.json), produced by `python train.py`
on the Lending Club Kaggle data.

## Model details

| | |
|---|---|
| Champion | LightGBM binary classifier (342 trees, early-stopped on 2014) + isotonic calibration (2014) |
| Challenger | Weight-of-evidence logistic scorecard (10 features) + isotonic calibration (2014) |
| Output | Calibrated probability that a 36-month loan charges off or defaults over its life |
| Downstream use | Approve/decline at PD ≤ 0.20; risk-based rate; top-4 adverse-action reason codes |
| Artifacts | `models/model_bundle.joblib`: fitted feature builder, champion, challenger, LGD, config |
| Code | `src/credit_risk/`, served by `api/main.py` |

## Intended use

- A portfolio and reference implementation of a PD → decision → price pipeline for unsecured
  consumer loans, demonstrating leakage control, out-of-time validation, calibration, reason codes,
  drift monitoring and a fair-lending proxy review.
- **Not intended** for real credit decisions. It has not been independently validated, it is trained
  on one lender's historical, already-approved book, and it has not had a compliance review under
  ECOA/Reg B or FCRA.

## Data

- **Source:** Lending Club accepted loans 2007–2018Q4 (Kaggle `wordsforthewise/lending-club`),
  2,260,701 rows.
- **Scope:** 36-month loans issued through December 2015, so every loan had its full term to pay
  off or default before the data ends in 2018Q4. That is 621,169 loans, of which 621,022 have a final
  outcome; the 147 still Current or Late were dropped.
- **Target:** 1 = Charged Off or Default, including "Does not meet the credit policy" variants.
  0 = Fully Paid. There are 86,604 bads and 534,418 goods.
- **Out-of-time split:**

| Split | Issue dates | Loans | Default rate | Used for |
|---|---|---|---|---|
| Train | 2007-06 – 2013-12 | 175,426 | 12.63% | fitting, WoE, imputation, category levels, LGD |
| Validation | 2014 | 162,570 | 13.73% | LightGBM early stopping, calibration, champion choice |
| Test | 2015 | 283,026 | 14.89% | final reporting only |

- **Features (24):** application-time fields only. They include FICO midpoint, credit history length,
  employment length (with a missing flag), loan-to-income, revolving balance-to-income,
  open-to-total accounts, DTI, revolving utilization, inquiries, delinquencies, public records,
  mortgage accounts, home ownership, verification status and purpose. `application_type` was
  dropped because it is constant in the training period.
- **Excluded by design:** post-origination fields (payments, recoveries, last FICO, settlement flags)
  and Lending Club's `int_rate`, `grade` and `sub_grade`. `installment` is excluded too, because it
  encodes `int_rate`. State and ZIP are used only for the fairness review. A unit test enforces all
  of these exclusions.

## Performance (2015 holdout, 283,026 loans)

| Model | PD | AUC | Gini | KS | Brier | Log loss |
|---|---|---|---|---|---|---|
| WoE scorecard | calibrated | 0.6495 | 0.2990 | 0.2139 | 0.1225 | 0.4037 |
| LightGBM | calibrated | 0.6655 | 0.3311 | 0.2391 | 0.1214 | 0.3991 |

Validation (2014, uncalibrated) Gini: scorecard 0.2969, LightGBM 0.3274. The champion rule requires
a validation Gini uplift of ≥ 0.03, fixed before training. LightGBM cleared it narrowly, with an uplift
of 0.0305. 2014 was also LightGBM's early-stopping set, so the 2015 uplift of 0.032 is the unbiased
check.

At the PD ≤ 0.20 policy cutoff the champion approves 80.96% of 2015 loans. The bad rate is 12.20%
among approved loans and 26.30% among declined ones, against 14.89% overall. The trade-off at
other cutoffs is in [cutoffs.md](reports/cutoffs.md).

## Calibration

The calibration chart is [reports/calibration.png](reports/calibration.png). Deciles track the
diagonal in shape, but the whole curve sits above it: mean calibrated PD on 2015 is 13.55% for
LightGBM and 13.28% for the scorecard, against 14.89% observed. This is **vintage drift in the default
level**: 12.63% → 13.73% → 14.89% across train, validation and test. It is not a ranking problem,
since score PSI is 0.0041. Isotonic calibration on 2014 cannot anticipate a worse 2015. In production
this would trigger recalibration on the most recent matured vintage, and pricing would carry a
margin-of-conservatism overlay until that is done.

## Pricing

- LGD = 0.898, exposure-weighted over 22,157 defaults issued 2007–2013.
- Rate = 4% + PD × LGD / 1.5 + 2% + 3%, floored at 6% and capped at 36%.
- Against Lending Club's actual rates by LC grade in 2015 ([table](reports/pricing_by_grade.md)), the
  model charges more for grades A–E, by +6.77 points in A down to +2.85 in E, and less for F (−0.34)
  and G (−2.34).
- Two causes:
  - The formula's 9 points of fixed spread, where Lending Club's rate is an investor coupon.
  - The model's PDs are compressed at the risky end. Mean PD in grade G is 26.04% against 46.12%
    observed, because Lending Club priced with bureau data that is not in the public file.

## Explainability

- Each decision returns the top 4 risk-increasing features relative to the average training
  applicant. For LightGBM these are exact TreeSHAP contributions in log-odds; for the scorecard,
  coefficient × (WoE − mean WoE).
- Isotonic calibration is monotone, so the ranking of drivers carries over to the calibrated PD.
- Among 2015 loans declined at the policy cutoff, the primary reasons are:

| Primary reason | Share |
|---|---|
| Credit score | 54.99% |
| Income | 12.32% |
| Loan amount relative to income | 8.69% |
| Loan purpose | 8.56% |
| Credit inquiries | 7.59% |

"Loan purpose" and "home ownership" appear as reasons. They are permitted inputs, but a compliance
review should confirm the adverse-action wording for them.

## Fairness review and its limits

**Lending Club data contains no protected attributes.** The review is therefore a geographic,
redlining-style proxy check, **not a fair-lending analysis**. It computes each group's approval rate
at PD ≤ 0.20 and its adverse impact ratio (AIR) against the best-treated group, for groups with at
least 500 loans in 2015, and flags AIR < 0.80 under the four-fifths rule.

| Grouping | Groups reviewed | Flagged | Lowest AIR |
|---|---|---|---|
| State | 47 | 0 | HI, 0.853 |
| 3-digit ZIP | 162 | 1 | 482 (Detroit, MI area), 0.765 |

- ZIP3 482 has 557 loans and an approval rate of 67.86%.
- Its observed default rate is 22.08%, against 14.89% overall, so the model's lower approval rate
  there tracks realised risk.
- That does not close the question. Detroit is a majority-Black city, and a geographic disparity there
  is the pattern redlining reviews exist to catch.
- Next steps before any real use:
  - Estimate protected-class composition (BISG).
  - Test for disparate impact on those estimates.
  - Search for a less discriminatory alternative model with comparable accuracy.
  - Check which features drive the gap. Credit score and income are expected; purpose or home
    ownership deserve scrutiny.
- "Approval rates" here are hypothetical decisions over loans Lending Club already approved, not over
  real applicants.

## Monitoring plan

| What | Metric | Thresholds | Action |
|---|---|---|---|
| Score distribution | PSI vs training scores, monthly | < 0.10 stable; 0.10–0.25 investigate; > 0.25 act | Investigate mix change; above 0.25, revalidate |
| Top features (8) | PSI per feature, monthly | same | Trace upstream data or policy changes |
| Calibration | Mean PD vs observed, by vintage as loans mature | gap > 1 point of PD | Recalibrate on latest matured vintage |
| Discrimination | Gini on matured vintages | drop > 0.03 vs 0.331 | Redevelop; re-run champion/challenger |
| Fair lending | AIR by state / ZIP3 at the live cutoff | any AIR < 0.80 | Fair-lending review (see above) |
| Overrides / reason codes | Primary-reason mix among declines | material shift | Review policy and explanations |

The baseline for 2015 against training is a score PSI of 0.0041, with every monitored feature below
0.10; the highest is DTI at 0.0865. Outcome-based checks lag, because 36-month defaults take up to
three years to mature. Until then, early-delinquency proxies such as 30+ days past due at 6 or 12 months
on book would be tracked.

## Known risks and limitations

- **Selection bias.** The model is trained only on loans Lending Club approved, with no reject
  inference.
- **Vintage drift.** 2015 PDs are under-predicted by about 1.3 points. Default levels are cyclical.
  The 2014–2015 validation and test years were a benign economy, and 2007–2009 vintages are only 4.7%
  of training loans, so performance in a downturn is untested.
- **Marginal champion.** LightGBM's advantage is small (+0.03 Gini) and it is not monotone-constrained.
  The scorecard remains a viable, more transparent alternative.
- **Thin features.** Only the public file is used, so discrimination is below what bureau-based
  models reach. Grade-level pricing diverges from Lending Club, especially in grades F–G.
- **Simplified pricing.** There is a single pooled LGD, average life is taken as half the term, and
  prepayment, funding curves and competition are ignored.
- **Scope.** The model covers 36-month loans only. The API rejects other terms, and applications must
  resemble 2007–2015 Lending Club borrowers.
