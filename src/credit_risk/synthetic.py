"""SYNTHETIC Lending-Club-shaped data. FOR TESTS AND CI ONLY.

*** THIS IS NOT REAL DATA. ***  No number produced from it may appear in the
README, model card or reports. It exists because CI cannot download the Kaggle
dataset. It reproduces the raw column names and string formats (e.g. "Dec-2015",
" 36 months", "10+ years", "123xx") so the real parsing code path is exercised,
and default risk is driven by application fields so models have signal to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import RAW_COLUMNS

SYNTHETIC = True

_STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "WA", "AZ", "MA"]
_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
             "small_business", "car", "medical", "moving", "vacation", "wedding", "renewable_energy"]
_PURPOSE_P = [0.55, 0.2, 0.06, 0.05, 0.03, 0.03, 0.02, 0.02, 0.015, 0.01, 0.01, 0.005]
_EMP = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
        "7 years", "8 years", "9 years", "10+ years"]


def make_synthetic_loans(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Return a SYNTHETIC raw DataFrame with exactly RAW_COLUMNS."""
    rng = np.random.default_rng(seed)

    months = pd.date_range("2008-01-01", "2016-12-01", freq="MS")
    w = np.linspace(1, 8, len(months))  # volume grows over time, like LC
    issue = pd.DatetimeIndex(rng.choice(months, size=n, p=w / w.sum()))

    fico_low = np.clip(rng.normal(695, 30, n).round() // 5 * 5, 660, 845).astype(int)
    annual_inc = np.round(np.exp(rng.normal(11.0, 0.5, n)), -2)
    loan_amnt = np.clip(np.round(rng.lognormal(9.4, 0.6, n), -2), 1000, 35000)
    dti = np.clip(rng.normal(18, 8, n), 0, 45).round(2)
    revol_bal = np.round(np.exp(rng.normal(9.3, 1.0, n)))
    revol_util = np.clip(rng.normal(55, 24, n), 0, 130).round(1)
    total_acc = np.clip(rng.poisson(24, n), 3, None)
    open_acc = np.minimum(np.clip(rng.poisson(11, n), 1, None), total_acc)
    inq = rng.poisson(0.8, n)
    delinq = rng.poisson(0.3, n)
    pub_rec = rng.poisson(0.15, n)
    bankr = np.minimum(pub_rec, rng.poisson(0.1, n))
    mort_acc = rng.poisson(1.5, n)
    hist_years = np.clip(rng.gamma(4, 4, n), 1, 50)
    earliest = issue - pd.to_timedelta(np.round(hist_years * 365.25), unit="D")
    emp_idx = rng.integers(0, len(_EMP), n)
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], n, p=[0.49, 0.40, 0.107, 0.003])
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], n, p=[0.3, 0.38, 0.32])
    purpose = rng.choice(_PURPOSES, n, p=_PURPOSE_P)
    app_type = rng.choice(["Individual", "Joint App"], n, p=[0.98, 0.02])
    term = rng.choice([" 36 months", " 60 months"], n, p=[0.75, 0.25])

    # Latent log-odds of default from application fields.
    logit = (
        -1.9
        - 0.018 * (fico_low - 695)
        + 0.025 * (dti - 18)
        + 0.007 * (revol_util - 55)
        + 0.15 * inq
        + 0.6 * np.log(loan_amnt / (annual_inc * 0.15))
        + 0.25 * (home == "RENT")
        + 0.4 * (purpose == "small_business")
        - 0.015 * hist_years
        + 0.2 * pub_rec
        + rng.normal(0, 0.5, n)
    )
    pd_true = 1 / (1 + np.exp(-logit))
    bad = rng.random(n) < pd_true

    # LC-like pricing from the latent risk, used only as a benchmark.
    risk_rank = pd.Series(logit).rank(pct=True).to_numpy()
    sub_idx = np.minimum((risk_rank * 35).astype(int), 34)
    grade = np.array(list("ABCDEFG"))[sub_idx // 5]
    sub_grade = np.char.add(grade, ((sub_idx % 5) + 1).astype(str))
    int_rate = np.round(5.5 + sub_idx * 0.72 + rng.normal(0, 0.3, n), 2)
    n_months = np.where(term == " 36 months", 36, 60)
    r = int_rate / 100 / 12
    installment = np.round(loan_amnt * r / (1 - (1 + r) ** -n_months), 2)

    # Outcomes. Recent loans may still be current / late (non-terminal).
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    months_since = (pd.Timestamp("2018-12-01") - issue).days / 30.4
    open_loan = (months_since < n_months + 6) & (rng.random(n) < 0.7)
    status[open_loan] = rng.choice(["Current", "Late (31-120 days)", "In Grace Period"], open_loan.sum(),
                                   p=[0.9, 0.07, 0.03])
    status[(~open_loan) & bad & (rng.random(n) < 0.02)] = "Default"
    policy = issue < pd.Timestamp("2010-06-01")
    flip = policy & (rng.random(n) < 0.05) & np.isin(status, ["Charged Off", "Fully Paid"])
    status[flip] = "Does not meet the credit policy. Status:" + status[flip].astype(str)

    funded = loan_amnt
    repaid_frac = np.where(bad, rng.uniform(0.1, 0.7, n), 1.0)
    total_rec_prncp = np.round(funded * repaid_frac, 2)
    recoveries = np.where(bad, np.round((funded - total_rec_prncp) * rng.uniform(0, 0.2, n), 2), 0.0)
    coll_fee = np.round(recoveries * 0.18, 2)

    zip3 = rng.integers(100, 999, n)
    states = rng.choice(_STATES, n)

    df = pd.DataFrame(
        {
            "issue_d": issue.strftime("%b-%Y"),
            "term": term,
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "emp_length": np.where(rng.random(n) < 0.05, None, np.array(_EMP)[emp_idx]),
            "home_ownership": home,
            "verification_status": verif,
            "purpose": purpose,
            "application_type": app_type,
            "dti": np.where(rng.random(n) < 0.01, np.nan, dti),
            "revol_bal": revol_bal,
            "revol_util": np.where(rng.random(n) < 0.01, np.nan, revol_util),
            "open_acc": open_acc,
            "total_acc": total_acc,
            "delinq_2yrs": delinq,
            "inq_last_6mths": inq,
            "pub_rec": pub_rec,
            "pub_rec_bankruptcies": bankr,
            "mort_acc": np.where(issue < pd.Timestamp("2010-01-01"), np.nan, mort_acc),
            "mths_since_last_delinq": np.where(delinq > 0, rng.integers(1, 80, n), np.nan),
            "earliest_cr_line": earliest.strftime("%b-%Y"),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "int_rate": int_rate,
            "grade": grade,
            "sub_grade": sub_grade,
            "installment": installment,
            "loan_status": status,
            "funded_amnt": funded,
            "total_rec_prncp": total_rec_prncp,
            "recoveries": recoveries,
            "collection_recovery_fee": coll_fee,
            "addr_state": states,
            "zip_code": [f"{z}xx" for z in zip3],
        }
    )
    return df[RAW_COLUMNS]
