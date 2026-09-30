"""Adverse-action reason codes from per-feature log-odds contributions.

A reason code is a feature that pushed THIS applicant's risk above the reference
(training-average) applicant. Only positive contributions qualify, ranked by size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REASON_TEXT = {
    "fico_mid": "Credit score",
    "credit_hist_years": "Length of credit history",
    "emp_length_years": "Length of employment",
    "emp_length_missing": "Employment length not provided",
    "loan_to_income": "Loan amount relative to income",
    "revol_bal_to_income": "Revolving balances relative to income",
    "open_to_total_acc": "Proportion of credit accounts currently open",
    "loan_amnt": "Loan amount requested",
    "annual_inc": "Income",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving credit utilization",
    "revol_bal": "Revolving credit balances",
    "open_acc": "Number of open credit accounts",
    "total_acc": "Total number of credit accounts",
    "delinq_2yrs": "Delinquencies in the last 2 years",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "pub_rec": "Derogatory public records",
    "pub_rec_bankruptcies": "Public record bankruptcies",
    "mort_acc": "Number of mortgage accounts",
    "mths_since_last_delinq": "Time since most recent delinquency",
    "no_delinq_history": "Delinquency history",
    "home_ownership": "Home ownership status",
    "verification_status": "Income verification status",
    "purpose": "Loan purpose",
    "application_type": "Application type",
}


def reason_codes(contrib: pd.DataFrame, n: int = 4) -> list[list[dict]]:
    """Top-n risk-increasing features per row, largest contribution first."""
    values = contrib.to_numpy()
    cols = np.asarray(contrib.columns)
    order = np.argsort(-values, axis=1)[:, :n]
    out = []
    for i, idx in enumerate(order):
        row = []
        for j in idx:
            v = float(values[i, j])
            if v <= 0:
                break
            row.append({"feature": str(cols[j]), "reason": REASON_TEXT.get(cols[j], cols[j]),
                        "contribution": round(v, 4)})
        out.append(row)
    return out


def primary_reason_frequency(contrib: pd.DataFrame, mask=None, n: int = 4) -> pd.DataFrame:
    """How often each feature appears as a reason code (e.g. among declined applicants)."""
    c = contrib if mask is None else contrib.loc[np.asarray(mask)]
    codes = reason_codes(c, n)
    first = pd.Series([r[0]["feature"] if r else None for r in codes]).value_counts()
    any_ = pd.Series([x["feature"] for r in codes for x in r]).value_counts()
    t = pd.DataFrame({"primary_reason_share": first / max(len(codes), 1),
                      "any_top_reason_share": any_ / max(len(codes), 1)}).fillna(0.0)
    t.index.name = "feature"
    t.insert(0, "reason", [REASON_TEXT.get(f, f) for f in t.index])
    return t.sort_values("primary_reason_share", ascending=False)
