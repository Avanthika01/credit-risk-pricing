"""Feature engineering from application-time fields only.

Two stages:
1. `engineer` - stateless row-wise derivations (ratios, date differences).
2. `FeatureBuilder` - fitted on TRAINING data only: imputation medians, category
   levels (rare -> OTHER). The same fitted object is used for validation, test
   and the API, so training and serving transforms cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import FeatureConfig

NUMERIC_FEATURES = [
    # engineered
    "fico_mid",
    "credit_hist_years",
    "emp_length_years",
    "loan_to_income",
    "revol_bal_to_income",
    "open_to_total_acc",
    # standard application fields
    "loan_amnt",
    "annual_inc",
    "dti",
    "revol_util",
    "revol_bal",
    "open_acc",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "pub_rec_bankruptcies",
    "mort_acc",
    "mths_since_last_delinq",
    # missingness indicators (missing carries meaning)
    "emp_length_missing",
    "no_delinq_history",
]

CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "application_type"]

# Raw inputs the builder needs. Anything else in the frame is ignored.
REQUIRED_INPUTS = [
    "issue_d", "earliest_cr_line", "emp_length", "fico_range_low", "fico_range_high",
    "loan_amnt", "annual_inc", "dti", "revol_util", "revol_bal", "open_acc", "total_acc",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
    "mths_since_last_delinq", *CATEGORICAL_FEATURES,
]

OTHER = "OTHER"
MISSING = "MISSING"


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den > 0)
    return num / den


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '1 year' -> 1, '10+ years' -> 10, 'n/a' / NaN -> NaN."""
    s = s.astype("string").str.strip()
    s = s.mask(s.str.startswith("<").fillna(False), "0")
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless derivations. Expects parsed dates (see data.parse_raw)."""
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")
    num = lambda c: pd.to_numeric(df[c], errors="coerce").astype("float64")  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["fico_mid"] = (num("fico_range_low") + num("fico_range_high")) / 2
    out["credit_hist_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    out["emp_length_years"] = parse_emp_length(df["emp_length"]).astype("float64")
    out["loan_to_income"] = _safe_ratio(num("loan_amnt"), num("annual_inc"))
    out["revol_bal_to_income"] = _safe_ratio(num("revol_bal"), num("annual_inc"))
    out["open_to_total_acc"] = _safe_ratio(num("open_acc"), num("total_acc"))
    for c in ["loan_amnt", "annual_inc", "dti", "revol_util", "revol_bal", "open_acc", "total_acc",
              "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies", "mort_acc",
              "mths_since_last_delinq"]:
        out[c] = num(c)
    out["emp_length_missing"] = out["emp_length_years"].isna().astype("float64")
    out["no_delinq_history"] = out["mths_since_last_delinq"].isna().astype("float64")
    for c in CATEGORICAL_FEATURES:
        out[c] = df[c].astype("string").str.strip().fillna(MISSING)
    return out


class FeatureBuilder:
    """Fit imputation and category levels on training data; apply everywhere."""

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()
        self.medians_: dict[str, float] = {}
        self.levels_: dict[str, list[str]] = {}
        self.dropped_constant_: list[str] = []
        self.numeric_: list[str] = []
        self.categorical_: list[str] = []

    def fit(self, train_raw: pd.DataFrame) -> "FeatureBuilder":
        eng = engineer(train_raw)
        self.medians_ = {c: float(eng[c].median()) for c in NUMERIC_FEATURES}
        for c in CATEGORICAL_FEATURES:
            share = eng[c].value_counts(normalize=True)
            keep = sorted(share[share >= self.cfg.rare_category_min_share].index.tolist())
            self.levels_[c] = keep + ([OTHER] if OTHER not in keep else [])
        # Drop features with no variation in training (e.g. a category that only
        # appears after the training period): they cannot be learned.
        X = self._apply(eng, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        self.dropped_constant_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        self.numeric_ = [c for c in NUMERIC_FEATURES if c not in self.dropped_constant_]
        self.categorical_ = [c for c in CATEGORICAL_FEATURES if c not in self.dropped_constant_]
        return self

    def _apply(self, eng: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
        X = pd.DataFrame(index=eng.index)
        for c in numeric:
            X[c] = eng[c].replace([np.inf, -np.inf], np.nan).fillna(self.medians_[c])
        for c in categorical:
            levels = self.levels_[c]
            vals = eng[c].where(eng[c].isin(levels), OTHER)
            X[c] = pd.Categorical(vals.astype(object), categories=levels)
        return X

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.levels_:
            raise RuntimeError("FeatureBuilder is not fitted")
        return self._apply(engineer(raw), self.numeric_, self.categorical_)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_ + self.categorical_
