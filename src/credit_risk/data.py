"""Loading, target definition, population filtering and out-of-time split.

Column lists are explicit so that (a) only ~35 of ~150 columns are read from
the 1.6 GB file and (b) the role of every column is documented: feature input,
benchmark only, outcome/LGD only, or fairness-review only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import Config

# Known at application time; inputs to feature engineering.
APPLICATION_COLS = [
    "issue_d",
    "term",
    "loan_amnt",
    "annual_inc",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "dti",
    "revol_bal",
    "revol_util",
    "open_acc",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "pub_rec_bankruptcies",
    "mort_acc",
    "mths_since_last_delinq",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
]

# Lending Club's own pricing outputs. Read ONLY to benchmark our pricing.
# installment is included: for a fixed term it is a deterministic function of
# loan_amnt and int_rate, so it would leak LC's rate into the model.
BENCHMARK_COLS = ["int_rate", "grade", "sub_grade", "installment"]

# Post-origination outcomes. Read ONLY for the target and LGD estimation.
OUTCOME_COLS = ["loan_status", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]

# Geography. Read ONLY for the fair-lending proxy review, never a feature.
FAIRNESS_COLS = ["addr_state", "zip_code"]

RAW_COLUMNS = APPLICATION_COLS + BENCHMARK_COLS + OUTCOME_COLS + FAIRNESS_COLS

# Post-origination fields that exist in the raw file and must never be features.
# Not read at all, but listed so the leakage test can guard against them.
POST_ORIGINATION_COLS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "out_prncp",
    "out_prncp_inv",
    "last_fico_range_high",
    "last_fico_range_low",
    "debt_settlement_flag",
    "settlement_status",
    "hardship_flag",
    "pymnt_plan",
    "funded_amnt_inv",
]

LEAKAGE_COLS = sorted(set(POST_ORIGINATION_COLS) | set(OUTCOME_COLS))
EXCLUDED_FROM_FEATURES = sorted(set(LEAKAGE_COLS) | set(BENCHMARK_COLS) | set(FAIRNESS_COLS))

TARGET = "default"


def map_target(status: pd.Series) -> pd.Series:
    """Charged Off / Default -> 1, Fully Paid -> 0, anything else -> NaN.

    Substring matching so that "Does not meet the credit policy. Status:Charged Off"
    maps like "Charged Off". Bad statuses are checked first.
    """
    s = status.astype("string")
    bad = s.str.contains("Charged Off", regex=False) | s.str.contains("Default", regex=False)
    good = s.str.contains("Fully Paid", regex=False)
    out = pd.Series(np.nan, index=status.index, dtype="float64")
    out[good.fillna(False).to_numpy()] = 0.0
    out[bad.fillna(False).to_numpy()] = 1.0
    return out


def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Type-normalise raw Lending Club strings (dates, term, percentages)."""
    out = df.copy()
    out["issue_d"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    out["earliest_cr_line"] = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["term"] = pd.to_numeric(out["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    for col in ["int_rate", "revol_util"]:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col].astype("string").str.rstrip("%").str.strip(), errors="coerce")
    return out


def select_population(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Terminal-outcome, 36-month loans issued through the maturity cut-off."""
    out = df.copy()
    out[TARGET] = map_target(out["loan_status"])
    mask = (
        out[TARGET].notna()
        & (out["term"] == cfg.split.term_months)
        & out["issue_d"].notna()
        & (out["issue_d"] <= pd.Timestamp(cfg.split.max_issue_date))
    )
    out = out.loc[mask].reset_index(drop=True)
    out[TARGET] = out[TARGET].astype("int8")
    return out


def time_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Out-of-time split by issue date: train <= 2013, val = 2014, test = 2015."""
    year = df["issue_d"].dt.year
    train = df.loc[df["issue_d"] <= pd.Timestamp(cfg.split.train_end)]
    val = df.loc[year == cfg.split.val_year]
    test = df.loc[year == cfg.split.test_year]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def find_raw_file(raw_dir: Path) -> Path:
    candidates = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and "accepted" in p.name.lower() and ".csv" in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No accepted-loans CSV under {raw_dir}. Run scripts/download_data.py first.")
    return candidates[0]


def scope_summary(parsed: pd.DataFrame, cfg: Config) -> dict:
    """Status counts for 36-month loans issued through the cut-off, before dropping
    non-terminal loans. Quantifies the selection effect of dropping Current/Late."""
    in_scope = parsed.loc[
        (parsed["term"] == cfg.split.term_months)
        & (parsed["issue_d"] <= pd.Timestamp(cfg.split.max_issue_date))
    ]
    t = map_target(in_scope["loan_status"])
    return {
        "n_raw_rows": int(len(parsed)),
        "n_in_scope": int(len(in_scope)),
        "n_good": int((t == 0).sum()),
        "n_bad": int((t == 1).sum()),
        "n_dropped_non_terminal": int(t.isna().sum()),
        "dropped_status_counts": {
            str(k): int(v) for k, v in in_scope.loc[t.isna(), "loan_status"].value_counts().items()
        },
    }


def load_population(cfg: Config, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read needed columns from the raw CSV once and cache the modelling population
    as parquet (plus a JSON scope summary). Returns (population, summary)."""
    cache = cfg.paths.processed_file
    summary_file = cache.with_suffix(".summary.json")
    if cache.exists() and summary_file.exists() and not refresh:
        return pd.read_parquet(cache), json.loads(summary_file.read_text())
    raw_path = find_raw_file(cfg.paths.raw_dir)
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, low_memory=False)
    parsed = parse_raw(raw)
    summary = scope_summary(parsed, cfg)
    pop = select_population(parsed, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(cache, index=False)
    summary_file.write_text(json.dumps(summary, indent=2))
    return pop, summary
