import numpy as np
import pandas as pd
import pytest

from credit_risk.data import BENCHMARK_COLS, EXCLUDED_FROM_FEATURES, LEAKAGE_COLS, TARGET
from credit_risk.features import OTHER, FeatureBuilder, parse_emp_length


@pytest.fixture(scope="module")
def fitted(splits, cfg):
    train, _, _ = splits
    return FeatureBuilder(cfg.features).fit(train)


def test_no_leakage_or_benchmark_column_in_feature_matrix(fitted, splits):
    X = fitted.transform(splits[2])
    forbidden = set(EXCLUDED_FROM_FEATURES) | set(LEAKAGE_COLS) | set(BENCHMARK_COLS) | {TARGET, "loan_status"}
    assert not forbidden & set(X.columns), forbidden & set(X.columns)
    for name in ["int_rate", "grade", "sub_grade", "installment", "total_pymnt", "recoveries",
                 "last_fico_range_high", "out_prncp", "addr_state", "zip_code"]:
        assert name not in X.columns


def test_features_do_not_depend_on_excluded_columns(fitted, splits):
    test = splits[2]
    X_full = fitted.transform(test)
    X_stripped = fitted.transform(test.drop(columns=[c for c in EXCLUDED_FROM_FEATURES + [TARGET] if c in test]))
    pd.testing.assert_frame_equal(X_full, X_stripped)

    scrambled = test.copy()
    rng = np.random.default_rng(0)
    for c in ["int_rate", "installment", "recoveries", "total_rec_prncp", "funded_amnt"]:
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    scrambled["grade"] = "G"
    scrambled["sub_grade"] = "G5"
    scrambled["loan_status"] = "Charged Off"
    pd.testing.assert_frame_equal(X_full, fitted.transform(scrambled))


def test_fit_uses_training_data_only(splits, cfg):
    train, val, test = splits
    fb = FeatureBuilder(cfg.features).fit(train)
    X_val = fb.transform(val)
    # Medians are unchanged by transforming other periods.
    fb2 = FeatureBuilder(cfg.features).fit(train)
    assert fb.medians_ == fb2.medians_
    # Imputed values equal the TRAIN median, not the val median.
    raw_dti_missing = val["dti"].isna()
    if raw_dti_missing.any():
        assert (X_val.loc[raw_dti_missing, "dti"] == fb.medians_["dti"]).all()


def test_no_missing_values_after_transform(fitted, splits):
    for part in splits:
        X = fitted.transform(part)
        assert not X.isna().any().any()


def test_rare_and_unseen_categories_map_to_other(fitted, splits):
    test = splits[2].copy()
    test.loc[test.index[:5], "purpose"] = "never_seen_purpose"
    X = fitted.transform(test)
    assert (X["purpose"].iloc[:5] == OTHER).all()
    # Rare synthetic purpose (share < threshold in train) is grouped.
    assert "renewable_energy" not in fitted.levels_["purpose"]


def test_emp_length_parsing():
    s = pd.Series(["< 1 year", "1 year", "5 years", "10+ years", "n/a", None])
    got = parse_emp_length(s).tolist()
    assert got[:4] == [0, 1, 5, 10]
    assert pd.isna(got[4]) and pd.isna(got[5])
