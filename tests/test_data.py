import pandas as pd

from credit_risk.data import RAW_COLUMNS, TARGET, map_target, parse_raw
from credit_risk.synthetic import SYNTHETIC, make_synthetic_loans
from scripts_helpers import pick_accepted_file


def test_target_mapping_uses_contains():
    s = pd.Series([
        "Fully Paid",
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
        "Current",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "In Grace Period",
        None,
    ])
    got = map_target(s).tolist()
    assert got[:5] == [0, 1, 1, 0, 1]
    assert all(pd.isna(x) for x in got[5:])


def test_parse_raw_formats():
    df = parse_raw(make_synthetic_loans(n=50, seed=3))
    assert set(df["term"].unique()) <= {36, 60}
    assert pd.api.types.is_datetime64_any_dtype(df["issue_d"])
    assert pd.api.types.is_numeric_dtype(df["revol_util"])


def test_synthetic_has_raw_schema_and_flag():
    assert SYNTHETIC is True
    assert list(make_synthetic_loans(n=10).columns) == RAW_COLUMNS


def test_population_filters(population, cfg):
    assert set(population[TARGET].unique()) <= {0, 1}
    assert (population["term"] == 36).all()
    assert population["issue_d"].max() <= pd.Timestamp(cfg.split.max_issue_date)
    assert not population["loan_status"].str.contains("Current|Late|Grace").any()


def test_time_split_has_no_date_overlap(splits):
    train, val, test = splits
    assert train["issue_d"].max() < val["issue_d"].min()
    assert val["issue_d"].max() < test["issue_d"].min()
    assert train["issue_d"].dt.year.max() == 2013
    assert set(val["issue_d"].dt.year) == {2014}
    assert set(test["issue_d"].dt.year) == {2015}
    assert len(train) and len(val) and len(test)


def test_download_script_prefers_gz():
    names = ["rejected_2007_to_2018Q4.csv.gz", "accepted_2007_to_2018Q4.csv.gz",
             "accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"]
    assert pick_accepted_file(names) == "accepted_2007_to_2018Q4.csv.gz"
