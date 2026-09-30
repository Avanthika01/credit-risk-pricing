import numpy as np
import pandas as pd

from credit_risk.explain import REASON_TEXT, primary_reason_frequency, reason_codes
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.monitoring import psi, psi_status


def test_reason_codes_are_top_positive_contributions():
    c = pd.DataFrame({"a": [0.5, -0.1], "b": [0.9, -0.2], "c": [-0.3, 0.05], "d": [0.1, -1.0], "e": [0.2, 0.0]})
    codes = reason_codes(c, n=4)
    assert [r["feature"] for r in codes[0]] == ["b", "a", "e", "d"]
    assert [r["feature"] for r in codes[1]] == ["c"]  # only risk-increasing features qualify
    assert all(r["contribution"] > 0 for row in codes for r in row)


def test_every_feature_has_reason_text():
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) <= set(REASON_TEXT)


def test_reason_frequency_table():
    c = pd.DataFrame({"a": [1.0, 1.0, -1.0], "b": [0.5, 2.0, 1.0]})
    t = primary_reason_frequency(c)
    assert t.loc["b", "primary_reason_share"] == 2 / 3


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    assert psi(x, rng.permutation(x)) < 1e-9
    assert psi(x, rng.normal(size=20_000)) < 0.01
    assert psi(x, rng.normal(1.0, 1, size=20_000)) > 0.25


def test_psi_categorical_and_thresholds():
    e = pd.Series(["A"] * 50 + ["B"] * 50)
    a = pd.Series(["A"] * 90 + ["B"] * 10)
    assert psi(e, a) > 0.25
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "moderate shift"
    assert psi_status(0.30) == "significant shift"


def test_model_reason_codes_on_synthetic(splits, cfg):
    from credit_risk.data import TARGET
    from credit_risk.features import FeatureBuilder
    from credit_risk.models import PDModel, WoEScorecard

    train, val, test = splits
    fb = FeatureBuilder(cfg.features).fit(train)
    m = PDModel("sc", WoEScorecard(cfg.features, cfg.model).fit(fb.transform(train), train[TARGET]))
    codes = reason_codes(m.contributions(fb.transform(test.iloc[:100])), n=4)
    assert all(len(r) <= 4 for r in codes)
    assert any(len(r) > 0 for r in codes)
