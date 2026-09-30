import numpy as np
import pytest

from credit_risk.data import TARGET
from credit_risk.evaluate import calibration_table, ks_statistic, pd_metrics
from credit_risk.features import FeatureBuilder
from credit_risk.models import PD_EPS, LGBMModel, PDModel, WoEBinner, WoEScorecard


@pytest.fixture(scope="module")
def data(splits, cfg):
    train, val, test = splits
    fb = FeatureBuilder(cfg.features).fit(train)
    return (fb.transform(train), train[TARGET], fb.transform(val), val[TARGET],
            fb.transform(test), test[TARGET])


@pytest.fixture(scope="module")
def models(data, cfg):
    Xtr, ytr, Xva, yva, _, _ = data
    sc = PDModel("scorecard", WoEScorecard(cfg.features, cfg.model).fit(Xtr, ytr)).calibrate(Xva, yva)
    gb = PDModel("lightgbm", LGBMModel(cfg.model).fit(Xtr, ytr, Xva, yva)).calibrate(Xva, yva)
    return sc, gb


def test_pds_are_probabilities(models, data):
    Xte = data[4]
    for m in models:
        p = m.predict_pd(Xte)
        assert np.isfinite(p).all()
        assert ((p >= PD_EPS) & (p <= 1 - PD_EPS)).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_models_beat_random_on_synthetic_signal(models, data):
    Xte, yte = data[4], data[5]
    for m in models:
        assert pd_metrics(yte, m.predict_pd(Xte))["auc"] > 0.6


def test_scorecard_coefficients_are_negative(models):
    sc = models[0].base
    assert (sc.coefficients < 0).all()


def test_woe_is_finite_with_empty_bins():
    import pandas as pd
    X = pd.DataFrame({"x": [1.0] * 50 + [2.0] * 50})
    y = pd.Series([0] * 50 + [1] * 50)  # bin 2 has zero goods
    b = WoEBinner(n_bins=4, smoothing=0.5).fit(X, y)
    assert np.isfinite(b.woe_["x"]).all()
    w = b.transform(X)["x"]
    assert w.iloc[0] > 0 > w.iloc[-1]  # all-good bin safer than all-bad bin


def test_contributions_shape(models, data):
    Xte = data[4].iloc[:20]
    for m in models:
        c = m.contributions(Xte)
        assert c.shape[0] == 20 and np.isfinite(c.to_numpy()).all()


def test_lgbm_contributions_sum_to_raw_logit(models, data):
    gb = models[1].base
    X = data[4].iloc[:50]
    full = gb.model_.predict_proba(X, pred_contrib=True)
    raw = gb.model_.predict_proba(X, raw_score=True)
    np.testing.assert_allclose(np.asarray(full).sum(axis=1), raw, rtol=1e-6, atol=1e-6)


def test_metric_definitions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = pd_metrics(y, p)
    assert m["auc"] == 1.0 and m["gini"] == 1.0 and ks_statistic(y, p) == 1.0
    assert calibration_table(y, p, n_bins=2)["n"].sum() == 4
