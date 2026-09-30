import numpy as np
import pandas as pd

from credit_risk.config import PricingConfig
from credit_risk.fairness import adverse_impact, fairness_summary, zip3
from credit_risk.pricing import benchmark_by_grade, cutoff_table, estimate_lgd, risk_based_rate


def test_pricing_is_monotonic_in_pd():
    cfg = PricingConfig()
    grid = np.linspace(0, 1, 1001)
    r = risk_based_rate(grid, lgd=0.9, cfg=cfg)
    assert np.all(np.diff(r) >= 0)
    assert r.min() >= cfg.rate_floor and r.max() <= cfg.rate_cap
    assert r[-1] == cfg.rate_cap


def test_pricing_formula_and_floor():
    cfg = PricingConfig()
    # PD 10%, LGD 90%, avg life 1.5y -> EL 6%/yr; 4% + 6% + 2% + 3% = 15%
    assert np.isclose(risk_based_rate(0.10, 0.9, cfg), 0.15)
    # zero-risk loan: 9% base cost > 6% floor
    assert np.isclose(risk_based_rate(0.0, 0.9, cfg), 0.09)
    assert np.isclose(risk_based_rate(0.0, 0.9, PricingConfig(margin=0.0, operating_cost=0.0)), 0.06)


def test_lgd_exposure_weighted():
    d = pd.DataFrame({"funded_amnt": [10_000, 1_000], "total_rec_prncp": [2_000, 900],
                      "recoveries": [1_000, 100], "collection_recovery_fee": [200, 0]})
    # EAD 8000 + 100 = 8100; net recoveries 800 + 100 = 900 -> LGD = 1 - 900/8100
    assert np.isclose(estimate_lgd(d)["lgd"], 1 - 900 / 8100)


def test_adverse_impact_ratio_and_min_group():
    df = pd.DataFrame({
        "state": ["A"] * 600 + ["B"] * 600 + ["C"] * 100,
        "pd": [0.1] * 600 + [0.1] * 300 + [0.3] * 300 + [0.3] * 100,
        "default": [0] * 1300,
    })
    t = adverse_impact(df, "state", cutoff=0.2, min_group_size=500)
    assert set(t["state"]) == {"A", "B"}  # C excluded (< 500)
    b = t.set_index("state").loc["B"]
    assert np.isclose(b["air"], 0.5) and b["flagged"]
    assert fairness_summary(t)["flagged_groups"] == ["B"]


def test_zip3():
    assert zip3(pd.Series(["945xx", "100xx"])).tolist() == ["945", "100"]


def test_cutoff_table_monotone_approval():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    t = cutoff_table(y, p, [0.1, 0.2, 0.3, 0.5])
    assert t["approval_rate"].is_monotonic_increasing
    assert t["bad_rate_approved"].is_monotonic_increasing


def test_benchmark_by_grade_units():
    df = pd.DataFrame({"grade": ["A", "A", "B"], "int_rate": [7.0, 8.0, 12.0], "model_rate": [0.08, 0.08, 0.13],
                       "pd": [0.05, 0.05, 0.1], "default": [0, 1, 0]})
    t = benchmark_by_grade(df).set_index("grade")
    assert np.isclose(t.loc["A", "lc_mean_rate"], 0.075)
