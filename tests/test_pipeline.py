"""End-to-end smoke test on SYNTHETIC data."""

import json

from credit_risk.config import Config
from credit_risk.pipeline import choose_champion, run


def test_pipeline_end_to_end_synthetic(tmp_path):
    m = run(Config(), synthetic_rows=25_000, out_dir=tmp_path / "reports", models_dir=tmp_path / "models",
            use_mlflow=False)
    assert m["data_source"] == "SYNTHETIC"
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["data_source"] == "SYNTHETIC"
    for f in ["roc.png", "calibration.png", "cutoff_curve.png", "pricing_by_grade.png", "model_performance.md",
              "fairness_addr_state.md", "psi.md"]:
        assert (tmp_path / "reports" / f).exists(), f
    assert "SYNTHETIC DATA" in (tmp_path / "reports" / "model_performance.md").read_text()
    assert (tmp_path / "models" / "model_bundle.joblib").exists()
    for name in ["scorecard", "lightgbm"]:
        assert 0 <= m["test_calibrated"][name]["mean_pd"] <= 1
    assert 0 <= m["lgd"]["lgd"] <= 1
    assert m["splits"]["train"]["issue_d_max"] < m["splits"]["validation"]["issue_d_min"]


def test_champion_rule():
    base = {"scorecard": {"gini": 0.40}}
    assert choose_champion({**base, "lightgbm": {"gini": 0.44}}, 0.03)[0] == "lightgbm"
    assert choose_champion({**base, "lightgbm": {"gini": 0.42}}, 0.03)[0] == "scorecard"
