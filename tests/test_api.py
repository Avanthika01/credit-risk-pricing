"""API tests using a model trained on SYNTHETIC data."""

import pytest
from fastapi.testclient import TestClient

from credit_risk.config import Config
from credit_risk.pipeline import run

VALID = {
    "loan_amnt": 12000, "annual_inc": 65000, "emp_length_years": 4, "home_ownership": "RENT",
    "verification_status": "Verified", "purpose": "debt_consolidation", "dti": 22.5, "revol_bal": 14000,
    "revol_util": 71.0, "open_acc": 9, "total_acc": 21, "delinq_2yrs": 0, "inq_last_6mths": 2, "pub_rec": 0,
    "mort_acc": 0, "earliest_cr_line": "2006-05-01", "fico_range_low": 675, "fico_range_high": 679,
    "application_date": "2015-06-01",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    tmp = tmp_path_factory.mktemp("api")
    run(Config(), synthetic_rows=20_000, out_dir=tmp / "reports", models_dir=tmp / "models", use_mlflow=False)
    monkeypatch_module.setenv("MODEL_PATH", str(tmp / "models" / "model_bundle.joblib"))
    from api import main

    main.load_bundle.cache_clear()
    yield TestClient(main.app)
    main.load_bundle.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["data_source"] == "SYNTHETIC"


def test_score_returns_valid_fields(client):
    r = client.post("/score", json=VALID)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0 <= b["pd"] <= 1
    assert 0.06 <= b["rate"] <= 0.36
    assert b["decision"] in {"approve", "decline"}
    assert (b["decision"] == "approve") == (b["pd"] <= b["pd_cutoff"])
    assert len(b["reason_codes"]) <= 4
    assert all(c["contribution"] > 0 for c in b["reason_codes"])


def test_riskier_applicant_gets_higher_pd_and_rate(client):
    safe = client.post("/score", json={**VALID, "fico_range_low": 790, "fico_range_high": 794, "dti": 5,
                                       "revol_util": 10, "annual_inc": 150000}).json()
    risky = client.post("/score", json={**VALID, "fico_range_low": 662, "fico_range_high": 666, "dti": 38,
                                        "revol_util": 98, "annual_inc": 30000}).json()
    assert risky["pd"] >= safe["pd"] and risky["rate"] >= safe["rate"]


def test_missing_optional_fields_ok(client):
    body = {k: v for k, v in VALID.items() if k not in {"emp_length_years", "revol_util", "mort_acc"}}
    assert client.post("/score", json=body).status_code == 200


@pytest.mark.parametrize("patch", [
    {"loan_amnt": -5}, {"term_months": 60}, {"fico_range_low": 700, "fico_range_high": 690},
    {"open_acc": 30}, {"home_ownership": "CASTLE"}, {"earliest_cr_line": "2020-01-01"},
])
def test_invalid_requests_rejected(client, patch):
    assert client.post("/score", json={**VALID, **patch}).status_code == 422
