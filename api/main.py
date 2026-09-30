"""FastAPI scoring service.

POST /score  - application -> PD, risk-based rate, decision, adverse-action reason codes
GET  /health - liveness + which model is loaded

The model bundle (models/model_bundle.joblib, produced by train.py) contains the
fitted FeatureBuilder, so serving uses exactly the training-time transform.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from credit_risk.explain import reason_codes
from credit_risk.pricing import risk_based_rate

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_bundle.joblib"

app = FastAPI(title="Credit risk PD & pricing API", version="0.1.0")


class Application(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=40_000, description="Requested amount, USD")
    term_months: Literal[36] = Field(36, description="Model covers 36-month loans only")
    annual_inc: float = Field(..., gt=0, le=10_000_000)
    emp_length_years: Optional[int] = Field(None, ge=0, le=10, description="10 means 10+; null if not provided")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"]
    verification_status: Literal["Not Verified", "Source Verified", "Verified"]
    purpose: str = Field(..., min_length=1, max_length=50, examples=["debt_consolidation"])
    application_type: Literal["Individual", "Joint App"] = "Individual"
    dti: float = Field(..., ge=0, le=100)
    revol_bal: float = Field(..., ge=0)
    revol_util: Optional[float] = Field(None, ge=0, le=200)
    open_acc: int = Field(..., ge=0)
    total_acc: int = Field(..., ge=0)
    delinq_2yrs: int = Field(0, ge=0)
    inq_last_6mths: int = Field(0, ge=0)
    pub_rec: int = Field(0, ge=0)
    pub_rec_bankruptcies: Optional[int] = Field(0, ge=0)
    mort_acc: Optional[int] = Field(None, ge=0)
    mths_since_last_delinq: Optional[int] = Field(None, ge=0, description="null if never delinquent")
    earliest_cr_line: date
    fico_range_low: int = Field(..., ge=300, le=850)
    fico_range_high: int = Field(..., ge=300, le=850)
    application_date: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def _consistency(self):
        if self.fico_range_high < self.fico_range_low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        if self.open_acc > self.total_acc:
            raise ValueError("open_acc cannot exceed total_acc")
        if self.earliest_cr_line >= self.application_date:
            raise ValueError("earliest_cr_line must be before application_date")
        return self

    def to_raw_frame(self) -> pd.DataFrame:
        """Convert to the raw Lending Club schema expected by FeatureBuilder."""
        d = self.model_dump()
        e = d.pop("emp_length_years")
        d["emp_length"] = None if e is None else ("< 1 year" if e == 0 else "10+ years" if e == 10
                                                  else f"{e} year" + ("s" if e > 1 else ""))
        d["issue_d"] = pd.Timestamp(d.pop("application_date"))
        d["earliest_cr_line"] = pd.Timestamp(d["earliest_cr_line"])
        d["term"] = d.pop("term_months")
        return pd.DataFrame([d])


class ReasonCode(BaseModel):
    feature: str
    reason: str
    contribution: float


class ScoreResponse(BaseModel):
    pd: float = Field(..., ge=0, le=1)
    rate: float
    decision: Literal["approve", "decline"]
    pd_cutoff: float
    reason_codes: list[ReasonCode]
    model: str
    data_source: str


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found at {p}. Run train.py first.")
    return joblib.load(p)


@app.get("/health")
def health() -> dict:
    try:
        b = load_bundle()
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}
    return {"status": "ok", "model_loaded": True, "model": b["champion"].name,
            "data_source": b["data_source"], "trained_at": b["trained_at"]}


@app.post("/score", response_model=ScoreResponse)
def score(application: Application) -> ScoreResponse:
    try:
        b = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    cfg = b["config"]
    X = b["feature_builder"].transform(application.to_raw_frame())
    model = b["champion"]
    pd_ = float(model.predict_pd(X)[0])
    codes = reason_codes(model.contributions(X), n=cfg.model.n_reason_codes)[0]
    return ScoreResponse(
        pd=pd_,
        rate=float(risk_based_rate(pd_, b["lgd"], cfg.pricing)),
        decision="approve" if pd_ <= cfg.decision.pd_cutoff else "decline",
        pd_cutoff=cfg.decision.pd_cutoff,
        reason_codes=[ReasonCode(**c) for c in codes],
        model=model.name,
        data_source=b["data_source"],
    )
