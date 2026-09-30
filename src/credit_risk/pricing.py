"""LGD estimation and risk-based pricing.

rate = cost_of_funds + PD * LGD / average_life + operating_cost + margin,
clipped to [floor, cap].

PD is the lifetime (36-month) probability of default, so PD * LGD is lifetime
expected loss per unit of exposure; dividing by the average life (half the term
for an amortising loan) converts it to an annual loss rate comparable to an APR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import PricingConfig


def estimate_lgd(defaults: pd.DataFrame) -> dict:
    """Exposure-weighted LGD from defaulted loans (use TRAINING-period defaults only).

    EAD = funded_amnt - total_rec_prncp  (principal outstanding at default)
    net recovery = recoveries - collection_recovery_fee
    LGD = 1 - sum(net recovery) / sum(EAD)
    Exposure weighting (ratio of sums) lets large losses count proportionally,
    rather than averaging per-loan ratios where tiny exposures are noisy.
    """
    ead = (defaults["funded_amnt"] - defaults["total_rec_prncp"]).astype(float)
    net_rec = (defaults["recoveries"] - defaults["collection_recovery_fee"]).astype(float)
    ok = ead > 0
    total_ead = float(ead[ok].sum())
    lgd = 1 - float(net_rec[ok].sum()) / total_ead
    return {"lgd": float(np.clip(lgd, 0, 1)), "n_defaults": int(ok.sum()), "total_ead": total_ead,
            "net_recoveries": float(net_rec[ok].sum())}


def risk_based_rate(pd_: np.ndarray | float, lgd: float, cfg: PricingConfig) -> np.ndarray:
    pd_ = np.asarray(pd_, dtype=float)
    annual_expected_loss = pd_ * lgd / cfg.average_life_years
    rate = cfg.cost_of_funds + annual_expected_loss + cfg.operating_cost + cfg.margin
    return np.clip(rate, cfg.rate_floor, cfg.rate_cap)


def benchmark_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Compare model rates with Lending Club's actual rates by LC grade.
    Expects columns: grade, int_rate (percent), model_rate (decimal), pd, default."""
    t = df.groupby("grade").agg(
        n=("default", "size"),
        observed_default_rate=("default", "mean"),
        mean_model_pd=("pd", "mean"),
        lc_mean_rate=("int_rate", lambda s: s.mean() / 100),
        model_mean_rate=("model_rate", "mean"),
    )
    t["rate_diff"] = t["model_mean_rate"] - t["lc_mean_rate"]
    return t.reset_index()


def cutoff_table(y, pd_, cutoffs) -> pd.DataFrame:
    """Approval rate vs bad rate among approved across PD cutoffs (approve if PD <= cutoff)."""
    y = np.asarray(y)
    pd_ = np.asarray(pd_, dtype=float)
    rows = []
    for c in cutoffs:
        appr = pd_ <= c
        rows.append({
            "pd_cutoff": float(c),
            "approval_rate": float(appr.mean()),
            "n_approved": int(appr.sum()),
            "bad_rate_approved": float(y[appr].mean()) if appr.any() else np.nan,
            "bad_rate_declined": float(y[~appr].mean()) if (~appr).any() else np.nan,
        })
    return pd.DataFrame(rows)
