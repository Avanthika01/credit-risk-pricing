"""Discrimination and calibration metrics for PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max separation between the score CDFs of bads and goods
    (equivalently max TPR - FPR over thresholds)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def pd_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_pd": float(p.mean()),
        "observed_default_rate": float(y.mean()),
    }


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs observed default rate by predicted-PD decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean"))
    return t.reset_index()
