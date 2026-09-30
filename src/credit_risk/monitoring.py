"""Population Stability Index (PSI) for score and feature drift.

PSI = sum_b (actual_b - expected_b) * ln(actual_b / expected_b)
with bins defined on the EXPECTED (training) distribution.
Conventional thresholds:
    < 0.10       stable, no action
    0.10 - 0.25  moderate shift, investigate
    > 0.25       significant shift, model review / possible redevelopment
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_EPS = 1e-4


def psi_status(v: float) -> str:
    if v < 0.10:
        return "stable"
    if v <= 0.25:
        return "moderate shift"
    return "significant shift"


def psi(expected, actual, n_bins: int = 10) -> float:
    e = pd.Series(expected)
    a = pd.Series(actual)
    if isinstance(e.dtype, pd.CategoricalDtype) or e.dtype == object or pd.api.types.is_string_dtype(e):
        levels = sorted(set(e.astype(str)) | set(a.astype(str)))
        pe = e.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
        pa = a.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
    else:
        edges = np.unique(np.quantile(e.to_numpy(float), np.linspace(0, 1, n_bins + 1)[1:-1]))
        be = np.searchsorted(edges, e.to_numpy(float), side="right")
        ba = np.searchsorted(edges, a.to_numpy(float), side="right")
        k = len(edges) + 1
        pe = np.bincount(be, minlength=k) / len(be)
        pa = np.bincount(ba, minlength=k) / len(ba)
    pe = np.clip(pe, PSI_EPS, None)
    pa = np.clip(pa, PSI_EPS, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_report(expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [{"variable": c, "psi": psi(expected[c], actual[c])} for c in columns]
    t = pd.DataFrame(rows)
    t["status"] = t["psi"].map(psi_status)
    return t
