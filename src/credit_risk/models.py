"""PD models: a Weight-of-Evidence scorecard and LightGBM, each with isotonic calibration.

Both expose the same interface:
    fit(...)                  -> self
    predict_raw(X)            -> uncalibrated P(default)
    contributions(X)          -> per-feature contribution to log-odds of default,
                                 relative to a training-population reference
PDModel wraps a base model with an isotonic calibrator and produces reason codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from credit_risk.config import FeatureConfig, ModelConfig

PD_EPS = 1e-4


# ---------------------------------------------------------------------------
# Weight of Evidence
# ---------------------------------------------------------------------------
class WoEBinner:
    """Quantile binning + smoothed WoE, fitted on training data.

    WoE_b = ln( (good_b + a) / (G + k*a) )  -  ln( (bad_b + a) / (B + k*a) )

    where k = number of bins and a = Laplace pseudo-count. Positive WoE = safer
    than average. Smoothing keeps sparse bins finite and shrinks them toward 0.
    """

    def __init__(self, n_bins: int = 10, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.edges_: dict[str, np.ndarray] = {}   # numeric: inner cut points
        self.woe_: dict[str, pd.Series] = {}      # bin label -> WoE
        self.iv_: dict[str, float] = {}
        self.bin_table_: dict[str, pd.DataFrame] = {}

    def _bin(self, col: str, x: pd.Series) -> pd.Series:
        if isinstance(x.dtype, pd.CategoricalDtype):
            return x.astype(str)
        idx = np.searchsorted(self.edges_[col], x.to_numpy(dtype=float), side="right")
        return pd.Series(idx, index=x.index).astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        y = pd.Series(np.asarray(y), index=X.index)
        a = self.smoothing
        for col in X.columns:
            x = X[col]
            if not isinstance(x.dtype, pd.CategoricalDtype):
                qs = np.quantile(x.to_numpy(dtype=float), np.linspace(0, 1, self.n_bins + 1)[1:-1])
                self.edges_[col] = np.unique(qs)
            b = self._bin(col, x)
            if isinstance(x.dtype, pd.CategoricalDtype):
                labels = [str(c) for c in x.cat.categories]
            else:
                labels = [str(i) for i in range(len(self.edges_[col]) + 1)]
            bad = y.groupby(b).sum().reindex(labels, fill_value=0).astype(float)
            n = b.value_counts().reindex(labels, fill_value=0).astype(float)
            good = n - bad
            k = len(labels)
            dist_good = (good + a) / (good.sum() + k * a)
            dist_bad = (bad + a) / (bad.sum() + k * a)
            woe = np.log(dist_good / dist_bad)
            self.woe_[col] = woe
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())
            self.bin_table_[col] = pd.DataFrame(
                {"n": n, "bad": bad, "bad_rate": bad / n.where(n > 0), "woe": woe}
            )
        return self

    def transform(self, X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = cols or list(self.woe_)
        out = pd.DataFrame(index=X.index)
        for col in cols:
            out[col] = self._bin(col, X[col]).map(self.woe_[col]).fillna(0.0).astype(float)
        return out

    def bin_ranges(self, col: str) -> list[str]:
        """Human-readable bin labels for a numeric feature."""
        e = self.edges_[col]
        bounds = [-np.inf, *e, np.inf]
        return [f"({bounds[i]:.4g}, {bounds[i + 1]:.4g}]" for i in range(len(bounds) - 1)]


class WoEScorecard:
    """WoE transform + logistic regression, with IV screening and a sign constraint."""

    def __init__(self, fcfg: FeatureConfig, mcfg: ModelConfig):
        self.binner = WoEBinner(fcfg.woe_bins, fcfg.woe_smoothing)
        self.C = mcfg.scorecard_C
        self.min_iv = mcfg.scorecard_min_iv
        self.features_: list[str] = []
        self.dropped_low_iv_: list[str] = []
        self.dropped_sign_: list[str] = []
        self.lr_: LogisticRegression | None = None
        self.woe_mean_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEScorecard":
        self.binner.fit(X, y)
        feats = [c for c in X.columns if self.binner.iv_[c] >= self.min_iv]
        self.dropped_low_iv_ = [c for c in X.columns if c not in feats]
        W = self.binner.transform(X, feats)
        # Positive WoE = safer, so every coefficient on P(default) must be negative.
        # A positive coefficient signals collinearity; drop the feature and refit.
        while True:
            lr = LogisticRegression(C=self.C, max_iter=2000)
            lr.fit(W[feats], y)
            coefs = pd.Series(lr.coef_[0], index=feats)
            wrong = coefs[coefs > 0]
            if wrong.empty:
                break
            worst = wrong.idxmax()
            self.dropped_sign_.append(worst)
            feats = [f for f in feats if f != worst]
        self.features_ = feats
        self.lr_ = lr
        self.woe_mean_ = W[feats].mean()
        return self

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.lr_.coef_[0], index=self.features_)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.binner.transform(X, self.features_))[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """coef * (WoE - training mean WoE): contribution to log-odds of default
        relative to the average training applicant. Positive = raises risk."""
        W = self.binner.transform(X, self.features_)
        return (W - self.woe_mean_) * self.coefficients


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
class LGBMModel:
    def __init__(self, mcfg: ModelConfig):
        self.params = dict(mcfg.lgbm_params, random_state=mcfg.seed)
        self.early_stopping_rounds = mcfg.early_stopping_rounds
        self.model_ = None
        self.best_iteration_: int | None = None

    def fit(self, X_tr, y_tr, X_val, y_val) -> "LGBMModel":
        import lightgbm as lgb

        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(
            X_tr, y_tr,
            eval_X=(X_val,),
            eval_y=(y_val,),
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        self.best_iteration_ = int(self.model_.best_iteration_)
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log-odds space (LightGBM pred_contrib).
        The last column LightGBM returns is the expected value (baseline); dropped."""
        contrib = self.model_.predict_proba(X, pred_contrib=True)
        return pd.DataFrame(np.asarray(contrib)[:, :-1], index=X.index, columns=list(X.columns))

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model_.booster_.feature_importance(importance_type="gain"), index=self.model_.feature_name_
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Calibration and wrapper
# ---------------------------------------------------------------------------
class PDModel:
    """Base model + isotonic calibration fitted on the validation period."""

    def __init__(self, name: str, base):
        self.name = name
        self.base = base
        self.calibrator_: IsotonicRegression | None = None

    def calibrate(self, X_val: pd.DataFrame, y_val) -> "PDModel":
        self.calibrator_ = IsotonicRegression(y_min=PD_EPS, y_max=1 - PD_EPS, out_of_bounds="clip")
        self.calibrator_.fit(self.base.predict_raw(X_val), np.asarray(y_val))
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        p = self.calibrator_.predict(self.base.predict_raw(X))
        return np.clip(p, PD_EPS, 1 - PD_EPS)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        # Isotonic calibration is monotone, so the ranking of drivers from the
        # base model's log-odds contributions carries over to the calibrated PD.
        return self.base.contributions(X)
