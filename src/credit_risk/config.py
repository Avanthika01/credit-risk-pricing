"""Central configuration: paths, split dates, modelling and pricing assumptions.

Every business assumption lives here as a dataclass field so it is visible,
reviewable and overridable in one place (a model-risk-management requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KAGGLE_DATASET = "wordsforthewise/lending-club"


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_file: Path = PROJECT_ROOT / "data" / "processed" / "loans_36m_2007_2015.parquet"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class SplitConfig:
    # Only 36-month loans issued through Dec 2015: the data runs to 2018Q4,
    # so every loan in scope has reached maturity and has a final outcome.
    term_months: int = 36
    max_issue_date: str = "2015-12-31"
    train_end: str = "2013-12-31"   # train: issued 2007..2013
    val_year: int = 2014            # validation: early stopping + calibration
    test_year: int = 2015           # untouched out-of-time holdout


@dataclass(frozen=True)
class FeatureConfig:
    # Categories below this share of training rows are grouped into OTHER.
    rare_category_min_share: float = 0.005
    # Number of quantile bins per numeric feature for the WoE scorecard.
    woe_bins: int = 10
    # Laplace pseudo-count added to goods and bads in every WoE bin.
    woe_smoothing: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_estimators": 3000,
            "verbose": -1,
        }
    )
    early_stopping_rounds: int = 100
    # L2 inverse regularisation for the scorecard logistic regression.
    scorecard_C: float = 1.0
    n_reason_codes: int = 4


@dataclass(frozen=True)
class DecisionConfig:
    # Approve if calibrated PD <= cutoff.
    pd_cutoff: float = 0.20
    # Fair-lending proxy review.
    min_group_size: int = 500
    air_threshold: float = 0.80  # four-fifths rule
    cutoff_grid: tuple = tuple(round(x, 2) for x in [0.05 + 0.025 * i for i in range(15)])


@dataclass(frozen=True)
class PricingConfig:
    cost_of_funds: float = 0.04
    operating_cost: float = 0.02
    margin: float = 0.03
    rate_floor: float = 0.06
    rate_cap: float = 0.36
    term_months: int = 36

    @property
    def average_life_years(self) -> float:
        # Simplifying assumption: average life = half the contractual term.
        return self.term_months / 12 / 2


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
