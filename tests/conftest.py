"""Shared fixtures. All data here is SYNTHETIC (credit_risk.synthetic)."""

import pytest

from credit_risk.config import Config
from credit_risk.data import parse_raw, select_population, time_split
from credit_risk.synthetic import make_synthetic_loans


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def synthetic_raw():
    return make_synthetic_loans(n=30_000, seed=1)


@pytest.fixture(scope="session")
def population(synthetic_raw, cfg):
    return select_population(parse_raw(synthetic_raw), cfg)


@pytest.fixture(scope="session")
def splits(population, cfg):
    return time_split(population, cfg)
