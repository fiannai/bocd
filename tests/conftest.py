"""Shared test fixtures — seeded, reproducible synthetic series."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def gaussian_step_series() -> NDArray[np.float64]:
    """Mean shifts 0 -> 5 at t=200. Total length 400."""
    rng = np.random.default_rng(42)
    return np.concatenate([
        rng.normal(0, 1, 200),
        rng.normal(5, 1, 200),
    ])


@pytest.fixture
def poisson_rate_change() -> NDArray[np.float64]:
    """Rate changes 3 -> 10 -> 3 at t=150, 300. Total length 450."""
    rng = np.random.default_rng(42)
    return np.concatenate([
        rng.poisson(3, 150),
        rng.poisson(10, 150),
        rng.poisson(3, 150),
    ]).astype(np.float64)


@pytest.fixture
def multi_changepoint_series() -> NDArray[np.float64]:
    """5 regime changes with varying gaps. Total length 600."""
    rng = np.random.default_rng(42)
    segments = []
    means = [0.0, 5.0, -3.0, 8.0, 1.0, 6.0]
    lengths = [100, 80, 120, 60, 150, 90]
    for mean, length in zip(means, lengths):
        segments.append(rng.normal(mean, 1, length))
    return np.concatenate(segments)


@pytest.fixture
def no_changepoint_series() -> NDArray[np.float64]:
    """Stationary — detector should stay quiet. Length 500."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 500)


@pytest.fixture
def gradual_drift_series() -> NDArray[np.float64]:
    """Slow linear drift — tests sensitivity boundary. Length 500."""
    rng = np.random.default_rng(42)
    t = np.arange(500, dtype=np.float64)
    return t * 0.01 + rng.normal(0, 1, 500)
