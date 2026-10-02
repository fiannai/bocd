"""Poisson observation model with Gamma conjugate prior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import nbinom

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PoissonModel:
    """Gamma-Poisson conjugate model for count/rate data.

    Sufficient statistics (sum_x, count) are numpy arrays indexed by
    run length.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self._alpha = alpha
        self._beta = beta
        self._sum_x: np.ndarray = np.array([0.0])
        self._count: np.ndarray = np.array([0.0])

    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        """Log-probability of x under Negative Binomial predictive."""
        alpha_n = self._alpha + self._sum_x
        beta_n = self._beta + self._count
        p = beta_n / (beta_n + 1.0)
        return np.asarray(nbinom.logpmf(x, n=alpha_n, p=p), dtype=np.float64)

    def _posterior_params(self) -> tuple[np.ndarray, np.ndarray]:
        return self._alpha + self._sum_x, self._beta + self._count

    def predictive_mean(self) -> NDArray[np.float64]:
        """Negative-binomial mean under each run length, alpha_n / beta_n."""
        alpha_n, beta_n = self._posterior_params()
        return np.asarray(alpha_n / beta_n, dtype=np.float64)

    def predictive_variance(self) -> NDArray[np.float64]:
        """Negative-binomial variance under each run length."""
        alpha_n, beta_n = self._posterior_params()
        return np.asarray(alpha_n * (beta_n + 1.0) / beta_n**2, dtype=np.float64)

    def update(self, x: float) -> None:
        """Add observation to sufficient statistics."""
        self._sum_x = np.concatenate([[0.0], self._sum_x + x])
        self._count = np.concatenate([[0.0], self._count + 1.0])

    def reset(self) -> None:
        """Reset to initial state."""
        self._sum_x = np.array([0.0])
        self._count = np.array([0.0])

    def copy(self) -> PoissonModel:
        """Deep copy with independent arrays."""
        m = PoissonModel.__new__(PoissonModel)
        m._alpha = self._alpha
        m._beta = self._beta
        m._sum_x = self._sum_x.copy()
        m._count = self._count.copy()
        return m

    def truncate(self, max_len: int) -> None:
        """Truncate sufficient statistics to max_len entries."""
        self._sum_x = self._sum_x[:max_len]
        self._count = self._count[:max_len]

    def validate_observation(self, x: float) -> None:
        """Reject negative or non-integer observations."""
        if x < 0:
            raise ValueError(f"PoissonModel requires x >= 0, got {x}")
        if x != int(x):
            raise ValueError(
                f"PoissonModel requires integer-valued observations, got {x}"
            )
