"""Gaussian observation model with Normal-Inverse-Gamma conjugate prior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import t as student_t

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GaussianModel:
    """Normal-Inverse-Gamma conjugate model for continuous data.

    Sufficient statistics (sum_x, sum_x2, count) are numpy arrays
    indexed by run length — no Python loop over hypotheses.
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self._mu_0 = mu_0
        self._kappa = kappa
        self._alpha = alpha
        self._beta = beta
        self._sum_x: np.ndarray = np.array([0.0])
        self._sum_x2: np.ndarray = np.array([0.0])
        self._count: np.ndarray = np.array([0.0])

    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        """Log-probability of x under each run-length hypothesis."""
        count = self._count
        count_safe = np.maximum(count, 1.0)

        kappa_n = self._kappa + count
        mu_n = (self._kappa * self._mu_0 + self._sum_x) / kappa_n
        alpha_n = self._alpha + count / 2.0

        ss = self._sum_x2 - self._sum_x**2 / count_safe
        prior_data = (
            self._kappa
            * count
            * (self._sum_x / count_safe - self._mu_0) ** 2
            / (2.0 * kappa_n)
        )
        beta_n = self._beta + 0.5 * ss + prior_data

        df = 2.0 * alpha_n
        loc = mu_n
        scale = np.sqrt(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n))

        return np.asarray(
            student_t.logpdf(x, df=df, loc=loc, scale=scale),
            dtype=np.float64,
        )

    def update(self, x: float) -> None:
        """Add observation to sufficient statistics."""
        self._sum_x = np.concatenate([[0.0], self._sum_x + x])
        self._sum_x2 = np.concatenate([[0.0], self._sum_x2 + x * x])
        self._count = np.concatenate([[0.0], self._count + 1.0])

    def reset(self) -> None:
        """Reset to initial state."""
        self._sum_x = np.array([0.0])
        self._sum_x2 = np.array([0.0])
        self._count = np.array([0.0])

    def copy(self) -> GaussianModel:
        """Deep copy with independent arrays."""
        m = GaussianModel.__new__(GaussianModel)
        m._mu_0 = self._mu_0
        m._kappa = self._kappa
        m._alpha = self._alpha
        m._beta = self._beta
        m._sum_x = self._sum_x.copy()
        m._sum_x2 = self._sum_x2.copy()
        m._count = self._count.copy()
        return m

    def truncate(self, max_len: int) -> None:
        """Truncate sufficient statistics to max_len entries."""
        self._sum_x = self._sum_x[:max_len]
        self._sum_x2 = self._sum_x2[:max_len]
        self._count = self._count[:max_len]

    def validate_observation(self, x: float) -> None:
        """No-op — Gaussian accepts all finite values."""
