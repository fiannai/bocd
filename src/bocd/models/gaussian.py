"""Gaussian observation model with Normal-Inverse-Gamma conjugate prior."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import t as student_t

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _beta_for_scale(scale: float, kappa: float, alpha: float) -> float:
    """NIG beta whose prior Student-t scale equals ``scale``.

    scale^2 = beta * (kappa + 1) / (alpha * kappa).
    """
    return scale**2 * alpha * kappa / (kappa + 1.0)


class GaussianModel:
    """Normal-Inverse-Gamma conjugate model for continuous data.

    Sufficient statistics are numpy arrays indexed by run length.
    ``_m2`` is Welford's sum of squared deviations from the running mean.
    The textbook ``sum(x^2) - sum(x)^2 / n`` cancels in float64 once
    observations reach about 1e8, and the predictive scale collapses.

    Defaults (``mu_0=0``, ``kappa=1``, ``alpha=1``, ``beta=1``) are a
    prior predictive Student-t centered at 0 with scale ``sqrt(2)``
    (degrees of freedom 2). Observations far from that scale inflate
    ``beta`` through prior-data conflict, and mean shifts stay hidden
    until the conflict is diluted. Use :meth:`from_scale` to match the
    prior to the series.
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
        self._mean: np.ndarray = np.array([0.0])
        self._m2: np.ndarray = np.array([0.0])
        self._count: np.ndarray = np.array([0.0])

    @classmethod
    def from_scale(
        cls,
        mean: float,
        scale: float,
        *,
        kappa: float = 1.0,
        alpha: float = 1.0,
    ) -> GaussianModel:
        """Prior whose predictive scale matches ``scale`` at ``mean``.

        ``scale`` is the Student-t scale of the prior predictive, not
        its standard deviation. With the default ``alpha=1`` the prior
        degrees of freedom are 2, so the variance is infinite and
        ``scale`` is the scale parameter SciPy uses. Pass ``alpha > 1``
        when a finite predictive variance is required: the prior
        degrees of freedom are ``2 * alpha``.

        ``GaussianModel()`` is ``from_scale(0, sqrt(2))``.
        """
        if not math.isfinite(mean):
            raise ValueError(f"mean must be finite, got {mean}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and > 0, got {scale}")
        beta = _beta_for_scale(scale, kappa, alpha)
        return cls(mu_0=mean, kappa=kappa, alpha=alpha, beta=beta)

    def _posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-run NIG posterior: kappa_n, mu_n, alpha_n, beta_n, df, scale."""
        count = self._count
        kappa_n = self._kappa + count
        sum_x = self._mean * count
        mu_n = (self._kappa * self._mu_0 + sum_x) / kappa_n
        alpha_n = self._alpha + count / 2.0

        x_bar = np.where(count > 0, self._mean, 0.0)
        prior_data = self._kappa * count * (x_bar - self._mu_0) ** 2 / (2.0 * kappa_n)
        beta_n = self._beta + 0.5 * self._m2 + prior_data

        df = 2.0 * alpha_n
        scale = np.sqrt(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n))
        return kappa_n, mu_n, alpha_n, beta_n, df, scale

    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        """Log-probability of x under each run-length hypothesis."""
        _kappa_n, mu_n, _alpha_n, _beta_n, df, scale = self._posterior()
        return np.asarray(
            student_t.logpdf(x, df=df, loc=mu_n, scale=scale),
            dtype=np.float64,
        )

    def predictive_mean(self) -> NDArray[np.float64]:
        """Student-t mean under each run length. NaN where df <= 1."""
        _kappa_n, mu_n, _alpha_n, _beta_n, df, _scale = self._posterior()
        mean = np.array(mu_n, dtype=np.float64, copy=True)
        mean[df <= 1.0] = np.nan
        return mean

    def predictive_variance(self) -> NDArray[np.float64]:
        """Student-t variance under each run length. Inf where df <= 2."""
        _kappa_n, _mu_n, _alpha_n, _beta_n, df, scale = self._posterior()
        variance = np.full(df.shape, np.inf, dtype=np.float64)
        finite = df > 2.0
        variance[finite] = scale[finite] ** 2 * df[finite] / (df[finite] - 2.0)
        return variance

    def update(self, x: float) -> None:
        """Add observation to sufficient statistics.

        The length-0 slot stores mean 0 as a placeholder. The first
        observation on that slot sets the mean to ``x`` and ``m2`` to 0.
        """
        n = self._count
        n_new = n + 1.0
        delta = x - self._mean
        mean_new = self._mean + delta / n_new
        m2_new = np.maximum(self._m2 + delta * (x - mean_new), 0.0)
        self._mean = np.concatenate([[0.0], mean_new])
        self._m2 = np.concatenate([[0.0], m2_new])
        self._count = np.concatenate([[0.0], n_new])

    def reset(self) -> None:
        """Reset to initial state."""
        self._mean = np.array([0.0])
        self._m2 = np.array([0.0])
        self._count = np.array([0.0])

    def copy(self) -> GaussianModel:
        """Deep copy with independent arrays."""
        m = GaussianModel.__new__(GaussianModel)
        m._mu_0 = self._mu_0
        m._kappa = self._kappa
        m._alpha = self._alpha
        m._beta = self._beta
        m._mean = self._mean.copy()
        m._m2 = self._m2.copy()
        m._count = self._count.copy()
        return m

    def truncate(self, max_len: int) -> None:
        """Truncate sufficient statistics to max_len entries."""
        self._mean = self._mean[:max_len]
        self._m2 = self._m2[:max_len]
        self._count = self._count[:max_len]

    def validate_observation(self, x: float) -> None:
        """No-op — Gaussian accepts all finite values."""
