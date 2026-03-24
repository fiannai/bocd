"""Protocol definitions for observation models and hazard functions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ObservationModel(Protocol):
    """Contract for BOCD observation models.

    Models maintain vectorized sufficient statistics indexed by run length.
    The core algorithm calls log_predictive_prob(x) before update(x) within
    each step, so predictive probabilities reflect state *before* the
    observation is incorporated.
    """

    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        """Log-probability of x under each run-length hypothesis.

        Returns shape (t,) where t is the current number of run-length
        hypotheses.
        """
        ...

    def update(self, x: float) -> None:
        """Update sufficient statistics with new observation.

        Mutates internal arrays in place. Grows arrays by one element
        (prepends a fresh run-length-0 hypothesis).
        """
        ...

    def reset(self) -> None:
        """Reset to initial state (prior parameters only)."""
        ...

    def copy(self) -> ObservationModel:
        """Deep copy with independent numpy arrays."""
        ...

    def truncate(self, max_len: int) -> None:
        """Truncate sufficient statistics arrays to max_len entries.

        Called by Detector after each step when max_run_length is active.
        Slice all run-length-indexed arrays to [:max_len].
        """
        ...

    def validate_observation(self, x: float) -> None:
        """Validate that x is in this model's input domain.

        Default: no-op (accepts all finite values). Override to enforce
        model-specific constraints. Raises ValueError if x is invalid.
        """
        ...


@runtime_checkable
class HazardFunction(Protocol):
    """Contract for BOCD hazard functions.

    Must return values in [0, 1]. Values outside this range produce
    invalid probabilities in the growth/changepoint equations.
    """

    def __call__(self, run_lengths: NDArray[np.int64]) -> NDArray[np.float64]:
        """P(changepoint | run_length) for each hypothesis."""
        ...
