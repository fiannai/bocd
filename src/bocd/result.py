"""Result dataclass returned by Detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Result:
    """Single-step BOCD result.

    Each Result contains an independent copy of the posterior array,
    safe to store and pass around after subsequent update() calls.
    """

    changepoint_prob: float
    """P(run_length = 0).

    With a constant hazard and no truncation this equals the hazard
    rate on every step, so it does not rise at a regime change. Use
    :meth:`mass_at_most` or :func:`bocd.detect.detect_changepoints`.
    """

    run_length_posterior: NDArray[np.float64]
    """Full posterior distribution over run lengths (probability space)."""

    most_likely_run_length: int
    """argmax of the posterior."""

    step: int
    """0-indexed observation number."""

    def mass_at_most(self, run_length: int) -> float:
        """Posterior probability that the current run length is <= ``run_length``."""
        if isinstance(run_length, bool) or not isinstance(run_length, int):
            raise TypeError(
                f"run_length must be an int, got {type(run_length).__name__}"
            )
        if run_length < 0:
            raise ValueError(f"run_length must be >= 0, got {run_length}")
        return float(self.run_length_posterior[: run_length + 1].sum())
