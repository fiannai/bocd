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
    """P(run_length = 0) — probability a changepoint just occurred."""

    run_length_posterior: NDArray[np.float64]
    """Full posterior distribution over run lengths (probability space)."""

    most_likely_run_length: int
    """argmax of the posterior."""

    step: int
    """0-indexed observation number."""
