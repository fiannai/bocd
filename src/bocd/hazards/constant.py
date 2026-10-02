"""Constant hazard rate (memoryless / geometric prior on run length)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConstantHazard:
    """Constant hazard: P(changepoint | run_length) = rate for all run lengths.

    Parameter: rate = 1 / expected_run_length. Default 1/200.

    Under this hazard, the posterior mass on run length 0 equals ``rate``
    at every step until truncation. Regime changes show up in the rest
    of the run-length posterior, not in that single bin.
    """

    def __init__(self, rate: float = 1 / 200) -> None:
        if not (0 < rate < 1):
            raise ValueError(f"rate must be in (0, 1) exclusive, got {rate}")
        self._rate = rate

    def __call__(self, run_lengths: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return constant hazard for each run length."""
        return np.full(run_lengths.shape, self._rate, dtype=np.float64)
