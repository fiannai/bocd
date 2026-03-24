"""Stateful BOCD detector with streaming and batch APIs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from bocd import _core
from bocd._typing import HazardFunction, ObservationModel
from bocd.result import Result

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Detector:
    """Stateful wrapper over the core BOCD algorithm.

    Provides streaming (update) and batch (fit) APIs, input validation,
    optional run-length truncation, and state management.
    """

    def __init__(
        self,
        model: ObservationModel | None = None,
        hazard: HazardFunction | None = None,
        *,
        max_run_length: int | None = None,
    ) -> None:
        if model is None:
            from bocd.models.gaussian import GaussianModel

            model = GaussianModel()
        if hazard is None:
            from bocd.hazards.constant import ConstantHazard

            hazard = ConstantHazard()

        if not isinstance(model, ObservationModel):
            raise TypeError(
                f"model must satisfy ObservationModel protocol, "
                f"got {type(model).__name__}"
            )
        if not isinstance(hazard, HazardFunction):
            raise TypeError(
                f"hazard must satisfy HazardFunction protocol, "
                f"got {type(hazard).__name__}"
            )

        self._model = model
        self._hazard = hazard
        self._max_run_length = max_run_length
        self._joint = np.array([0.0], dtype=np.float64)
        self._step_count = 0

    # --- Streaming ---

    def update(self, x: float) -> Result:
        """Process one observation. Raises ValueError if x is non-finite."""
        if not math.isfinite(x):
            raise ValueError(f"x must be finite, got {x}")
        self._model.validate_observation(x)

        self._joint, _ = _core.step(x, self._joint, self._model, self._hazard)

        if self._max_run_length is not None and len(self._joint) > self._max_run_length:
            self._joint = self._joint[: self._max_run_length]
            self._joint -= logsumexp(self._joint)
            self._model.truncate(self._max_run_length)

        posterior = np.exp(self._joint)
        result = Result(
            changepoint_prob=float(posterior[0]),
            run_length_posterior=posterior,
            most_likely_run_length=int(np.argmax(self._joint)),
            step=self._step_count,
        )
        self._step_count += 1
        return result

    def reset(self) -> None:
        """Restore to initial state."""
        self._joint = np.array([0.0], dtype=np.float64)
        self._model.reset()
        self._step_count = 0

    # --- Batch ---

    def fit(self, observations: ArrayLike) -> list[Result]:
        """Batch mode. Resets state, then processes all observations."""
        try:
            arr = np.asarray(observations, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"observations must be convertible to a 1-D numeric array: {e}"
            ) from e

        if arr.ndim != 1:
            raise TypeError(f"observations must be 1-D, got shape {arr.shape}")

        if len(arr) == 0:
            self.reset()
            return []

        if not np.all(np.isfinite(arr)):
            raise ValueError("All observations must be finite (no NaN or Inf)")

        self.reset()
        results: list[Result] = []
        for x in arr:
            results.append(self.update(float(x)))
        return results

    # --- Snapshot ---

    def copy(self) -> Detector:
        """Independent clone."""
        d = Detector.__new__(Detector)
        d._model = self._model.copy()
        d._hazard = self._hazard
        d._max_run_length = self._max_run_length
        d._joint = self._joint.copy()
        d._step_count = self._step_count
        return d

    # --- Inspection ---

    @property
    def step_count(self) -> int:
        """Number of observations processed since last reset."""
        return self._step_count

    @property
    def run_length_posterior(self) -> NDArray[np.float64]:
        """Current posterior (probability space, independent copy)."""
        return np.exp(self._joint.copy())
