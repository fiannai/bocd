"""Stateful BOCD detector with streaming and batch APIs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from bocd import _core
from bocd._typing import HazardFunction, ObservationModel
from bocd.hazards.constant import ConstantHazard
from bocd.models.gaussian import GaussianModel
from bocd.result import Result

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Detector:
    """Stateful wrapper over the core BOCD algorithm.

    Provides streaming (update) and batch (fit) APIs, input validation,
    optional run-length truncation, and state management.

    The default model is ``GaussianModel()``, a prior for observations
    near 0 with noise on the order of 1. For other units pass
    ``GaussianModel.from_scale(mean, scale)``.

    ``changepoint_prob`` on each result is ``P(run length = 0)``. With a
    constant hazard and no truncation that equals the hazard rate on
    every step. Detection uses the run-length posterior; see
    ``bocd.detect``.
    """

    def __init__(
        self,
        model: ObservationModel | None = None,
        hazard: HazardFunction | None = None,
        *,
        max_run_length: int | None = None,
        prune_threshold: float | None = None,
    ) -> None:
        if model is None:
            model = GaussianModel()
        if hazard is None:
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
        if max_run_length is not None and (
            isinstance(max_run_length, bool)
            or not isinstance(max_run_length, int)
            or max_run_length < 2
        ):
            raise ValueError(
                f"max_run_length must be an int >= 2, got {max_run_length}"
            )
        if prune_threshold is not None and (
            isinstance(prune_threshold, bool)
            or not isinstance(prune_threshold, (int, float))
            or not math.isfinite(float(prune_threshold))
            or not 0.0 < float(prune_threshold) < 1.0
        ):
            raise ValueError(
                f"prune_threshold must be in (0, 1), got {prune_threshold}"
            )

        self._model = model
        self._hazard = hazard
        self._max_run_length = max_run_length
        self._prune_threshold = (
            None if prune_threshold is None else float(prune_threshold)
        )
        self._joint = np.array([0.0], dtype=np.float64)
        self._step_count = 0

    # --- Streaming ---

    def update(self, x: float) -> Result:
        """Process one observation. Raises ValueError if x is non-finite."""
        if not math.isfinite(x):
            raise ValueError(f"x must be finite, got {x}")
        self._model.validate_observation(x)

        self._joint, _ = _core.step(x, self._joint, self._model, self._hazard)
        self._apply_truncation()

        posterior = np.exp(self._joint)
        result = Result(
            changepoint_prob=float(posterior[0]),
            run_length_posterior=posterior,
            most_likely_run_length=int(np.argmax(self._joint)),
            step=self._step_count,
        )
        self._step_count += 1
        return result

    def _apply_truncation(self) -> None:
        """Drop a negligible long-run tail, then enforce the hard cap.

        Tail pruning discards run lengths whose combined mass is below
        ``prune_threshold``. That is safe when a changepoint has already
        killed the long runs. It does not bound a single regime whose
        mode sits on the longest run.

        The hard cap folds probability for run lengths beyond
        ``max_run_length`` into the last kept bin. That bin's sufficient
        statistics stay the statistics of run length ``max_run_length - 1``
        (the most recent points only). Longer runs do not share one
        statistic vector, so the folded bin is an approximation that
        keeps the belief "this regime is already long" instead of
        deleting it.
        """
        if self._prune_threshold is not None:
            self._prune_tail(self._prune_threshold)
        cap = self._max_run_length
        if cap is not None and len(self._joint) > cap:
            self._fold_cap(cap)

    def _prune_tail(self, threshold: float) -> None:
        probs = np.exp(self._joint)
        tail = np.cumsum(probs[::-1])[::-1]
        # tail[k] = P(r >= k). Keep the shortest prefix whose discarded
        # suffix has mass below the threshold. tail[0] is 1, so the cut
        # is at least 1.
        cuts = np.flatnonzero(tail < threshold)
        if len(cuts) == 0:
            return
        keep = int(cuts[0])
        if keep < 1 or keep >= len(self._joint):
            return
        self._joint = self._joint[:keep]
        self._joint -= logsumexp(self._joint)
        self._model.truncate(keep)

    def _fold_cap(self, cap: int) -> None:
        kept = self._joint[:cap].copy()
        kept[-1] = logsumexp(self._joint[cap - 1 :])
        kept -= logsumexp(kept)
        self._joint = kept
        self._model.truncate(cap)

    def reset(self) -> None:
        """Restore to initial state."""
        self._joint = np.array([0.0], dtype=np.float64)
        self._model.reset()
        self._step_count = 0

    # --- Batch ---

    def fit(self, observations: ArrayLike) -> list[Result]:
        """Batch mode. Resets state, then processes all observations.

        All validation runs before any state mutation. On failure,
        detector state is unchanged.
        """
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

        # Pre-validate all observations against model domain before
        # resetting state, so a mid-batch domain error leaves the
        # detector unchanged.
        for x in arr:
            self._model.validate_observation(float(x))

        self.reset()
        results: list[Result] = []
        for x in arr:
            results.append(self.update(float(x)))
        return results

    # --- Prediction ---

    def predictive_log_prob(self, x: float) -> float:
        """Log density of one candidate next observation under the mixture.

        Mixes each run length's predictive with the current posterior.
        This is the filtering predictive from Adams & MacKay, equation (1).
        """
        if not math.isfinite(x):
            raise ValueError(f"x must be finite, got {x}")
        self._model.validate_observation(x)
        log_components = self._model.log_predictive_prob(x)
        return float(logsumexp(self._joint + log_components))

    def predictive_mean(self) -> float:
        """Mean of the one-step mixture predictive.

        NaN when any run length with positive posterior mass has an
        undefined mean (Student-t degrees of freedom at most 1).
        """
        means = self._moment("predictive_mean")
        weights = np.exp(self._joint)
        if np.any((weights > 0.0) & ~np.isfinite(means)):
            return math.nan
        return float(np.sum(weights * means))

    def predictive_variance(self) -> float:
        """Variance of the one-step mixture predictive.

        Infinite when any run length with positive mass has degrees of
        freedom at most 2 (the default Gaussian prior, before data).
        """
        means = self._moment("predictive_mean")
        variances = self._moment("predictive_variance")
        weights = np.exp(self._joint)
        if np.any((weights > 0.0) & ~np.isfinite(variances)):
            return math.inf
        if np.any((weights > 0.0) & ~np.isfinite(means)):
            return math.nan
        mean = float(np.sum(weights * means))
        within = float(np.sum(weights * variances))
        between = float(np.sum(weights * (means - mean) ** 2))
        return within + between

    def _moment(self, name: str) -> NDArray[np.float64]:
        method = getattr(self._model, name, None)
        if not callable(method):
            raise TypeError(
                f"{type(self._model).__name__} does not provide {name}(); "
                "predictive moments are defined for the built-in models"
            )
        return np.asarray(method(), dtype=np.float64)

    # --- Snapshot ---

    def copy(self) -> Detector:
        """Independent clone."""
        d = Detector.__new__(Detector)
        d._model = self._model.copy()
        d._hazard = self._hazard
        d._max_run_length = self._max_run_length
        d._prune_threshold = self._prune_threshold
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
