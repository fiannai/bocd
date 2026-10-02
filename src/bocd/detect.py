"""Changepoint decisions from the run-length posterior.

With a constant hazard, ``Result.changepoint_prob`` equals the hazard
rate on every observation (until truncation moves mass). The data-dependent
signal is the run-length posterior: a regime change moves the MAP run
length down and puts mass on short runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bocd.result import Result


@dataclass(frozen=True, slots=True)
class Changepoint:
    """A regime boundary confirmed from the MAP run length.

    ``step`` is the first observation of the new regime. ``detected_at``
    is the later index at which the collapse had persisted, so online
    callers see the boundary after a delay of ``detected_at - step``.
    """

    step: int
    detected_at: int
    run_length: int


def _check_detection_params(drop: int, persistence: int) -> None:
    if isinstance(drop, bool) or not isinstance(drop, int) or drop < 1:
        raise ValueError(f"drop must be an int >= 1, got {drop}")
    if (
        isinstance(persistence, bool)
        or not isinstance(persistence, int)
        or persistence < 1
    ):
        raise ValueError(f"persistence must be an int >= 1, got {persistence}")


class ChangepointStream:
    """Online decisions from successive detector results.

    ``push`` returns a :class:`Changepoint` when the MAP run length has
    fallen by more than ``drop`` and stayed down for ``persistence``
    steps. If the MAP jumps back up by more than ``drop`` before that,
    the collapse was a flicker between modes and is discarded.

    The boundary index is ``detected_at - run_length + 1``: a MAP run
    length of ``r`` means the current run holds the last ``r`` points.
    """

    def __init__(self, *, drop: int = 20, persistence: int = 5) -> None:
        _check_detection_params(drop, persistence)
        self._drop = drop
        self._persistence = persistence
        self._prev: int | None = None
        self._pending_t: int | None = None
        self._seen = 0

    def push(self, result: Result) -> Changepoint | None:
        """Consume one result. Returns a boundary once it has persisted."""
        return self._push_run_length(result.most_likely_run_length)

    def _push_run_length(self, run_length: int) -> Changepoint | None:
        t = self._seen
        emitted: Changepoint | None = None
        if self._prev is not None:
            if self._pending_t is None and self._prev - run_length > self._drop:
                self._pending_t = t
            elif self._pending_t is not None and run_length - self._prev > self._drop:
                self._pending_t = None
            if (
                self._pending_t is not None
                and t - self._pending_t + 1 >= self._persistence
            ):
                start = t - run_length + 1
                if start < 0:
                    start = 0
                if start > t:
                    start = t
                emitted = Changepoint(step=start, detected_at=t, run_length=run_length)
                self._pending_t = None
        self._prev = run_length
        self._seen = t + 1
        return emitted


def detect_changepoints(
    results: Sequence[Result],
    *,
    drop: int = 20,
    persistence: int = 5,
) -> list[Changepoint]:
    """Batch segmentation. Same rule as :class:`ChangepointStream`.

    A collapse in the last ``persistence - 1`` observations is
    unconfirmed: the helper waits until the short run has held.
    """
    stream = ChangepointStream(drop=drop, persistence=persistence)
    found: list[Changepoint] = []
    for result in results:
        changepoint = stream.push(result)
        if changepoint is not None:
            found.append(changepoint)
    return found
