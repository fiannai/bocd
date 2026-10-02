"""Tests for run-length changepoint decisions."""

from __future__ import annotations

import numpy as np
import pytest

from bocd import (
    Changepoint,
    ChangepointStream,
    Detector,
    GaussianModel,
    detect_changepoints,
)
from bocd.result import Result


def _spike(step: int, run_length: int) -> Result:
    length = max(step + 2, run_length + 1)
    posterior = np.zeros(length, dtype=np.float64)
    posterior[run_length] = 1.0
    return Result(
        changepoint_prob=float(posterior[0]),
        run_length_posterior=posterior,
        most_likely_run_length=run_length,
        step=step,
    )


def _growing(start: int, length: int, run0: int) -> list[Result]:
    return [_spike(start + i, run0 + i) for i in range(length)]


class TestDetectChangepoints:
    def test_confirms_a_persistent_drop(self) -> None:
        results = _growing(0, 40, 1)
        # MAP falls from 40 to 1 and then grows.
        results.extend(_growing(40, 15, 1))
        found = detect_changepoints(results, drop=20, persistence=5)
        assert len(found) == 1
        assert found[0].step == 40
        assert found[0].detected_at == 44
        assert found[0].run_length == 5

    def test_discards_a_flicker(self) -> None:
        results = _growing(0, 30, 1)
        results.append(_spike(30, 2))
        results.append(_spike(31, 3))
        # MAP jumps back to the long run.
        results.extend(_growing(32, 10, 33))
        assert detect_changepoints(results, drop=20, persistence=5) == []

    def test_unconfirmed_at_end_of_series(self) -> None:
        results = _growing(0, 30, 1)
        results.extend(_growing(30, 3, 1))
        assert detect_changepoints(results, drop=20, persistence=5) == []

    def test_drop_must_be_positive_int(self) -> None:
        with pytest.raises(ValueError, match="drop"):
            detect_changepoints([], drop=0)

    def test_persistence_must_be_positive_int(self) -> None:
        with pytest.raises(ValueError, match="persistence"):
            ChangepointStream(persistence=0)

    def test_zero_run_length_boundary_is_the_confirmation_step(self) -> None:
        results = [_spike(0, 30)]
        results.extend(_spike(i, 0) for i in range(1, 8))
        found = detect_changepoints(results, drop=20, persistence=5)
        assert found == [Changepoint(step=5, detected_at=5, run_length=0)]

    def test_overlong_run_length_clamps_to_start(self) -> None:
        # Increases of exactly `drop` are not a heal. The MAP can then
        # name a run longer than the series; the boundary clamps to 0.
        rls = [40, 1, 21, 41, 61, 81]
        results = [_spike(i, rl) for i, rl in enumerate(rls)]
        found = detect_changepoints(results, drop=20, persistence=5)
        assert len(found) == 1
        assert found[0].step == 0

    def test_stream_matches_batch(self) -> None:
        results = _growing(0, 25, 1)
        results.extend(_growing(25, 10, 1))
        batch = detect_changepoints(results, drop=10, persistence=4)
        stream = ChangepointStream(drop=10, persistence=4)
        online = [cp for result in results if (cp := stream.push(result)) is not None]
        assert online == batch

    def test_real_series_ignores_seed_blip(
        self, gaussian_step_series: np.ndarray
    ) -> None:
        results = Detector().fit(gaussian_step_series)
        found = detect_changepoints(results)
        assert len(found) == 1
        assert abs(found[0].step - 200) <= 5
        # The MAP flicker near t=139 must not be reported.
        assert found[0].step > 180

    def test_mass_at_most_jumps_at_boundary(
        self, gaussian_step_series: np.ndarray
    ) -> None:
        results = Detector().fit(gaussian_step_series)
        assert results[199].mass_at_most(15) < 0.1
        assert results[200].mass_at_most(15) > 0.9

    def test_mass_at_most_rejects_negative(self) -> None:
        result = _spike(0, 0)
        with pytest.raises(ValueError, match="run_length"):
            result.mass_at_most(-1)

    def test_mass_at_most_rejects_bool(self) -> None:
        result = _spike(0, 0)
        with pytest.raises(TypeError, match="run_length"):
            result.mass_at_most(True)

    def test_large_magnitude_shift_with_matched_scale(self) -> None:
        rng = np.random.default_rng(0)
        level = 1e8
        series = np.concatenate(
            [rng.normal(level, 1.0, 80), rng.normal(level + 10.0, 1.0, 80)]
        )
        results = Detector(model=GaussianModel.from_scale(level, 1.0)).fit(series)
        found = detect_changepoints(results)
        assert len(found) == 1
        assert abs(found[0].step - 80) <= 8
