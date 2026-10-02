"""Tests for the Detector class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bocd.detect import detect_changepoints
from bocd.detector import Detector
from bocd.hazards.constant import ConstantHazard
from bocd.models.gaussian import GaussianModel
from bocd.models.poisson import PoissonModel
from bocd.models.studentt import StudentTModel
from bocd.result import Result

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestDetectorInit:
    """Tests for Detector construction and defaults."""

    def test_zero_args_creates_detector(self) -> None:
        d = Detector()
        assert d.step_count == 0

    def test_custom_model_and_hazard(self) -> None:
        d = Detector(model=GaussianModel(mu_0=5.0), hazard=ConstantHazard(0.1))
        assert d.step_count == 0

    def test_max_run_length_param(self) -> None:
        d = Detector(max_run_length=100)
        assert d.step_count == 0

    def test_invalid_model_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            Detector(model="not a model")  # type: ignore[arg-type]

    def test_invalid_hazard_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            Detector(hazard="not a hazard")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Streaming update
# ---------------------------------------------------------------------------


class TestDetectorUpdate:
    """Tests for streaming update."""

    def test_returns_result(self) -> None:
        d = Detector()
        r = d.update(1.0)
        assert isinstance(r, Result)

    def test_step_counter_increments(self) -> None:
        d = Detector()
        d.update(1.0)
        assert d.step_count == 1
        d.update(2.0)
        assert d.step_count == 2

    def test_result_step_is_zero_indexed(self) -> None:
        d = Detector()
        r0 = d.update(1.0)
        r1 = d.update(2.0)
        assert r0.step == 0
        assert r1.step == 1

    def test_cp_prob_in_unit_interval(self) -> None:
        d = Detector()
        for x in [1.0, 2.0, 3.0]:
            r = d.update(x)
            assert 0.0 <= r.changepoint_prob <= 1.0

    def test_posterior_sums_to_one(self) -> None:
        d = Detector()
        for x in [1.0, 2.0, 3.0]:
            r = d.update(x)
            assert sum(r.run_length_posterior) == pytest.approx(1.0, abs=1e-10)

    def test_posterior_shape_grows(self) -> None:
        d = Detector()
        r0 = d.update(1.0)
        assert r0.run_length_posterior.shape == (2,)
        r1 = d.update(2.0)
        assert r1.run_length_posterior.shape == (3,)

    def test_non_finite_nan_raises(self) -> None:
        d = Detector()
        with pytest.raises(ValueError):
            d.update(float("nan"))

    def test_non_finite_inf_raises(self) -> None:
        d = Detector()
        with pytest.raises(ValueError):
            d.update(float("inf"))

    def test_non_finite_neg_inf_raises(self) -> None:
        d = Detector()
        with pytest.raises(ValueError):
            d.update(float("-inf"))

    def test_state_unchanged_after_validation_failure(self) -> None:
        d = Detector()
        d.update(1.0)
        assert d.step_count == 1
        with pytest.raises(ValueError):
            d.update(float("nan"))
        assert d.step_count == 1

    def test_poisson_rejects_negative(self) -> None:
        d = Detector(model=PoissonModel())
        with pytest.raises(ValueError, match="x >= 0"):
            d.update(-1.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestDetectorReset:
    """Tests for reset()."""

    def test_reset_clears_step_count(self) -> None:
        d = Detector()
        d.update(1.0)
        d.update(2.0)
        d.reset()
        assert d.step_count == 0

    def test_reset_restores_prior_posterior(self) -> None:
        d = Detector()
        d.update(1.0)
        d.reset()
        posterior = d.run_length_posterior
        np.testing.assert_array_almost_equal(posterior, [1.0])


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestDetectorFit:
    """Tests for batch mode."""

    def test_returns_list_of_results(self) -> None:
        d = Detector()
        results = d.fit([1.0, 2.0, 3.0])
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, Result) for r in results)

    def test_fit_resets_before_processing(self) -> None:
        d = Detector()
        d.update(100.0)
        d.update(200.0)
        results = d.fit([1.0, 2.0])
        assert results[0].step == 0
        assert d.step_count == 2

    def test_empty_input_returns_empty(self) -> None:
        d = Detector()
        results = d.fit([])
        assert results == []

    def test_non_array_raises_type_error(self) -> None:
        d = Detector()
        with pytest.raises(TypeError):
            d.fit("not an array")  # type: ignore[arg-type]

    def test_2d_array_raises_type_error(self) -> None:
        d = Detector()
        with pytest.raises(TypeError):
            d.fit(np.array([[1, 2], [3, 4]]))

    def test_non_finite_raises_value_error(self) -> None:
        d = Detector()
        with pytest.raises(ValueError):
            d.fit([1.0, float("nan"), 3.0])

    def test_state_unchanged_on_validation_failure(self) -> None:
        d = Detector()
        d.update(1.0)
        with pytest.raises(ValueError):
            d.fit([1.0, float("nan")])
        assert d.step_count == 1  # still from the previous update

    def test_detector_continues_after_fit(self) -> None:
        d = Detector()
        d.fit([1.0, 2.0, 3.0])
        r = d.update(4.0)
        assert r.step == 3  # continues from fit

    def test_state_unchanged_on_model_domain_error(self) -> None:
        """Regression: fit([3.0, -1.0]) with Poisson left detector at step 1."""
        d = Detector(model=PoissonModel())
        d.update(5.0)
        assert d.step_count == 1
        with pytest.raises(ValueError, match="x >= 0"):
            d.fit([3.0, -1.0])
        assert d.step_count == 1  # unchanged — not 0, not 2

    def test_fit_rejects_fractional_poisson(self) -> None:
        """Regression: Poisson with x=1.5 collapsed posterior to nan."""
        d = Detector(model=PoissonModel())
        with pytest.raises(ValueError, match="integer-valued"):
            d.fit([1.0, 2.0, 1.5])


# ---------------------------------------------------------------------------
# Streaming / batch equivalence
# ---------------------------------------------------------------------------


class TestStreamingBatchEquivalence:
    """Streaming and batch must produce identical results."""

    def test_equivalence(self) -> None:
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        d_stream = Detector()
        streaming = [d_stream.update(x) for x in obs]

        d_batch = Detector()
        batch = d_batch.fit(obs)

        for s, b in zip(streaming, batch, strict=False):
            assert s.changepoint_prob == pytest.approx(b.changepoint_prob)
            np.testing.assert_array_almost_equal(
                s.run_length_posterior, b.run_length_posterior
            )
            assert s.most_likely_run_length == b.most_likely_run_length
            assert s.step == b.step


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestMaxRunLength:
    """Tests for truncation."""

    def test_posterior_capped_at_max(self) -> None:
        d = Detector(max_run_length=5)
        for i in range(20):
            r = d.update(float(i))
        assert len(r.run_length_posterior) <= 5

    def test_posterior_still_sums_to_one(self) -> None:
        d = Detector(max_run_length=10)
        for i in range(50):
            r = d.update(float(i))
            assert sum(r.run_length_posterior) == pytest.approx(1.0, abs=1e-10)

    def test_truncation_does_not_break_detection(self) -> None:
        """A hard cap still reports a real mean shift, via the run length."""
        rng = np.random.default_rng(42)
        series = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)])
        results = Detector(max_run_length=50).fit(series)
        found = detect_changepoints(results)
        assert len(found) == 1
        assert 95 <= found[0].step <= 110

    def test_cap_keeps_long_regime(self) -> None:
        """Folding the tail must not manufacture changepoints on noise."""
        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 400)
        results = Detector(max_run_length=50).fit(series)
        assert detect_changepoints(results) == []
        assert results[-1].most_likely_run_length == 49

    def test_max_run_length_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="max_run_length"):
            Detector(max_run_length=1)

    def test_prune_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="prune_threshold"):
            Detector(prune_threshold=0.0)

    def test_prune_drops_dead_tail_after_changepoint(self) -> None:
        rng = np.random.default_rng(0)
        series = np.concatenate([rng.normal(0, 1, 80), rng.normal(8, 1, 80)])
        results = Detector(prune_threshold=1e-4).fit(series)
        last = results[-1]
        # The new regime is 80 points. Pruning drops the dead first regime,
        # so the posterior is about that long rather than the full series.
        assert 70 <= last.most_likely_run_length <= 85
        assert len(last.run_length_posterior) <= last.most_likely_run_length + 5
        assert len(last.run_length_posterior) < last.step
        assert sum(last.run_length_posterior) == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


class TestDetectorCopy:
    """Tests for Detector.copy()."""

    def test_copy_is_independent(self) -> None:
        d = Detector()
        d.update(1.0)
        d.update(2.0)
        c = d.copy()

        # Advance original
        d.update(100.0)
        assert d.step_count == 3

        # Copy is unaffected
        assert c.step_count == 2
        r = c.update(3.0)
        assert r.step == 2

    def test_copy_produces_same_results(self) -> None:
        d = Detector()
        d.update(1.0)
        c = d.copy()

        r_orig = d.update(2.0)
        r_copy = c.update(2.0)
        assert r_orig.changepoint_prob == pytest.approx(r_copy.changepoint_prob)


# ---------------------------------------------------------------------------
# run_length_posterior property
# ---------------------------------------------------------------------------


class TestRunLengthPosteriorProperty:
    """Tests for the run_length_posterior property."""

    def test_returns_copy(self) -> None:
        d = Detector()
        d.update(1.0)
        p1 = d.run_length_posterior
        p2 = d.run_length_posterior
        assert p1 is not p2
        np.testing.assert_array_equal(p1, p2)

    def test_initial_posterior_is_prior(self) -> None:
        d = Detector()
        np.testing.assert_array_almost_equal(d.run_length_posterior, [1.0])


# ---------------------------------------------------------------------------
# Integration: known changepoint detection
# ---------------------------------------------------------------------------


class TestIntegrationGaussianStep:
    """Detect a known mean shift with default Gaussian model."""

    def test_detects_mean_shift(self, gaussian_step_series: np.ndarray) -> None:
        d = Detector()
        results = d.fit(gaussian_step_series)
        # Detection signal: most_likely_run_length drops near the changepoint.
        # With constant hazard, cp_prob stays near the hazard rate — the run
        # length posterior is the correct detection metric for BOCD.
        rls = [r.most_likely_run_length for r in results]
        # Before changepoint at t=200, run length should be growing
        assert rls[195] > 100
        # After changepoint, run length resets to small values
        assert rls[202] < 10

    def test_stationary_stays_quiet(self, no_changepoint_series: np.ndarray) -> None:
        d = Detector()
        results = d.fit(no_changepoint_series)
        rls = [r.most_likely_run_length for r in results]
        # After initial transient, run length should grow monotonically
        # (no resets = no detected changepoints)
        assert rls[-1] > 400


class TestIntegrationPoissonRate:
    """Detect rate changes with Poisson model."""

    def test_detects_rate_change(self, poisson_rate_change: np.ndarray) -> None:
        d = Detector(model=PoissonModel(), hazard=ConstantHazard(rate=1 / 100))
        results = d.fit(poisson_rate_change)
        rls = [r.most_likely_run_length for r in results]
        # Run length should reset near t=150 and t=300
        assert rls[145] > 50  # before first change
        assert rls[155] < 15  # after first change
        assert rls[295] > 50  # before second change
        assert rls[305] < 15  # after second change


class TestIntegrationMultiChangepoint:
    """Multiple regime changes."""

    def test_detects_multiple_changes(
        self, multi_changepoint_series: np.ndarray
    ) -> None:
        d = Detector(hazard=ConstantHazard(rate=1 / 50))
        results = d.fit(multi_changepoint_series)
        rls = [r.most_likely_run_length for r in results]
        # Count run-length resets (drops of >20 between consecutive steps)
        drops = sum(1 for i in range(1, len(rls)) if rls[i - 1] - rls[i] > 20)
        # Should detect at least 3 of the 5 true changepoints
        assert drops >= 3


class TestIntegrationCustomModel:
    """A minimal Protocol-satisfying model works with Detector."""

    def test_custom_model_runs(self) -> None:
        class ConstantModel:
            """Always predicts uniform — trivial but valid."""

            def __init__(self) -> None:
                self.reset()

            def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
                return self._log_probs

            def update(self, x: float) -> None:
                self._log_probs = np.concatenate([[np.log(0.5)], self._log_probs])

            def reset(self) -> None:
                self._log_probs = np.array([np.log(0.5)])

            def copy(self) -> ConstantModel:
                m = ConstantModel()
                m._log_probs = self._log_probs.copy()
                return m

            def truncate(self, max_len: int) -> None:
                self._log_probs = self._log_probs[:max_len]

            def validate_observation(self, x: float) -> None:
                pass

        d = Detector(model=ConstantModel())
        results = d.fit([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(results) == 5
        for r in results:
            assert 0 <= r.changepoint_prob <= 1
        with pytest.raises(TypeError, match="predictive_mean"):
            d.predictive_mean()


# ---------------------------------------------------------------------------
# Constant-hazard identity, prior scale, mixture predictive
# ---------------------------------------------------------------------------


class TestChangepointProbIdentity:
    """P(r=0) equals a constant hazard until truncation."""

    def test_equals_hazard_on_every_step(self) -> None:
        rng = np.random.default_rng(0)
        results = Detector(hazard=ConstantHazard(rate=1 / 50)).fit(rng.normal(0, 1, 40))
        for result in results:
            assert result.changepoint_prob == pytest.approx(1 / 50)


class TestPriorScale:
    """Default prior misses a shift far from 0; from_scale does not."""

    def test_from_scale_detects_shift_default_misses(self) -> None:
        rng = np.random.default_rng(0)
        series = np.concatenate([rng.normal(100, 1, 200), rng.normal(105, 1, 150)])
        assert detect_changepoints(Detector().fit(series)) == []
        found = detect_changepoints(
            Detector(model=GaussianModel.from_scale(100.0, 1.0)).fit(series)
        )
        assert len(found) == 1
        assert abs(found[0].step - 200) <= 8


class TestMixturePredictive:
    """One-step filtering predictive."""

    def test_prior_log_prob_matches_model(self) -> None:
        detector = Detector()
        model = GaussianModel()
        x = 0.5
        assert detector.predictive_log_prob(x) == pytest.approx(
            float(model.log_predictive_prob(x)[0])
        )

    def test_default_variance_is_infinite(self) -> None:
        detector = Detector()
        assert detector.predictive_mean() == pytest.approx(0.0)
        assert detector.predictive_variance() == float("inf")

    def test_finite_variance_when_prior_df_exceeds_two(self) -> None:
        detector = Detector(model=GaussianModel.from_scale(0.0, 1.0, alpha=2.0))
        assert detector.predictive_variance() == pytest.approx(2.0)

    def test_student_prior_mean_undefined(self) -> None:
        detector = Detector(model=StudentTModel())
        assert np.isnan(detector.predictive_mean())

    def test_mean_tracks_regime(self) -> None:
        rng = np.random.default_rng(0)
        series = rng.normal(5, 1, 80)
        detector = Detector(model=GaussianModel.from_scale(0.0, 1.0, alpha=2.0))
        detector.fit(series)
        assert detector.predictive_mean() == pytest.approx(5.0, abs=0.5)
        assert math.isfinite(detector.predictive_variance())
        near = detector.predictive_log_prob(5.0)
        far = detector.predictive_log_prob(30.0)
        assert near > far

    def test_non_finite_predictive_input_raises(self) -> None:
        detector = Detector()
        with pytest.raises(ValueError):
            detector.predictive_log_prob(float("nan"))
