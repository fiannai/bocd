"""Property-based tests using Hypothesis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bocd.detector import Detector
from bocd.models.gaussian import GaussianModel
from bocd.models.poisson import PoissonModel
from bocd.models.studentt import StudentTModel

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
count_floats = st.floats(
    min_value=0, max_value=100, allow_nan=False, allow_infinity=False
).map(lambda x: float(int(x)))  # integer-valued for Poisson

observation_lists = st.lists(finite_floats, min_size=1, max_size=50)
count_lists = st.lists(count_floats, min_size=1, max_size=50)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class TestGaussianProperties:
    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_posterior_sums_to_one(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel())
        for x in obs:
            r = d.update(x)
            total = float(np.sum(r.run_length_posterior))
            assert total == pytest.approx(1.0, abs=1e-8)

    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_no_nan_or_inf(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel())
        for x in obs:
            r = d.update(x)
            assert np.all(np.isfinite(r.run_length_posterior))
            assert np.isfinite(r.changepoint_prob)

    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_cp_prob_in_unit_interval(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel())
        for x in obs:
            r = d.update(x)
            assert 0.0 <= r.changepoint_prob <= 1.0

    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_posterior_shape_matches_step(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel())
        for i, x in enumerate(obs):
            r = d.update(x)
            assert r.run_length_posterior.shape == (i + 2,)


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------


class TestStudentTProperties:
    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_posterior_sums_to_one(self, obs: list[float]) -> None:
        d = Detector(model=StudentTModel())
        for x in obs:
            r = d.update(x)
            assert float(np.sum(r.run_length_posterior)) == pytest.approx(1.0, abs=1e-8)

    @given(obs=observation_lists)
    @settings(max_examples=50)
    def test_no_nan_or_inf(self, obs: list[float]) -> None:
        d = Detector(model=StudentTModel())
        for x in obs:
            r = d.update(x)
            assert np.all(np.isfinite(r.run_length_posterior))


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


class TestPoissonProperties:
    @given(obs=count_lists)
    @settings(max_examples=50)
    def test_posterior_sums_to_one(self, obs: list[float]) -> None:
        d = Detector(model=PoissonModel())
        for x in obs:
            r = d.update(x)
            assert float(np.sum(r.run_length_posterior)) == pytest.approx(1.0, abs=1e-8)

    @given(obs=count_lists)
    @settings(max_examples=50)
    def test_no_nan_or_inf(self, obs: list[float]) -> None:
        d = Detector(model=PoissonModel())
        for x in obs:
            r = d.update(x)
            assert np.all(np.isfinite(r.run_length_posterior))

    @given(obs=count_lists)
    @settings(max_examples=50)
    def test_cp_prob_in_unit_interval(self, obs: list[float]) -> None:
        d = Detector(model=PoissonModel())
        for x in obs:
            r = d.update(x)
            assert 0.0 <= r.changepoint_prob <= 1.0


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncationProperties:
    @given(obs=observation_lists)
    @settings(max_examples=30)
    def test_truncated_posterior_sums_to_one(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel(), max_run_length=10)
        for x in obs:
            r = d.update(x)
            assert float(np.sum(r.run_length_posterior)) == pytest.approx(1.0, abs=1e-8)

    @given(obs=observation_lists)
    @settings(max_examples=30)
    def test_truncated_posterior_shape_capped(self, obs: list[float]) -> None:
        max_rl = 10
        d = Detector(model=GaussianModel(), max_run_length=max_rl)
        for i, x in enumerate(obs):
            r = d.update(x)
            expected = min(i + 2, max_rl)
            assert r.run_length_posterior.shape == (expected,)


# ---------------------------------------------------------------------------
# Copy independence
# ---------------------------------------------------------------------------


class TestCopyProperties:
    @given(obs=observation_lists)
    @settings(max_examples=30)
    def test_copy_evolves_independently(self, obs: list[float]) -> None:
        d = Detector(model=GaussianModel())
        # Feed half the observations
        mid = len(obs) // 2
        for x in obs[:mid]:
            d.update(x)
        c = d.copy()

        # Feed remaining to original only
        for x in obs[mid:]:
            d.update(x)

        # Copy should still be at mid
        assert c.step_count == mid


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_observation(self) -> None:
        d = Detector()
        r = d.update(0.0)
        assert r.run_length_posterior.shape == (2,)
        assert np.all(np.isfinite(r.run_length_posterior))

    @given(
        x=st.floats(
            min_value=-1e10,
            max_value=1e10,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=50)
    def test_extreme_values_produce_valid_output(self, x: float) -> None:
        d = Detector(model=GaussianModel())
        r = d.update(x)
        assert np.all(np.isfinite(r.run_length_posterior))
        assert 0.0 <= r.changepoint_prob <= 1.0
