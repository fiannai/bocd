"""Tests for core BOCD algorithm functions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from bocd._core import run, step
from bocd.hazards.constant import ConstantHazard
from bocd.models.gaussian import GaussianModel


class TestStep:
    """Tests for _core.step()."""

    def test_returns_tuple(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        result = step(1.0, joint, model, hazard)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_new_joint_shape_grows_by_one(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])  # shape (1,)
        new_joint, _ = step(1.0, joint, model, hazard)
        assert new_joint.shape == (2,)

    def test_posterior_sums_to_one(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        new_joint, _ = step(1.0, joint, model, hazard)
        total = np.exp(logsumexp(new_joint))
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_cp_prob_in_unit_interval(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        _, cp_prob = step(1.0, joint, model, hazard)
        assert 0.0 <= cp_prob <= 1.0

    def test_cp_prob_matches_joint_zero(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        new_joint, cp_prob = step(1.0, joint, model, hazard)
        assert cp_prob == pytest.approx(float(np.exp(new_joint[0])), abs=1e-12)

    def test_does_not_mutate_input_joint(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        original = joint.copy()
        step(1.0, joint, model, hazard)
        np.testing.assert_array_equal(joint, original)

    def test_multi_step_shape_evolution(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        for i in range(5):
            joint, _ = step(float(i), joint, model, hazard)
            assert joint.shape == (i + 2,)

    def test_multi_step_normalization(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        joint = np.array([0.0])
        for i in range(10):
            joint, _ = step(float(i), joint, model, hazard)
            total = np.exp(logsumexp(joint))
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_high_hazard_increases_cp_prob(self) -> None:
        """Higher hazard rate should produce higher changepoint probability."""
        model_lo = GaussianModel()
        model_hi = GaussianModel()
        hazard_lo = ConstantHazard(rate=0.01)
        hazard_hi = ConstantHazard(rate=0.5)
        joint = np.array([0.0])
        _, cp_lo = step(1.0, joint, model_lo, hazard_lo)
        _, cp_hi = step(1.0, joint, model_hi, hazard_hi)
        assert cp_hi > cp_lo


class TestRun:
    """Tests for _core.run()."""

    def test_returns_correct_types(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([1.0, 2.0, 3.0])
        cp_probs, posteriors = run(obs, model, hazard)
        assert isinstance(cp_probs, np.ndarray)
        assert isinstance(posteriors, list)

    def test_output_lengths_match_input(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        cp_probs, posteriors = run(obs, model, hazard)
        assert len(cp_probs) == 4
        assert len(posteriors) == 4

    def test_posterior_shapes_grow(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([1.0, 2.0, 3.0])
        _, posteriors = run(obs, model, hazard)
        assert posteriors[0].shape == (2,)
        assert posteriors[1].shape == (3,)
        assert posteriors[2].shape == (4,)

    def test_all_posteriors_normalized(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, posteriors = run(obs, model, hazard)
        for post in posteriors:
            total = np.exp(logsumexp(post))
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_cp_probs_match_posterior_zeros(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([1.0, 2.0, 3.0])
        cp_probs, posteriors = run(obs, model, hazard)
        for cp, post in zip(cp_probs, posteriors, strict=False):
            assert cp == pytest.approx(float(np.exp(post[0])), abs=1e-12)

    def test_empty_input(self) -> None:
        model = GaussianModel()
        hazard = ConstantHazard()
        obs = np.array([], dtype=np.float64)
        cp_probs, posteriors = run(obs, model, hazard)
        assert len(cp_probs) == 0
        assert len(posteriors) == 0
