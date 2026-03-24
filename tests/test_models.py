"""Tests for observation models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom
from scipy.stats import t as student_t

from bocd.models.gaussian import GaussianModel
from bocd.models.poisson import PoissonModel
from bocd.models.studentt import StudentTModel


# ---------------------------------------------------------------------------
# GaussianModel
# ---------------------------------------------------------------------------


class TestGaussianModel:
    """Tests for GaussianModel (Normal-Inverse-Gamma conjugate)."""

    # --- Constructor validation ---

    def test_default_params(self) -> None:
        m = GaussianModel()
        # Should not raise; initial stats have shape (1,)
        assert m.log_predictive_prob(0.0).shape == (1,)

    def test_invalid_kappa_zero(self) -> None:
        with pytest.raises(ValueError):
            GaussianModel(kappa=0.0)

    def test_invalid_kappa_negative(self) -> None:
        with pytest.raises(ValueError):
            GaussianModel(kappa=-1.0)

    def test_invalid_alpha_zero(self) -> None:
        with pytest.raises(ValueError):
            GaussianModel(alpha=0.0)

    def test_invalid_beta_negative(self) -> None:
        with pytest.raises(ValueError):
            GaussianModel(beta=-0.5)

    # --- log_predictive_prob ---

    def test_log_pred_prior_matches_scipy(self) -> None:
        """At initial state, predictive should match prior Student-t."""
        m = GaussianModel(mu_0=0.0, kappa=1.0, alpha=1.0, beta=1.0)
        x = 1.5
        log_prob = m.log_predictive_prob(x)
        assert log_prob.shape == (1,)
        # Prior: df=2*1=2, loc=0, scale=sqrt(1*(1+1)/(1*1))=sqrt(2)
        expected = student_t.logpdf(x, df=2, loc=0, scale=np.sqrt(2))
        np.testing.assert_almost_equal(log_prob[0], expected, decimal=10)

    def test_log_pred_after_one_observation(self) -> None:
        """After one update, verify against hand-computed scipy reference."""
        m = GaussianModel(mu_0=0.0, kappa=1.0, alpha=1.0, beta=1.0)
        m.update(3.0)
        # Now stats have shape (2,): [run_len=0 (fresh), run_len=1 (saw 3.0)]
        x = 2.0
        log_prob = m.log_predictive_prob(x)
        assert log_prob.shape == (2,)

        # run_len=0: prior params (no data) — same as prior
        df0 = 2 * 1.0
        loc0 = 0.0
        scale0 = np.sqrt(1.0 * (1.0 + 1) / (1.0 * 1.0))
        expected_0 = student_t.logpdf(x, df=df0, loc=loc0, scale=scale0)
        np.testing.assert_almost_equal(log_prob[0], expected_0, decimal=10)

        # run_len=1: one observation (3.0)
        n = 1.0
        kappa_n = 1.0 + n  # 2
        mu_n = (1.0 * 0.0 + 3.0) / kappa_n  # 1.5
        alpha_n = 1.0 + n / 2  # 1.5
        x_bar = 3.0
        ss = 0.0  # sum of squared deviations = 0 (one obs)
        beta_n = 1.0 + 0.5 * ss + 1.0 * n * (x_bar - 0.0) ** 2 / (2 * kappa_n)
        # beta_n = 1.0 + 0 + 1*1*9/(2*2) = 1.0 + 2.25 = 3.25
        df1 = 2 * alpha_n  # 3
        loc1 = mu_n  # 1.5
        scale1 = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
        expected_1 = student_t.logpdf(x, df=df1, loc=loc1, scale=scale1)
        np.testing.assert_almost_equal(log_prob[1], expected_1, decimal=10)

    def test_log_pred_with_custom_prior(self) -> None:
        """Non-default hyperparameters match scipy reference."""
        m = GaussianModel(mu_0=5.0, kappa=2.0, alpha=3.0, beta=2.0)
        x = 4.0
        log_prob = m.log_predictive_prob(x)
        # Prior: df=6, loc=5, scale=sqrt(2*(2+1)/(3*2))=sqrt(1)=1
        expected = student_t.logpdf(x, df=6, loc=5, scale=1.0)
        np.testing.assert_almost_equal(log_prob[0], expected, decimal=10)

    # --- update ---

    def test_update_grows_arrays(self) -> None:
        m = GaussianModel()
        assert m.log_predictive_prob(0.0).shape == (1,)
        m.update(1.0)
        assert m.log_predictive_prob(0.0).shape == (2,)
        m.update(2.0)
        assert m.log_predictive_prob(0.0).shape == (3,)

    # --- reset ---

    def test_reset_returns_to_prior(self) -> None:
        m = GaussianModel(mu_0=1.0, kappa=2.0, alpha=3.0, beta=4.0)
        prior_pred = m.log_predictive_prob(0.0).copy()
        m.update(5.0)
        m.update(10.0)
        m.reset()
        after_reset = m.log_predictive_prob(0.0)
        np.testing.assert_array_equal(prior_pred, after_reset)

    # --- copy ---

    def test_copy_is_independent(self) -> None:
        m = GaussianModel()
        m.update(1.0)
        c = m.copy()
        # Mutate original
        m.update(100.0)
        # Copy should be unaffected
        pred_copy = c.log_predictive_prob(0.0)
        assert pred_copy.shape == (2,)  # still has 1 observation

    # --- truncate ---

    def test_truncate_slices_arrays(self) -> None:
        m = GaussianModel()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.update(x)
        assert m.log_predictive_prob(0.0).shape == (6,)
        m.truncate(3)
        assert m.log_predictive_prob(0.0).shape == (3,)

    # --- validate_observation ---

    def test_validate_observation_accepts_any_finite(self) -> None:
        m = GaussianModel()
        # Should not raise for any finite value
        m.validate_observation(0.0)
        m.validate_observation(-1e10)
        m.validate_observation(1e10)


# ---------------------------------------------------------------------------
# StudentTModel
# ---------------------------------------------------------------------------


class TestStudentTModel:
    """Tests for StudentTModel (robust-prior preset of GaussianModel)."""

    def test_is_subclass_of_gaussian(self) -> None:
        assert issubclass(StudentTModel, GaussianModel)

    def test_default_alpha_is_half(self) -> None:
        m = StudentTModel()
        # Prior: df = 2*0.5 = 1 (Cauchy-like), loc=0, scale=sqrt(1*(1+1)/(0.5*1))=2
        x = 1.0
        log_prob = m.log_predictive_prob(x)
        expected = student_t.logpdf(x, df=1.0, loc=0.0, scale=2.0)
        np.testing.assert_almost_equal(log_prob[0], expected, decimal=10)

    def test_equivalent_to_gaussian_with_alpha_half(self) -> None:
        m1 = StudentTModel()
        m2 = GaussianModel(alpha=0.5)
        x = 3.0
        np.testing.assert_array_almost_equal(
            m1.log_predictive_prob(x),
            m2.log_predictive_prob(x),
        )

    def test_custom_params_override(self) -> None:
        m = StudentTModel(mu_0=2.0, kappa=3.0, alpha=0.8, beta=2.0)
        x = 1.0
        expected = student_t.logpdf(
            x,
            df=2 * 0.8,
            loc=2.0,
            scale=np.sqrt(2.0 * (3.0 + 1) / (0.8 * 3.0)),
        )
        np.testing.assert_almost_equal(
            m.log_predictive_prob(x)[0], expected, decimal=10
        )

    def test_heavier_tails_than_gaussian(self) -> None:
        """StudentTModel should assign higher probability to extreme values."""
        g = GaussianModel()
        s = StudentTModel()
        x_extreme = 10.0
        assert s.log_predictive_prob(x_extreme)[0] > g.log_predictive_prob(x_extreme)[0]


# ---------------------------------------------------------------------------
# PoissonModel
# ---------------------------------------------------------------------------


class TestPoissonModel:
    """Tests for PoissonModel (Gamma-Poisson conjugate)."""

    # --- Constructor validation ---

    def test_default_params(self) -> None:
        m = PoissonModel()
        assert m.log_predictive_prob(0.0).shape == (1,)

    def test_invalid_alpha_zero(self) -> None:
        with pytest.raises(ValueError):
            PoissonModel(alpha=0.0)

    def test_invalid_beta_negative(self) -> None:
        with pytest.raises(ValueError):
            PoissonModel(beta=-1.0)

    # --- log_predictive_prob ---

    def test_log_pred_prior_matches_scipy(self) -> None:
        """At initial state, predictive matches Negative Binomial."""
        m = PoissonModel(alpha=2.0, beta=1.0)
        x = 3.0
        log_prob = m.log_predictive_prob(x)
        # Prior: r=alpha=2, p=beta/(beta+1)=0.5
        expected = nbinom.logpmf(x, n=2.0, p=0.5)
        np.testing.assert_almost_equal(log_prob[0], expected, decimal=10)

    def test_log_pred_after_one_observation(self) -> None:
        """After one update, verify run_len=0 and run_len=1."""
        m = PoissonModel(alpha=2.0, beta=1.0)
        m.update(5.0)
        x = 3.0
        log_prob = m.log_predictive_prob(x)
        assert log_prob.shape == (2,)

        # run_len=0 (fresh): r=2, p=1/(1+1)=0.5
        expected_0 = nbinom.logpmf(x, n=2.0, p=0.5)
        np.testing.assert_almost_equal(log_prob[0], expected_0, decimal=10)

        # run_len=1 (saw 5.0): r=2+5=7, p=(1+1)/(1+1+1)=2/3
        expected_1 = nbinom.logpmf(x, n=7.0, p=2.0 / 3.0)
        np.testing.assert_almost_equal(log_prob[1], expected_1, decimal=10)

    # --- update ---

    def test_update_grows_arrays(self) -> None:
        m = PoissonModel()
        assert m.log_predictive_prob(0.0).shape == (1,)
        m.update(1.0)
        assert m.log_predictive_prob(0.0).shape == (2,)

    # --- reset ---

    def test_reset_returns_to_prior(self) -> None:
        m = PoissonModel(alpha=2.0, beta=3.0)
        prior_pred = m.log_predictive_prob(1.0).copy()
        m.update(5.0)
        m.reset()
        np.testing.assert_array_equal(prior_pred, m.log_predictive_prob(1.0))

    # --- copy ---

    def test_copy_is_independent(self) -> None:
        m = PoissonModel()
        m.update(3.0)
        c = m.copy()
        m.update(100.0)
        assert c.log_predictive_prob(0.0).shape == (2,)

    # --- truncate ---

    def test_truncate_slices_arrays(self) -> None:
        m = PoissonModel()
        for x in [1.0, 2.0, 3.0, 4.0]:
            m.update(x)
        assert m.log_predictive_prob(0.0).shape == (5,)
        m.truncate(2)
        assert m.log_predictive_prob(0.0).shape == (2,)

    # --- validate_observation ---

    def test_validate_observation_rejects_negative(self) -> None:
        m = PoissonModel()
        with pytest.raises(ValueError, match="x >= 0"):
            m.validate_observation(-1.0)

    def test_validate_observation_accepts_zero(self) -> None:
        m = PoissonModel()
        m.validate_observation(0.0)  # should not raise

    def test_validate_observation_accepts_positive(self) -> None:
        m = PoissonModel()
        m.validate_observation(5.0)  # should not raise
