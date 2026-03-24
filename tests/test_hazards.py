"""Tests for hazard functions."""

from __future__ import annotations

import numpy as np
import pytest

from bocd.hazards.constant import ConstantHazard


class TestConstantHazard:
    def test_returns_constant_array(self) -> None:
        h = ConstantHazard(rate=1 / 200)
        rl = np.arange(10, dtype=np.int64)
        result = h(rl)
        expected = np.full(10, 1 / 200)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape_matches_input(self) -> None:
        h = ConstantHazard()
        rl = np.arange(5, dtype=np.int64)
        assert h(rl).shape == (5,)

    def test_output_dtype_is_float64(self) -> None:
        h = ConstantHazard()
        rl = np.arange(3, dtype=np.int64)
        assert h(rl).dtype == np.float64

    def test_default_rate(self) -> None:
        h = ConstantHazard()
        rl = np.array([0], dtype=np.int64)
        assert h(rl)[0] == pytest.approx(1 / 200)

    def test_custom_rate(self) -> None:
        h = ConstantHazard(rate=0.1)
        rl = np.array([0, 5, 100], dtype=np.int64)
        np.testing.assert_array_almost_equal(h(rl), np.full(3, 0.1))

    def test_single_element(self) -> None:
        h = ConstantHazard(rate=0.05)
        rl = np.array([42], dtype=np.int64)
        assert h(rl).shape == (1,)
        assert h(rl)[0] == pytest.approx(0.05)

    def test_invalid_rate_zero(self) -> None:
        with pytest.raises(ValueError):
            ConstantHazard(rate=0.0)

    def test_invalid_rate_one(self) -> None:
        with pytest.raises(ValueError):
            ConstantHazard(rate=1.0)

    def test_invalid_rate_negative(self) -> None:
        with pytest.raises(ValueError):
            ConstantHazard(rate=-0.1)

    def test_invalid_rate_greater_than_one(self) -> None:
        with pytest.raises(ValueError):
            ConstantHazard(rate=1.5)
