"""Bayesian Online Change Point Detection.

Public API — all names importable from the top-level ``bocd`` package::

    from bocd import Detector, Result
    from bocd import GaussianModel, StudentTModel, PoissonModel
    from bocd import ConstantHazard
    from bocd import ObservationModel, HazardFunction

Submodule imports also work::

    from bocd.models import GaussianModel
    from bocd.hazards import ConstantHazard
    from bocd.core import step, run
"""

from bocd._typing import HazardFunction, ObservationModel
from bocd.detector import Detector
from bocd.hazards.constant import ConstantHazard
from bocd.models.gaussian import GaussianModel
from bocd.models.poisson import PoissonModel
from bocd.models.studentt import StudentTModel
from bocd.result import Result

__all__ = [
    "ConstantHazard",
    "Detector",
    "GaussianModel",
    "HazardFunction",
    "ObservationModel",
    "PoissonModel",
    "Result",
    "StudentTModel",
]
