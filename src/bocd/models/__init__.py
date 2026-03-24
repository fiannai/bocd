"""Built-in observation models."""

from bocd.models.gaussian import GaussianModel
from bocd.models.poisson import PoissonModel
from bocd.models.studentt import StudentTModel

__all__ = ["GaussianModel", "PoissonModel", "StudentTModel"]
