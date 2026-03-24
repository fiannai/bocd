"""Student-t observation model — robust-prior preset of GaussianModel."""

from __future__ import annotations

from bocd.models.gaussian import GaussianModel


class StudentTModel(GaussianModel):
    """Robust-prior preset: GaussianModel with alpha=0.5 (df=1, Cauchy-like tails).

    Reduces false changepoint detections from extreme observations.
    Equivalent to GaussianModel(alpha=0.5) — provided as a separate
    importable name for discoverability.
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        super().__init__(mu_0=mu_0, kappa=kappa, alpha=alpha, beta=beta)
