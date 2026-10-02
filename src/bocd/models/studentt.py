"""Student-t observation model — wider-prior preset of GaussianModel."""

from __future__ import annotations

import math

from bocd.models.gaussian import GaussianModel, _beta_for_scale


class StudentTModel(GaussianModel):
    """GaussianModel preset with a heavier prior (``alpha=0.5``).

    The predictive is Student-t with degrees of freedom ``1 + n`` on a
    run of length ``n``. Tails are heavier than ``GaussianModel``
    (``alpha=1``, degrees of freedom ``2 + n``) at the start of a run
    and approach a Gaussian as the run grows. A spike in a long regime
    is still evidence for a changepoint. Equivalent to
    ``GaussianModel(alpha=0.5)``.
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        super().__init__(mu_0=mu_0, kappa=kappa, alpha=alpha, beta=beta)

    @classmethod
    def from_scale(
        cls,
        mean: float,
        scale: float,
        *,
        kappa: float = 1.0,
        alpha: float = 0.5,
    ) -> StudentTModel:
        """Prior whose predictive scale matches ``scale`` at ``mean``.

        Default ``alpha=0.5`` keeps the heavier prior. ``StudentTModel()``
        is ``from_scale(0, 2)``.
        """
        if not math.isfinite(mean):
            raise ValueError(f"mean must be finite, got {mean}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and > 0, got {scale}")
        beta = _beta_for_scale(scale, kappa, alpha)
        return cls(mu_0=mean, kappa=kappa, alpha=alpha, beta=beta)
