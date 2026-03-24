"""Pure BOCD algorithm functions (log-space throughout)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bocd._typing import HazardFunction, ObservationModel


def step(
    x: float,
    joint: NDArray[np.float64],
    model: ObservationModel,
    hazard: HazardFunction,
) -> tuple[NDArray[np.float64], float]:
    """Single BOCD step. Returns (new_joint, changepoint_probability).

    Mutates model via model.update(). All other arguments are not
    modified — new_joint is a fresh array.
    """
    t = len(joint)

    # 1. Predictive log-probabilities under each run-length hypothesis
    log_pred = model.log_predictive_prob(x)  # shape (t,)

    # 2. Hazard values
    run_lengths = np.arange(t, dtype=np.int64)
    h = hazard(run_lengths)  # shape (t,)

    # 3. Growth probabilities (log-space): each run continues
    log_growth = joint + log_pred + np.log1p(-h)  # shape (t,)

    # 4. Changepoint probability (log-space): a new run starts
    log_cp = logsumexp(joint + log_pred + np.log(h))  # scalar

    # 5. Assemble new joint distribution
    new_joint = np.empty(t + 1, dtype=np.float64)
    new_joint[0] = log_cp
    new_joint[1:] = log_growth

    # 6. Normalize
    log_evidence = logsumexp(new_joint)
    new_joint -= log_evidence

    # 7. Update model sufficient statistics
    model.update(x)

    # 8. Changepoint probability
    cp_prob = float(np.exp(new_joint[0]))

    return new_joint, cp_prob


def run(
    observations: NDArray[np.float64],
    model: ObservationModel,
    hazard: HazardFunction,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Batch mode. Returns (changepoint_probs, run_length_posteriors).

    Lower-level API — does not reset model state or apply truncation.
    Callers must provide a fresh/reset model. For managed batch
    processing with reset and truncation, use Detector.fit().
    """
    n = len(observations)
    if n == 0:
        return np.array([], dtype=np.float64), []

    joint = np.array([0.0], dtype=np.float64)
    cp_probs = np.empty(n, dtype=np.float64)
    posteriors: list[NDArray[np.float64]] = []

    for i in range(n):
        joint, cp = step(float(observations[i]), joint, model, hazard)
        cp_probs[i] = cp
        posteriors.append(joint.copy())

    return cp_probs, posteriors
