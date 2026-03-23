# BOCD — Bayesian Online Change Point Detection Library

**Date:** 2026-03-23
**Status:** Draft
**Author:** Scott (design), Claude (spec writing)

## Purpose

A standalone, pip-installable Python library implementing the Adams & MacKay (2007) Bayesian Online Change Point Detection algorithm. Designed to be the community-standard BOCD implementation on PyPI — easy defaults, full extensibility, production-grade quality.

Primary consumer: GLADyS memory service (episode boundary detection). But the library has no GLADyS awareness — it's a general-purpose tool for detecting regime changes in any time series.

## Decisions Log

| Decision | Choice | Rationale |
| --- | --- | --- |
| Scope | General-purpose, no GLADyS awareness | Broader adoption; GLADyS imports it like any other consumer |
| Observation models (v1) | Gaussian + Student-t + Poisson | Covers continuous, outlier-robust, and count/rate data |
| Hazard functions (v1) | Constant + user callable | Easy default; any callable satisfies the Protocol for custom hazards |
| API shape | Streaming + batch | Streaming for production; batch for notebooks/research |
| Dependencies | numpy + scipy | scipy.stats gives clean model implementations; universal in data science |
| Python version | 3.11+ | 3.10 EOL Oct 2026; 3.11 performance gains matter for numerics |
| Architecture | Protocol-based composition + vectorized core | Modern, type-safe, extensible without inheritance; vectorized for performance |
| Purity | Mutate-in-place by default, copy-on-demand | Fast path for streaming; pure path available via `model.copy()` |
| Testing | TDD, unit + integration + property-based | Math is known upfront; tests-first catches implementation errors early |
| Documentation | Docstrings + README + Jupyter notebook + Read the Docs | Full docs site is the mark of a serious library |
| Build system | Hatch | Modern, PEP 621 native |
| Name | `bocd` | Exact algorithm name; maximum PyPI discoverability |
| License | MIT | Standard for scientific Python |
| Rust rewrite | No | Bottleneck is numpy (already C/Fortran); algorithmic optimizations (truncation) first |

## Architecture

### Overview

Three layers with strict separation of concerns:

1. **Core (`_core.py`)** — pure functions implementing the BOCD algorithm on numpy arrays. No state management, no side effects except controlled model mutation. Independently testable.
2. **Protocols & Models (`_typing.py`, `models/`, `hazards/`)** — extension contracts (Protocols) and built-in implementations. Each model is a self-contained file with its own conjugate math.
3. **Detector (`detector.py`)** — stateful wrapper providing the public streaming and batch API. Thin layer over the core.

### Module Dependency Graph

```text
__init__.py          (re-exports public API)
    ↓
detector.py          (stateful wrapper)
result.py            (Result dataclass)
    ↓
_core.py             (pure algorithm functions)
_typing.py           (Protocol definitions)
    ↑
models/*             (implement ObservationModel Protocol)
hazards/*            (implement HazardFunction Protocol)
```

No cycles. Core has zero internal dependencies.

### Package Structure

```text
bocd/
├── pyproject.toml
├── LICENSE
├── README.md
├── CITATION.cff
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── api.rst
│   ├── guide/
│   │   ├── quickstart.rst
│   │   └── custom-models.rst
│   └── notebooks/
│       └── tutorial.ipynb
├── src/
│   └── bocd/
│       ├── __init__.py
│       ├── py.typed
│       ├── _core.py
│       ├── _typing.py
│       ├── detector.py
│       ├── result.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gaussian.py
│       │   ├── studentt.py
│       │   └── poisson.py
│       └── hazards/
│           ├── __init__.py
│           └── constant.py
└── tests/
    ├── conftest.py
    ├── test_core.py
    ├── test_detector.py
    ├── test_models.py
    ├── test_hazards.py
    └── test_properties.py
```

## Core Algorithm

Adams & MacKay (2007) maintains a probability distribution over run lengths — how many observations since the last changepoint. At each time step (formulas shown in probability space for clarity; implementation uses log-space equivalents throughout):

1. **Predict** — compute predictive log-probability of new observation under each run-length hypothesis via `model.log_predictive_prob(x)`. Returns numpy array `pred` of shape `(t,)`.
2. **Growth** — probability each run continues: `growth[r] = joint[r] × pred[r] × (1 − H[r])`
3. **Changepoint** — probability a new run starts: `cp = Σ(joint[r] × pred[r] × H[r])`
4. **Update joint** — new joint distribution: `[cp, growth[0], growth[1], ...]`
5. **Normalize** — ensure distribution sums to 1
6. **Update model** — add observation to sufficient statistics arrays

Changepoint probability = `joint[0]` — the posterior mass on run length zero.

### Vectorization Strategy

Sufficient statistics are numpy arrays indexed by run length, not object-per-hypothesis copies. For each built-in model:

| Model | Sufficient Stats | Predictive Distribution |
| --- | --- | --- |
| Gaussian (NIG) | `sum_x`, `sum_x2`, `count` — each shape `(t,)` | Student-t: df=2α, loc=μ₀, scale=β·(κ+1)/(α·κ) (prior form; posterior substitutes updated hyperparameters per standard NIG conjugate update rules) |
| Student-t | Same NIG stats, default α=0.5 (df=1, Cauchy-like tails) | Student-t with heavier tails than Gaussian default |
| Poisson (Gamma) | `sum_x`, `count` — each shape `(t,)` | Negative Binomial: r=α+sum_x, p=(β+count)/(β+count+1) |

All operations are numpy broadcasts — no Python loop over run lengths. This is the 10-100x performance advantage over naive implementations.

### Numerical Stability

- **Log-space arithmetic** throughout — all probability computations use log-probabilities to avoid underflow on long series. `scipy.special.logsumexp` for normalization.
- **Run-length truncation** — optional `max_run_length` parameter on `Detector` caps the distribution after each `step()` call. Tail mass is folded into the longest retained run length. The core `step()` function is unaware of truncation — this is the Detector layer's responsibility. Essential for infinite streams.
- **Minimum probability floor** — small epsilon prevents log(0) in edge cases.

### Core Function Signatures

```python
def step(
    x: float,
    joint: NDArray[np.float64],
    model: ObservationModel,
    hazard: HazardFunction,
) -> tuple[NDArray[np.float64], float]:
    """Single BOCD step. Returns (new_joint, changepoint_probability).

    Pure function — no side effects except model.update().
    """

def run(
    observations: NDArray[np.float64],
    model: ObservationModel,
    hazard: HazardFunction,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Batch mode. Returns (changepoint_probs, run_length_posteriors)."""
```

### Purity Model

`step()` mutates the observation model's sufficient statistics in place for performance. Users who need pure operation call `model.copy()` before stepping:

```python
# Fast path (streaming, production)
result = step(x, joint, model, hazard)

# Pure path (analysis, branching, checkpointing)
snapshot = model.copy()
result = step(x, joint, model, hazard)
# snapshot is untouched
```

Performance cost of copying scales with series length (O(t) arrays). Negligible for series under ~1000 steps; infeasible for infinite streams.

## Protocols

```python
@runtime_checkable
class ObservationModel(Protocol):
    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        """Log-probability of x under each run-length hypothesis.
        Returns shape (t,).
        """
        ...

    def update(self, x: float) -> None:
        """Update sufficient statistics with new observation.
        Mutates internal arrays in place.
        """
        ...

    def reset(self) -> None:
        """Reset to initial state (prior parameters only)."""
        ...

    def copy(self) -> "ObservationModel":
        """Deep copy with independent numpy arrays."""
        ...


@runtime_checkable
class HazardFunction(Protocol):
    def __call__(self, run_lengths: NDArray[np.int64]) -> NDArray[np.float64]:
        """P(changepoint | run_length) for each hypothesis."""
        ...
```

`runtime_checkable` enables `isinstance` validation at `Detector.__init__`, catching misconfigured models at construction rather than mid-run.

## Built-in Observation Models

### GaussianModel

Normal-Inverse-Gamma conjugate prior. Hyperparameters: `mu_0`, `kappa`, `alpha`, `beta`. Predictive distribution is Student-t. Use for continuous data with roughly normal variation within a regime (temperatures, latencies, stock returns).

### StudentTModel

Same NIG conjugate structure as `GaussianModel` with different default hyperparameters: `alpha=0.5` (giving df=1, Cauchy-like tails) vs. Gaussian's `alpha=1.0` (df=2). The lower degrees of freedom produce heavier tails, making the predictive distribution less sensitive to outliers and reducing false changepoint detections from extreme observations. All other mechanics (sufficient statistics, update logic) are identical — `StudentTModel` is effectively a robust preset of the NIG conjugate. Use for data with occasional spikes (network traffic, sensor data).

### PoissonModel

Gamma-Poisson conjugate prior. Hyperparameters: `alpha`, `beta`. Predictive distribution is Negative Binomial. Use for count/rate data — events per time window, requests per second, error counts. This is what GLADyS feeds it.

## Built-in Hazard Functions

### ConstantHazard

Constant hazard rate (memoryless / geometric prior on run length). Parameter: `rate` (default `1/200`) = `1/expected_run_length`. Changepoints are equally likely at any run length. This is the standard hazard function from Adams & MacKay 2007 and covers the vast majority of use cases.

### Custom Hazard

Any callable satisfying the `HazardFunction` Protocol works — no class needed. This is the extension point for time-dependent hazards (Weibull, logistic, etc.):

```python
# Hazard that decreases as run length grows
detector = Detector(model=m, hazard=lambda rl: 1 / (rl + 200))

# Increasing hazard (changepoints more likely over time)
detector = Detector(model=m, hazard=lambda rl: 1 - np.exp(-rl / 100))
```

## Detector API

### Result Dataclass

```python
@dataclass(frozen=True, slots=True)
class Result:
    changepoint_prob: float        # P(run_length = 0)
    run_length_posterior: NDArray   # Full distribution, shape (step+1,) — one entry per possible run length 0..step
    most_likely_run_length: int    # argmax of posterior
    step: int                      # 0-indexed observation number
```

`frozen=True` — immutable, safe to store and pass around. `slots=True` — low memory footprint for batch collection.

### Detector Class

```python
class Detector:
    def __init__(
        self,
        model: ObservationModel = GaussianModel(),
        hazard: HazardFunction = ConstantHazard(),
        *,
        max_run_length: int | None = None,
    ) -> None: ...

    # Streaming
    def update(self, x: float) -> Result: ...
    def reset(self) -> None: ...

    # Batch
    def fit(self, observations: ArrayLike) -> list[Result]: ...

    # Inspection
    @property
    def step_count(self) -> int: ...
    @property
    def run_length_posterior(self) -> NDArray[np.float64]: ...
```

### Design Decisions

- **`Detector()` works with zero arguments** — GaussianModel + ConstantHazard(1/200) as defaults. Two lines from import to changepoint detection.
- **`fit()` resets state** — prevents leftover-state bugs in batch mode. Use `update()` in a loop to continue from existing state.
- **`fit()` returns `list[Result]`** — standard Python type, works with all existing tooling.
- **`fit()` calls `step()` in a loop** — resets state, then calls `_core.step()` for each observation, wrapping each step's output in a `Result`. The core `run()` function exists as a lower-level batch API returning raw arrays for users who want to avoid `Result` overhead.
- **`run_length_posterior` property returns a copy** — prevents accidental mutation of internal state. `Result.run_length_posterior` should be treated as read-only by convention (frozen dataclass prevents attribute reassignment but not array mutation).
- **No `predict()` method** — BOCD detects, it doesn't forecast. Keeps the library focused.

### Usage Examples

**Streaming (production):**

```python
from bocd import Detector
from bocd.models import PoissonModel
from bocd.hazards import ConstantHazard

detector = Detector(
    model=PoissonModel(),
    hazard=ConstantHazard(rate=1/100),
    max_run_length=500,
)

for event in event_stream:
    result = detector.update(event.density)
    if result.changepoint_prob > 0.6:
        create_episode_boundary(confidence=result.changepoint_prob)
```

**Batch (notebook):**

```python
from bocd import Detector

results = Detector().fit(time_series)
changepoints = [r.step for r in results if r.changepoint_prob > 0.5]
```

## Testing Strategy

### Development Method: TDD

Tests are written before implementation. The algorithm's math is fully specified by Adams & MacKay 2007, and the API is designed upfront — both preconditions for effective TDD.

### Three Layers

**Unit tests** (`test_core.py`, `test_models.py`, `test_hazards.py`):

- Each model's `log_predictive_prob` against `scipy.stats` reference values
- Sufficient statistics update correctly after each observation
- `reset()` returns model to prior state
- `copy()` produces independent snapshot
- Hazard functions return correct shapes and values
- Core `step()` produces valid probability distribution

**Integration tests** (`test_detector.py`):

- Detector finds known changepoints in synthetic series (step function, mean shift, rate change)
- Streaming and batch produce identical results on same input
- `fit()` resets state correctly between calls
- `max_run_length` truncation doesn't break detection
- Each model × hazard combination runs end-to-end
- Custom Protocol-satisfying model works with Detector

**Property-based tests** (`test_properties.py`, Hypothesis):

- Posterior always sums to 1.0 (±epsilon) for arbitrary inputs
- No NaN or Inf in output for any finite input
- `changepoint_prob` always in [0, 1]
- `run_length_posterior` shape matches `step + 1`
- Empty or single-observation series don't crash
- Extreme values produce valid output
- `copy()` is truly independent

### Synthetic Test Series

Defined in `conftest.py` as seeded, reproducible fixtures:

- `gaussian_step_series` — mean shifts 0→5 at t=200
- `poisson_rate_change` — rate changes 3→10→3 at t=150, 300
- `multi_changepoint_series` — 5 regime changes with varying gaps
- `no_changepoint_series` — stationary (detector should stay quiet)
- `gradual_drift_series` — slow linear drift (tests sensitivity boundary)

### CI Matrix

- Python 3.11, 3.12, 3.13 × ubuntu-latest, windows-latest, macos-latest
- 95%+ line coverage enforced
- ruff (format + lint), mypy strict mode
- Pre-commit hooks: ruff, mypy, pytest (fast subset)

## Documentation

### README.md

One-sentence description, badges (PyPI, CI, coverage, Python versions), 3-line quickstart, feature highlights, comparison table vs. alternatives (ruptures, changefinder), link to full docs.

### Read the Docs (Sphinx + Furo + MyST-NB)

- **Quickstart guide** — install, first detection, plot results
- **Concepts** — what BOCD is, how it works, when to use it
- **API Reference** — auto-generated from docstrings
- **Custom models guide** — Protocol implementation walkthrough
- **Tutorial notebook** — embedded, executable
- **Changelog** — keep-a-changelog format

### Tutorial Notebook

Three parts:

1. **Streaming detection** — synthetic series, overlay changepoint probabilities, threshold tuning
2. **Batch analysis** — run-length posterior heatmap, compare Gaussian vs. Student-t, Poisson on count data
3. **Custom models** — implement ExponentialModel step by step, custom hazard as lambda

Runnable on Google Colab (badge link in README).

## Packaging & Release

### pyproject.toml

- Build system: Hatch
- Dependencies: `numpy>=1.24`, `scipy>=1.10`
- Optional: `dev` (pytest, hypothesis, ruff, mypy, pre-commit), `docs` (sphinx, furo, myst-nb, sphinx-copybutton)
- Classifiers: Science/Research, Mathematics, Typed
- Keywords: changepoint, bayesian, online, detection, time-series

### Release Pipeline

```text
on push/PR:
  ruff check + format → mypy --strict → pytest (matrix) → coverage → sphinx build

on tag (v*):
  hatch build → publish to PyPI (trusted publisher / OIDC) → trigger Read the Docs
```

### Discoverability

- PyPI classifiers and keywords
- GitHub topics matching PyPI keywords
- `py.typed` marker (PEP 561)
- `CITATION.cff` referencing Adams & MacKay 2007
- Tutorial notebook runnable on Colab
- Read the Docs indexed by search engines

## Extension Example

No inheritance required — implement the Protocol methods:

```python
class ExponentialModel:
    """Gamma-Exponential conjugate for inter-arrival times."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self._a0, self._b0 = alpha, beta
        self.reset()

    def log_predictive_prob(self, x: float) -> NDArray[np.float64]:
        a = self._a0 + self._count
        b = self._b0 + self._sum_x
        return np.log(a) + a * np.log(b) - (a + 1) * np.log(b + x)

    def update(self, x: float) -> None:
        self._sum_x = np.concatenate([[0.0], self._sum_x + x])
        self._count = np.concatenate([[0.0], self._count + 1])

    def reset(self) -> None:
        self._sum_x = np.array([0.0])
        self._count = np.array([0.0])

    def copy(self) -> "ExponentialModel":
        m = ExponentialModel(self._a0, self._b0)
        m._sum_x, m._count = self._sum_x.copy(), self._count.copy()
        return m

# Works immediately:
detector = Detector(model=ExponentialModel(), hazard=ConstantHazard())
```

## References

- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.
