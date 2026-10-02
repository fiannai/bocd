# bocd

Bayesian Online Change Point Detection for Python.

[![CI](https://github.com/fiannai/bocd/actions/workflows/ci.yml/badge.svg)](https://github.com/fiannai/bocd/actions)
[![Python](https://img.shields.io/pypi/pyversions/bocd)](https://pypi.org/project/bocd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pip-installable implementation of the Adams & MacKay (2007) algorithm for detecting regime changes in streaming or batch time series data. Easy defaults, full extensibility, production-grade quality.

## Install

From PyPI (when published):

```bash
pip install bocd
```

From source:

```bash
git clone https://github.com/fiannai/bocd.git
cd bocd
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quickstart

```python
from bocd import Detector, detect_changepoints

results = Detector().fit(your_time_series)

# A boundary is a MAP run-length collapse that persists for several steps.
# With a constant hazard, Result.changepoint_prob equals the hazard rate
# on every step, so thresholding it does not detect anything.
for cp in detect_changepoints(results):
    print(f"Changepoint at step {cp.step} (confirmed at {cp.detected_at})")
```

The default `GaussianModel` is a prior for data near 0 with noise on the order of 1. Match it to the series when that is not true:

```python
from bocd import Detector, GaussianModel

detector = Detector(model=GaussianModel.from_scale(mean=100.0, scale=1.0))
```

## Features

- **Streaming and batch** — `update()` for production streams, `fit()` for notebooks
- **Three built-in models** — Gaussian, Student-t prior (heavier tails early in a run), Poisson (count data)
- **Protocol-based extensibility** — implement the `ObservationModel` protocol to add any model
- **Vectorized** — numpy broadcasts over run lengths, no Python loops
- **Numerically stable** — log-space arithmetic throughout, optional run-length truncation
- **Typed** — full type annotations, py.typed marker, mypy strict

## Streaming Example

```python
from bocd import ChangepointStream, Detector
from bocd.models import PoissonModel
from bocd.hazards import ConstantHazard

detector = Detector(
    model=PoissonModel(),
    hazard=ConstantHazard(rate=1/100),
    max_run_length=500,
)
alarm = ChangepointStream(drop=20, persistence=5)

for observation in stream:
    result = detector.update(observation)
    changepoint = alarm.push(result)
    if changepoint is not None:
        handle_changepoint(changepoint)
```

`max_run_length` caps memory by folding probability for longer runs into the last bin, so a long regime is not discarded. `prune_threshold` (for example `1e-4`) drops a long-run tail only when its mass is already negligible, which shrinks the state after a changepoint.

## Custom Models

Any class implementing the `ObservationModel` protocol works — no inheritance required:

```python
from bocd import Detector

class MyModel:
    def log_predictive_prob(self, x): ...
    def update(self, x): ...
    def reset(self): ...
    def copy(self): ...
    def truncate(self, max_len): ...
    def validate_observation(self, x): ...

detector = Detector(model=MyModel())
```

See the [design doc](docs/design.md) for the full protocol specification and a worked `ExponentialModel` example.

## Why bocd?

| | bocd | ruptures | changefinder |
|---|---|---|---|
| Online streaming | Yes | No (batch only) | Yes |
| Bayesian posterior | Yes | No | No |
| Run-length distribution | Yes | No | No |
| Typed (PEP 561) | Yes | No | No |
| Custom models via Protocol | Yes | N/A | No |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and feedback are welcome.

## Citation

If you use this library in research, please cite the original paper:

> Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.

## License

[MIT](LICENSE)
