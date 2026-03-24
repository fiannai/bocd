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
from bocd import Detector

results = Detector().fit(your_time_series)

# Detection via run-length reset (most reliable with constant hazard)
for i in range(1, len(results)):
    if results[i - 1].most_likely_run_length - results[i].most_likely_run_length > 20:
        print(f"Changepoint detected at step {results[i].step}")
```

## Features

- **Streaming and batch** — `update()` for production streams, `fit()` for notebooks
- **Three built-in models** — Gaussian, Student-t (outlier-robust), Poisson (count data)
- **Protocol-based extensibility** — implement the `ObservationModel` protocol to add any model
- **Vectorized** — numpy broadcasts over run lengths, no Python loops
- **Numerically stable** — log-space arithmetic throughout, optional run-length truncation
- **Typed** — full type annotations, py.typed marker, mypy strict

## Streaming Example

```python
from bocd import Detector
from bocd.models import PoissonModel
from bocd.hazards import ConstantHazard

detector = Detector(
    model=PoissonModel(),
    hazard=ConstantHazard(rate=1/100),
    max_run_length=500,
)

prev_rl = 0
for observation in stream:
    result = detector.update(observation)
    if prev_rl - result.most_likely_run_length > 20:
        handle_changepoint(result)
    prev_rl = result.most_likely_run_length
```

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
