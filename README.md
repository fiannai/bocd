# bocd

Bayesian Online Change Point Detection for Python.

[![PyPI](https://img.shields.io/pypi/v/bocd)](https://pypi.org/project/bocd/)
[![CI](https://github.com/fiannai/bocd/actions/workflows/ci.yml/badge.svg)](https://github.com/fiannai/bocd/actions)
[![codecov](https://img.shields.io/codecov/c/github/fiannai/bocd)](https://codecov.io/gh/fiannai/bocd)
[![Python](https://img.shields.io/pypi/pyversions/bocd)](https://pypi.org/project/bocd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pip-installable implementation of the Adams & MacKay (2007) algorithm for detecting regime changes in streaming or batch time series data. Easy defaults, full extensibility, production-grade quality.

## Quickstart

```bash
pip install bocd
```

```python
from bocd import Detector

results = Detector().fit(your_time_series)
changepoints = [r.step for r in results if r.changepoint_prob > 0.5]
```

## Features

- **Streaming and batch** — `update()` for production streams, `fit()` for notebooks
- **Three built-in models** — Gaussian, Student-t (outlier-robust), Poisson (count data)
- **Protocol-based extensibility** — implement 4 methods to add any observation model
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

for observation in stream:
    result = detector.update(observation)
    if result.changepoint_prob > 0.6:
        handle_changepoint(result)
```

## Why bocd?

| | bocd | ruptures | changefinder |
|---|---|---|---|
| Online streaming | Yes | No (batch only) | Yes |
| Bayesian posterior | Yes | No | No |
| Run-length distribution | Yes | No | No |
| Typed (PEP 561) | Yes | No | No |
| Custom models via Protocol | Yes | N/A | No |

## Documentation

Full docs at [bocd.readthedocs.io](https://bocd.readthedocs.io), including:

- [Quickstart guide](https://bocd.readthedocs.io/en/latest/guide/quickstart.html)
- [API reference](https://bocd.readthedocs.io/en/latest/api.html)
- [Custom models guide](https://bocd.readthedocs.io/en/latest/guide/custom-models.html)
- [Tutorial notebook](https://bocd.readthedocs.io/en/latest/notebooks/tutorial.html) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiannai/bocd/blob/main/docs/notebooks/tutorial.ipynb)

## Citation

If you use this library in research, please cite the original paper:

> Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.

## License

[MIT](LICENSE)
