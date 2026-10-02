# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added
- `detect_changepoints` and `ChangepointStream` — boundaries from a MAP run-length collapse that persists, at `step - run_length + 1`
- `Result.mass_at_most` — posterior mass on short runs
- `GaussianModel.from_scale` / `StudentTModel.from_scale` — prior mean and predictive scale in observation units
- `Detector.predictive_log_prob`, `predictive_mean`, `predictive_variance` — one-step mixture predictive
- `prune_threshold` — drop a long-run tail whose mass is below a threshold
- Core BOCD algorithm (`step()`, `run()`) with log-space arithmetic

### Changed
- `max_run_length` folds discarded tail mass into the last bin instead of deleting the mode. Values below 2 raise `ValueError`
- `GaussianModel` variance uses Welford's running sum of squares, so large-magnitude series keep a real scale
- `StudentTModel` is documented as a heavier prior whose tails approach a Gaussian as the run grows
- Documented that `changepoint_prob` equals a constant hazard on every untruncated step
- `Detector` class with streaming (`update`) and batch (`fit`) APIs
- `GaussianModel` — Normal-Inverse-Gamma conjugate for continuous data
- `StudentTModel` — heavier-prior preset of `GaussianModel` (`alpha=0.5`)
- `PoissonModel` — Gamma-Poisson conjugate for count/rate data
- `ConstantHazard` — memoryless hazard function
- `ObservationModel` and `HazardFunction` Protocols for custom extensions
- Optional run-length truncation (`max_run_length`) for infinite streams
- Input validation at Detector layer (finiteness, model domain checks)
- Full type annotations with `py.typed` marker (PEP 561)
- Property-based tests with Hypothesis
