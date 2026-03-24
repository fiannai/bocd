# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added
- Core BOCD algorithm (`step()`, `run()`) with log-space arithmetic
- `Detector` class with streaming (`update`) and batch (`fit`) APIs
- `GaussianModel` — Normal-Inverse-Gamma conjugate for continuous data
- `StudentTModel` — robust-prior preset (heavy tails, outlier-resistant)
- `PoissonModel` — Gamma-Poisson conjugate for count/rate data
- `ConstantHazard` — memoryless hazard function
- `ObservationModel` and `HazardFunction` Protocols for custom extensions
- Optional run-length truncation (`max_run_length`) for infinite streams
- Input validation at Detector layer (finiteness, model domain checks)
- Full type annotations with `py.typed` marker (PEP 561)
- Property-based tests with Hypothesis
