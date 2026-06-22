# Changelog

## [0.3.0] — 2026-06-22

### Added
- Unit test framework (`tests/test_pipeline.py`) with pytest
- Dockerfile for containerized deployment
- GitHub Actions CI/CD (lint + test)
- Ruff linting configuration in `pyproject.toml`
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- Checkpoint/resume support in `run_pipeline.py` (`--resume` flag)
- `CHANGELOG.md` for tracking project history

### Fixed
- VR metadata injection (spherical_injector.py) — rewrote to use Google spatial-media CLI with sv3d+st3d boxes
- run_pipeline.py API mismatch with depth_estimator and stereo_renderer
- vr_metadata.py missing codec/bitrate/hardware-encoder parameters
- upscaler.py file content corruption (embedded XML artifacts)

### Changed
- `.gitignore` expanded to cover temp dirs, coverage files, test cache
- `requirements.txt` now uses version ranges with comments and optional deps

## [0.2.0] — 2026-06-20

### Added
- Initial pipeline: depth estimation → stereo rendering → equirectangular projection → VR metadata
- Depth-Anything V2 integration
- Pixel upscaler module (Real-ESRGAN + OpenCV fallback)
- CLI argument parsing in `run_pipeline.py`

## [0.1.0] — 2026-06-18

### Added
- Project scaffolding
- Architecture documentation