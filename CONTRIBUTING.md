# Contributing to Cytologick

This repository is a research-oriented Python application for working with whole-slide cytology images (MRXS), running PyTorch inference, and training segmentation models.

Contributions are welcome, but please keep changes pragmatic and easy to review.

## What to contribute

- Bug fixes and robustness improvements (I/O, parsing, preprocessing, inference)
- Documentation fixes (README/config hints, troubleshooting)
- Tests that cover real regressions
- Training and evaluation improvements that do not change UX unexpectedly

If you plan a larger change, open an issue first and describe the goal and the intended user impact.

## Development setup (recommended)

Follow the project README for system prerequisites (OpenSlide binaries are required; `openslide-python` alone is not enough).

Typical local setup uses conda:

```bash
conda create -n cyto python=3.12
conda activate cyto

# Project deps (PyTorch path)
pip install -r requirements-pytorch.txt

# OpenSlide Python bindings (often easier via conda)
conda install openslide-python

# Dev tooling
pip install -r requirements-dev.txt
```

Note: TensorFlow training is considered legacy in this repo. New work should target the PyTorch path unless explicitly discussed.

## Running locally

```bash
# Desktop app
python run.py

# Web app (experimental)
python run_web.py

# Training (uses config.yaml)
python model_train.py
```

## Code style

Use the existing style; keep diffs small and focused.

```bash
black .
ruff check .

# Optional (useful for larger refactors)
mypy clogic parsers
```

If you use pre-commit locally:

```bash
pre-commit install
```

## Tests

Pytest is configured in [pytest.ini](pytest.ini) and runs coverage by default.

```bash
pytest

# Example: run a specific test module
pytest tests/test_pytorch_inference.py
```

Some tests are marked (see `pytest.ini`): `pytorch`, `tensorflow`, `gui`, `slow`, `integration`.

## Data and binaries

- Do not commit patient data.
- Do not commit large slide files (`.mrxs`, `.svs`) or model checkpoints (`.pth`). If you must reference them, use reproducible download instructions or a separate storage.
- Keep example artifacts small and anonymized/synthetic.

## Pull requests

Please include:

- A short summary of what changed and why
- How you tested it (commands and environment)
- Any config changes needed (`config.yaml` keys)

PRs that touch training/inference should include at least one reproducible check (e.g., a small unit test, or a deterministic smoke test on synthetic input).
