# Contributing

Bug reports, questions and pull requests are welcome. Open an issue first for anything larger than a small fix, so we can agree on the approach.

## Set up

```bash
make install   # virtualenv + pinned runtime and dev dependencies (Python 3.11+)
make models    # trained checkpoints and the GTSRB cache, needed for the dashboard and its page tests
make lint test
```

## Ground rules

- **Logic lives in `src/`.** Notebooks and the dashboard pages only call functions and display results.
- **Every parameter comes from `configs/base.yaml`.** No hard-coded paths, thresholds or class ids.
- **The test set is never modified**, and images stay in [0, 1]; normalisation is a layer inside the model.
- **Detection never uses the true poison indices** to decide or to choose a parameter. They are only used to score a finished detection. A method tuned after seeing its results must be labelled post-hoc.
- **Report numbers as measured.** One seed per model means differences of about one point are noise; say so rather than rounding a result into a finding.
- Seeds are always set; metrics are saved as JSON and figures as PNG under `results/`.

## Code style

- `ruff check .` must pass (config in `pyproject.toml`, line length 120).
- Type hints and a one-line docstring on public functions.
- Comments explain why, not what.
- Use `logging`, not `print`.

## Tests

- `pytest` runs on synthetic data only; nothing is downloaded. Tests that need the checkpoints are skipped when they are missing.
- Test behaviour, not implementation. A bug fix comes with a test that fails without it.
- Coverage of `src/` must stay at or above 90% (`make test` fails below).

## Commits and pull requests

- Branch from `main` as `type/short-description`, for example `feat/clean-label-attack`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`, imperative, lowercase, no final period, 72 characters at most. Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `build`, `chore`, `perf`.
- One logical change per commit; do not mix a refactor with a feature.
- `main` is protected: a pull request needs green CI (lint, tests on Python 3.11 and 3.12) and resolved review threads.
