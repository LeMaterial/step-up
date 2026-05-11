# step-up

A benchmark for 2D to 3D conformer generation models.

## Getting Started

This project uses [`uv`](https://docs.astral.sh/uv/) for Python, dependency, and
environment management.

```bash
git clone https://github.com/LeMaterial/step-up.git
cd step-up
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

Run formatting and lint checks:

```bash
uv run ruff format --check
uv run ruff check
```

## Usage

The benchmark is in early development. As functionality is added, reusable code
will live under the `step_up` Python package and can be run through `uv`:

```bash
uv run python
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, pre-commit hooks,
and contribution checks.
