# Contributing to RTSA

Thank you for your interest in improving RTSA! This document provides guidelines for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Fengrru/rtsa.git
cd rtsa

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=analysis --cov=extractors --cov-report=html
```

## Code Style

- Follow PEP 8 guidelines.
- Use type hints for public functions and methods.
- Keep functions focused and modular.
- Add docstrings to modules, classes, and public functions.

## Adding a New Extractor

1. Create a new file in `rtsa/extractors/` inheriting from the base extractor interface.
2. Implement `extract(self, text: str, **kwargs) -> ReasoningTraceGraph`.
3. Register it in `rtsa/extractors/__init__.py`.
4. Add tests in `tests/test_extractors/`.

## Adding a New Analysis Module

1. Create a new file in `rtsa/analysis/`.
2. Use `ReasoningTraceGraph` from `core.types` as the input type.
3. Return structured results using `dataclass` or Pydantic models.
4. Add unit tests in `tests/test_analysis/`.

## Adding a New Dataset Adapter

1. Add a parser to `rtsa/utils/hf_adapter.py` and register it in `COT_PARSERS`.
2. Keep parsers pure (no I/O): `(text: str) -> (cot, answer)`.
3. Add unit tests in `tests/test_hf_adapter.py` covering the parser and auto-sniffing.

## Running Experiments

All experiments go through the unified entrypoint:

```bash
python -m experiments.run extract   --dataset gsm8k --max-traces 50
python -m experiments.run analyze   --dataset gsm8k
python -m experiments.run prune     --dataset synthetic --n 50
python -m experiments.run calibrate --synthetic
python -m experiments.run annotate
python -m experiments.run all       --dataset gsm8k
```

Every run writes to `rtsa/experiments/results/runs/<command>_<timestamp>/` with a
`manifest.json` recording the git commit, Python version, arguments, and UTC
timestamp. Never commit run outputs to the repository; they are ignored via
`.gitignore`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` new functionality
- `fix: ...` bug fixes
- `docs: ...` documentation changes
- `test: ...` test-only changes
- `refactor: ...` behavior-preserving restructuring
- `chore: ...` tooling / maintenance

Keep the subject under 72 characters and add a body listing the affected areas.

## Documentation Obligations

- Update `README.md` when the CLI, pipeline, or feature tables change.
- Update `CHANGELOG.md` (Keep a Changelog format) for every user-visible change.
- Update `docs/api.md` when public symbols are added, renamed, or removed.
- Add module-level docstrings and type hints to new modules and public functions.
- Do not use emoji in documentation or commit messages.

## Reporting Issues

When reporting bugs, please include:

- A minimal reproducible example.
- The expected vs. actual behavior.
- Your Python version and OS.
- The output of `pip list | grep -E "rtsa|numpy|networkx|pydantic"`.

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Make your changes and ensure tests pass.
3. Update documentation if applicable.
4. Submit a pull request with a clear description of the changes.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
