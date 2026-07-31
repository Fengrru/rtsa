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

1. Create a new file in `extractors/` inheriting from the base extractor interface.
2. Implement `extract(self, text: str, **kwargs) -> ReasoningTraceGraph`.
3. Register it in `extractors/__init__.py`.
4. Add tests in `tests/test_extractors/`.

## Adding a New Analysis Module

1. Create a new file in `analysis/`.
2. Use `ReasoningTraceGraph` from `core.types` as the input type.
3. Return structured results using `dataclass` or Pydantic models.
4. Add unit tests in `tests/test_analysis/`.

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
