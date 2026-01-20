#!/bin/bash
set -e

echo "🔍 Pre-publish checklist for PyPI..."

# Check poetry.lock is up to date
echo "✅ Checking poetry.lock..."
poetry check

# Run tests
echo "✅ Running tests..."
poetry run pytest tests/ --cov=sekha_llm_bridge --cov-report=term-missing

# Check formatting
echo "✅ Checking code formatting..."
poetry run black --check src/

# Run linting
echo "✅ Running linter..."
poetry run ruff check .

# Type checking
echo "✅ Type checking..."
poetry run mypy src/sekha_llm_bridge --ignore-missing-imports

# Build package
echo "✅ Building package..."
poetry build

# Check package
echo "✅ Checking package..."
poetry run twine check dist/*

echo ""
echo "🎉 All checks passed! Ready to publish with:"
echo "   poetry publish --dry-run  # Test first"
echo "   poetry publish             # Real publish"
