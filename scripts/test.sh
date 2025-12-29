#!/bin/bash
set -e

echo "🤖 Running Sekha LLM Bridge Test Suite..."

TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
  "lint")
    echo "🔍 Running ruff and black..."
    ruff check .
    black --check .
    ;;
  "unit")
    echo "Running unit tests..."
    pytest tests/ -v
    ;;
  "integration")
    echo "Running integration tests..."
    pytest tests/integration/ -v
    ;;
  "all"|*)
    echo "Running linting, unit, and integration tests..."
    ruff check .
    black --check .
    pytest tests/ -v --cov=sekha_llm_bridge --cov-report=html
    ;;
esac

echo "✅ Tests complete!"