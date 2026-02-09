#!/bin/bash
# Sekha LLM Bridge Test Runner
# Provides easy commands for running different test suites

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
    print_success "Python $(python3 --version) found"
    
    # Check pytest
    if ! python3 -m pytest --version &> /dev/null; then
        print_error "pytest not found. Installing test dependencies..."
        pip install -e ".[test]"
    fi
    print_success "pytest $(python3 -m pytest --version | head -n1)"
    
    # Check if package is installed
    if ! python3 -c "import sekha_llm_bridge" &> /dev/null; then
        print_warning "Package not installed in development mode"
        print_warning "Installing with: pip install -e ."
        pip install -e .
    fi
    print_success "sekha_llm_bridge package installed"
}

function run_unit_tests() {
    print_header "Running Unit Tests (Fast)"
    python3 -m pytest tests/ -m "not integration and not e2e" -v
}

function run_integration_tests() {
    print_header "Running Integration Tests"
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_warning "Ollama not detected at localhost:11434"
        print_warning "Some integration tests may be skipped"
        print_warning "Start Ollama with: ollama serve"
    else
        print_success "Ollama is running"
    fi
    
    python3 -m pytest tests/integration/ -m integration -v
}

function run_e2e_tests() {
    print_header "Running E2E Tests"
    
    # Check services
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_warning "Ollama not running (localhost:11434)"
    else
        print_success "Ollama is running"
    fi
    
    python3 -m pytest tests/ -m e2e -v
}

function run_all_tests() {
    print_header "Running All Tests"
    python3 -m pytest tests/ -v
}

function run_with_coverage() {
    print_header "Running Tests with Coverage"
    python3 -m pytest tests/ -m "not integration and not e2e" \
        --cov=sekha_llm_bridge \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-fail-under=80
    
    print_success "Coverage report generated in htmlcov/index.html"
}

function run_specific_test() {
    print_header "Running Specific Test: $1"
    python3 -m pytest "$1" -v
}

function run_failed_tests() {
    print_header "Re-running Failed Tests"
    python3 -m pytest tests/ --lf -v
}

function run_watch_mode() {
    print_header "Running Tests in Watch Mode"
    print_warning "Press Ctrl+C to stop"
    
    if ! command -v ptw &> /dev/null; then
        print_error "pytest-watch not installed"
        print_warning "Installing pytest-watch..."
        pip install pytest-watch
    fi
    
    ptw tests/ -- -m "not integration and not e2e" --tb=short
}

function show_test_structure() {
    print_header "Test Structure"
    tree tests/ -L 2 -I '__pycache__|*.pyc' || find tests/ -type f -name "test_*.py" | sort
}

function show_help() {
    echo -e "${BLUE}Sekha LLM Bridge Test Runner${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  unit           Run unit tests only (fast, no external dependencies)"
    echo "  integration    Run integration tests (requires Ollama)"
    echo "  e2e            Run end-to-end tests (requires full stack)"
    echo "  all            Run all tests"
    echo "  coverage       Run tests with coverage report"
    echo "  failed         Re-run only failed tests from last run"
    echo "  watch          Run tests in watch mode (auto-rerun on changes)"
    echo "  structure      Show test directory structure"
    echo "  specific PATH  Run specific test file or function"
    echo ""
    echo "Examples:"
    echo "  $0 unit"
    echo "  $0 integration"
    echo "  $0 coverage"
    echo "  $0 specific tests/test_resilience.py"
    echo "  $0 specific tests/test_resilience.py::TestCircuitBreaker::test_opens_after_failure_threshold"
    echo ""
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    check)
        check_dependencies
        ;;
    unit)
        check_dependencies
        run_unit_tests
        ;;
    integration)
        check_dependencies
        run_integration_tests
        ;;
    e2e)
        check_dependencies
        run_e2e_tests
        ;;
    all)
        check_dependencies
        run_all_tests
        ;;
    coverage)
        check_dependencies
        run_with_coverage
        ;;
    failed)
        check_dependencies
        run_failed_tests
        ;;
    watch)
        check_dependencies
        run_watch_mode
        ;;
    structure)
        show_test_structure
        ;;
    specific)
        if [ $# -eq 0 ]; then
            print_error "Please provide a test path"
            echo "Example: $0 specific tests/test_resilience.py"
            exit 1
        fi
        check_dependencies
        run_specific_test "$1"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

print_success "Done!"
