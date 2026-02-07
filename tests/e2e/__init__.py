"""
End-to-End (E2E) Tests for Sekha v2.0

This package contains E2E tests that verify the complete system behavior
across multiple components (controller, bridge, proxy).

Test Categories:
- test_full_flow.py: Complete conversation lifecycle tests
- test_resilience.py: Failure handling and recovery tests

Running E2E Tests:
    pytest tests/e2e/ -v -m e2e -s

Requirements:
- Full Sekha stack running (controller, bridge, ChromaDB, Ollama)
- Environment variables configured:
  - SEKHA_CONTROLLER_URL
  - SEKHA_BRIDGE_URL
  - SEKHA_API_KEY

Note: E2E tests are marked with @pytest.mark.e2e and @pytest.mark.slow
"""
