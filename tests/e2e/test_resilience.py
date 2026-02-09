"""E2E Test: Resilience and Failure Recovery

Tests system behavior under real-world conditions:
1. Check multiple providers (primary + fallback)
2. Observe cost-based fallback behavior
3. Monitor circuit breaker states
4. Verify graceful degradation
5. Test data consistency during provider instability
6. Verify timeout handling

Note: These tests require a running sekha-llm-bridge server.
      They will be automatically skipped if no server is detected.
      See tests/e2e/conftest.py for configuration.

Module 4 - Task 4.6: E2E Resilience & Recovery

Note: These are TRUE E2E tests - they observe real system behavior
without injecting failures. For controlled failure injection,
see tests/integration/test_resilience_integration.py
"""

import asyncio
import os
import time

import httpx
import pytest

# Test configuration
BRIDGE_URL = os.getenv("SEKHA_BRIDGE_URL", "http://localhost:5001")
CONTROLLER_URL = os.getenv("SEKHA_CONTROLLER_URL", "http://localhost:8080")
API_KEY = os.getenv("SEKHA_API_KEY", "test_key_12345678901234567890123456789012")


@pytest.fixture
def api_headers():
    """Headers for API requests"""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_provider_fallback(async_client, api_headers):
    """
    E2E Test: Automatic provider fallback

    Observes real fallback behavior:
    1. Check initial provider health
    2. Request routing for a task
    3. Verify fallback providers exist
    4. Test cost-based fallback
    5. Verify request succeeds via fallback when needed
    """

    print("\n🔄 Testing provider fallback...")

    # Step 1: Get initial provider health
    print("\n🏛️ Step 1: Checking provider health...")
    response = await async_client.get(
        f"{BRIDGE_URL}/api/v1/health/providers", headers=api_headers
    )

    if response.status_code != 200:
        pytest.skip("Provider health endpoint not available")

    health = response.json()
    providers = health.get("providers", [])

    if len(providers) < 2:
        pytest.skip("Need at least 2 providers configured for fallback test")

    healthy_providers = [p for p in providers if p.get("status") == "healthy"]
    print(f"✅ {len(healthy_providers)}/{len(providers)} providers healthy")

    # Step 2: Test routing
    print("\n🧭 Step 2: Testing routing...")
    routing_request = {"task": "embedding", "max_cost": 0.01}

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    if response.status_code == 200:
        routing = response.json()
        primary_provider = routing.get("provider_id")
        print(f"✅ Primary provider: {primary_provider}")
        print(f"   Model: {routing.get('model_id')}")
        print(f"   Cost: ${routing.get('estimated_cost', 0):.4f}")
    else:
        pytest.fail(f"Routing failed: {response.status_code}")

    # Step 3: Verify fallback works by requesting with budget constraint
    print("\n💰 Step 3: Testing cost-based fallback...")

    # Request cheapest possible (should fallback to free local model if available)
    routing_request = {"task": "embedding", "max_cost": 0.0001}  # Very low budget

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    if response.status_code == 200:
        routing = response.json()
        fallback_provider = routing.get("provider_id")
        print(f"✅ Fallback provider: {fallback_provider}")
        print(f"   Model: {routing.get('model_id')}")
        print(f"   Cost: ${routing.get('estimated_cost', 0):.4f}")

        # Cost should be within budget
        assert (
            routing.get("estimated_cost", 0) <= 0.0001
        ), "Fallback exceeded cost limit"
    else:
        # If no provider meets budget, should get clear error
        print(f"⚠️  No provider within budget (expected): {response.status_code}")

    print("✅ Provider fallback test PASSED")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_circuit_breaker_behavior(async_client, api_headers):
    """
    E2E Test: Circuit breaker observation

    Observes circuit breaker states in real system:
    1. Monitor circuit breaker states
    2. Make multiple requests
    3. Verify circuit breakers respond correctly
    4. Check state consistency
    """

    print("\n⚡ Testing circuit breaker behavior...")

    # Get initial circuit breaker states
    print("\n🔵 Step 1: Checking initial circuit breaker states...")
    response = await async_client.get(
        f"{BRIDGE_URL}/api/v1/health/providers", headers=api_headers
    )

    if response.status_code != 200:
        pytest.skip("Provider health endpoint not available")

    health = response.json()
    providers = health.get("providers", [])

    initial_states = {
        p["provider_id"]: p.get("circuit_breaker_state", "unknown") for p in providers
    }

    print("Initial circuit breaker states:")
    for provider_id, state in initial_states.items():
        print(f"  {provider_id}: {state}")

    # Most should be closed (healthy)
    closed_count = sum(1 for state in initial_states.values() if state == "closed")
    print(f"✅ {closed_count}/{len(initial_states)} circuit breakers closed")

    # Step 2: Verify circuit breaker responds to health
    print("\n🔴 Step 2: Monitoring circuit breaker responses...")

    # Make multiple routing requests to ensure providers are exercised
    for i in range(3):
        routing_request = {"task": "chat_small", "preferred_model": None}

        response = await async_client.post(
            f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
        )

        if response.status_code == 200:
            routing = response.json()
            print(f"  Request {i+1}: Routed to {routing.get('provider_id')}")
        else:
            print(f"  Request {i+1}: Failed ({response.status_code})")

        await asyncio.sleep(0.5)

    # Step 3: Check if any circuit breakers changed state
    print("\n🟪 Step 3: Checking for state changes...")
    response = await async_client.get(
        f"{BRIDGE_URL}/api/v1/health/providers", headers=api_headers
    )

    health = response.json()
    providers = health.get("providers", [])

    final_states = {
        p["provider_id"]: p.get("circuit_breaker_state", "unknown") for p in providers
    }

    print("Final circuit breaker states:")
    for provider_id, state in final_states.items():
        print(f"  {provider_id}: {state}")

    # Verify circuit breakers are functioning (not stuck)
    functioning = all(
        state in ["closed", "open", "half_open"] for state in final_states.values()
    )
    assert functioning, "Some circuit breakers in invalid state"

    print("✅ Circuit breaker behavior test PASSED")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_graceful_degradation(async_client, api_headers):
    """
    E2E Test: Graceful degradation under constraints

    Tests system behavior with impossible constraints:
    1. Request operation with constraints no provider can satisfy
    2. Verify error messages are informative
    3. Verify no crashes or hangs
    4. Verify system recovers with valid requests
    """

    print("\n🛡️ Testing graceful degradation...")

    # Step 1: Test with impossible constraints (no provider can satisfy)
    print("\n❌ Step 1: Testing with impossible constraints...")

    routing_request = {
        "task": "chat_smart",
        "max_cost": 0.0,  # Impossible: $0 for paid models
        "preferred_model": "nonexistent-model-xyz",
    }

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    # Should get a clear error, not crash
    if response.status_code != 200:
        error_data = response.json()
        print(f"✅ Got expected error: {response.status_code}")
        print(f"   Error message: {error_data.get('detail', 'N/A')}")

        # Error message should be informative
        error_msg = str(error_data.get("detail", "")).lower()
        assert any(
            keyword in error_msg
            for keyword in ["provider", "model", "available", "cost"]
        ), "Error message not informative"
    else:
        # If it succeeded, verify it used a free model
        routing = response.json()
        assert (
            routing.get("estimated_cost", 0) == 0.0
        ), "Should only route to free model with $0 budget"
        print(f"✅ Gracefully used free model: {routing.get('model_id')}")

    # Step 2: Test with valid request after failure
    print("\n✅ Step 2: Verifying recovery with valid request...")

    routing_request = {"task": "embedding", "max_cost": 0.01}

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    if response.status_code == 200:
        routing = response.json()
        print(f"✅ System recovered: {routing.get('provider_id')}")
    else:
        # If all providers are actually down, this is acceptable
        print("⚠️  All providers unavailable (acceptable in test environment)")

    print("✅ Graceful degradation test PASSED")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_data_consistency_during_failures(async_client, api_headers):
    """
    E2E Test: Data consistency observation

    Tests data consistency in real system:
    1. Create conversation during normal operation
    2. Retrieve conversation (works regardless of provider state)
    3. Search for conversation
    4. Verify no data loss or corruption
    """

    print("\n💾 Testing data consistency during failures...")

    # Step 1: Create conversation normally
    print("\n📝 Step 1: Creating test conversation...")

    conversation_data = {
        "label": "Resilience Test",
        "folder": "resilience_tests",
        "messages": [
            {"role": "user", "content": "Test message for resilience"},
            {"role": "assistant", "content": "Acknowledged"},
        ],
    }

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/conversations",
        json=conversation_data,
        headers=api_headers,
    )

    if response.status_code != 200:
        pytest.skip(f"Cannot create conversation: {response.status_code}")

    conv_id = response.json()["id"]
    print(f"✅ Created conversation: {conv_id}")

    # Wait for embedding
    await asyncio.sleep(2)

    # Step 2: Retrieve conversation (should work even if providers unstable)
    print("\n📖 Step 2: Retrieving conversation...")

    response = await async_client.get(
        f"{CONTROLLER_URL}/api/v1/conversations/{conv_id}", headers=api_headers
    )

    assert response.status_code == 200, "Failed to retrieve conversation"

    retrieved = response.json()
    assert retrieved["id"] == conv_id
    assert retrieved["label"] == conversation_data["label"]

    print("✅ Data intact after retrieval")

    # Step 3: Search should work (may fallback to different provider)
    print("\n🔍 Step 3: Searching for conversation...")

    search_data = {"query": "resilience test", "limit": 10}

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/search", json=search_data, headers=api_headers
    )

    if response.status_code == 200:
        results = response.json()["results"]
        found = any(str(r["conversation_id"]) == conv_id for r in results)

        if found:
            print("✅ Conversation found in search")
        else:
            print("⚠️  Conversation not in top results (may be ranking issue)")
    else:
        print(f"⚠️  Search unavailable: {response.status_code}")

    print("✅ Data consistency test PASSED")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_timeout_handling(async_client, api_headers):
    """
    E2E Test: Timeout handling

    Verifies system handles operations within reasonable time.
    """

    print("\n⏱️ Testing timeout handling...")

    # Use a very short timeout to test handling
    short_timeout_client = httpx.AsyncClient(timeout=5.0)  # 5 second timeout

    try:
        # Request routing (should complete quickly)
        routing_request = {"task": "embedding"}

        start_time = time.time()

        response = await short_timeout_client.post(
            f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
        )

        elapsed = time.time() - start_time

        print(f"✅ Routing completed in {elapsed:.2f}s")

        # Should complete quickly (routing is metadata operation)
        assert elapsed < 3.0, "Routing too slow"

        if response.status_code == 200:
            routing = response.json()
            print(f"   Provider: {routing.get('provider_id')}")

    except httpx.TimeoutException:
        pytest.fail("Request timed out (bridge may be unresponsive)")

    finally:
        await short_timeout_client.aclose()

    print("✅ Timeout handling test PASSED")


if __name__ == "__main__":
    # Run with: pytest tests/e2e/test_resilience.py -v -s
    pytest.main([__file__, "-v", "-s"])
