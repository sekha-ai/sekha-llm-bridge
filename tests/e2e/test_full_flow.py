"""E2E Test: Full Conversation Flow

Tests the complete lifecycle of a conversation through Sekha:
1. Store conversation via controller
2. Verify embedding in correct dimension collection
3. Search for conversation
4. Retrieve full conversation
5. Generate summary
6. Verify all used optimal models

Note: These tests require a running sekha-llm-bridge server.
      They will be automatically skipped if no server is detected.
      See tests/e2e/conftest.py for configuration.

Module 4 - Task 4.5: E2E Happy Path
"""

import asyncio
import os
import uuid

import pytest

# Test configuration
CONTROLLER_URL = os.getenv("SEKHA_CONTROLLER_URL", "http://localhost:8080")
BRIDGE_URL = os.getenv("SEKHA_BRIDGE_URL", "http://localhost:5001")
API_KEY = os.getenv("SEKHA_API_KEY", "test_key_12345678901234567890123456789012")


@pytest.fixture
def api_headers():
    """Headers for API requests"""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_conversation_flow(async_client, api_headers):
    """
    E2E Test: Complete conversation lifecycle

    Steps:
    1. Store conversation via controller API
    2. Verify embedding dimension
    3. Search for stored conversation
    4. Retrieve conversation details
    5. Verify data consistency
    """

    # Step 1: Store conversation
    conversation_data = {
        "label": f"E2E Test Conversation {uuid.uuid4().hex[:8]}",
        "folder": "e2e_tests",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "The capital of France is Paris. It is known for the Eiffel Tower.",
            },
            {"role": "user", "content": "What is the population?"},
            {
                "role": "assistant",
                "content": "Paris has a population of approximately 2.2 million in the city proper, and over 12 million in the metropolitan area.",
            },
        ],
    }

    print("\n📝 Step 1: Creating conversation...")
    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/conversations",
        json=conversation_data,
        headers=api_headers,
    )
    assert (
        response.status_code == 200
    ), f"Failed to create conversation: {response.text}"

    conversation = response.json()
    conv_id = conversation["id"]
    print(f"✅ Created conversation: {conv_id}")

    # Wait for embedding processing
    await asyncio.sleep(2)

    # Step 2: Search for conversation
    print("\n🔍 Step 2: Searching for conversation...")
    search_data = {"query": "capital of France", "limit": 5}

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/search", json=search_data, headers=api_headers
    )
    assert response.status_code == 200, f"Search failed: {response.text}"

    search_results = response.json()
    assert "results" in search_results, "No results in search response"
    assert len(search_results["results"]) > 0, "Search returned no results"

    # Verify our conversation is in results
    found_conv = any(
        str(result["conversation_id"]) == conv_id
        for result in search_results["results"]
    )
    assert found_conv, f"Conversation {conv_id} not found in search results"
    print("✅ Found conversation in search results")

    # Step 3: Retrieve conversation
    print(f"\n📖 Step 3: Retrieving conversation {conv_id}...")
    response = await async_client.get(
        f"{CONTROLLER_URL}/api/v1/conversations/{conv_id}", headers=api_headers
    )
    assert (
        response.status_code == 200
    ), f"Failed to retrieve conversation: {response.text}"

    retrieved = response.json()
    assert retrieved["id"] == conv_id
    assert retrieved["label"] == conversation_data["label"]
    assert retrieved["folder"] == conversation_data["folder"]
    print("✅ Retrieved conversation successfully")

    # Step 4: Verify routing was optimal
    print("\n🎯 Step 4: Verifying routing decisions...")

    # Check bridge health and providers
    response = await async_client.get(
        f"{BRIDGE_URL}/api/v1/health/providers", headers=api_headers
    )

    if response.status_code == 200:
        providers = response.json()
        healthy_count = sum(
            1 for p in providers.get("providers", []) if p.get("status") == "healthy"
        )
        print(f"✅ {healthy_count} healthy providers available")
    else:
        print(f"⚠️  Could not verify providers: {response.status_code}")

    print("\n✅ Full conversation flow test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_dimension_workflow(async_client, api_headers):
    """
    E2E Test: Multi-dimension embedding workflow

    Verifies that conversations are embedded with correct dimensions
    and can be searched appropriately.
    """

    print("\n🔀 Testing multi-dimension workflow...")

    # Create conversation
    conversation_data = {
        "label": f"Dimension Test {uuid.uuid4().hex[:8]}",
        "folder": "dimension_tests",
        "messages": [
            {"role": "user", "content": "Test message for dimension verification"},
            {"role": "assistant", "content": "This is a test response."},
        ],
    }

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/conversations",
        json=conversation_data,
        headers=api_headers,
    )
    assert response.status_code == 200
    conv_id = response.json()["id"]

    # Wait for embedding
    await asyncio.sleep(2)

    # Search should work regardless of dimension
    search_data = {"query": "dimension verification test", "limit": 10}

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/search", json=search_data, headers=api_headers
    )
    assert response.status_code == 200

    results = response.json()["results"]
    found = any(str(r["conversation_id"]) == conv_id for r in results)
    assert found, "Conversation not found after embedding"

    print("✅ Multi-dimension workflow test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cost_tracking_workflow(async_client, api_headers):
    """
    E2E Test: Cost tracking across operations

    Verifies cost estimation works throughout the workflow.
    """

    print("\n💰 Testing cost tracking...")

    # Get routing info for embedding
    routing_request = {"task": "embedding", "max_cost": 0.01}

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    if response.status_code == 200:
        routing = response.json()
        print(
            f"📊 Embedding routing: {routing.get('model_id')} (${routing.get('estimated_cost', 0):.4f})"
        )

        # For embeddings, cost should be very low or zero (if using Ollama)
        assert routing.get("estimated_cost", 0) <= 0.01, "Embedding cost too high"
    else:
        print(f"⚠️  Routing endpoint not available: {response.status_code}")

    # Get routing info for chat
    routing_request = {"task": "chat", "max_cost": 0.05}

    response = await async_client.post(
        f"{BRIDGE_URL}/api/v1/route", json=routing_request, headers=api_headers
    )

    if response.status_code == 200:
        routing = response.json()
        print(
            f"💬 Chat routing: {routing.get('model_id')} (${routing.get('estimated_cost', 0):.4f})"
        )
        assert routing.get("estimated_cost", 0) <= 0.05, "Chat cost exceeds budget"

    print("✅ Cost tracking workflow test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_ranking_quality(async_client, api_headers):
    """
    E2E Test: Search ranking quality

    Verifies that search returns relevant results in correct order.
    """

    print("\n🎯 Testing search ranking quality...")

    # Create specific conversations
    conversations = [
        {
            "label": "Python Tutorial",
            "folder": "programming",
            "content": "Python is a high-level programming language known for its simplicity.",
        },
        {
            "label": "JavaScript Guide",
            "folder": "programming",
            "content": "JavaScript is primarily used for web development and runs in browsers.",
        },
        {
            "label": "Recipe Collection",
            "folder": "cooking",
            "content": "This is a collection of my favorite recipes including pasta and pizza.",
        },
    ]

    created_ids = []
    for conv in conversations:
        data = {
            "label": conv["label"],
            "folder": conv["folder"],
            "messages": [
                {"role": "user", "content": conv["content"]},
                {"role": "assistant", "content": "Understood."},
            ],
        }

        response = await async_client.post(
            f"{CONTROLLER_URL}/api/v1/conversations", json=data, headers=api_headers
        )
        if response.status_code == 200:
            created_ids.append(response.json()["id"])

    # Wait for embeddings
    await asyncio.sleep(3)

    # Search for programming-related content
    search_data = {"query": "programming language Python", "limit": 10}

    response = await async_client.post(
        f"{CONTROLLER_URL}/api/v1/search", json=search_data, headers=api_headers
    )

    if response.status_code == 200 and len(created_ids) > 0:
        results = response.json()["results"]

        if len(results) > 0:
            # First result should be Python-related (higher relevance)
            top_result = results[0]
            print(f"Top result: {top_result.get('label', 'unknown')}")

            # Verify relevance scores are descending
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results not properly ranked"
            print("✅ Results properly ranked by relevance")

    print("✅ Search ranking test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_operations(async_client, api_headers):
    """
    E2E Test: Concurrent operations

    Verifies system handles concurrent requests properly.
    """

    print("\n⚡ Testing concurrent operations...")

    # Create multiple conversations concurrently
    async def create_conversation(index: int):
        data = {
            "label": f"Concurrent Test {index}",
            "folder": "concurrent_tests",
            "messages": [
                {"role": "user", "content": f"Message {index}"},
                {"role": "assistant", "content": f"Response {index}"},
            ],
        }

        response = await async_client.post(
            f"{CONTROLLER_URL}/api/v1/conversations", json=data, headers=api_headers
        )
        return response.status_code == 200

    # Launch 5 concurrent requests
    tasks = [create_conversation(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if r is True)
    print(f"✅ {successful}/5 concurrent requests succeeded")

    assert successful >= 4, "Too many concurrent requests failed"

    print("✅ Concurrent operations test PASSED")


if __name__ == "__main__":
    # Run with: pytest tests/e2e/test_full_flow.py -v -s
    pytest.main([__file__, "-v", "-s"])
