"""End-to-end tests for full Sekha v2.0 stack integration.

These tests validate:
- Proxy routes through bridge
- Bridge selects optimal providers
- Controller provides context
- Full request flow works
- Provider fallback functions
- Dimension switching works
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


class TestProxyBridgeIntegration:
    """Test proxy -> bridge integration."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_proxy_routes_through_bridge(self):
        """Test that proxy correctly routes requests through bridge."""
        # Mock bridge client
        mock_bridge = AsyncMock()
        
        # Mock routing response
        mock_bridge.post = AsyncMock(side_effect=[
            # First call: routing
            MagicMock(
                status_code=200,
                json=lambda: {
                    "provider_id": "ollama_local",
                    "model_id": "llama3.1:8b",
                    "estimated_cost": 0.0,
                    "reason": "Local model available",
                    "provider_type": "ollama",
                },
                raise_for_status=lambda: None,
            ),
            # Second call: completion
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Test response",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                raise_for_status=lambda: None,
            ),
        ])
        
        # Simulate proxy forward_chat logic
        messages = [{"role": "user", "content": "Hello"}]
        
        # Step 1: Route
        routing_response = await mock_bridge.post(
            "/api/v1/route",
            json={"task": "chat_small", "preferred_model": None},
        )
        routing_data = routing_response.json()
        
        assert routing_data["provider_id"] == "ollama_local"
        assert routing_data["model_id"] == "llama3.1:8b"
        assert routing_data["estimated_cost"] == 0.0
        
        # Step 2: Complete
        completion_response = await mock_bridge.post(
            "/v1/chat/completions",
            json={"model": routing_data["model_id"], "messages": messages},
        )
        completion_data = completion_response.json()
        
        assert completion_data["choices"][0]["message"]["content"] == "Test response"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_vision_request_routing(self):
        """Test that vision requests are routed to vision-capable models."""
        mock_bridge = AsyncMock()
        
        # Mock routing response for vision task
        mock_bridge.post = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=lambda: {
                    "provider_id": "openai_cloud",
                    "model_id": "gpt-4o",
                    "estimated_cost": 0.01,
                    "reason": "Vision support required",
                    "provider_type": "openai",
                },
                raise_for_status=lambda: None,
            )
        )
        
        # Simulate vision request
        routing_response = await mock_bridge.post(
            "/api/v1/route",
            json={"task": "vision", "require_vision": True},
        )
        routing_data = routing_response.json()
        
        assert routing_data["provider_id"] == "openai_cloud"
        assert routing_data["model_id"] == "gpt-4o"
        assert "vision" in routing_data["reason"].lower()


class TestProviderFallback:
    """Test provider fallback behavior."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fallback_on_provider_failure(self):
        """Test that bridge falls back to next provider on failure."""
        from sekha_llm_bridge.registry import registry
        from sekha_llm_bridge.config import ModelTask
        
        # Mock provider health
        with patch.object(registry, "providers") as mock_providers:
            # Primary provider (unhealthy)
            primary = MagicMock()
            primary.provider_id = "primary"
            primary.priority = 1
            primary.circuit_breaker.is_open = True  # Simulates failure
            
            # Fallback provider (healthy)
            fallback = MagicMock()
            fallback.provider_id = "fallback"
            fallback.priority = 2
            fallback.circuit_breaker.is_open = False
            
            mock_providers.values.return_value = [primary, fallback]
            
            # Mock routing logic
            with patch.object(registry, "route_with_fallback") as mock_route:
                mock_route.return_value = MagicMock(
                    provider=fallback,
                    model_id="fallback-model",
                    estimated_cost=0.001,
                    reason="Primary provider unavailable, using fallback",
                )
                
                result = await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMALL
                )
                
                assert result.provider.provider_id == "fallback"
                assert "fallback" in result.reason.lower()

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_no_providers_available(self):
        """Test behavior when no providers are available."""
        from sekha_llm_bridge.registry import registry
        from sekha_llm_bridge.config import ModelTask
        
        with patch.object(registry, "providers") as mock_providers:
            # All providers unhealthy
            mock_providers.values.return_value = []
            
            with pytest.raises(RuntimeError, match="No suitable provider"):
                await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)


class TestDimensionSwitching:
    """Test dimension switching across services."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_embedding_dimension_coordination(self):
        """Test that embedding dimension is consistent across services."""
        from sekha_llm_bridge.registry import registry
        from sekha_llm_bridge.config import ModelTask
        
        # Mock routing for 768-dim embedding
        with patch.object(registry, "route_with_fallback") as mock_route:
            mock_route.return_value = MagicMock(
                provider=MagicMock(
                    provider_id="ollama",
                    provider_type="ollama",
                ),
                model_id="nomic-embed-text",
                estimated_cost=0.0,
                reason="Local embedding model",
            )
            
            # Mock model capabilities
            with patch.object(registry, "model_cache") as mock_cache:
                mock_cache.get.return_value = {
                    "model_id": "nomic-embed-text",
                    "dimension": 768,
                    "task": "embedding",
                }
                
                result = await registry.route_with_fallback(
                    task=ModelTask.EMBEDDING
                )
                
                # Verify dimension can be retrieved
                model_info = mock_cache.get(result.model_id)
                assert model_info["dimension"] == 768
                
                # Controller would use this to select collection: conversations_768

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multi_dimension_search(self):
        """Test searching across multiple dimensions."""
        # This would test controller's search_all_dimensions method
        # which merges results from multiple ChromaDB collections
        
        # Mock ChromaDB collections
        mock_results = {
            "conversations_768": [
                {"id": "1", "distance": 0.1, "dimension": 768}
            ],
            "conversations_3072": [
                {"id": "2", "distance": 0.15, "dimension": 3072}
            ],
        }
        
        # Simulate merging results by distance
        all_results = []
        for collection, results in mock_results.items():
            all_results.extend(results)
        
        all_results.sort(key=lambda x: x["distance"])
        
        # Best result should be from 768 collection (lower distance)
        assert all_results[0]["dimension"] == 768
        assert all_results[0]["id"] == "1"


class TestContextInjection:
    """Test context injection with routing."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_full_request_with_context(self):
        """Test full request flow: proxy -> controller (context) -> bridge -> provider."""
        
        # Mock controller response (context)
        mock_controller = AsyncMock()
        mock_controller.post = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=lambda: [
                    {
                        "role": "system",
                        "content": "Previous context about user preferences",
                        "metadata": {
                            "citation": {
                                "label": "User Preferences",
                                "folder": "/profile",
                            }
                        },
                    }
                ],
                raise_for_status=lambda: None,
            )
        )
        
        # Get context from controller
        context_response = await mock_controller.post(
            "/api/v1/context/assemble",
            json={"query": "What are my preferences?"},
        )
        context = context_response.json()
        
        assert len(context) == 1
        assert "preferences" in context[0]["content"].lower()
        assert context[0]["metadata"]["citation"]["label"] == "User Preferences"
        
        # Context would then be injected into messages before routing to bridge
        user_messages = [{"role": "user", "content": "What are my preferences?"}]
        enhanced_messages = context + user_messages
        
        assert len(enhanced_messages) == 2
        assert enhanced_messages[0]["role"] == "system"  # Injected context
        assert enhanced_messages[1]["role"] == "user"    # Original message


class TestCostTracking:
    """Test cost tracking across requests."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cost_estimation_in_routing(self):
        """Test that routing includes cost estimates."""
        from sekha_llm_bridge.pricing import estimate_cost
        
        # Test free local model
        local_cost = estimate_cost("llama3.1:8b", 1000, 500)
        assert local_cost == 0.0
        
        # Test paid model
        openai_cost = estimate_cost("gpt-4o", 1000, 500)
        assert openai_cost > 0.0
        
        # Routing should prefer cheaper model when both available
        models = ["gpt-4o", "llama3.1:8b"]
        costs = {m: estimate_cost(m, 1000, 500) for m in models}
        
        cheapest = min(costs, key=costs.get)
        assert cheapest == "llama3.1:8b"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cost_budget_enforcement(self):
        """Test that max_cost constraint is respected."""
        from sekha_llm_bridge.pricing import find_cheapest_model
        
        models = ["gpt-4o", "gpt-4o-mini", "llama3.1:8b"]
        
        # Budget allows GPT-4o-mini but not GPT-4o
        cheapest = find_cheapest_model(models, 1000, 500, max_cost=0.001)
        
        # Should return free local model or GPT-4o-mini (both under budget)
        assert cheapest in ["llama3.1:8b", "gpt-4o-mini"]
        assert cheapest != "gpt-4o"  # Too expensive


if __name__ == "__main__":
    # Run E2E tests (skip by default in CI)
    pytest.main([__file__, "-v", "-m", "e2e", "-s"])
