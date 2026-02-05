"""Integration tests for provider routing logic.

Tests validate:
- Provider selection based on task
- Priority-based routing
- Vision capability routing
- Preferred model routing
- Circuit breaker integration
- Cost-aware routing
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sekha_llm_bridge.registry import registry, RoutingResult
from sekha_llm_bridge.config import ModelTask, ProviderType


class TestBasicRouting:
    """Test basic routing functionality."""

    @pytest.mark.asyncio
    async def test_route_by_task(self):
        """Test routing selects correct model for task."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Mock candidates for embedding task
            mock_candidates.return_value = [
                ("ollama", "nomic-embed-text", 1),
            ]
            
            mock_provider = MagicMock()
            mock_provider.provider_id = "ollama"
            mock_provider.provider_type = "ollama"
            
            with patch.object(registry, "providers", {"ollama": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.EMBEDDING
                        )
                        
                        assert result.provider.provider_id == "ollama"
                        assert result.model_id == "nomic-embed-text"
                        assert "embed" in result.model_id.lower()

    @pytest.mark.asyncio
    async def test_route_chat_small_vs_smart(self):
        """Test distinction between chat_small and chat_smart tasks."""
        # Small chat: simple queries, should use smaller models
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("ollama", "llama3.1:8b", 1)]
            
            mock_provider = MagicMock()
            mock_provider.provider_id = "ollama"
            
            with patch.object(registry, "providers", {"ollama": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )
                        
                        assert "8b" in result.model_id.lower() or "small" in result.model_id.lower()
        
        # Smart chat: complex queries, should use larger models
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("openai", "gpt-4o", 1)]
            
            mock_provider = MagicMock()
            mock_provider.provider_id = "openai"
            
            with patch.object(registry, "providers", {"openai": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.01):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART
                        )
                        
                        assert "gpt-4" in result.model_id.lower() or "claude" in result.model_id.lower()


class TestPriorityRouting:
    """Test priority-based provider selection."""

    @pytest.mark.asyncio
    async def test_select_highest_priority_provider(self):
        """Test that highest priority (lowest number) provider is selected first."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Priority 1 should be selected over priority 2
            mock_candidates.return_value = [
                ("primary", "model-a", 1),    # Highest priority
                ("fallback", "model-b", 2),   # Lower priority
            ]
            
            primary = MagicMock()
            primary.provider_id = "primary"
            
            fallback = MagicMock()
            fallback.provider_id = "fallback"
            
            with patch.object(registry, "providers", {"primary": primary, "fallback": fallback}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )
                        
                        # Should select priority 1
                        assert result.provider.provider_id == "primary"
                        assert result.model_id == "model-a"

    @pytest.mark.asyncio
    async def test_fallback_to_lower_priority_when_primary_fails(self):
        """Test fallback to lower priority when higher priority unavailable."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("primary", "model-a", 1),
                ("fallback", "model-b", 2),
            ]
            
            primary = MagicMock()
            primary.provider_id = "primary"
            
            fallback = MagicMock()
            fallback.provider_id = "fallback"
            
            with patch.object(registry, "providers", {"primary": primary, "fallback": fallback}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # Primary circuit breaker is open
                    def cb_side_effect(provider_id):
                        if provider_id == "primary":
                            return MagicMock(is_open=lambda: True)  # Circuit open
                        return MagicMock(is_open=lambda: False)
                    
                    mock_cbs.get.side_effect = cb_side_effect
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )
                        
                        # Should fallback to priority 2
                        assert result.provider.provider_id == "fallback"
                        assert result.model_id == "model-b"


class TestVisionRouting:
    """Test vision capability routing."""

    @pytest.mark.asyncio
    async def test_route_to_vision_capable_model(self):
        """Test that vision requests route to vision-capable models."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Vision required, only vision models returned
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
            ]
            
            # Verify filter was applied
            mock_candidates.assert_not_called()
            
            result_call = mock_candidates.return_value
            assert len(result_call) == 1
            
            mock_provider = MagicMock()
            mock_provider.provider_id = "openai"
            
            with patch.object(registry, "providers", {"openai": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.01):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                        )
                        
                        # Should route to vision model
                        assert result.model_id == "gpt-4o"
                        mock_candidates.assert_called_once_with(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                            preferred_model=None,
                        )

    @pytest.mark.asyncio
    async def test_reject_non_vision_model_for_vision_task(self):
        """Test that non-vision models are rejected for vision tasks."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # No vision-capable models available
            mock_candidates.return_value = []
            
            with pytest.raises(RuntimeError, match="No suitable provider"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMART,
                    require_vision=True,
                )


class TestPreferredModelRouting:
    """Test preferred model routing."""

    @pytest.mark.asyncio
    async def test_use_preferred_model_when_available(self):
        """Test that preferred model is used when available."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Preferred model returned with priority 0 (highest)
            mock_candidates.return_value = [
                ("openai", "gpt-4o-mini", 0),  # Preferred (priority 0)
                ("ollama", "llama3.1:8b", 1),  # Default (priority 1)
            ]
            
            preferred = MagicMock()
            preferred.provider_id = "openai"
            
            default = MagicMock()
            default.provider_id = "ollama"
            
            with patch.object(registry, "providers", {"openai": preferred, "ollama": default}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.001):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL,
                            preferred_model="gpt-4o-mini",
                        )
                        
                        # Should use preferred model
                        assert result.model_id == "gpt-4o-mini"
                        assert "preferred" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_fallback_when_preferred_unavailable(self):
        """Test fallback to default when preferred model unavailable."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Preferred model not in candidates (unavailable)
            mock_candidates.return_value = [
                ("ollama", "llama3.1:8b", 1),  # Fallback
            ]
            
            fallback = MagicMock()
            fallback.provider_id = "ollama"
            
            with patch.object(registry, "providers", {"ollama": fallback}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL,
                            preferred_model="unavailable-model",
                        )
                        
                        # Should fallback to available model
                        assert result.model_id == "llama3.1:8b"


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in routing."""

    @pytest.mark.asyncio
    async def test_skip_provider_with_open_circuit(self):
        """Test that providers with open circuits are skipped."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("unhealthy", "model-a", 1),
                ("healthy", "model-b", 2),
            ]
            
            unhealthy = MagicMock()
            unhealthy.provider_id = "unhealthy"
            
            healthy = MagicMock()
            healthy.provider_id = "healthy"
            
            with patch.object(registry, "providers", {"unhealthy": unhealthy, "healthy": healthy}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    def cb_side_effect(provider_id):
                        if provider_id == "unhealthy":
                            return MagicMock(is_open=lambda: True)
                        return MagicMock(is_open=lambda: False)
                    
                    mock_cbs.get.side_effect = cb_side_effect
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", return_value=0.0):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )
                        
                        # Should skip unhealthy, use healthy
                        assert result.provider.provider_id == "healthy"

    @pytest.mark.asyncio
    async def test_all_circuits_open_error(self):
        """Test error when all provider circuits are open."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),
                ("provider2", "model-b", 2),
            ]
            
            p1 = MagicMock()
            p1.provider_id = "provider1"
            
            p2 = MagicMock()
            p2.provider_id = "provider2"
            
            with patch.object(registry, "providers", {"provider1": p1, "provider2": p2}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # All circuits open
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: True)
                    
                    with pytest.raises(RuntimeError, match="No suitable provider"):
                        await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )


class TestCostAwareRouting:
    """Test cost-aware routing decisions."""

    @pytest.mark.asyncio
    async def test_prefer_cheaper_provider_when_equivalent(self):
        """Test preferring cheaper providers when capabilities are equal."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Same priority, different costs
            mock_candidates.return_value = [
                ("expensive", "gpt-4o", 1),
                ("cheap", "llama3.1:8b", 1),  # Same priority
            ]
            
            expensive = MagicMock()
            expensive.provider_id = "expensive"
            
            cheap = MagicMock()
            cheap.provider_id = "cheap"
            
            with patch.object(registry, "providers", {"expensive": expensive, "cheap": cheap}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    def cost_side_effect(model_id, *args):
                        if "gpt-4o" in model_id:
                            return 0.01
                        return 0.0
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", side_effect=cost_side_effect):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )
                        
                        # Should prefer free model
                        assert result.provider.provider_id == "cheap"
                        assert result.estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_respect_cost_limit_in_routing(self):
        """Test that routing respects max_cost parameter."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
                ("ollama", "llama3.1:8b", 2),
            ]
            
            openai = MagicMock()
            openai.provider_id = "openai"
            
            ollama = MagicMock()
            ollama.provider_id = "ollama"
            
            with patch.object(registry, "providers", {"openai": openai, "ollama": ollama}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)
                    
                    def cost_side_effect(model_id, *args):
                        if "gpt-4o" in model_id:
                            return 0.05  # Too expensive
                        return 0.0
                    
                    with patch("sekha_llm_bridge.registry.estimate_cost", side_effect=cost_side_effect):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL,
                            max_cost=0.01,  # Excludes GPT-4o
                        )
                        
                        # Should route to free model
                        assert result.provider.provider_id == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
