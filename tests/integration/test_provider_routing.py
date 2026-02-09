"""Integration tests for provider routing logic.

Tests validate:
- Task-based routing
- Priority-based provider selection
- Cost-aware routing
- Circuit breaker integration
- Vision requirement filtering
- Fallback behavior
"""

from unittest.mock import MagicMock, patch

import pytest

from sekha_llm_bridge.config import ModelTask, ProviderConfig, ProviderType
from sekha_llm_bridge.registry import CachedModelInfo, registry


class TestTaskBasedRouting:
    """Test routing based on task type."""

    @pytest.mark.asyncio
    async def test_route_embedding_task(self):
        """Test routing for embedding tasks."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("openai", "text-embedding-3-large", 1)]

            mock_provider = MagicMock()
            mock_provider.provider_id = "openai"

            with patch.object(registry, "providers", {"openai": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.0001
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.EMBEDDING
                        )

                        assert result.model_id == "text-embedding-3-large"
                        mock_candidates.assert_called_once_with(
                            task=ModelTask.EMBEDDING,
                            require_vision=False,
                            preferred_model=None,
                        )

    @pytest.mark.asyncio
    async def test_route_chat_small_task(self):
        """Test routing for fast chat tasks."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("ollama", "llama3.1:8b", 1)]

            mock_provider = MagicMock()
            mock_provider.provider_id = "ollama"

            with patch.object(registry, "providers", {"ollama": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.0
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        assert result.model_id == "llama3.1:8b"

    @pytest.mark.asyncio
    async def test_route_chat_smart_task(self):
        """Test routing for smart/complex chat tasks."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("anthropic", "claude-3.5-sonnet", 1)]

            mock_provider = MagicMock()
            mock_provider.provider_id = "anthropic"

            with patch.object(registry, "providers", {"anthropic": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.015
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART
                        )

                        assert result.model_id == "claude-3.5-sonnet"


class TestPriorityRouting:
    """Test priority-based provider selection."""

    @pytest.mark.asyncio
    async def test_prefer_higher_priority_provider(self):
        """Test that higher priority providers are tried first."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Return candidates in priority order
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Highest priority
                ("provider2", "model-b", 2),
                ("provider3", "model-c", 3),
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            with patch.object(registry, "providers", {"provider1": provider1}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        # Should select highest priority provider
                        assert result.provider.provider_id == "provider1"
                        assert result.model_id == "model-a"

    @pytest.mark.asyncio
    async def test_fallback_to_lower_priority_when_primary_fails(self):
        """Test fallback to lower priority when higher fails."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Will fail
                ("provider2", "model-b", 2),  # Fallback
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            provider2 = MagicMock()
            provider2.provider_id = "provider2"

            with patch.object(
                registry, "providers", {"provider1": provider1, "provider2": provider2}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # Provider1 circuit breaker is open (failed)
                    def cb_side_effect(pid):
                        if pid == "provider1":
                            return MagicMock(is_open=lambda: True)
                        return MagicMock(is_open=lambda: False)

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        # Should fallback to provider2
                        assert result.provider.provider_id == "provider2"
                        assert result.model_id == "model-b"


class TestVisionRouting:
    """Test vision-specific routing logic."""

    @pytest.mark.asyncio
    async def test_require_vision_filters_candidates(self):
        """Test that require_vision filters non-vision models."""
        # Mock model_cache as a dict with items()
        mock_cache = {
            "openai:gpt-4o": CachedModelInfo(
                model_id="gpt-4o",
                provider_id="openai",
                task=ModelTask.CHAT_SMART,
                context_window=128000,
                supports_vision=True,
            ),
            "ollama:llama3.1:8b": CachedModelInfo(
                model_id="llama3.1:8b",
                provider_id="ollama",
                task=ModelTask.CHAT_SMART,
                context_window=8000,
                supports_vision=False,
            ),
        }

        # Mock settings with provider configs
        mock_settings = MagicMock()
        mock_settings.providers = [
            ProviderConfig(
                id="openai",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                priority=1,
                models=[],
            ),
            ProviderConfig(
                id="ollama",
                provider_type=ProviderType.OLLAMA,
                base_url="http://localhost:11434",
                priority=2,
                models=[],
            ),
        ]

        with patch.object(registry, "model_cache", mock_cache):
            with patch(
                "sekha_llm_bridge.registry.get_settings", return_value=mock_settings
            ):
                candidates = registry._get_candidates(
                    task=ModelTask.CHAT_SMART,
                    require_vision=True,
                )

                model_ids = [c[1] for c in candidates]
                assert "gpt-4o" in model_ids
                assert "llama3.1:8b" not in model_ids

    @pytest.mark.asyncio
    async def test_reject_non_vision_model_for_vision_task(self):
        """Test error when no vision models available."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # No candidates (all filtered out)
            mock_candidates.return_value = []

            with pytest.raises(RuntimeError, match="No providers available for task"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMART,
                    require_vision=True,
                )


class TestCostAwareRouting:
    """Test cost-aware routing decisions."""

    @pytest.mark.asyncio
    async def test_prefer_cheaper_provider_when_equivalent(self):
        """Test preference for cheaper providers with same capabilities."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Two equivalent providers
            mock_candidates.return_value = [
                ("openai", "gpt-4o-mini", 1),  # Cheaper
                ("openai", "gpt-4o", 1),  # More expensive, same priority
            ]

            provider = MagicMock()
            provider.provider_id = "openai"

            with patch.object(registry, "providers", {"openai": provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    # Mock costs - accept **kwargs
                    def cost_side_effect(model, **kwargs):
                        if "mini" in model:
                            return 0.001
                        return 0.015

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=cost_side_effect,
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART
                        )

                        # Should prefer cheaper model
                        assert "mini" in result.model_id
                        assert result.estimated_cost < 0.01

    @pytest.mark.asyncio
    async def test_respect_cost_limit_in_routing(self):
        """Test that cost limits are respected during routing."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),  # Expensive
                ("ollama", "llama3.1:8b", 2),  # Free
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            ollama = MagicMock()
            ollama.provider_id = "ollama"

            with patch.object(
                registry, "providers", {"openai": openai, "ollama": ollama}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    # Mock costs - accept **kwargs
                    def cost_side_effect(model, **kwargs):
                        if "gpt" in model:
                            return 0.05
                        return 0.0

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=cost_side_effect,
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Forces free provider
                        )

                        # Should route to free provider
                        assert result.provider.provider_id == "ollama"
                        assert result.estimated_cost == 0.0


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with routing."""

    @pytest.mark.asyncio
    async def test_skip_provider_with_open_circuit_breaker(self):
        """Test that providers with open circuit breakers are skipped."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Circuit breaker open
                ("provider2", "model-b", 2),  # Circuit breaker closed
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            provider2 = MagicMock()
            provider2.provider_id = "provider2"

            with patch.object(
                registry, "providers", {"provider1": provider1, "provider2": provider2}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:

                    def cb_side_effect(pid):
                        if pid == "provider1":
                            return MagicMock(is_open=lambda: True)  # Open
                        return MagicMock(is_open=lambda: False)  # Closed

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        # Should skip provider1, use provider2
                        assert result.provider.provider_id == "provider2"

    @pytest.mark.asyncio
    async def test_all_circuit_breakers_open(self):
        """Test error when all providers have open circuit breakers."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),
                ("provider2", "model-b", 2),
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            provider2 = MagicMock()
            provider2.provider_id = "provider2"

            with patch.object(
                registry, "providers", {"provider1": provider1, "provider2": provider2}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # All circuit breakers open
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: True)

                    with pytest.raises(
                        RuntimeError, match="No suitable providers available"
                    ):
                        await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)


class TestPreferredModelRouting:
    """Test routing with preferred model specified."""

    @pytest.mark.asyncio
    async def test_route_to_preferred_model_when_available(self):
        """Test that preferred model is selected when available."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
                ("openai", "gpt-4o-mini", 1),
            ]

            provider = MagicMock()
            provider.provider_id = "openai"

            with patch.object(registry, "providers", {"openai": provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.015
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            preferred_model="gpt-4o",
                        )

                        # Should use preferred model
                        assert result.model_id == "gpt-4o"
                        mock_candidates.assert_called_once_with(
                            task=ModelTask.CHAT_SMART,
                            require_vision=False,
                            preferred_model="gpt-4o",
                        )

    @pytest.mark.asyncio
    async def test_fallback_when_preferred_model_unavailable(self):
        """Test fallback when preferred model is unavailable."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Preferred model not in candidates
            mock_candidates.return_value = [
                ("ollama", "llama3.1:8b", 1),
            ]

            provider = MagicMock()
            provider.provider_id = "ollama"

            with patch.object(registry, "providers", {"ollama": provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.0
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL,
                            preferred_model="unavailable-model",
                        )

                        # Should fallback to available model
                        assert result.model_id == "llama3.1:8b"


class TestMultiProviderFallback:
    """Test comprehensive multi-provider fallback scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_through_multiple_providers(self):
        """Test fallback cascades through multiple providers."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Will fail
                ("provider2", "model-b", 2),  # Will fail
                ("provider3", "model-c", 3),  # Will succeed
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            provider2 = MagicMock()
            provider2.provider_id = "provider2"

            provider3 = MagicMock()
            provider3.provider_id = "provider3"

            with patch.object(
                registry,
                "providers",
                {
                    "provider1": provider1,
                    "provider2": provider2,
                    "provider3": provider3,
                },
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:

                    def cb_side_effect(pid):
                        # First two fail, third succeeds
                        if pid in ["provider1", "provider2"]:
                            return MagicMock(is_open=lambda: True)
                        return MagicMock(is_open=lambda: False)

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        # Should cascade to provider3
                        assert result.provider.provider_id == "provider3"
                        assert result.model_id == "model-c"

    @pytest.mark.asyncio
    async def test_no_providers_available_error(self):
        """Test error when no providers are available."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # No candidates returned
            mock_candidates.return_value = []

            with pytest.raises(RuntimeError, match="No providers available for task"):
                await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
