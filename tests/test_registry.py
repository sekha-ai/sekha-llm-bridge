"""Comprehensive tests for ModelRegistry."""

from unittest.mock import Mock, patch

import pytest

from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.registry import ModelRegistry, registry


class TestModelRegistryInitialization:
    """Test registry initialization and configuration loading."""

    def test_registry_singleton_exists(self):
        """Test that global registry instance exists."""
        assert registry is not None
        assert isinstance(registry, ModelRegistry)


class TestRoutingWithFallback:
    """Test routing with fallback behavior."""

    @pytest.mark.asyncio
    async def test_route_with_circuit_breaker_open(self):
        """Test routing when circuit breaker is open."""
        # When circuit breaker is open, routing should fallback to other providers
        # or return None/error if no providers available
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Return candidates, but circuit breaker will block primary
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Circuit breaker open
                ("provider2", "model-b", 2),  # Circuit breaker closed
            ]

            provider1 = Mock()
            provider1.provider_id = "provider1"

            provider2 = Mock()
            provider2.provider_id = "provider2"

            with patch.object(
                registry, "providers", {"provider1": provider1, "provider2": provider2}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:

                    def cb_side_effect(pid):
                        if pid == "provider1":
                            return Mock(is_open=lambda: True)  # Open - blocked
                        return Mock(is_open=lambda: False)  # Closed - available

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        # Should fallback to provider2
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        assert result is not None
                        assert result.provider.provider_id == "provider2"

    @pytest.mark.asyncio
    async def test_route_with_max_cost_constraint(self):
        """Test routing with max cost constraint."""
        # Routes should respect max_cost and choose affordable providers
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),  # Expensive
                ("ollama", "llama3.1:8b", 2),  # Free
            ]

            openai = Mock()
            openai.provider_id = "openai"

            ollama = Mock()
            ollama.provider_id = "ollama"

            with patch.object(
                registry, "providers", {"openai": openai, "ollama": ollama}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = Mock(is_open=lambda: False)

                    def cost_side_effect(model, **kwargs):
                        if "gpt" in model:
                            return 0.05  # Too expensive
                        return 0.0  # Free

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=cost_side_effect,
                    ):
                        # Should route to cheap provider
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Forces cheap provider
                        )

                        assert result is not None
                        assert result.provider.provider_id == "ollama"
                        assert result.estimated_cost <= 0.01
