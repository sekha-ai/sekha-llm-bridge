"""Integration tests for cost limit enforcement in routing.

Tests validate:
- Cost-aware routing
- Max cost constraints
- Provider selection based on cost
- Fallback when cost limits exceeded
"""

from unittest.mock import MagicMock, patch

import pytest

from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.registry import registry


class TestCostLimitEnforcement:
    """Test enforcement of cost limits in routing."""

    @pytest.mark.asyncio
    async def test_respect_max_cost_during_routing(self):
        """Test that routing respects max_cost parameter."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Two candidates: expensive and cheap
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

                    def cost_side_effect(model, **kwargs):
                        if "gpt" in model:
                            return 0.05  # Too expensive
                        return 0.0  # Free

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=cost_side_effect,
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Low limit
                        )

                        # Should pick free provider
                        assert result.provider.provider_id == "ollama"
                        assert result.estimated_cost <= 0.01

    @pytest.mark.asyncio
    async def test_all_providers_too_expensive(self):
        """Test error when all providers exceed cost limit."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # All candidates expensive
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
                ("anthropic", "claude-3.5-sonnet", 2),
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry,
                "providers",
                {"openai": openai, "anthropic": anthropic},
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    # All models expensive
                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.10
                    ):
                        with pytest.raises(
                            RuntimeError,
                            match="No suitable providers available",
                        ):
                            await registry.route_with_fallback(
                                task=ModelTask.CHAT_SMART,
                                max_cost=0.001,  # Very low limit
                            )

    @pytest.mark.asyncio
    async def test_cost_estimation_accuracy(self):
        """Test that cost estimation is reasonably accurate."""
        from sekha_llm_bridge.pricing import estimate_cost

        # Test known model costs
        gpt4_cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert gpt4_cost > 0.0

        # Mini should be cheaper
        gpt4_mini_cost = estimate_cost(
            "gpt-4o-mini", input_tokens=1000, output_tokens=500
        )
        assert gpt4_mini_cost < gpt4_cost

        # Ollama is free
        ollama_cost = estimate_cost("llama3.1:8b", input_tokens=1000, output_tokens=500)
        assert ollama_cost == 0.0
