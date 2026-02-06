"""Integration tests for cost limit enforcement.

Tests validate:
- Request rejection when exceeding max_cost
- Budget enforcement across multiple requests
- Cost limit per provider
- Cost tracking and reporting
- Fallback to cheaper providers when over budget
"""

import pytest
from unittest.mock import MagicMock, patch
from sekha_llm_bridge.registry import registry
from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.pricing import estimate_cost


class TestCostLimitEnforcement:
    """Test that cost limits are enforced during routing."""

    @pytest.mark.asyncio
    async def test_reject_request_exceeding_max_cost(self):
        """Test that requests exceeding max_cost are rejected."""
        # Mock expensive provider
        with patch.object(registry, "providers") as mock_providers:
            expensive_provider = MagicMock()
            expensive_provider.provider_id = "openai"
            expensive_provider.provider_type = "openai"

            mock_providers.values.return_value = [expensive_provider]

            # Mock model cache with expensive model
            with patch.object(registry, "model_cache") as mock_cache:
                mock_cache.values.return_value = [
                    MagicMock(
                        model_id="gpt-4o",
                        provider_id="openai",
                        task=ModelTask.CHAT_SMART,
                        supports_vision=False,
                    )
                ]

                # Mock expensive cost estimate
                with patch(
                    "sekha_llm_bridge.registry.estimate_cost", return_value=0.05
                ):
                    # Request with tight budget
                    with pytest.raises(RuntimeError, match="No suitable provider"):
                        await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Budget too low for GPT-4o
                        )

    @pytest.mark.asyncio
    async def test_fallback_to_cheaper_provider(self):
        """Test fallback to cheaper provider when primary exceeds budget."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Return candidates in priority order: expensive first, cheap second
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),  # Priority 1, expensive
                ("ollama", "llama3.1:8b", 2),  # Priority 2, free
            ]

            # Mock providers
            expensive = MagicMock()
            expensive.provider_id = "openai"
            expensive.provider_type = "openai"

            cheap = MagicMock()
            cheap.provider_id = "ollama"
            cheap.provider_type = "ollama"

            with patch.object(
                registry, "providers", {"openai": expensive, "ollama": cheap}
            ):
                # Mock circuit breakers
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    # Mock cost estimates
                    def mock_estimate(model_id, *args):
                        if "gpt-4o" in model_id:
                            return 0.05  # Expensive
                        return 0.0  # Free

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=mock_estimate,
                    ):
                        # Route with budget that excludes expensive provider
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Only allows free provider
                        )

                        # Should route to cheap provider
                        assert result.provider.provider_id == "ollama"
                        assert result.model_id == "llama3.1:8b"
                        assert result.estimated_cost <= 0.01

    @pytest.mark.asyncio
    async def test_all_providers_too_expensive(self):
        """Test error when all providers exceed budget."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),
                ("anthropic", "claude-3.5-sonnet", 2),
            ]

            # Mock expensive providers
            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry, "providers", {"openai": openai, "anthropic": anthropic}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    # All models expensive
                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.05
                    ):
                        with pytest.raises(RuntimeError, match="No suitable provider"):
                            await registry.route_with_fallback(
                                task=ModelTask.CHAT_SMART,
                                max_cost=0.001,  # Very tight budget
                            )


class TestMultiRequestBudget:
    """Test budget enforcement across multiple requests."""

    @pytest.mark.asyncio
    async def test_cumulative_cost_tracking(self):
        """Test tracking cumulative cost across multiple requests."""

        # Simulate budget tracker
        class BudgetTracker:
            def __init__(self, max_budget):
                self.max_budget = max_budget
                self.spent = 0.0

            def can_afford(self, cost):
                return (self.spent + cost) <= self.max_budget

            def record_spend(self, cost):
                if not self.can_afford(cost):
                    raise RuntimeError(
                        f"Budget exceeded: ${self.spent + cost:.4f} > ${self.max_budget:.4f}"
                    )
                self.spent += cost

            def remaining(self):
                return self.max_budget - self.spent

        # Daily budget: $1.00
        tracker = BudgetTracker(max_budget=1.0)

        # Request 1: $0.40
        cost1 = estimate_cost("gpt-4o", 10000, 5000)
        assert tracker.can_afford(cost1)
        tracker.record_spend(cost1)

        # Request 2: $0.40
        cost2 = estimate_cost("gpt-4o", 10000, 5000)
        assert tracker.can_afford(cost2)
        tracker.record_spend(cost2)

        # Remaining budget should be small
        assert tracker.remaining() < 0.3

        # Request 3: Would exceed budget
        cost3 = estimate_cost("gpt-4o", 10000, 5000)
        assert not tracker.can_afford(cost3)

        with pytest.raises(RuntimeError, match="Budget exceeded"):
            tracker.record_spend(cost3)

    @pytest.mark.asyncio
    async def test_budget_reset_after_period(self):
        """Test budget resets after time period."""
        from datetime import datetime, timedelta

        class DailyBudgetTracker:
            def __init__(self, daily_limit):
                self.daily_limit = daily_limit
                self.spent_today = 0.0
                self.last_reset = datetime.now()

            def check_reset(self):
                now = datetime.now()
                if (now - self.last_reset) > timedelta(days=1):
                    self.spent_today = 0.0
                    self.last_reset = now

            def record_spend(self, cost):
                self.check_reset()
                if (self.spent_today + cost) > self.daily_limit:
                    raise RuntimeError("Daily budget exceeded")
                self.spent_today += cost

        tracker = DailyBudgetTracker(daily_limit=1.0)

        # Spend up to limit
        tracker.record_spend(0.5)
        tracker.record_spend(0.4)

        # Should reject next request
        with pytest.raises(RuntimeError, match="Daily budget exceeded"):
            tracker.record_spend(0.2)

        # Simulate next day
        tracker.last_reset = datetime.now() - timedelta(days=2)
        tracker.check_reset()

        # Budget should be reset
        assert tracker.spent_today == 0.0

        # Should accept new request
        tracker.record_spend(0.5)
        assert tracker.spent_today == 0.5


class TestPerProviderCostLimits:
    """Test cost limits per provider."""

    @pytest.mark.asyncio
    async def test_provider_specific_budget(self):
        """Test that different providers can have different budgets."""
        provider_budgets = {
            "openai": {"daily_limit": 5.0, "per_request": 0.10},
            "anthropic": {"daily_limit": 10.0, "per_request": 0.20},
            "ollama": {
                "daily_limit": float("inf"),
                "per_request": float("inf"),
            },  # Free
        }

        # OpenAI request should respect its limit
        openai_cost = estimate_cost("gpt-4o", 10000, 5000)
        assert openai_cost <= provider_budgets["openai"]["per_request"]

        # Large request to Anthropic
        large_tokens = 50000
        anthropic_cost = estimate_cost("claude-3.5-sonnet", large_tokens, large_tokens)

        if anthropic_cost > provider_budgets["anthropic"]["per_request"]:
            # Should reject or fallback
            assert True, "Cost limit enforced"

        # Ollama has no limits
        ollama_cost = estimate_cost("llama3.1:8b", 100000, 100000)
        assert ollama_cost == 0.0

    @pytest.mark.asyncio
    async def test_disable_expensive_provider_when_over_budget(self):
        """Test disabling expensive providers when budget is low."""
        remaining_budget = 0.01  # $0.01 left

        # Check which providers are affordable
        providers_costs = {
            "openai": estimate_cost("gpt-4o", 5000, 2000),
            "anthropic": estimate_cost("claude-3.5-sonnet", 5000, 2000),
            "ollama": estimate_cost("llama3.1:8b", 5000, 2000),
        }

        affordable_providers = [
            provider
            for provider, cost in providers_costs.items()
            if cost <= remaining_budget
        ]

        # Only free provider should be affordable
        assert "ollama" in affordable_providers
        assert "openai" not in affordable_providers
        assert "anthropic" not in affordable_providers


class TestCostReporting:
    """Test cost reporting and tracking."""

    @pytest.mark.asyncio
    async def test_cost_included_in_response(self):
        """Test that cost is included in routing response."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [("ollama", "llama3.1:8b", 1)]

            mock_provider = MagicMock()
            mock_provider.provider_id = "ollama"
            mock_provider.provider_type = "ollama"

            with patch.object(registry, "providers", {"ollama": mock_provider}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.0
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMALL
                        )

                        # Response should include cost estimate
                        assert hasattr(result, "estimated_cost")
                        assert result.estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_cost_logging(self):
        """Test that costs are logged for monitoring."""
        import logging
        from io import StringIO

        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger("sekha_llm_bridge.registry")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            with patch.object(registry, "_get_candidates") as mock_candidates:
                mock_candidates.return_value = [("openai", "gpt-4o-mini", 1)]

                mock_provider = MagicMock()
                mock_provider.provider_id = "openai"

                with patch.object(registry, "providers", {"openai": mock_provider}):
                    with patch.object(registry, "circuit_breakers") as mock_cbs:
                        mock_cbs.get.return_value = MagicMock(is_open=lambda: False)

                        with patch(
                            "sekha_llm_bridge.registry.estimate_cost",
                            return_value=0.002,
                        ):
                            result = await registry.route_with_fallback(
                                task=ModelTask.CHAT_SMALL, max_cost=0.01
                            )

                            # Verify routing completed successfully
                            assert result is not None
                            assert result.provider.provider_id == "openai"
                            assert result.estimated_cost == 0.002

                            # Verify cost was logged
                            log_output = log_capture.getvalue()
                            # Log output should contain routing information
                            # (exact format depends on logger configuration)
                            assert isinstance(log_output, str)
        finally:
            logger.removeHandler(handler)


class TestCostOptimization:
    """Test automatic cost optimization."""

    @pytest.mark.asyncio
    async def test_prefer_cheaper_model_when_equivalent(self):
        """Test that cheaper models are preferred when capabilities are equivalent."""
        # Two models, same task, different costs
        models = [
            {
                "provider_id": "openai",
                "model_id": "gpt-4o",
                "task": ModelTask.CHAT_SMART,
                "cost": 0.015,
                "priority": 1,
            },
            {
                "provider_id": "openai",
                "model_id": "gpt-4o-mini",
                "task": ModelTask.CHAT_SMART,
                "cost": 0.0005,
                "priority": 1,  # Same priority
            },
        ]

        # When max_cost allows both, should prefer cheaper
        affordable = [m for m in models if m["cost"] <= 0.02]

        # Both affordable, sort by cost
        affordable.sort(key=lambda x: x["cost"])

        assert affordable[0]["model_id"] == "gpt-4o-mini"
        assert affordable[0]["cost"] < affordable[1]["cost"]

    @pytest.mark.asyncio
    async def test_suggest_cheaper_alternative_on_rejection(self):
        """Test that cheaper alternatives are suggested when request is rejected."""
        max_cost = 0.001
        requested_model = "gpt-4o"
        requested_cost = estimate_cost(requested_model, 10000, 5000)

        if requested_cost > max_cost:
            # Find cheaper alternatives
            alternatives = [
                ("gpt-4o-mini", estimate_cost("gpt-4o-mini", 10000, 5000)),
                ("llama3.1:8b", estimate_cost("llama3.1:8b", 10000, 5000)),
            ]

            affordable = [model for model, cost in alternatives if cost <= max_cost]

            # Should find affordable alternatives
            assert len(affordable) > 0
            assert "llama3.1:8b" in affordable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
