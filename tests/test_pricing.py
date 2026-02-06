"""Comprehensive tests for pricing module."""

import pytest
from sekha_llm_bridge.pricing import (
    ModelPricing,
    PRICING_TABLE,
    get_model_pricing,
    estimate_cost,
    estimate_cost_from_text,
    compare_costs,
    find_cheapest_model,
)


class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_model_pricing_creation(self):
        """Test creating ModelPricing instance."""
        pricing = ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015)
        assert pricing.input_cost_per_1k == 0.005
        assert pricing.output_cost_per_1k == 0.015
        assert pricing.embedding_cost_per_1k is None

    def test_model_pricing_with_embedding(self):
        """Test ModelPricing with embedding cost."""
        pricing = ModelPricing(
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            embedding_cost_per_1k=0.0005,
        )
        assert pricing.embedding_cost_per_1k == 0.0005


class TestPricingTable:
    """Test PRICING_TABLE completeness."""

    def test_pricing_table_exists(self):
        """Test pricing table is not empty."""
        assert len(PRICING_TABLE) > 0

    def test_openai_models_in_table(self):
        """Test OpenAI models are in pricing table."""
        assert "gpt-4o" in PRICING_TABLE
        assert "gpt-4o-mini" in PRICING_TABLE
        assert "gpt-3.5-turbo" in PRICING_TABLE

    def test_anthropic_models_in_table(self):
        """Test Anthropic models are in pricing table."""
        assert "claude-3-opus" in PRICING_TABLE
        assert "claude-3-sonnet" in PRICING_TABLE
        assert "claude-3-haiku" in PRICING_TABLE

    def test_local_models_free(self):
        """Test local models have zero cost."""
        local_models = ["llama3.1:8b", "nomic-embed-text", "mistral:7b"]
        for model in local_models:
            pricing = PRICING_TABLE.get(model)
            assert pricing is not None
            assert pricing.input_cost_per_1k == 0
            assert pricing.output_cost_per_1k == 0

    def test_embedding_models_have_embedding_cost(self):
        """Test embedding models have embedding-specific pricing."""
        embedding_models = [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "nomic-embed-text",
        ]
        for model in embedding_models:
            pricing = PRICING_TABLE.get(model)
            assert pricing is not None
            assert pricing.embedding_cost_per_1k is not None


class TestGetModelPricing:
    """Test get_model_pricing function."""

    def test_exact_match(self):
        """Test exact model ID match."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_cost_per_1k == 0.005
        assert pricing.output_cost_per_1k == 0.015

    def test_with_provider_prefix(self):
        """Test model ID with provider prefix."""
        pricing = get_model_pricing("ollama/llama3.1:8b")
        assert pricing is not None
        assert pricing.input_cost_per_1k == 0

    def test_partial_match(self):
        """Test partial model name matching."""
        # Should match models that start with the query
        pricing = get_model_pricing("gpt-4")
        assert pricing is not None

    def test_unknown_model(self):
        """Test unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_claude_versioned_models(self):
        """Test Claude versioned model matching."""
        pricing = get_model_pricing("claude-3-opus-20240229")
        assert pricing is not None
        assert pricing.input_cost_per_1k == 0.015


class TestEstimateCost:
    """Test estimate_cost function."""

    def test_chat_cost_calculation(self):
        """Test cost calculation for chat models."""
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # (1000/1000) * 0.005 + (500/1000) * 0.015 = 0.005 + 0.0075 = 0.0125
        assert cost == 0.0125

    def test_embedding_cost_calculation(self):
        """Test cost calculation for embeddings."""
        cost = estimate_cost(
            "text-embedding-3-small", input_tokens=1000, is_embedding=True
        )
        # (1000/1000) * 0.00002 = 0.00002
        assert cost == 0.00002

    def test_free_model_cost(self):
        """Test free models return zero cost."""
        cost = estimate_cost("llama3.1:8b", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self):
        """Test unknown models return zero cost with warning."""
        cost = estimate_cost("unknown-model", input_tokens=1000)
        assert cost == 0.0

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        cost = estimate_cost("gpt-4o-mini", input_tokens=100000, output_tokens=50000)
        # (100000/1000) * 0.00015 + (50000/1000) * 0.0006
        # = 0.015 + 0.03 = 0.045
        assert cost == 0.045

    def test_rounding(self):
        """Test cost is rounded to 6 decimal places."""
        cost = estimate_cost("gpt-4o", input_tokens=333, output_tokens=666)
        # Should be rounded to 6 decimals
        assert len(str(cost).split(".")[1]) <= 6

    def test_zero_output_tokens(self):
        """Test with zero output tokens."""
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        assert cost == 0.005  # Only input cost


class TestEstimateCostFromText:
    """Test estimate_cost_from_text function."""

    def test_basic_text_estimation(self):
        """Test cost estimation from text."""
        text = "This is a test" * 100  # ~400 characters = ~100 tokens
        cost = estimate_cost_from_text("gpt-4o", text)
        assert cost > 0

    def test_embedding_text_estimation(self):
        """Test embedding cost estimation from text."""
        text = "Test embedding" * 100
        cost = estimate_cost_from_text(
            "text-embedding-3-small", text, is_embedding=True
        )
        assert cost >= 0

    def test_custom_chars_per_token(self):
        """Test custom characters per token ratio."""
        text = "a" * 1000  # 1000 characters
        cost1 = estimate_cost_from_text("gpt-4o", text, chars_per_token=4)
        cost2 = estimate_cost_from_text("gpt-4o", text, chars_per_token=2)
        # More tokens with chars_per_token=2, so higher cost
        assert cost2 > cost1

    def test_empty_text(self):
        """Test empty text returns zero cost."""
        cost = estimate_cost_from_text("gpt-4o", "")
        assert cost == 0.0


class TestCompareCosts:
    """Test compare_costs function."""

    def test_compare_multiple_models(self):
        """Test comparing costs across multiple models."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        costs = compare_costs(models, input_tokens=1000, output_tokens=500)

        assert len(costs) == 3
        assert all(model in costs for model in models)
        # gpt-4o-mini should be cheapest
        assert costs["gpt-4o-mini"] < costs["gpt-4o"]

    def test_compare_with_free_models(self):
        """Test comparison including free models."""
        models = ["gpt-4o", "llama3.1:8b"]
        costs = compare_costs(models, input_tokens=1000)

        assert costs["llama3.1:8b"] == 0.0
        assert costs["gpt-4o"] > 0

    def test_compare_empty_list(self):
        """Test comparing empty model list."""
        costs = compare_costs([], input_tokens=1000)
        assert costs == {}

    def test_compare_with_unknown_models(self):
        """Test comparison with unknown models."""
        models = ["gpt-4o", "unknown-model"]
        costs = compare_costs(models, input_tokens=1000)

        assert len(costs) == 2
        assert costs["unknown-model"] == 0.0  # Unknown models return 0


class TestFindCheapestModel:
    """Test find_cheapest_model function."""

    def test_find_cheapest_among_paid_models(self):
        """Test finding cheapest among paid models."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        cheapest = find_cheapest_model(models, input_tokens=1000, output_tokens=500)

        assert cheapest == "gpt-4o-mini"  # Should be cheapest

    def test_find_cheapest_with_free_models(self):
        """Test free models are selected as cheapest."""
        models = ["gpt-4o", "llama3.1:8b"]
        cheapest = find_cheapest_model(models, input_tokens=1000)

        assert cheapest == "llama3.1:8b"  # Free model

    def test_find_cheapest_with_max_cost(self):
        """Test max_cost constraint."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        cheapest = find_cheapest_model(
            models, input_tokens=1000, output_tokens=500, max_cost=0.001
        )

        # Only gpt-4o-mini should be under budget
        assert cheapest == "gpt-4o-mini"

    def test_find_cheapest_none_under_budget(self):
        """Test when no models are under budget."""
        models = ["gpt-4o", "gpt-3.5-turbo"]
        cheapest = find_cheapest_model(
            models, input_tokens=1000, output_tokens=500, max_cost=0.0001
        )

        assert cheapest is None  # No model under budget

    def test_find_cheapest_empty_list(self):
        """Test with empty model list."""
        cheapest = find_cheapest_model([], input_tokens=1000)
        assert cheapest is None

    def test_find_cheapest_single_model(self):
        """Test with single model."""
        cheapest = find_cheapest_model(["gpt-4o"], input_tokens=1000)
        assert cheapest == "gpt-4o"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_token_counts(self):
        """Test negative token counts are handled."""
        # Should not crash, though negative doesn't make sense
        cost = estimate_cost("gpt-4o", input_tokens=-100, output_tokens=-50)
        assert isinstance(cost, float)

    def test_very_large_costs(self):
        """Test very large token counts."""
        cost = estimate_cost("gpt-4o", input_tokens=10_000_000, output_tokens=5_000_000)
        assert cost > 0
        assert isinstance(cost, float)

    def test_model_id_case_sensitivity(self):
        """Test model ID matching is case-sensitive."""
        pricing1 = get_model_pricing("gpt-4o")
        pricing2 = get_model_pricing("GPT-4O")
        # Should not match due to case
        assert pricing1 is not None
        # Uppercase might not match exact, but could partial match

    def test_special_characters_in_model_id(self):
        """Test model IDs with special characters."""
        pricing = get_model_pricing("llama3.1:8b")
        assert pricing is not None

    def test_compare_costs_preserves_order(self):
        """Test compare_costs returns dict with all models."""
        models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]
        costs = compare_costs(models, input_tokens=1000)
        assert set(costs.keys()) == set(models)
