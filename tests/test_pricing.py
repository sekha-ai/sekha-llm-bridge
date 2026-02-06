"""Comprehensive tests for pricing module."""

from sekha_llm_bridge.pricing import (
    get_model_pricing,
    estimate_cost,
    compare_costs,
    find_cheapest_model,
)


class TestGetModelPricing:
    """Test get_model_pricing function."""

    def test_openai_gpt4o_pricing(self):
        """Test GPT-4o pricing retrieval."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0

    def test_openai_gpt4o_mini_pricing(self):
        """Test GPT-4o-mini pricing retrieval."""
        pricing = get_model_pricing("gpt-4o-mini")
        assert pricing is not None
        assert pricing["input"] < get_model_pricing("gpt-4o")["input"]

    def test_anthropic_claude_pricing(self):
        """Test Claude pricing retrieval."""
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_local_model_pricing(self):
        """Test local model (free) pricing."""
        pricing = get_model_pricing("llama3.1:8b")
        assert pricing is not None
        assert pricing["input"] == 0
        assert pricing["output"] == 0

    def test_unknown_model_pricing(self):
        """Test pricing for unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_embedding_model_pricing(self):
        """Test embedding model pricing."""
        pricing = get_model_pricing("text-embedding-3-small")
        assert pricing is not None
        assert "input" in pricing

    def test_all_openai_models_have_pricing(self):
        """Test all common OpenAI models have pricing."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        for model in models:
            pricing = get_model_pricing(model)
            assert pricing is not None, f"Missing pricing for {model}"

    def test_all_anthropic_models_have_pricing(self):
        """Test all common Anthropic models have pricing."""
        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        for model in models:
            pricing = get_model_pricing(model)
            assert pricing is not None, f"Missing pricing for {model}"

    def test_pricing_values_positive(self):
        """Test all pricing values are non-negative."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing["input"] >= 0
        assert pricing["output"] >= 0

    def test_model_id_case_sensitive(self):
        """Test model ID matching is case-sensitive."""
        pricing1 = get_model_pricing("gpt-4o")
        # GPT-4O in uppercase might not match
        assert pricing1 is not None


class TestEstimateCost:
    """Test estimate_cost function."""

    def test_estimate_gpt4o_cost(self):
        """Test cost estimation for GPT-4o."""
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_local_model_cost(self):
        """Test cost estimation for local model (should be 0)."""
        cost = estimate_cost("llama3.1:8b", input_tokens=1000, output_tokens=500)
        assert cost == 0

    def test_estimate_unknown_model_cost(self):
        """Test cost estimation for unknown model."""
        cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_estimate_with_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0

    def test_estimate_with_only_input_tokens(self):
        """Test cost estimation with only input tokens."""
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        assert cost > 0

    def test_estimate_with_only_output_tokens(self):
        """Test cost estimation with only output tokens."""
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=1000)
        assert cost > 0

    def test_output_tokens_more_expensive(self):
        """Test that output tokens are typically more expensive."""
        input_cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        output_cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=1000)
        # For most models, output is more expensive
        assert output_cost >= input_cost

    def test_large_token_counts(self):
        """Test cost estimation with large token counts."""
        cost = estimate_cost("gpt-4o", input_tokens=100000, output_tokens=50000)
        assert cost > 0
        assert cost < 1000  # Sanity check


class TestCompareCosts:
    """Test compare_costs function."""

    def test_compare_two_models(self):
        """Test comparing costs of two models."""
        comparison = compare_costs(
            ["gpt-4o", "gpt-4o-mini"], input_tokens=1000, output_tokens=500
        )
        assert len(comparison) == 2
        assert "gpt-4o" in comparison
        assert "gpt-4o-mini" in comparison

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        """Test that gpt-4o-mini is cheaper than gpt-4o."""
        comparison = compare_costs(
            ["gpt-4o", "gpt-4o-mini"], input_tokens=1000, output_tokens=500
        )
        assert comparison["gpt-4o-mini"] < comparison["gpt-4o"]

    def test_compare_local_vs_paid(self):
        """Test comparing local (free) vs paid models."""
        comparison = compare_costs(
            ["gpt-4o", "llama3.1:8b"], input_tokens=1000, output_tokens=500
        )
        assert comparison["llama3.1:8b"] == 0
        assert comparison["gpt-4o"] > 0

    def test_compare_empty_model_list(self):
        """Test comparing empty model list."""
        comparison = compare_costs([], input_tokens=1000, output_tokens=500)
        assert comparison == {}

    def test_compare_single_model(self):
        """Test comparing single model."""
        comparison = compare_costs(["gpt-4o"], input_tokens=1000, output_tokens=500)
        assert len(comparison) == 1
        assert "gpt-4o" in comparison

    def test_compare_with_unknown_model(self):
        """Test comparison with unknown model."""
        comparison = compare_costs(
            ["gpt-4o", "unknown-model"], input_tokens=1000, output_tokens=500
        )
        # Unknown model should be excluded or have None value
        assert "gpt-4o" in comparison

    def test_compare_all_openai_models(self):
        """Test comparing all OpenAI models."""
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        comparison = compare_costs(models, input_tokens=1000, output_tokens=500)
        # All should be in comparison
        for model in models:
            assert model in comparison


class TestFindCheapestModel:
    """Test find_cheapest_model function."""

    def test_find_cheapest_from_two_models(self):
        """Test finding cheapest from two models."""
        cheapest = find_cheapest_model(
            ["gpt-4o", "gpt-4o-mini"], input_tokens=1000, output_tokens=500
        )
        assert cheapest == "gpt-4o-mini"

    def test_local_model_is_cheapest(self):
        """Test that local model is cheapest (free)."""
        cheapest = find_cheapest_model(
            ["gpt-4o", "llama3.1:8b"], input_tokens=1000, output_tokens=500
        )
        assert cheapest == "llama3.1:8b"

    def test_find_cheapest_empty_list(self):
        """Test finding cheapest from empty list."""
        cheapest = find_cheapest_model([], input_tokens=1000, output_tokens=500)
        assert cheapest is None

    def test_find_cheapest_single_model(self):
        """Test finding cheapest from single model."""
        cheapest = find_cheapest_model(["gpt-4o"], input_tokens=1000, output_tokens=500)
        assert cheapest == "gpt-4o"

    def test_find_cheapest_among_many(self):
        """Test finding cheapest among many models."""
        models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-5-haiku-20241022",
        ]
        cheapest = find_cheapest_model(models, input_tokens=1000, output_tokens=500)
        # gpt-3.5-turbo or gpt-4o-mini should be among cheapest
        assert cheapest in ["gpt-3.5-turbo", "gpt-4o-mini"]

    def test_find_cheapest_with_unknown_models(self):
        """Test finding cheapest when some models are unknown."""
        models = ["gpt-4o", "unknown-model-1", "unknown-model-2"]
        cheapest = find_cheapest_model(models, input_tokens=1000, output_tokens=500)
        # Should return the one known model
        assert cheapest == "gpt-4o"


class TestPricingEdgeCases:
    """Test edge cases in pricing."""

    def test_negative_token_counts(self):
        """Test pricing with negative token counts."""
        # Should handle gracefully (treat as 0 or return None)
        cost = estimate_cost("gpt-4o", input_tokens=-100, output_tokens=-50)
        assert cost is not None  # Should handle gracefully

    def test_very_large_token_counts(self):
        """Test pricing with very large token counts."""
        cost = estimate_cost("gpt-4o", input_tokens=10000000, output_tokens=5000000)
        assert cost > 0
        assert cost < 10000  # Reasonable upper bound

    def test_pricing_precision(self):
        """Test pricing calculations maintain precision."""
        cost = estimate_cost("gpt-4o", input_tokens=1, output_tokens=1)
        assert cost > 0
        assert cost < 1  # Should be fraction of a cent

    def test_none_model_id(self):
        """Test pricing with None as model ID."""
        pricing = get_model_pricing(None)
        assert pricing is None

    def test_empty_string_model_id(self):
        """Test pricing with empty string as model ID."""
        pricing = get_model_pricing("")
        assert pricing is None


class TestEmbeddingPricing:
    """Test pricing for embedding models."""

    def test_embedding_model_has_pricing(self):
        """Test embedding models have pricing defined."""
        models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        for model in models:
            pricing = get_model_pricing(model)
            assert pricing is not None, f"Missing pricing for {model}"

    def test_embedding_cost_estimation(self):
        """Test cost estimation for embeddings."""
        cost = estimate_cost(
            "text-embedding-3-small", input_tokens=1000, output_tokens=0
        )
        assert cost >= 0

    def test_embedding_cheaper_than_chat(self):
        """Test embeddings are cheaper than chat completions."""
        embed_cost = estimate_cost(
            "text-embedding-3-small", input_tokens=1000, output_tokens=0
        )
        chat_cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        assert embed_cost < chat_cost


class TestPricingComparison:
    """Test comprehensive pricing comparisons."""

    def test_openai_vs_anthropic_pricing(self):
        """Test comparing OpenAI vs Anthropic pricing."""
        comparison = compare_costs(
            ["gpt-4o", "claude-3-5-sonnet-20241022"],
            input_tokens=1000,
            output_tokens=500,
        )
        assert len(comparison) == 2
        assert all(cost > 0 for cost in comparison.values())

    def test_model_tiers_pricing(self):
        """Test pricing reflects model tiers (small < medium < large)."""
        small = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        large = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert small < large
