"""Cost estimation for LLM model usage."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_1k: float  # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens
    embedding_cost_per_1k: Optional[float] = None  # USD per 1K tokens for embeddings


# Pricing table for common models (USD per 1K tokens)
# Prices as of February 2026 - update regularly
PRICING_TABLE: Dict[str, ModelPricing] = {
    # OpenAI Models
    "gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
    "gpt-4": ModelPricing(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
    "gpt-3.5-turbo": ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "text-embedding-3-large": ModelPricing(
        input_cost_per_1k=0.00013, output_cost_per_1k=0, embedding_cost_per_1k=0.00013
    ),
    "text-embedding-3-small": ModelPricing(
        input_cost_per_1k=0.00002, output_cost_per_1k=0, embedding_cost_per_1k=0.00002
    ),
    "text-embedding-ada-002": ModelPricing(
        input_cost_per_1k=0.0001, output_cost_per_1k=0, embedding_cost_per_1k=0.0001
    ),
    # Anthropic Models
    "claude-3-opus-20240229": ModelPricing(
        input_cost_per_1k=0.015, output_cost_per_1k=0.075
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        input_cost_per_1k=0.003, output_cost_per_1k=0.015
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_cost_per_1k=0.00025, output_cost_per_1k=0.00125
    ),
    "claude-3-opus": ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "claude-3-sonnet": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-haiku": ModelPricing(
        input_cost_per_1k=0.00025, output_cost_per_1k=0.00125
    ),
    # Moonshot (Kimi)
    "kimi-2.5": ModelPricing(input_cost_per_1k=0.0001, output_cost_per_1k=0.0002),
    "moonshot-v1-8k": ModelPricing(
        input_cost_per_1k=0.00012, output_cost_per_1k=0.00012
    ),
    "moonshot-v1-32k": ModelPricing(
        input_cost_per_1k=0.00024, output_cost_per_1k=0.00024
    ),
    "moonshot-v1-128k": ModelPricing(
        input_cost_per_1k=0.00060, output_cost_per_1k=0.00060
    ),
    # DeepSeek
    "deepseek-v3": ModelPricing(input_cost_per_1k=0.00014, output_cost_per_1k=0.00028),
    "deepseek-chat": ModelPricing(
        input_cost_per_1k=0.00014, output_cost_per_1k=0.00028
    ),
    # Local models (Ollama, etc.) - FREE
    "llama3.1:8b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llama3.1:70b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llama3.1:405b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llama3.2:1b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llama3.2:3b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "mistral:7b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "mixtral:8x7b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "mixtral:8x22b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "nomic-embed-text": ModelPricing(
        input_cost_per_1k=0, output_cost_per_1k=0, embedding_cost_per_1k=0
    ),
    "llava:7b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llava:13b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
    "llava:34b": ModelPricing(input_cost_per_1k=0, output_cost_per_1k=0),
}


def get_model_pricing(model_id: str) -> Optional[ModelPricing]:
    """Get pricing for a model.

    Args:
        model_id: Model identifier (e.g., 'gpt-4o', 'llama3.1:8b')

    Returns:
        ModelPricing if found, None otherwise
    """
    # Try exact match first
    if model_id in PRICING_TABLE:
        return PRICING_TABLE[model_id]

    # Try without provider prefix (e.g., 'ollama/llama3.1:8b' -> 'llama3.1:8b')
    if "/" in model_id:
        base_model = model_id.split("/", 1)[1]
        if base_model in PRICING_TABLE:
            return PRICING_TABLE[base_model]

    # Try partial matches for versioned models
    for known_model, pricing in PRICING_TABLE.items():
        if model_id.startswith(known_model) or known_model.startswith(model_id):
            return pricing

    return None


def estimate_cost(
    model_id: str, input_tokens: int, output_tokens: int = 0, is_embedding: bool = False
) -> float:
    """Estimate cost for a model usage.

    Args:
        model_id: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (for chat)
        is_embedding: Whether this is an embedding request

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing(model_id)

    if not pricing:
        logger.warning(f"No pricing information for model: {model_id}")
        return 0.0

    if is_embedding and pricing.embedding_cost_per_1k is not None:
        # Use embedding-specific pricing
        cost = (input_tokens / 1000) * pricing.embedding_cost_per_1k
    else:
        # Use chat pricing
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        cost = input_cost + output_cost

    return round(cost, 6)


def estimate_cost_from_text(
    model_id: str, text: str, is_embedding: bool = False, chars_per_token: int = 4
) -> float:
    """Estimate cost from text (approximate token count).

    Args:
        model_id: Model identifier
        text: Input text
        is_embedding: Whether this is an embedding request
        chars_per_token: Average characters per token (default: 4)

    Returns:
        Estimated cost in USD
    """
    estimated_tokens = len(text) // chars_per_token
    return estimate_cost(model_id, estimated_tokens, 0, is_embedding)


def compare_costs(
    model_ids: list[str], input_tokens: int, output_tokens: int = 0
) -> Dict[str, float]:
    """Compare costs across multiple models.

    Args:
        model_ids: List of model identifiers
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Dictionary mapping model_id to estimated cost
    """
    costs = {}
    for model_id in model_ids:
        costs[model_id] = estimate_cost(model_id, input_tokens, output_tokens)
    return costs


def find_cheapest_model(
    model_ids: list[str],
    input_tokens: int,
    output_tokens: int = 0,
    max_cost: Optional[float] = None,
) -> Optional[str]:
    """Find the cheapest model from a list.

    Args:
        model_ids: List of model identifiers
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        max_cost: Maximum acceptable cost (optional)

    Returns:
        Model ID of cheapest model, or None if none meet criteria
    """
    costs = compare_costs(model_ids, input_tokens, output_tokens)

    # Filter by max_cost if specified
    if max_cost is not None:
        costs = {m: c for m, c in costs.items() if c <= max_cost}

    if not costs:
        return None

    # Return model with minimum cost
    return min(costs, key=costs.get)
