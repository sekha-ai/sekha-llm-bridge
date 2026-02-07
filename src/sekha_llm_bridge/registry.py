"""Model registry for provider and model management."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .config import ModelTask, ProviderConfig, get_settings
from .pricing import estimate_cost
from .providers.base import LlmProvider
from .providers.litellm_provider import LiteLlmProvider
from .resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class CachedModelInfo:
    """Cached information about a model."""

    model_id: str
    provider_id: str
    task: ModelTask
    context_window: int
    dimension: Optional[int] = None
    supports_vision: bool = False
    supports_audio: bool = False
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RoutingResult:
    """Result of model routing decision."""

    provider: LlmProvider
    model_id: str
    estimated_cost: float
    reason: str  # Why this provider/model was selected


class ModelRegistry:
    """Central registry for LLM providers and models.

    Responsibilities:
    - Track all configured providers
    - Maintain model capability cache
    - Route requests to optimal provider/model
    - Handle fallback on provider failures
    - Integrate circuit breakers
    - Estimate costs
    """

    def __init__(self):
        """Initialize the model registry."""
        self.providers: Dict[str, LlmProvider] = {}
        self.model_cache: Dict[str, CachedModelInfo] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._last_cache_refresh: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        # Initialize providers from settings
        self._initialize_providers()
        logger.info(
            f"Model registry initialized with {len(self.providers)} provider(s)"
        )

    def _initialize_providers(self) -> None:
        """Initialize providers from configuration."""
        settings = get_settings()
        for provider_config in settings.providers:
            try:
                # Create provider instance
                provider = self._create_provider(provider_config)
                self.providers[provider_config.id] = provider

                # Create circuit breaker for this provider
                cb_config = settings.routing.circuit_breaker
                self.circuit_breakers[provider_config.id] = CircuitBreaker(
                    failure_threshold=cb_config.failure_threshold,
                    timeout_secs=cb_config.timeout_secs,
                    success_threshold=cb_config.success_threshold,
                )

                # Cache model information
                for model in provider_config.models:
                    cache_key = f"{provider_config.id}:{model.model_id}"
                    self.model_cache[cache_key] = CachedModelInfo(
                        model_id=model.model_id,
                        provider_id=provider_config.id,
                        task=ModelTask(model.task.value),
                        context_window=model.context_window,
                        dimension=model.dimension,
                        supports_vision=model.supports_vision,
                        supports_audio=model.supports_audio,
                    )

                logger.info(
                    f"Initialized provider '{provider_config.id}' with "
                    f"{len(provider_config.models)} model(s)"
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize provider '{provider_config.id}': {e}"
                )

    def _create_provider(self, provider_config: ProviderConfig) -> LlmProvider:
        """Create a provider instance from configuration."""
        config_dict = {
            "provider_type": provider_config.provider_type.value,
            "base_url": provider_config.base_url,
            "api_key": provider_config.api_key,
            "timeout": provider_config.timeout_secs,
            "models": [
                {
                    "model_id": m.model_id,
                    "task": m.task.value,
                    "context_window": m.context_window,
                }
                for m in provider_config.models
            ],
        }

        # For now, we use LiteLlmProvider for all provider types
        return LiteLlmProvider(provider_config.id, config_dict)

    async def route_with_fallback(
        self,
        task: ModelTask,
        preferred_model: Optional[str] = None,
        require_vision: bool = False,
        max_cost: Optional[float] = None,
        **kwargs,
    ) -> RoutingResult:
        """Route a request to the best available provider/model.

        Args:
            task: The task type (embedding, chat_small, etc.)
            preferred_model: Optional preferred model ID
            require_vision: Whether vision support is required
            max_cost: Maximum acceptable cost per request
            **kwargs: Additional routing hints

        Returns:
            RoutingResult with provider, model, and cost estimate

        Raises:
            RuntimeError: If no suitable provider is available
        """
        # Get candidate providers sorted by priority
        candidates = self._get_candidates(
            task=task,
            require_vision=require_vision,
            preferred_model=preferred_model,
        )

        if not candidates:
            raise RuntimeError(
                f"No providers available for task '{task.value}' "
                f"(vision={require_vision})"
            )

        # Try each candidate until one succeeds
        last_error = None

        for provider_id, model_id, priority in candidates:
            # Check circuit breaker
            cb = self.circuit_breakers.get(provider_id)
            if cb and cb.is_open():
                logger.warning(
                    f"Skipping provider '{provider_id}' - circuit breaker is open"
                )
                continue

            # Get provider
            provider = self.providers.get(provider_id)
            if not provider:
                continue

            # Estimate cost
            # For now, use a basic estimate - actual cost will be calculated after request
            estimated_cost = estimate_cost(
                model_id,
                input_tokens=1000,  # Placeholder
                output_tokens=500,  # Placeholder
            )

            # Check cost limit
            if max_cost is not None and estimated_cost > max_cost:
                logger.info(
                    f"Skipping {provider_id}/{model_id} - "
                    f"estimated cost ${estimated_cost:.4f} exceeds limit ${max_cost:.4f}"
                )
                continue

            # Found a suitable provider
            reason = f"Selected by priority ({priority})"
            if preferred_model:
                reason = "Matched preferred model"

            return RoutingResult(
                provider=provider,
                model_id=model_id,
                estimated_cost=estimated_cost,
                reason=reason,
            )

        # No providers available
        if last_error:
            raise RuntimeError(
                f"All providers failed for task '{task.value}': {last_error}"
            )
        else:
            raise RuntimeError(
                f"No suitable providers available for task '{task.value}'"
            )

    def _get_candidates(
        self,
        task: ModelTask,
        require_vision: bool = False,
        preferred_model: Optional[str] = None,
    ) -> List[Tuple[str, str, int]]:
        """Get candidate (provider_id, model_id, priority) tuples.

        Returns:
            List of (provider_id, model_id, priority) sorted by priority
        """
        settings = get_settings()
        candidates = []

        # If preferred model specified, try to find it first
        if preferred_model:
            for cache_key, model_info in self.model_cache.items():
                if model_info.model_id == preferred_model:
                    provider_config = next(
                        (
                            p
                            for p in settings.providers
                            if p.id == model_info.provider_id
                        ),
                        None,
                    )
                    if provider_config:
                        candidates.append(
                            (
                                model_info.provider_id,
                                model_info.model_id,
                                0,  # Highest priority
                            )
                        )

        # Add models matching task
        for cache_key, model_info in self.model_cache.items():
            # Skip if doesn't match task
            if model_info.task != task:
                continue

            # Skip if vision required but not supported
            if require_vision and not model_info.supports_vision:
                continue

            # Get provider priority
            provider_config = next(
                (p for p in settings.providers if p.id == model_info.provider_id), None
            )
            if not provider_config:
                continue

            # Skip if already added as preferred
            if preferred_model and model_info.model_id == preferred_model:
                continue

            candidates.append(
                (
                    model_info.provider_id,
                    model_info.model_id,
                    provider_config.priority,
                )
            )

        # Sort by priority (lower number = higher priority)
        candidates.sort(key=lambda x: x[2])

        return candidates

    async def execute_with_circuit_breaker(
        self, provider_id: str, operation, *args, **kwargs
    ):
        """Execute an operation with circuit breaker protection.

        Args:
            provider_id: Provider identifier
            operation: Async function to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result from the operation

        Raises:
            Exception: If operation fails
        """
        cb = self.circuit_breakers.get(provider_id)

        try:
            result = await operation(*args, **kwargs)

            # Record success
            if cb:
                cb.record_success()

            return result

        except Exception as e:
            # Record failure
            if cb:
                cb.record_failure(e)
            raise

    def get_provider_health(self) -> Dict[str, Dict]:
        """Get health status of all providers.

        Returns:
            Dictionary mapping provider_id to health info
        """
        health = {}

        for provider_id, provider in self.providers.items():
            cb = self.circuit_breakers.get(provider_id)
            cb_stats = cb.get_stats() if cb else {}

            health[provider_id] = {
                "provider_type": provider.provider_type,
                "circuit_breaker": cb_stats,
                "models_count": len(
                    [
                        m
                        for m in self.model_cache.values()
                        if m.provider_id == provider_id
                    ]
                ),
            }

        return health

    def list_all_models(self) -> List[Dict]:
        """List all available models across all providers.

        Returns:
            List of model dictionaries
        """
        models = []

        for model_info in self.model_cache.values():
            models.append(
                {
                    "model_id": model_info.model_id,
                    "provider_id": model_info.provider_id,
                    "task": model_info.task.value,
                    "context_window": model_info.context_window,
                    "dimension": model_info.dimension,
                    "supports_vision": model_info.supports_vision,
                    "supports_audio": model_info.supports_audio,
                }
            )

        return models


# Global registry instance
registry = ModelRegistry()
