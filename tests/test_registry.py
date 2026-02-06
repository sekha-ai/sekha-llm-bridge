"""Comprehensive tests for model registry."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.registry import (CachedModelInfo, ModelRegistry,
                                       RoutingResult)


class TestCachedModelInfo:
    """Test CachedModelInfo dataclass."""

    def test_cached_model_info_creation(self):
        """Test creating CachedModelInfo."""
        info = CachedModelInfo(
            model_id="gpt-4o",
            provider_id="openai",
            task=ModelTask.CHAT_SMART,
            context_window=128000,
        )
        assert info.model_id == "gpt-4o"
        assert info.provider_id == "openai"
        assert info.task == ModelTask.CHAT_SMART
        assert info.context_window == 128000
        assert info.dimension is None
        assert not info.supports_vision
        assert not info.supports_audio
        assert isinstance(info.last_updated, datetime)

    def test_cached_model_info_with_optionals(self):
        """Test CachedModelInfo with optional fields."""
        info = CachedModelInfo(
            model_id="text-embedding-3-large",
            provider_id="openai",
            task=ModelTask.EMBEDDING,
            context_window=8191,
            dimension=3072,
            supports_vision=False,
            supports_audio=False,
        )
        assert info.dimension == 3072


class TestRoutingResult:
    """Test RoutingResult dataclass."""

    def test_routing_result_creation(self):
        """Test creating RoutingResult."""
        mock_provider = Mock()
        result = RoutingResult(
            provider=mock_provider,
            model_id="gpt-4o",
            estimated_cost=0.025,
            reason="Selected by priority",
        )
        assert result.provider == mock_provider
        assert result.model_id == "gpt-4o"
        assert result.estimated_cost == 0.025
        assert result.reason == "Selected by priority"


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    with patch("sekha_llm_bridge.registry.settings") as mock:
        # Mock provider config
        provider_config = Mock()
        provider_config.id = "test-provider"
        provider_config.provider_type.value = "litellm"
        provider_config.base_url = "http://localhost:11434"
        provider_config.api_key = None
        provider_config.timeout_secs = 30
        provider_config.priority = 1

        # Mock model config
        model_config = Mock()
        model_config.model_id = "llama3.1:8b"
        model_config.task = Mock()
        model_config.task.value = "chat_small"
        model_config.context_window = 8192
        model_config.dimension = None
        model_config.supports_vision = False
        model_config.supports_audio = False

        provider_config.models = [model_config]
        mock.providers = [provider_config]

        # Mock routing config
        mock.routing = Mock()
        mock.routing.circuit_breaker = Mock()
        mock.routing.circuit_breaker.failure_threshold = 5
        mock.routing.circuit_breaker.timeout_secs = 60
        mock.routing.circuit_breaker.success_threshold = 2

        yield mock


class TestModelRegistryInitialization:
    """Test ModelRegistry initialization."""

    def test_registry_initializes_successfully(self, mock_settings):
        """Test registry initializes with mock settings."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider_class.return_value = Mock(provider_id="test-provider")
            registry = ModelRegistry()

            assert len(registry.providers) == 1
            assert "test-provider" in registry.providers
            assert len(registry.circuit_breakers) == 1
            assert len(registry.model_cache) == 1

    def test_registry_handles_provider_init_failure(self, mock_settings):
        """Test registry handles provider initialization failures gracefully."""
        with patch(
            "sekha_llm_bridge.registry.LiteLlmProvider", side_effect=Exception("Failed")
        ):
            registry = ModelRegistry()
            # Should initialize but with no providers
            assert len(registry.providers) == 0

    def test_provider_creation(self, mock_settings):
        """Test _create_provider method."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider_class.return_value = Mock()
            registry = ModelRegistry()

            provider_config = mock_settings.providers[0]
            provider = registry._create_provider(provider_config)

            assert provider is not None
            mock_provider_class.assert_called_once()

    def test_cache_ttl_initialization(self):
        """Test cache TTL is properly initialized."""
        with patch("sekha_llm_bridge.registry.settings") as mock:
            mock.providers = []
            registry = ModelRegistry()

            assert registry._cache_ttl == timedelta(minutes=5)
            assert registry._last_cache_refresh is None


class TestRoutingWithFallback:
    """Test route_with_fallback method."""

    @pytest.mark.asyncio
    async def test_route_with_preferred_model(self, mock_settings):
        """Test routing with preferred model."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()

            result = await registry.route_with_fallback(
                task=ModelTask.CHAT_SMALL, preferred_model="llama3.1:8b"
            )

            assert result.model_id == "llama3.1:8b"
            assert result.reason == "Matched preferred model"
            assert result.provider == mock_provider

    @pytest.mark.asyncio
    async def test_route_no_suitable_provider(self):
        """Test routing when no suitable provider is available."""
        with patch("sekha_llm_bridge.registry.settings") as mock:
            mock.providers = []
            registry = ModelRegistry()

            with pytest.raises(RuntimeError, match="No providers available"):
                await registry.route_with_fallback(task=ModelTask.EMBEDDING)

    @pytest.mark.asyncio
    async def test_route_with_circuit_breaker_open(self, mock_settings):
        """Test routing skips providers with open circuit breakers."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()

            # Open the circuit breaker
            cb = registry.circuit_breakers["test-provider"]
            cb._state = "open"

            with pytest.raises(RuntimeError, match="No suitable providers"):
                await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)

    @pytest.mark.asyncio
    async def test_route_with_max_cost_constraint(self, mock_settings):
        """Test routing respects max_cost constraint."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()

            # Set max_cost very low to exclude all models
            with pytest.raises(RuntimeError, match="No suitable providers"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMALL, max_cost=0.00001
                )

    @pytest.mark.asyncio
    async def test_route_with_vision_requirement(self, mock_settings):
        """Test routing with vision requirement."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()

            # Model doesn't support vision, should fail
            with pytest.raises(RuntimeError, match="No providers available"):
                await registry.route_with_fallback(
                    task=ModelTask.CHAT_SMALL, require_vision=True
                )

    @pytest.mark.asyncio
    async def test_route_by_priority(self, mock_settings):
        """Test routing selects by provider priority."""
        # Create two providers with different priorities
        provider1 = Mock()
        provider1.id = "provider1"
        provider1.provider_type.value = "litellm"
        provider1.priority = 2
        provider1.models = [
            Mock(
                model_id="model1",
                task=Mock(value="chat_small"),
                context_window=4096,
                dimension=None,
                supports_vision=False,
                supports_audio=False,
            )
        ]

        provider2 = Mock()
        provider2.id = "provider2"
        provider2.provider_type.value = "litellm"
        provider2.priority = 1  # Higher priority (lower number)
        provider2.models = [
            Mock(
                model_id="model2",
                task=Mock(value="chat_small"),
                context_window=4096,
                dimension=None,
                supports_vision=False,
                supports_audio=False,
            )
        ]

        with patch("sekha_llm_bridge.registry.settings") as mock:
            provider1.base_url = "http://test1"
            provider1.api_key = None
            provider1.timeout_secs = 30
            provider2.base_url = "http://test2"
            provider2.api_key = None
            provider2.timeout_secs = 30

            mock.providers = [provider1, provider2]
            mock.routing = Mock()
            mock.routing.circuit_breaker = Mock(
                failure_threshold=5, timeout_secs=60, success_threshold=2
            )

            with patch(
                "sekha_llm_bridge.registry.LiteLlmProvider"
            ) as mock_provider_class:
                mock_provider_class.side_effect = lambda id, _: Mock(provider_id=id)

                registry = ModelRegistry()
                result = await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)

                # Should select model2 from provider2 (priority 1)
                assert result.model_id == "model2"


class TestGetCandidates:
    """Test _get_candidates method."""

    def test_get_candidates_for_task(self, mock_settings):
        """Test getting candidates for a specific task."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            candidates = registry._get_candidates(task=ModelTask.CHAT_SMALL)

            assert len(candidates) > 0
            assert all(len(c) == 3 for c in candidates)  # (provider, model, priority)

    def test_get_candidates_with_preferred_model(self, mock_settings):
        """Test candidates prioritize preferred model."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            candidates = registry._get_candidates(
                task=ModelTask.CHAT_SMALL, preferred_model="llama3.1:8b"
            )

            # First candidate should be the preferred model with priority 0
            assert candidates[0][1] == "llama3.1:8b"
            assert candidates[0][2] == 0

    def test_get_candidates_filters_by_vision(self, mock_settings):
        """Test candidates filtered by vision support."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            candidates = registry._get_candidates(
                task=ModelTask.CHAT_SMALL, require_vision=True
            )

            # No candidates should match (mock model doesn't support vision)
            assert len(candidates) == 0

    def test_get_candidates_sorting(self, mock_settings):
        """Test candidates are sorted by priority."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            candidates = registry._get_candidates(task=ModelTask.CHAT_SMALL)

            # Check priorities are in ascending order
            priorities = [c[2] for c in candidates]
            assert priorities == sorted(priorities)


class TestExecuteWithCircuitBreaker:
    """Test execute_with_circuit_breaker method."""

    @pytest.mark.asyncio
    async def test_execute_successful_operation(self, mock_settings):
        """Test successful operation execution."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            async def mock_operation():
                return "success"

            result = await registry.execute_with_circuit_breaker(
                "test-provider", mock_operation
            )

            assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_records_failure(self, mock_settings):
        """Test failed operation records failure."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            async def mock_operation():
                raise Exception("Operation failed")

            with pytest.raises(Exception, match="Operation failed"):
                await registry.execute_with_circuit_breaker(
                    "test-provider", mock_operation
                )

            # Circuit breaker should have recorded the failure
            cb = registry.circuit_breakers["test-provider"]
            assert cb.failure_count > 0

    @pytest.mark.asyncio
    async def test_execute_with_args_and_kwargs(self, mock_settings):
        """Test operation with arguments."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()

            async def mock_operation(a, b, c=None):
                return f"{a}-{b}-{c}"

            result = await registry.execute_with_circuit_breaker(
                "test-provider", mock_operation, "x", "y", c="z"
            )

            assert result == "x-y-z"


class TestProviderHealth:
    """Test get_provider_health method."""

    def test_get_provider_health(self, mock_settings):
        """Test getting provider health status."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider.provider_type = "litellm"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()
            health = registry.get_provider_health()

            assert "test-provider" in health
            assert "provider_type" in health["test-provider"]
            assert "circuit_breaker" in health["test-provider"]
            assert "models_count" in health["test-provider"]

    def test_health_includes_circuit_breaker_stats(self, mock_settings):
        """Test health includes circuit breaker statistics."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.provider_id = "test-provider"
            mock_provider.provider_type = "litellm"
            mock_provider_class.return_value = mock_provider

            registry = ModelRegistry()
            health = registry.get_provider_health()

            cb_stats = health["test-provider"]["circuit_breaker"]
            assert "state" in cb_stats
            assert "failure_count" in cb_stats

    def test_health_with_multiple_providers(self):
        """Test health status with multiple providers."""
        # Mock multiple providers
        provider1 = Mock(
            id="provider1",
            provider_type=Mock(value="litellm"),
            priority=1,
            models=[],
        )
        provider2 = Mock(
            id="provider2",
            provider_type=Mock(value="litellm"),
            priority=2,
            models=[],
        )

        with patch("sekha_llm_bridge.registry.settings") as mock:
            mock.providers = [provider1, provider2]
            mock.routing = Mock(
                circuit_breaker=Mock(
                    failure_threshold=5, timeout_secs=60, success_threshold=2
                )
            )

            with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
                registry = ModelRegistry()
                health = registry.get_provider_health()

                assert len(health) == 2
                assert "provider1" in health
                assert "provider2" in health


class TestListAllModels:
    """Test list_all_models method."""

    def test_list_all_models(self, mock_settings):
        """Test listing all available models."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()
            models = registry.list_all_models()

            assert len(models) > 0
            assert all(isinstance(m, dict) for m in models)

    def test_list_models_includes_required_fields(self, mock_settings):
        """Test listed models include all required fields."""
        with patch("sekha_llm_bridge.registry.LiteLlmProvider"):
            registry = ModelRegistry()
            models = registry.list_all_models()

            required_fields = [
                "model_id",
                "provider_id",
                "task",
                "context_window",
                "dimension",
                "supports_vision",
                "supports_audio",
            ]

            for model in models:
                for field in required_fields:
                    assert field in model

    def test_list_models_empty_registry(self):
        """Test listing models from empty registry."""
        with patch("sekha_llm_bridge.registry.settings") as mock:
            mock.providers = []
            registry = ModelRegistry()
            models = registry.list_all_models()

            assert models == []
