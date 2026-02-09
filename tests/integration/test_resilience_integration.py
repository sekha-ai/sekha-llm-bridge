"""Integration tests for resilience patterns with controlled failure injection.

These tests use mocking to inject specific failure scenarios that would be
difficult or impossible to test reliably in E2E tests.

Tests validate:
- Provider failure and automatic fallback
- Circuit breaker state transitions
- Circuit breaker recovery
- Multi-provider fallback cascades
- Graceful degradation
- Cost-based fallback under failure

For true E2E resilience tests, see tests/e2e/test_resilience.py
"""

from unittest.mock import MagicMock, patch

import pytest

from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.registry import registry
from sekha_llm_bridge.resilience import CircuitBreaker, CircuitState


class TestProviderFailureAndFallback:
    """Test provider failure scenarios with automatic fallback."""

    @pytest.mark.asyncio
    async def test_fallback_when_primary_provider_fails(self):
        """Test automatic fallback when primary provider fails."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Primary provider available, secondary as backup
            mock_candidates.return_value = [
                ("provider1", "model-a", 1),  # Will fail (circuit breaker open)
                ("provider2", "model-b", 2),  # Backup
            ]

            provider1 = MagicMock()
            provider1.provider_id = "provider1"

            provider2 = MagicMock()
            provider2.provider_id = "provider2"

            with patch.object(
                registry, "providers", {"provider1": provider1, "provider2": provider2}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    # Provider1 circuit breaker is OPEN (failed)
                    def cb_side_effect(pid):
                        if pid == "provider1":
                            cb = MagicMock(spec=CircuitBreaker)
                            cb.is_open.return_value = True  # Failed
                            return cb
                        cb = MagicMock(spec=CircuitBreaker)
                        cb.is_open.return_value = False  # Healthy
                        return cb

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
                        assert "priority" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_fallback_cascade_through_multiple_providers(self):
        """Test fallback cascades through multiple failed providers."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Three providers, first two fail
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
                        # First two providers failed
                        if pid in ["provider1", "provider2"]:
                            cb = MagicMock(spec=CircuitBreaker)
                            cb.is_open.return_value = True
                            return cb
                        # Third provider healthy
                        cb = MagicMock(spec=CircuitBreaker)
                        cb.is_open.return_value = False
                        return cb

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
    async def test_all_providers_failed(self):
        """Test error when all providers have failed."""
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
                    cb = MagicMock(spec=CircuitBreaker)
                    cb.is_open.return_value = True
                    mock_cbs.get.return_value = cb

                    with pytest.raises(
                        RuntimeError, match="No suitable providers available"
                    ):
                        await registry.route_with_fallback(task=ModelTask.CHAT_SMALL)


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions."""

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, timeout_secs=60)

        # Initial state: closed
        assert cb.get_state() == CircuitState.CLOSED
        assert not cb.is_open()

        # Record failures
        cb.record_failure(Exception("Error 1"))
        assert cb.get_state() == CircuitState.CLOSED  # Still closed

        cb.record_failure(Exception("Error 2"))
        assert cb.get_state() == CircuitState.CLOSED  # Still closed

        cb.record_failure(Exception("Error 3"))
        assert cb.get_state() == CircuitState.OPEN  # Now open
        assert cb.is_open()

    def test_circuit_breaker_transitions_to_half_open_after_timeout(self):
        """Test circuit breaker moves to half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout_secs=0)  # Instant timeout

        # Force circuit breaker to open
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.get_state() == CircuitState.OPEN

        # After timeout, should transition to half-open
        assert not cb.is_open()  # is_open() triggers transition
        assert cb.get_state() == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_half_open_successes(self):
        """Test circuit breaker closes after successes in half-open."""
        cb = CircuitBreaker(
            failure_threshold=2, timeout_secs=0, success_threshold=2
        )

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.get_state() == CircuitState.OPEN

        # Transition to half-open
        cb.is_open()  # Triggers transition after timeout
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.get_state() == CircuitState.HALF_OPEN  # Still half-open

        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED  # Now closed
        assert not cb.is_open()

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Test circuit breaker reopens immediately on half-open failure."""
        cb = CircuitBreaker(failure_threshold=2, timeout_secs=0)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.get_state() == CircuitState.OPEN

        # Transition to half-open
        cb.is_open()
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Failure in half-open immediately reopens
        cb.record_failure(Exception("Error 3"))
        assert cb.get_state() == CircuitState.OPEN

    def test_circuit_breaker_resets_failure_count_on_success(self):
        """Test failure count resets on success in closed state."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record some failures
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.get_state() == CircuitState.CLOSED
        assert cb.failure_count == 2

        # Success resets count
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.get_state() == CircuitState.CLOSED

        # Should take 3 more failures to open
        cb.record_failure(Exception("Error 3"))
        cb.record_failure(Exception("Error 4"))
        assert cb.get_state() == CircuitState.CLOSED

        cb.record_failure(Exception("Error 5"))
        assert cb.get_state() == CircuitState.OPEN


class TestCircuitBreakerIntegrationWithRouting:
    """Test circuit breaker integration with routing logic."""

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_success(self):
        """Test successful operation records success in circuit breaker."""
        provider_id = "test_provider"

        # Create a real circuit breaker
        cb = CircuitBreaker(failure_threshold=3)
        registry.circuit_breakers[provider_id] = cb

        async def successful_operation():
            return "success"

        try:
            result = await registry.execute_with_circuit_breaker(
                provider_id, successful_operation
            )

            assert result == "success"
            assert cb.failure_count == 0
            assert cb.get_state() == CircuitState.CLOSED

        finally:
            # Cleanup
            if provider_id in registry.circuit_breakers:
                del registry.circuit_breakers[provider_id]

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_failure(self):
        """Test failed operation records failure in circuit breaker."""
        provider_id = "test_provider"

        # Create a real circuit breaker
        cb = CircuitBreaker(failure_threshold=3)
        registry.circuit_breakers[provider_id] = cb

        async def failing_operation():
            raise Exception("Operation failed")

        try:
            with pytest.raises(Exception, match="Operation failed"):
                await registry.execute_with_circuit_breaker(
                    provider_id, failing_operation
                )

            # Failure should be recorded
            assert cb.failure_count == 1
            assert cb.get_state() == CircuitState.CLOSED  # Not open yet

        finally:
            # Cleanup
            if provider_id in registry.circuit_breakers:
                del registry.circuit_breakers[provider_id]

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_repeated_failures(self):
        """Test circuit breaker opens after threshold failures."""
        provider_id = "test_provider"

        # Create a real circuit breaker
        cb = CircuitBreaker(failure_threshold=2)  # Low threshold for testing
        registry.circuit_breakers[provider_id] = cb

        async def failing_operation():
            raise Exception("Operation failed")

        try:
            # First failure
            with pytest.raises(Exception):
                await registry.execute_with_circuit_breaker(
                    provider_id, failing_operation
                )
            assert cb.get_state() == CircuitState.CLOSED

            # Second failure - should open circuit
            with pytest.raises(Exception):
                await registry.execute_with_circuit_breaker(
                    provider_id, failing_operation
                )
            assert cb.get_state() == CircuitState.OPEN
            assert cb.is_open()

        finally:
            # Cleanup
            if provider_id in registry.circuit_breakers:
                del registry.circuit_breakers[provider_id]


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""

    @pytest.mark.asyncio
    async def test_cost_based_fallback_when_primary_exceeds_budget(self):
        """Test fallback to cheaper provider when primary exceeds cost limit."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4", 1),  # Expensive
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
                    cb = MagicMock(spec=CircuitBreaker)
                    cb.is_open.return_value = False
                    mock_cbs.get.return_value = cb

                    def cost_side_effect(model, **kwargs):
                        if "gpt" in model:
                            return 0.05  # Expensive
                        return 0.0  # Free

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost",
                        side_effect=cost_side_effect,
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            max_cost=0.01,  # Tight budget
                        )

                        # Should fallback to free provider
                        assert result.provider.provider_id == "ollama"
                        assert result.estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_no_providers_within_cost_limit(self):
        """Test error when no providers meet cost constraint."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            mock_candidates.return_value = [
                ("openai", "gpt-4", 1),
                ("anthropic", "claude-3", 2),
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry, "providers", {"openai": openai, "anthropic": anthropic}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    cb = MagicMock(spec=CircuitBreaker)
                    cb.is_open.return_value = False
                    mock_cbs.get.return_value = cb

                    # All models expensive
                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.05
                    ):
                        with pytest.raises(
                            RuntimeError, match="No suitable providers available"
                        ):
                            await registry.route_with_fallback(
                                task=ModelTask.CHAT_SMART,
                                max_cost=0.001,  # Impossible budget
                            )

    @pytest.mark.asyncio
    async def test_vision_requirement_with_fallback(self):
        """Test fallback when vision required but primary doesn't support."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Only vision-capable model returned when vision required
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 1),  # Vision capable
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            with patch.object(registry, "providers", {"openai": openai}):
                with patch.object(registry, "circuit_breakers") as mock_cbs:
                    cb = MagicMock(spec=CircuitBreaker)
                    cb.is_open.return_value = False
                    mock_cbs.get.return_value = cb

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                        )

                        # Should route to vision-capable model
                        assert result.provider.provider_id == "openai"
                        assert result.model_id == "gpt-4o"

                        # Verify _get_candidates was called with require_vision=True
                        mock_candidates.assert_called_once_with(
                            task=ModelTask.CHAT_SMART,
                            require_vision=True,
                            preferred_model=None,
                        )


class TestPreferredModelWithFallback:
    """Test preferred model routing with fallback scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_when_preferred_model_provider_fails(self):
        """Test fallback when preferred model's provider fails."""
        with patch.object(registry, "_get_candidates") as mock_candidates:
            # Preferred model first, then fallback
            mock_candidates.return_value = [
                ("openai", "gpt-4o", 0),  # Preferred, will fail
                ("anthropic", "claude-3", 1),  # Fallback
            ]

            openai = MagicMock()
            openai.provider_id = "openai"

            anthropic = MagicMock()
            anthropic.provider_id = "anthropic"

            with patch.object(
                registry, "providers", {"openai": openai, "anthropic": anthropic}
            ):
                with patch.object(registry, "circuit_breakers") as mock_cbs:

                    def cb_side_effect(pid):
                        if pid == "openai":
                            cb = MagicMock(spec=CircuitBreaker)
                            cb.is_open.return_value = True  # Failed
                            return cb
                        cb = MagicMock(spec=CircuitBreaker)
                        cb.is_open.return_value = False  # Healthy
                        return cb

                    mock_cbs.get.side_effect = cb_side_effect

                    with patch(
                        "sekha_llm_bridge.registry.estimate_cost", return_value=0.01
                    ):
                        result = await registry.route_with_fallback(
                            task=ModelTask.CHAT_SMART,
                            preferred_model="gpt-4o",
                        )

                        # Should fallback despite preference
                        assert result.provider.provider_id == "anthropic"
                        assert result.model_id == "claude-3"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
