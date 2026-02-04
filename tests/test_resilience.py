"""Unit tests for circuit breaker resilience patterns."""

import pytest
import asyncio
from datetime import datetime, timedelta
from sekha_llm_bridge.resilience import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state_is_closed(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker()
        assert cb.get_state() == CircuitState.CLOSED
        assert not cb.is_open()
    
    def test_opens_after_failure_threshold(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED  # Still closed
        
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED  # Still closed
        
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN  # Now open
        assert cb.is_open()
    
    def test_rejects_requests_when_open(self):
        """Open circuit should reject requests."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        
        assert cb.is_open()
        assert cb.get_state() == CircuitState.OPEN
    
    def test_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=1, timeout_secs=1)
        
        # Open the circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        
        # Manually set last_failure_time to past
        cb.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        # Should now be half-open
        assert cb.get_state() == CircuitState.HALF_OPEN
        assert not cb.is_open()
    
    def test_closes_after_success_in_half_open(self):
        """Circuit should close after successes in half-open state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            timeout_secs=1,
            success_threshold=2
        )
        
        # Open the circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        
        # Transition to half-open
        cb.last_failure_time = datetime.now() - timedelta(seconds=2)
        assert cb.get_state() == CircuitState.HALF_OPEN
        
        # Record successes
        cb.record_success()
        assert cb.get_state() == CircuitState.HALF_OPEN  # Still half-open
        
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED  # Now closed
    
    def test_reopens_on_failure_in_half_open(self):
        """Circuit should reopen immediately on failure in half-open."""
        cb = CircuitBreaker(failure_threshold=1, timeout_secs=1)
        
        # Open the circuit
        cb.record_failure()
        
        # Transition to half-open
        cb.last_failure_time = datetime.now() - timedelta(seconds=2)
        cb.get_state()  # Updates to half-open
        
        # Failure in half-open should reopen
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
    
    def test_resets_failure_count_on_success_when_closed(self):
        """Success in closed state should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.get_state() == CircuitState.CLOSED
    
    def test_manual_reset(self):
        """Manual reset should close circuit and clear counts."""
        cb = CircuitBreaker(failure_threshold=1)
        
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        assert cb.failure_count > 0
        
        cb.reset()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    def test_get_stats_returns_correct_info(self):
        """Stats should include state, counts, and timing info."""
        cb = CircuitBreaker()
        
        stats = cb.get_stats()
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "last_failure" in stats
        assert "last_state_change" in stats
        
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
    
    def test_records_exception_in_failure(self):
        """Circuit breaker should accept exception in record_failure."""
        cb = CircuitBreaker()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            cb.record_failure(e)
            assert cb.failure_count == 1
    
    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        cb = CircuitBreaker(
            failure_threshold=5,
            timeout_secs=120,
            success_threshold=3
        )
        
        # Should take 5 failures to open
        for i in range(4):
            cb.record_failure()
            assert cb.get_state() == CircuitState.CLOSED
        
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
    
    def test_concurrent_failures(self):
        """Circuit breaker should handle rapid failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Rapid failures
        for _ in range(5):
            cb.record_failure()
        
        assert cb.get_state() == CircuitState.OPEN
        assert cb.failure_count >= 3


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with async operations."""
    
    @pytest.mark.asyncio
    async def test_protects_failing_service(self):
        """Circuit breaker should protect against failing service."""
        cb = CircuitBreaker(failure_threshold=2, timeout_secs=1)
        call_count = 0
        
        async def failing_service():
            nonlocal call_count
            call_count += 1
            raise Exception("Service unavailable")
        
        # First two calls fail and open circuit
        for _ in range(2):
            try:
                if not cb.is_open():
                    await failing_service()
                    cb.record_success()
            except Exception as e:
                cb.record_failure(e)
        
        assert cb.get_state() == CircuitState.OPEN
        
        # Next calls should be rejected without calling service
        for _ in range(3):
            if cb.is_open():
                # Don't call service
                pass
        
        # Service should only have been called twice (not 5 times)
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self):
        """Circuit should recover after timeout if service is healthy."""
        cb = CircuitBreaker(failure_threshold=1, timeout_secs=1, success_threshold=1)
        
        # Fail once to open circuit
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Should transition to half-open
        assert cb.get_state() == CircuitState.HALF_OPEN
        
        # Success should close it
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
