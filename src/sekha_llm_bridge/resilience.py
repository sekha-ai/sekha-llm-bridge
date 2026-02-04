"""Resilience patterns for LLM provider failures."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for graceful provider failure handling.
    
    Prevents cascading failures by:
    - Opening circuit after N consecutive failures
    - Keeping circuit open for timeout period
    - Attempting recovery via half-open state
    
    Args:
        failure_threshold: Number of failures before opening circuit (default: 3)
        timeout_secs: Seconds to wait before attempting recovery (default: 60)
        success_threshold: Successes needed in half-open to close (default: 2)
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        timeout_secs: int = 60,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_secs)
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
    
    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failure and potentially open the circuit.
        
        Args:
            error: Optional exception that caused the failure
        """
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately reopens circuit
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit breaker reopened after failure in half-open state: {error}"
            )
        elif self.state == CircuitState.CLOSED:
            # Check if we've hit threshold
            if self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
    
    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                logger.info(
                    f"Circuit breaker closed after {self.success_count} successes"
                )
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests).
        
        Returns:
            True if circuit is open and should reject requests
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._should_attempt_recovery():
                self._transition_to(CircuitState.HALF_OPEN)
                return False
            return True
        
        return False
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self.failure_count = 0
        self.success_count = 0
        self._transition_to(CircuitState.CLOSED)
        logger.info("Circuit breaker manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        # Update to half-open if timeout passed
        if self.state == CircuitState.OPEN and self._should_attempt_recovery():
            self._transition_to(CircuitState.HALF_OPEN)
        
        return self.state
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics.
        
        Returns:
            Dictionary with state, failure count, and timing info
        """
        return {
            "state": self.get_state().value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": (
                self.last_failure_time.isoformat() 
                if self.last_failure_time 
                else None
            ),
            "last_state_change": self.last_state_change.isoformat(),
            "time_since_failure": (
                str(datetime.now() - self.last_failure_time)
                if self.last_failure_time
                else None
            ),
        }
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return False
        
        return datetime.now() - self.last_failure_time >= self.timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
        
        logger.info(f"Circuit breaker: {old_state.value} -> {new_state.value}")
