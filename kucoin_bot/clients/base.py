"""Base client with retry logic and common functionality."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Known rate limit error codes
RATE_LIMIT_CODES = {"429001", "429000", "100013"}
TIMEOUT_ERROR_CODES = {"timeout", "408"}
RETRYABLE_ERROR_CODES = RATE_LIMIT_CODES | TIMEOUT_ERROR_CODES | {"500", "502", "503", "504"}


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    rate_limit_delay: float = 5.0  # Specific delay for rate limits
    timeout_delay: float = 2.0  # Specific delay for timeouts
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class APIMetrics:
    """Metrics for API calls."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    rate_limited_calls: int = 0
    timeout_calls: int = 0
    last_call_time: float = 0.0
    avg_response_time: float = 0.0
    _response_times: list[float] = field(default_factory=list)

    def record_call(self, success: bool, response_time: float, retried: bool = False,
                    rate_limited: bool = False, timeout: bool = False) -> None:
        """Record a call metric."""
        self.total_calls += 1
        self.last_call_time = time.time()
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        if retried:
            self.retried_calls += 1
        if rate_limited:
            self.rate_limited_calls += 1
        if timeout:
            self.timeout_calls += 1
        
        # Track response time (keep last 100)
        self._response_times.append(response_time)
        if len(self._response_times) > 100:
            self._response_times.pop(0)
        self.avg_response_time = sum(self._response_times) / len(self._response_times)

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary for export."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "retried_calls": self.retried_calls,
            "rate_limited_calls": self.rate_limited_calls,
            "timeout_calls": self.timeout_calls,
            "avg_response_time_ms": self.avg_response_time * 1000,
            "last_call_time": self.last_call_time,
        }


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    error_str = str(error).lower()
    return any(code.lower() in error_str for code in RATE_LIMIT_CODES) or "rate limit" in error_str


def is_timeout_error(error: Exception) -> bool:
    """Check if error is a timeout error."""
    return isinstance(error, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)) or "timeout" in str(error).lower()


def is_retryable_error(error: Exception) -> bool:
    """Check if error should be retried."""
    error_str = str(error).lower()
    if is_rate_limit_error(error) or is_timeout_error(error):
        return True
    return any(code.lower() in error_str for code in RETRYABLE_ERROR_CODES)


def with_retry(retry_config: RetryConfig | None = None, metrics: APIMetrics | None = None) -> Callable:
    """Decorator for adding retry logic to async functions with enhanced error handling."""
    config = retry_config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            start_time = time.time()
            retried = False
            rate_limited = False
            timeout = False

            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    # Record success
                    if metrics:
                        metrics.record_call(True, time.time() - start_time, retried, rate_limited, timeout)
                    return result
                except Exception as e:
                    last_exception = e
                    
                    # Categorize error
                    if is_rate_limit_error(e):
                        rate_limited = True
                        delay = config.rate_limit_delay
                        logger.warning(f"Rate limit hit for {func.__name__}, waiting {delay}s")
                    elif is_timeout_error(e):
                        timeout = True
                        delay = config.timeout_delay
                        logger.warning(f"Timeout for {func.__name__}, waiting {delay}s")
                    elif is_retryable_error(e):
                        delay = min(
                            config.base_delay * (config.exponential_base ** (attempt - 1)),
                            config.max_delay,
                        )
                    else:
                        # Non-retryable error
                        if metrics:
                            metrics.record_call(False, time.time() - start_time, retried, rate_limited, timeout)
                        raise
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for "
                            f"{func.__name__}: {e}"
                        )
                        if metrics:
                            metrics.record_call(False, time.time() - start_time, retried, rate_limited, timeout)
                        raise
                    
                    # Add jitter if enabled
                    if config.jitter:
                        delay = delay * (0.5 + random.random())
                    
                    retried = True
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            # Should not reach here, but type checker needs this
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


class RateLimiter:
    """Rate limiter for API calls with burst support."""

    def __init__(self, calls_per_second: float = 10.0, burst_size: int = 5):
        self.min_interval = 1.0 / calls_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed / self.min_interval
            )
            self.last_refill = now
            
            if self.tokens < 1:
                # Wait for token to become available
                wait_time = (1 - self.tokens) * self.min_interval
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1


class BaseClient:
    """Base class for API clients with common functionality."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        sandbox: bool = True,
        retry_config: RetryConfig | None = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.sandbox = sandbox
        self.retry_config = retry_config or RetryConfig()
        self.rate_limiter = RateLimiter()
        self.metrics = APIMetrics()
        self._session = None
        self._safe_mode = False

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    def enable_safe_mode(self) -> None:
        """Enable safe mode - reduces API call frequency."""
        self._safe_mode = True
        self.rate_limiter = RateLimiter(calls_per_second=2.0, burst_size=2)
        logger.warning("API client entered safe mode")

    def disable_safe_mode(self) -> None:
        """Disable safe mode - restore normal operation."""
        self._safe_mode = False
        self.rate_limiter = RateLimiter()
        logger.info("API client exited safe mode")

    def get_metrics(self) -> dict[str, Any]:
        """Get API metrics for monitoring."""
        return {
            **self.metrics.get_metrics_dict(),
            "safe_mode": self._safe_mode,
        }
