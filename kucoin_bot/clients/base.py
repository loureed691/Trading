"""Base client with retry logic and common functionality."""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base


def with_retry(retry_config: RetryConfig | None = None) -> Callable:
    """Decorator for adding retry logic to async functions."""
    config = retry_config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for "
                            f"{func.__name__}: {e}"
                        )
                        raise

                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )
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
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 10.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()


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
        self._session = None

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
