"""Coordinated async rate limiter for external API requests."""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Per-hostname async rate limiter using asyncio.Lock + timestamp tracking.

    Ensures that concurrent async tasks respect per-hostname rate limits
    (e.g. Nominatim's 1-request-per-second policy).
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request_time: dict[str, float] = {}

    async def wait(self, hostname: str, min_interval: float) -> None:
        """Wait until at least ``min_interval`` seconds since last request.

        Parameters
        ----------
        hostname
            The hostname to rate-limit against.
        min_interval
            Minimum seconds between requests to this hostname.
        """
        if hostname not in self._locks:
            self._locks[hostname] = asyncio.Lock()

        async with self._locks[hostname]:
            elapsed = time.monotonic() - self._last_request_time.get(hostname, 0.0)
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_time[hostname] = time.monotonic()


# module-level singleton shared across all async operations
_rate_limiter = AsyncRateLimiter()
