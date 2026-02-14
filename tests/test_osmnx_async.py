#!/usr/bin/env python
# ruff: noqa: PLR2004, S101
"""Test suite for the osmnx.aio async subpackage."""

from __future__ import annotations

import time
from collections import OrderedDict
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest

import osmnx as ox
from osmnx import aio
from osmnx.aio import _nominatim
from osmnx.aio import _overpass
from osmnx.aio import geocoder as aio_geocoder
from osmnx.aio._http import _async_retrieve_from_cache
from osmnx.aio._http import _async_save_to_cache
from osmnx.aio._http import _build_request_kwargs
from osmnx.aio._http import _get_http_headers
from osmnx.aio._http import _parse_response
from osmnx.aio._http import _resolve_url_to_ip
from osmnx.aio._rate_limit import AsyncRateLimiter

# configure osmnx settings for tests
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = ".temp/cache"


# ---------------------------------------------------------------------------
# Settings infrastructure
# ---------------------------------------------------------------------------


class TestContextVarSettings:
    """Test the contextvars-based settings override for async safety."""

    def test_get_returns_module_default(self) -> None:
        """_get returns the module-level default when no override is set."""
        assert ox.settings._get("use_cache") is True
        assert ox.settings._get("overpass_url") == "https://overpass-api.de/api"

    def test_get_returns_override(self) -> None:
        """_get returns the override value when one is set."""
        token = ox.settings._settings_overrides.set({"use_cache": False})
        try:
            assert ox.settings._get("use_cache") is False
            # non-overridden setting still returns module default
            assert ox.settings._get("overpass_url") == "https://overpass-api.de/api"
        finally:
            ox.settings._settings_overrides.reset(token)

    def test_override_isolation(self) -> None:
        """Overrides are isolated per-context (simulated with set/reset)."""
        assert ox.settings._get("use_cache") is True
        token = ox.settings._settings_overrides.set({"use_cache": False})
        assert ox.settings._get("use_cache") is False
        ox.settings._settings_overrides.reset(token)
        assert ox.settings._get("use_cache") is True


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestAsyncRateLimiter:
    """Test the AsyncRateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_interval(self) -> None:
        """Rate limiter enforces minimum interval between calls."""
        limiter = AsyncRateLimiter()
        start = time.monotonic()

        await limiter.wait("test-host", min_interval=0.1)
        await limiter.wait("test-host", min_interval=0.1)

        elapsed = time.monotonic() - start
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_different_hosts_independent(self) -> None:
        """Different hostnames are rate-limited independently."""
        limiter = AsyncRateLimiter()
        start = time.monotonic()

        await limiter.wait("host-a", min_interval=0.5)
        await limiter.wait("host-b", min_interval=0.5)

        elapsed = time.monotonic() - start
        # both should complete nearly immediately since they're independent
        assert elapsed < 0.4


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class TestAsyncHttp:
    """Test async HTTP helper functions."""

    def test_get_http_headers(self) -> None:
        """Headers are built correctly from settings."""
        headers = _get_http_headers()
        assert "User-Agent" in headers
        assert "referer" in headers
        assert "Accept-Language" in headers

    def test_get_http_headers_custom(self) -> None:
        """Custom header values override settings."""
        headers = _get_http_headers(
            user_agent="test-agent",
            referer="test-referer",
            accept_language="fr",
        )
        assert headers["User-Agent"] == "test-agent"
        assert headers["referer"] == "test-referer"
        assert headers["Accept-Language"] == "fr"

    def test_build_request_kwargs_empty(self) -> None:
        """Empty requests_kwargs produces empty dicts."""
        client_kw, request_kw = _build_request_kwargs()
        assert client_kw == {}
        assert request_kw == {}

    def test_build_request_kwargs_proxies_deprecation(self) -> None:
        """Proxies dict triggers deprecation warning."""
        original = ox.settings.requests_kwargs
        try:
            ox.settings.requests_kwargs = {"proxies": {"https": "http://proxy:8080"}}
            with pytest.warns(DeprecationWarning, match="proxies"):
                client_kw, request_kw = _build_request_kwargs()
            assert client_kw["proxy"] == "http://proxy:8080"
            assert "proxies" not in request_kw
        finally:
            ox.settings.requests_kwargs = original

    def test_build_request_kwargs_client_vs_request(self) -> None:
        """Client-level kwargs are separated from per-request kwargs."""
        original = ox.settings.requests_kwargs
        try:
            ox.settings.requests_kwargs = {"verify": False, "custom_param": "value"}
            client_kw, request_kw = _build_request_kwargs()
            assert client_kw == {"verify": False}
            assert request_kw == {"custom_param": "value"}
        finally:
            ox.settings.requests_kwargs = original

    @pytest.mark.asyncio
    async def test_resolve_url_to_ip(self) -> None:
        """Resolve returns original URL unchanged (no rewrite for TLS safety)."""
        url = "https://overpass-api.de/api/interpreter"
        resolved_url, headers = await _resolve_url_to_ip(url)

        # URL must be returned unchanged so TLS cert verification succeeds
        assert resolved_url == url
        assert headers == {}

    @pytest.mark.asyncio
    async def test_parse_response_success(self) -> None:
        """_parse_response correctly parses a successful JSON response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.url = "https://example.com/api"
        mock_response.content = b'{"key": "value"}'
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"key": "value"}

        result = _parse_response(mock_response)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestAsyncCache:
    """Test async cache wrappers."""

    @pytest.mark.asyncio
    async def test_cache_round_trip(self) -> None:
        """Save and retrieve from cache produces consistent results."""
        test_url = "https://test.example.com/async-cache-test-unique-url-12345"
        test_data = {"test": "async_cache_data"}

        # save to cache
        await _async_save_to_cache(test_url, test_data, ok=True)

        # retrieve from cache
        result = await _async_retrieve_from_cache(test_url)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self) -> None:
        """Cache miss returns None."""
        result = await _async_retrieve_from_cache(
            "https://test.example.com/nonexistent-url-67890",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Test that the aio subpackage is importable."""

    def test_aio_importable(self) -> None:
        """osmnx.aio can be imported when httpx is installed."""
        assert hasattr(aio, "graph")
        assert hasattr(aio, "features")
        assert hasattr(aio, "geocoder")
        assert hasattr(aio, "elevation")


# ---------------------------------------------------------------------------
# Nominatim async (mocked)
# ---------------------------------------------------------------------------


class TestAsyncNominatim:
    """Test async Nominatim request logic with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_nominatim_request_uses_cache(self) -> None:
        """Nominatim request returns cached data when available."""
        cached_data = [{"place_id": 1, "lat": "37.0", "lon": "-122.0"}]

        with patch(
            "osmnx.aio._http._async_retrieve_from_cache",
            new_callable=AsyncMock,
            return_value=cached_data,
        ):
            params: OrderedDict[str, int | str] = OrderedDict()
            params["format"] = "json"
            params["q"] = "test query"
            result = await _nominatim._nominatim_request(params=params)

        assert result == cached_data


# ---------------------------------------------------------------------------
# Overpass async (mocked)
# ---------------------------------------------------------------------------


class TestAsyncOverpass:
    """Test async Overpass request logic with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_overpass_request_uses_cache(self) -> None:
        """Overpass request returns cached data when available."""
        cached_data = {"elements": []}

        with (
            patch(
                "osmnx.aio._http._async_retrieve_from_cache",
                new_callable=AsyncMock,
                return_value=cached_data,
            ),
            patch(
                "osmnx.aio._http._resolve_url_to_ip",
                new_callable=AsyncMock,
                return_value=("https://overpass-api.de/api/interpreter", {}),
            ),
        ):
            result = await _overpass._overpass_request(OrderedDict(data="test"))

        assert result == cached_data


# ---------------------------------------------------------------------------
# Geocoder async (mocked)
# ---------------------------------------------------------------------------


class TestAsyncGeocoder:
    """Test async geocoder with mocked Nominatim."""

    @pytest.mark.asyncio
    async def test_geocode(self) -> None:
        """Async geocode returns (lat, lon) tuple."""
        mock_response = [{"lat": "37.7952", "lon": "-122.4028"}]

        with patch(
            "osmnx.aio._nominatim._nominatim_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await aio_geocoder.geocode("test address")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert abs(result[0] - 37.7952) < 0.001
        assert abs(result[1] - (-122.4028)) < 0.001

    @pytest.mark.asyncio
    async def test_geocode_no_results_raises(self) -> None:
        """Async geocode raises InsufficientResponseError on empty results."""
        with (
            patch(
                "osmnx.aio._nominatim._nominatim_request",
                new_callable=AsyncMock,
                return_value=[],
            ),
            pytest.raises(ox._errors.InsufficientResponseError),
        ):
            await aio_geocoder.geocode("nonexistent place xyz123")
