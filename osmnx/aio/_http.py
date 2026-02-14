"""Async HTTP transport layer using httpx."""

from __future__ import annotations

import asyncio
import logging as lg
import socket
import warnings
from json import JSONDecodeError
from typing import Any
from urllib.parse import urlparse

import httpx

from osmnx import _http
from osmnx import settings
from osmnx import utils
from osmnx._errors import InsufficientResponseError
from osmnx._errors import ResponseStatusCodeError

# lock protecting concurrent cache file reads/writes
_cache_lock = asyncio.Lock()


async def _async_retrieve_from_cache(
    url: str,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Retrieve a HTTP response JSON object from the cache if it exists.

    Thread-safe wrapper around the sync cache retrieval, protected by an
    ``asyncio.Lock`` to prevent concurrent file access.

    Parameters
    ----------
    url
        The URL of the request.

    Returns
    -------
    response_json
        The cached response for ``url`` if it exists, otherwise None.
    """
    async with _cache_lock:
        return await asyncio.to_thread(_http._retrieve_from_cache, url)


async def _async_save_to_cache(
    url: str,
    response_json: dict[str, Any] | list[dict[str, Any]],
    ok: bool,  # noqa: FBT001
) -> None:
    """Save a HTTP response JSON object to a file in the cache folder.

    Thread-safe wrapper around the sync cache save, protected by an
    ``asyncio.Lock`` to prevent concurrent file access.

    Parameters
    ----------
    url
        The URL of the request.
    response_json
        The JSON HTTP response.
    ok
        Whether the HTTP status was successful.
    """
    async with _cache_lock:
        await asyncio.to_thread(_http._save_to_cache, url, response_json, ok)


async def _resolve_host_via_doh(hostname: str) -> str:
    """Resolve hostname to IP address via DNS-over-HTTPS.

    Async equivalent of ``osmnx._http._resolve_host_via_doh`` using
    ``httpx.AsyncClient`` instead of ``requests``.

    Parameters
    ----------
    hostname
        The hostname to resolve.

    Returns
    -------
    ip_address
        Resolved IP address, or hostname itself if resolution failed.
    """
    doh_url_template = settings._get("doh_url_template")
    if doh_url_template is None:
        msg = "User set `doh_url_template=None`, requesting host by name"
        utils.log(msg, level=lg.WARNING)
        return hostname

    err_msg = f"Failed to resolve {hostname!r} IP via DoH, requesting host by name"
    try:
        url = doh_url_template.format(hostname=hostname)
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=settings._get("requests_timeout"))
            data = response.json()

    except httpx.HTTPError:  # pragma: no cover
        utils.log(err_msg, level=lg.ERROR)
        return hostname

    else:
        if response.is_success and data["Status"] == 0:
            ip_address: str = data["Answer"][0]["data"]
            return ip_address

        utils.log(err_msg, level=lg.ERROR)
        return hostname


async def _resolve_url_to_ip(url: str) -> tuple[str, dict[str, str]]:
    """Resolve a URL's hostname to an IP address for logging purposes.

    The sync ``_config_dns()`` mutates global ``socket.getaddrinfo`` to pin
    DNS so that the Overpass status check and subsequent query hit the same
    server.  We cannot safely rewrite the URL to contain the raw IP because
    TLS certificate verification requires the original hostname.  Instead we
    resolve the IP for informational logging and return the original URL
    unchanged, letting httpx perform standard DNS resolution.

    Parameters
    ----------
    url
        The URL whose hostname should be resolved.

    Returns
    -------
    url, extra_headers
        The original URL (unchanged) and an empty headers dict.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    try:
        ip = await asyncio.to_thread(socket.gethostbyname, hostname)
    except socket.gaierror:  # pragma: no cover
        msg = (
            f"Encountered gaierror while trying to resolve {hostname!r},"
            " trying again via DoH..."
        )
        utils.log(msg, level=lg.ERROR)
        ip = await _resolve_host_via_doh(hostname)

    msg = f"Resolved {hostname!r} to {ip!r}"
    utils.log(msg, level=lg.INFO)
    return url, {}


def _build_request_kwargs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate ``settings.requests_kwargs`` to httpx equivalents.

    Splits the kwargs into client-level kwargs (passed to
    ``httpx.AsyncClient()``) and per-request kwargs (passed to
    ``.get()``/``.post()``).

    Returns
    -------
    client_kwargs, request_kwargs
        Separated keyword arguments for client construction and per-request
        calls.
    """
    raw = dict(settings._get("requests_kwargs"))
    client_kwargs: dict[str, Any] = {}
    request_kwargs: dict[str, Any] = {}

    # keys that must be set on the httpx.AsyncClient, not per-request
    client_keys = {"verify", "cert", "auth", "proxy", "proxies"}

    for key, value in raw.items():
        if key == "proxies":
            # requests uses proxies={"https": "..."}, httpx uses proxy="..."
            warnings.warn(
                "The 'proxies' key in `settings.requests_kwargs` is deprecated "
                "for the async API. Use 'proxy' with a single URL string instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(value, dict):
                # take the first available proxy URL
                proxy_url = next(iter(value.values()), None)
                if proxy_url is not None:
                    client_kwargs["proxy"] = proxy_url
            else:
                client_kwargs["proxy"] = value
        elif key in client_keys:
            client_kwargs[key] = value
        else:
            request_kwargs[key] = value

    return client_kwargs, request_kwargs


def _get_http_headers(
    *,
    user_agent: str | None = None,
    referer: str | None = None,
    accept_language: str | None = None,
) -> dict[str, str]:
    """Build HTTP headers from settings, using async-safe ``settings._get``.

    Parameters
    ----------
    user_agent
        The user agent. If None, use ``settings.http_user_agent`` value.
    referer
        The referer. If None, use ``settings.http_referer`` value.
    accept_language
        The accept language. If None, use ``settings.http_accept_language``
        value.

    Returns
    -------
    headers
        The HTTP request headers.
    """
    if user_agent is None:
        user_agent = settings._get("http_user_agent")
    if referer is None:
        referer = settings._get("http_referer")
    if accept_language is None:
        accept_language = settings._get("http_accept_language")

    return {
        "User-Agent": user_agent,
        "referer": referer,
        "Accept-Language": accept_language,
    }


def _parse_response(
    response: httpx.Response,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse JSON from an httpx response and log the details.

    Parameters
    ----------
    response
        The httpx response object.

    Returns
    -------
    response_json
        Value will be a dict if the response is from the Google or Overpass
        APIs, and a list if the response is from the Nominatim API.
    """
    hostname = _http._hostname_from_url(str(response.url))
    size_kb = len(response.content) / 1000
    msg = f"Downloaded {size_kb:,.1f}kB from {hostname!r} with status {response.status_code}"
    utils.log(msg, level=lg.INFO)

    try:
        response_json: dict[str, Any] | list[dict[str, Any]] = response.json()
    except JSONDecodeError as e:  # pragma: no cover
        msg = (
            f"{hostname!r} responded: {response.status_code}"
            f" {response.reason_phrase} {response.text}"
        )
        utils.log(msg, level=lg.ERROR)
        if response.is_success:
            raise InsufficientResponseError(msg) from e
        raise ResponseStatusCodeError(msg) from e

    if isinstance(response_json, dict) and "remark" in response_json:  # pragma: no cover
        msg = f"{hostname!r} remarked: {response_json['remark']!r}"
        utils.log(msg, level=lg.WARNING)

    if not response.is_success:
        msg = f"{hostname!r} returned HTTP status code {response.status_code}"
        utils.log(msg, level=lg.WARNING)

    return response_json
