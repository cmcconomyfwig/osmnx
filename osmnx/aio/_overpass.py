"""Async tools to work with the Overpass API."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging as lg
from collections import OrderedDict
from typing import TYPE_CHECKING
from typing import Any

import httpx
import numpy as np

from osmnx import _http
from osmnx import _overpass as _overpass_sync
from osmnx import settings
from osmnx import utils
from osmnx._errors import InsufficientResponseError

from . import _http as _ahttp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shapely import MultiPolygon
    from shapely import Polygon

# re-export pure-computation helpers from sync module
_get_network_filter = _overpass_sync._get_network_filter
_make_overpass_settings = _overpass_sync._make_overpass_settings
_make_overpass_polygon_coord_strs = _overpass_sync._make_overpass_polygon_coord_strs
_create_overpass_features_query = _overpass_sync._create_overpass_features_query


async def _get_overpass_pause(
    base_endpoint: str,
    *,
    recursion_pause: float = 5,
    default_pause: float = 60,
) -> float:
    """Retrieve a pause duration from the Overpass API status endpoint.

    Async equivalent of ``osmnx._overpass._get_overpass_pause``.

    Parameters
    ----------
    base_endpoint
        Base Overpass API URL (without "/status" at the end).
    recursion_pause
        How long to wait between recursive calls if the server is currently
        running a query.
    default_pause
        If a fatal error occurs, fall back on this liberal pause duration.

    Returns
    -------
    pause
        The current pause duration specified by the Overpass status endpoint.
    """
    if not settings._get("overpass_rate_limit"):
        return 0

    url = base_endpoint.rstrip("/") + "/status"
    client_kwargs, request_kwargs = _ahttp._build_request_kwargs()

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(
                url,
                headers=_ahttp._get_http_headers(),
                timeout=settings._get("requests_timeout"),
                **request_kwargs,
            )
            response_text = response.text
    except httpx.ConnectError as e:  # pragma: no cover
        msg = f"Unable to reach {url}, {e}"
        utils.log(msg, level=lg.ERROR)
        return default_pause

    try:
        status = response_text.split("\n")[4]
        status_first_part = status.split(" ")[0]
    except (AttributeError, IndexError, ValueError):  # pragma: no cover
        msg = f"Unable to parse {url} response: {response_text}"
        utils.log(msg, level=lg.ERROR)
        return default_pause

    try:
        _ = int(status_first_part)
        pause: float = 0

    except ValueError:  # pragma: no cover
        if status_first_part == "Slot":
            utc_time_str = status.split(" ")[3]
            pattern = "%Y-%m-%dT%H:%M:%SZ,"
            utc_time = dt.datetime.strptime(utc_time_str, pattern).replace(tzinfo=dt.UTC)
            utc_now = dt.datetime.now(tz=dt.UTC)
            seconds = int(np.ceil((utc_time - utc_now).total_seconds()))
            pause = max(seconds, 1)

        elif status_first_part == "Currently":
            await asyncio.sleep(recursion_pause)
            pause = await _get_overpass_pause(base_endpoint)

        else:
            msg = f"Unrecognized server status: {status!r}"
            utils.log(msg, level=lg.ERROR)
            return default_pause

    return pause


async def _overpass_request(data: OrderedDict[str, Any]) -> dict[str, Any]:
    """Send a HTTP POST request to the Overpass API and return response.

    Async equivalent of ``osmnx._overpass._overpass_request``. Uses
    URL-rewriting instead of ``_config_dns()`` for DNS pinning, and
    ``asyncio.sleep`` instead of ``time.sleep`` for rate limiting.

    Parameters
    ----------
    data
        Key-value pairs of parameters.

    Returns
    -------
    response_json
        The Overpass API's response.
    """
    overpass_url = settings._get("overpass_url")

    # resolve URL to same IP via URL-rewriting (no global state mutation)
    base_url = overpass_url.rstrip("/") + "/interpreter"
    resolved_url, host_headers = await _ahttp._resolve_url_to_ip(base_url)

    # prepare the URL for cache key (use original hostname URL like sync does)
    url = overpass_url.rstrip("/") + "/interpreter"
    prepared_url = str(httpx.Request("GET", url, params=data).url)
    cached_response_json = await _ahttp._async_retrieve_from_cache(prepared_url)
    if isinstance(cached_response_json, dict):
        return cached_response_json

    # pause then request this URL
    pause = await _get_overpass_pause(overpass_url)
    hostname = _http._hostname_from_url(url)
    msg = f"Pausing {pause} second(s) before making HTTP POST request to {hostname!r}"
    utils.log(msg, level=lg.INFO)
    await asyncio.sleep(pause)

    # transmit the HTTP POST request
    msg = f"Post {prepared_url} with timeout={settings._get('requests_timeout')}"
    utils.log(msg, level=lg.INFO)

    client_kwargs, request_kwargs = _ahttp._build_request_kwargs()
    headers = {**_ahttp._get_http_headers(), **host_headers}

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post(
            resolved_url,
            data=data,
            timeout=settings._get("requests_timeout"),
            headers=headers,
            **request_kwargs,
        )

    # handle 429 and 504 errors by pausing then re-trying request
    if response.status_code in {429, 504}:  # pragma: no cover
        error_pause = 55
        msg = (
            f"{hostname!r} responded {response.status_code} {response.reason_phrase}: "
            f"we'll retry in {error_pause} secs"
        )
        utils.log(msg, level=lg.WARNING)
        await asyncio.sleep(error_pause)
        return await _overpass_request(data)

    response_json = _ahttp._parse_response(response)
    if not isinstance(response_json, dict):  # pragma: no cover
        msg = "Overpass API did not return a dict of results."
        raise InsufficientResponseError(msg)
    await _ahttp._async_save_to_cache(prepared_url, response_json, response.is_success)
    return response_json


async def _download_overpass_network(
    polygon: Polygon | MultiPolygon,
    network_type: str,
    custom_filter: str | list[str] | None,
) -> AsyncIterator[dict[str, Any]]:
    """Retrieve networked ways and nodes within boundary from the Overpass API.

    Async equivalent of ``osmnx._overpass._download_overpass_network``.

    Parameters
    ----------
    polygon
        The boundary to fetch the network ways/nodes within.
    network_type
        What type of street network to get if ``custom_filter`` is None.
    custom_filter
        A custom "ways" filter to be used instead of ``network_type`` presets.

    Yields
    ------
    response_json
        JSON response from the Overpass server.
    """
    way_filters = []
    if isinstance(custom_filter, list):
        way_filters = custom_filter
    elif isinstance(custom_filter, str):
        way_filters = [custom_filter]
    else:
        way_filters = [_get_network_filter(network_type)]

    overpass_settings = _make_overpass_settings()
    polygon_coord_strs = _make_overpass_polygon_coord_strs(polygon)
    msg = f"Requesting data from API in {len(polygon_coord_strs)} request(s)"
    utils.log(msg, level=lg.INFO)

    for polygon_coord_str in polygon_coord_strs:
        for way_filter in way_filters:
            query_str = (
                f"{overpass_settings};(way{way_filter}"
                f"(poly:{polygon_coord_str!r});>;);out;"
            )
            yield await _overpass_request(OrderedDict(data=query_str))


async def _download_overpass_features(
    polygon: Polygon,
    tags: dict[str, bool | str | list[str]],
) -> AsyncIterator[dict[str, Any]]:
    """Retrieve OSM features within boundary polygon from the Overpass API.

    Async equivalent of ``osmnx._overpass._download_overpass_features``.

    Parameters
    ----------
    polygon
        Boundary to retrieve elements within.
    tags
        Tags used for finding elements in the selected area.

    Yields
    ------
    response_json
        JSON response from the Overpass server.
    """
    polygon_coord_strs = _make_overpass_polygon_coord_strs(polygon)
    msg = f"Requesting data from API in {len(polygon_coord_strs)} request(s)"
    utils.log(msg, level=lg.INFO)

    for polygon_coord_str in polygon_coord_strs:
        query_str = _create_overpass_features_query(polygon_coord_str, tags)
        yield await _overpass_request(OrderedDict(data=query_str))
