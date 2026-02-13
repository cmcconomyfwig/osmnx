"""Async tools to work with the Nominatim API."""

from __future__ import annotations

import asyncio
import logging as lg
from collections import OrderedDict
from typing import Any

import httpx

from osmnx import _http
from osmnx import settings
from osmnx import utils
from osmnx._errors import InsufficientResponseError

from . import _http as _ahttp
from ._rate_limit import _rate_limiter


async def _download_nominatim_element(
    query: str | dict[str, str],
    *,
    by_osmid: bool = False,
    limit: int = 1,
    polygon_geojson: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve an OSM element from the Nominatim API.

    Async equivalent of ``osmnx._nominatim._download_nominatim_element``.

    Parameters
    ----------
    query
        Query string or structured query dict.
    by_osmid
        If True, treat ``query`` as an OSM ID lookup rather than text search.
    limit
        Max number of results to return.
    polygon_geojson
        Whether to retrieve the place's geometry from the API.

    Returns
    -------
    response_json
        The Nominatim API's response.
    """
    params: OrderedDict[str, int | str] = OrderedDict()
    params["format"] = "json"
    params["polygon_geojson"] = int(polygon_geojson)

    if by_osmid:
        if not isinstance(query, str):
            msg = "`query` must be a string if `by_osmid` is True."
            raise TypeError(msg)
        request_type = "lookup"
        params["osm_ids"] = query

    else:
        request_type = "search"
        params["dedupe"] = 0
        params["limit"] = limit

        if isinstance(query, str):
            params["q"] = query
        elif isinstance(query, dict):
            for key in sorted(query):
                params[key] = query[key]
        else:  # pragma: no cover
            msg = "Each query must be a dict or a string."  # type: ignore[unreachable]
            raise TypeError(msg)

    return await _nominatim_request(params=params, request_type=request_type)


async def _nominatim_request(
    params: OrderedDict[str, int | str],
    *,
    request_type: str = "search",
) -> list[dict[str, Any]]:
    """Send a HTTP GET request to the Nominatim API and return response.

    Async equivalent of ``osmnx._nominatim._nominatim_request``. Uses an
    ``AsyncRateLimiter`` to enforce Nominatim's 1-request-per-second policy
    across concurrent tasks.

    Parameters
    ----------
    params
        Key-value pairs of parameters.
    request_type
        Which Nominatim API endpoint to query, one of {"search", "reverse",
        "lookup"}.

    Returns
    -------
    response_json
        The Nominatim API's response.
    """
    if request_type not in {"search", "reverse", "lookup"}:  # pragma: no cover
        msg = "Nominatim `request_type` must be 'search', 'reverse', or 'lookup'."
        raise ValueError(msg)

    nominatim_key = settings._get("nominatim_key")
    if nominatim_key is not None:
        params["key"] = nominatim_key

    # prepare URL and check cache
    nominatim_url = settings._get("nominatim_url")
    url = nominatim_url.rstrip("/") + "/" + request_type
    prepared_url = str(httpx.Request("GET", url, params=params).url)
    cached_response_json = await _ahttp._async_retrieve_from_cache(prepared_url)
    if isinstance(cached_response_json, list):
        return cached_response_json

    # enforce Nominatim's rate limit: 1 request per second
    hostname = _http._hostname_from_url(url)
    await _rate_limiter.wait(hostname, min_interval=1.0)

    msg = f"Pausing before making HTTP GET request to {hostname!r}"
    utils.log(msg, level=lg.INFO)

    # transmit the HTTP GET request
    msg = f"Get {prepared_url} with timeout={settings._get('requests_timeout')}"
    utils.log(msg, level=lg.INFO)

    client_kwargs, request_kwargs = _ahttp._build_request_kwargs()
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.get(
            url,
            params=params,
            timeout=settings._get("requests_timeout"),
            headers=_ahttp._get_http_headers(),
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
        return await _nominatim_request(params, request_type=request_type)

    response_json = _ahttp._parse_response(response)
    if not isinstance(response_json, list):
        msg = "Nominatim API did not return a list of results."
        raise InsufficientResponseError(msg)
    await _ahttp._async_save_to_cache(prepared_url, response_json, response.is_success)
    return response_json
