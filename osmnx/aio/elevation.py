"""Async functions to add node elevations from web APIs.

Async equivalents of the elevation API functions in ``osmnx.elevation``.
"""

from __future__ import annotations

import asyncio
import logging as lg
from typing import Any

import httpx
import networkx as nx
import numpy as np
import pandas as pd

from osmnx import _http
from osmnx import settings
from osmnx import utils
from osmnx._errors import InsufficientResponseError

from . import _http as _ahttp


async def add_node_elevations_google(
    G: nx.MultiDiGraph,
    *,
    api_key: str | None = None,
    batch_size: int = 512,
    pause: float = 0,
) -> nx.MultiDiGraph:
    """Add ``elevation`` (meters) attributes to all nodes using a web API.

    Async equivalent of ``osmnx.elevation.add_node_elevations_google``.
    When ``pause`` is 0, batched requests are issued concurrently via
    ``asyncio.gather`` for improved throughput.

    Parameters
    ----------
    G
        Graph to add elevation data to.
    api_key
        A valid API key. Can be None if the API does not require a key.
    batch_size
        Max number of coordinate pairs to submit in each request.
    pause
        How long to pause in seconds between API calls.

    Returns
    -------
    G
        Graph with ``elevation`` attributes on the nodes.
    """
    node_points = pd.Series(
        {n: f"{d['y']:.6f},{d['x']:.6f}" for n, d in G.nodes(data=True)},
    )
    n_calls = int(np.ceil(len(node_points) / batch_size))
    hostname = _http._hostname_from_url(settings._get("elevation_url_template"))

    msg = f"Requesting node elevations from {hostname!r} in {n_calls} request(s)"
    utils.log(msg, level=lg.INFO)

    # build all URLs up front
    urls = []
    for i in range(0, len(node_points), batch_size):
        chunk = node_points.iloc[i : i + batch_size]
        locations = "|".join(chunk)
        url = settings._get("elevation_url_template").format(
            locations=locations, key=api_key,
        )
        urls.append(url)

    # fetch all elevation results
    if pause == 0:
        # concurrent requests when no pause needed
        tasks = [_elevation_request(url, 0) for url in urls]
        responses = await asyncio.gather(*tasks)
    else:
        # sequential with pause
        responses = []
        for url in urls:
            response_json = await _elevation_request(url, pause)
            responses.append(response_json)

    results = []
    for response_json in responses:
        if "results" in response_json and len(response_json["results"]) > 0:
            results.extend(response_json["results"])
        else:
            raise InsufficientResponseError(str(response_json))

    msg = (
        f"Graph has {len(G):,} nodes and we received {len(results):,}"
        f" results from {hostname!r}"
    )
    utils.log(msg, level=lg.INFO)
    if not (len(results) == len(G) == len(node_points)):  # pragma: no cover
        err_msg = f"{msg}\n{response_json}"
        raise InsufficientResponseError(err_msg)

    df_elev = pd.DataFrame(node_points, columns=["node_points"])
    df_elev["elevation"] = [result["elevation"] for result in results]
    nx.set_node_attributes(G, name="elevation", values=df_elev["elevation"].to_dict())
    msg = f"Added elevation data from {hostname!r} to all nodes."
    utils.log(msg, level=lg.INFO)

    return G


async def _elevation_request(url: str, pause: float) -> dict[str, Any]:
    """Send a HTTP GET request to a Google Maps-style elevation API.

    Async equivalent of ``osmnx.elevation._elevation_request``.

    Parameters
    ----------
    url
        URL of API endpoint, populated with request data.
    pause
        How long to pause in seconds before request.

    Returns
    -------
    response_json
        The elevation API's response.
    """
    cached_response_json = await _ahttp._async_retrieve_from_cache(url)
    if isinstance(cached_response_json, dict):
        return cached_response_json

    hostname = _http._hostname_from_url(url)
    msg = f"Pausing {pause} second(s) before making HTTP GET request to {hostname!r}"
    utils.log(msg, level=lg.INFO)
    await asyncio.sleep(pause)

    msg = f"Get {url} with timeout={settings._get('requests_timeout')}"
    utils.log(msg, level=lg.INFO)

    client_kwargs, request_kwargs = _ahttp._build_request_kwargs()
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.get(
            url,
            timeout=settings._get("requests_timeout"),
            headers=_ahttp._get_http_headers(),
            **request_kwargs,
        )

    response_json = _ahttp._parse_response(response)
    if not isinstance(response_json, dict):  # pragma: no cover
        msg = "Elevation API did not return a dict of results."
        raise InsufficientResponseError(msg)
    await _ahttp._async_save_to_cache(url, response_json, response.is_success)
    return response_json
