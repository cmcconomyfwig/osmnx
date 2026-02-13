"""Async geocoding via the Nominatim API.

Async equivalents of the functions in ``osmnx.geocoder``.
"""

from __future__ import annotations

import logging as lg
from collections import OrderedDict
from typing import Any

import geopandas as gpd
import pandas as pd

from osmnx import geocoder as _geocoder_sync
from osmnx import settings
from osmnx import utils
from osmnx._errors import InsufficientResponseError

from . import _nominatim


async def geocode(query: str) -> tuple[float, float]:
    """Geocode place names or addresses to ``(lat, lon)`` with the Nominatim API.

    Async equivalent of ``osmnx.geocoder.geocode``.

    Parameters
    ----------
    query
        The query string to geocode.

    Returns
    -------
    point
        The ``(lat, lon)`` coordinates returned by the geocoder.
    """
    params: OrderedDict[str, int | str] = OrderedDict()
    params["format"] = "json"
    params["limit"] = 1
    params["dedupe"] = 0
    params["q"] = query
    response_json = await _nominatim._nominatim_request(params=params)

    if response_json and "lat" in response_json[0] and "lon" in response_json[0]:
        lat = float(response_json[0]["lat"])
        lon = float(response_json[0]["lon"])
        point = (lat, lon)

        msg = f"Geocoded {query!r} to {point}"
        utils.log(msg, level=lg.INFO)
        return point

    msg = f"Nominatim could not geocode query {query!r}."
    raise InsufficientResponseError(msg)


async def geocode_to_gdf(
    query: str | dict[str, str] | list[str | dict[str, str]],
    *,
    which_result: int | None | list[int | None] = None,
    by_osmid: bool = False,
) -> gpd.GeoDataFrame:
    """Retrieve OSM elements by place name or OSM ID with the Nominatim API.

    Async equivalent of ``osmnx.geocoder.geocode_to_gdf``.

    Parameters
    ----------
    query
        The query or queries to geocode to retrieve place boundary polygon(s).
    which_result
        Which search result to return. If None, auto-select the first
        (Multi)Polygon or raise an error if OSM doesn't return one. To get
        the top match (sorted by importance) regardless of geometry type, set
        ``which_result=1``. Ignored if ``by_osmid=True``.
    by_osmid
        If True, treat query as an OSM ID lookup rather than text search.

    Returns
    -------
    gdf
        GeoDataFrame with one row for each query result.
    """
    if isinstance(query, list):
        q_list = query
        wr_list = which_result if isinstance(which_result, list) else [which_result] * len(query)
    else:
        q_list = [query]
        wr_list = [which_result[0]] if isinstance(which_result, list) else [which_result]

    if len(q_list) != len(wr_list):  # pragma: no cover
        msg = "`which_result` length must equal `query` length."
        raise ValueError(msg)

    # geocode each query sequentially (rate limited by Nominatim 1 req/s)
    results = []
    for q, wr in zip(q_list, wr_list, strict=True):
        gdf_row = await _async_geocode_query_to_gdf(q, wr, by_osmid)
        results.append(gdf_row)

    gdf = pd.concat(results, ignore_index=True).set_crs(settings._get("default_crs"))

    msg = f"Created GeoDataFrame with {len(gdf)} rows from {len(q_list)} queries"
    utils.log(msg, level=lg.INFO)
    return gdf


async def _async_geocode_query_to_gdf(
    query: str | dict[str, str],
    which_result: int | None,
    by_osmid: bool,  # noqa: FBT001
) -> gpd.GeoDataFrame:
    """Geocode a single place query to a GeoDataFrame.

    Parameters
    ----------
    query
        Query string or structured dict to geocode.
    which_result
        Which search result to return.
    by_osmid
        If True, treat query as an OSM ID lookup rather than text search.

    Returns
    -------
    gdf
        GeoDataFrame with one row containing the geocoding result.
    """
    limit = 50 if which_result is None else which_result
    results = await _nominatim._download_nominatim_element(
        query, by_osmid=by_osmid, limit=limit,
    )

    # reuse the sync module's result-selection and GeoDataFrame-building logic
    # by sorting and selecting here, then delegating to the sync helper
    results = sorted(results, key=lambda x: x["importance"], reverse=True)

    if len(results) == 0:
        msg = f"Nominatim geocoder returned 0 results for query {query!r}."
        raise InsufficientResponseError(msg)

    if by_osmid:
        result = results[0]

    elif which_result is None:
        try:
            result = _geocoder_sync._get_first_polygon(results)
        except TypeError as e:
            msg = (
                f"Nominatim did not geocode query {query!r} to a geometry"
                " of type (Multi)Polygon."
            )
            raise TypeError(msg) from e

    elif len(results) >= which_result:
        result = results[which_result - 1]

    else:  # pragma: no cover
        msg = f"Nominatim returned {len(results)} result(s) but `which_result={which_result}`."
        raise InsufficientResponseError(msg)

    geom_type = result["geojson"]["type"]
    if geom_type not in {"Polygon", "MultiPolygon"}:
        msg = (
            f"Nominatim geocoder returned a {geom_type} as the geometry"
            f" for query {query!r}"
        )
        utils.log(msg, level=lg.WARNING)

    bottom, top, left, right = result["boundingbox"]
    feature: dict[str, Any] = {
        "type": "Feature",
        "geometry": result["geojson"],
        "properties": {
            "bbox_west": left,
            "bbox_south": bottom,
            "bbox_east": right,
            "bbox_north": top,
        },
    }

    for attr in result:
        if attr not in {"address", "boundingbox", "geojson", "icon", "licence"}:
            feature["properties"][attr] = result[attr]

    gdf = gpd.GeoDataFrame.from_features([feature])
    cols = ["lat", "lon", "bbox_north", "bbox_south", "bbox_east", "bbox_west"]
    gdf[cols] = gdf[cols].astype(float)
    return gdf
