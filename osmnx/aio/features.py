"""Async functions to download and create GeoDataFrames from OSM features.

Async equivalents of the functions in ``osmnx.features``.
"""

from __future__ import annotations

import logging as lg
from typing import TYPE_CHECKING

from shapely import MultiPolygon
from shapely import Polygon

from osmnx import features as _features_sync
from osmnx import utils
from osmnx import utils_geo

from . import _overpass
from . import geocoder

if TYPE_CHECKING:
    import geopandas as gpd


async def features_from_bbox(
    bbox: tuple[float, float, float, float],
    tags: dict[str, bool | str | list[str]],
) -> gpd.GeoDataFrame:
    """Download OSM features within a lat-lon bounding box.

    Async equivalent of ``osmnx.features.features_from_bbox``.

    Parameters
    ----------
    bbox
        Bounding box as ``(left, bottom, right, top)``.
    tags
        Tags for finding elements in the selected area.

    Returns
    -------
    gdf
        The features, multi-indexed by element type and OSM ID.
    """
    polygon = utils_geo.bbox_to_poly(bbox)
    return await features_from_polygon(polygon, tags)


async def features_from_point(
    center_point: tuple[float, float],
    tags: dict[str, bool | str | list[str]],
    dist: float,
) -> gpd.GeoDataFrame:
    """Download OSM features within some distance of a lat-lon point.

    Async equivalent of ``osmnx.features.features_from_point``.

    Parameters
    ----------
    center_point
        The ``(lat, lon)`` center point.
    tags
        Tags for finding elements in the selected area.
    dist
        Distance in meters from ``center_point``.

    Returns
    -------
    gdf
        The features, multi-indexed by element type and OSM ID.
    """
    bbox = utils_geo.bbox_from_point(center_point, dist)
    return await features_from_bbox(bbox, tags)


async def features_from_address(
    address: str,
    tags: dict[str, bool | str | list[str]],
    dist: float,
) -> gpd.GeoDataFrame:
    """Download OSM features within some distance of an address.

    Async equivalent of ``osmnx.features.features_from_address``.

    Parameters
    ----------
    address
        The address to geocode and use as the center point.
    tags
        Tags for finding elements in the selected area.
    dist
        Distance in meters from ``address``.

    Returns
    -------
    gdf
        The features, multi-indexed by element type and OSM ID.
    """
    center_point = await geocoder.geocode(address)
    return await features_from_point(center_point, tags, dist)


async def features_from_place(
    query: str | dict[str, str] | list[str | dict[str, str]],
    tags: dict[str, bool | str | list[str]],
    *,
    which_result: int | None | list[int | None] = None,
) -> gpd.GeoDataFrame:
    """Download OSM features within the boundaries of some place(s).

    Async equivalent of ``osmnx.features.features_from_place``.

    Parameters
    ----------
    query
        The query or queries to geocode to retrieve place boundary polygon(s).
    tags
        Tags for finding elements in the selected area.
    which_result
        Which search result to return.

    Returns
    -------
    gdf
        The features, multi-indexed by element type and OSM ID.
    """
    gdf = await geocoder.geocode_to_gdf(query, which_result=which_result)
    polygon = gdf.union_all()
    msg = "Constructed place geometry polygon(s) to query Overpass"
    utils.log(msg, level=lg.INFO)

    return await features_from_polygon(polygon, tags)


async def features_from_polygon(
    polygon: Polygon | MultiPolygon,
    tags: dict[str, bool | str | list[str]],
) -> gpd.GeoDataFrame:
    """Download OSM features within the boundaries of a (Multi)Polygon.

    Async equivalent of ``osmnx.features.features_from_polygon``. The HTTP
    download is performed asynchronously; the CPU-bound GeoDataFrame
    construction reuses the sync ``osmnx.features._create_gdf`` function.

    Parameters
    ----------
    polygon
        The geometry within which to retrieve features.
    tags
        Tags for finding elements in the selected area.

    Returns
    -------
    gdf
        The features, multi-indexed by element type and OSM ID.
    """
    if not polygon.is_valid:
        msg = "The geometry of `polygon` is invalid."
        raise ValueError(msg)

    if not isinstance(polygon, (Polygon, MultiPolygon)):
        msg = (
            "Boundaries must be a Polygon or MultiPolygon. If you requested "
            "`features_from_place`, ensure your query geocodes to a Polygon "
            "or MultiPolygon. See the documentation for details."
        )
        raise TypeError(msg)

    # download data asynchronously, collecting all responses
    response_jsons = [
        rj async for rj in _overpass._download_overpass_features(polygon, tags)
    ]

    # reuse sync CPU-bound GeoDataFrame creation
    return _features_sync._create_gdf(iter(response_jsons), polygon, tags)
