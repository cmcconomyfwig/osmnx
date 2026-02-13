"""Async functions to download and create graphs from OpenStreetMap data.

Async equivalents of the functions in ``osmnx.graph``.
"""

from __future__ import annotations

import logging as lg

import networkx as nx
from shapely import MultiPolygon
from shapely import Polygon

from osmnx import distance
from osmnx import graph as _graph_sync
from osmnx import projection
from osmnx import settings
from osmnx import simplification
from osmnx import stats
from osmnx import truncate
from osmnx import utils
from osmnx import utils_geo

from . import _overpass
from . import geocoder


async def graph_from_bbox(
    bbox: tuple[float, float, float, float],
    *,
    network_type: str = "all",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = False,
    custom_filter: str | list[str] | None = None,
) -> nx.MultiDiGraph:
    """Download and create a graph within a lat-lon bounding box.

    Async equivalent of ``osmnx.graph.graph_from_bbox``.

    Parameters
    ----------
    bbox
        Bounding box as ``(left, bottom, right, top)``.
    network_type
        What type of street network to retrieve if ``custom_filter`` is None.
    simplify
        If True, simplify graph topology via the ``simplify_graph`` function.
    retain_all
        If True, return the entire graph even if it is not connected.
    truncate_by_edge
        If True, retain nodes outside bounding box if at least one neighbor
        lies within it.
    custom_filter
        A custom ways filter to be used instead of ``network_type`` presets.

    Returns
    -------
    G
        The resulting MultiDiGraph.
    """
    polygon = utils_geo.bbox_to_poly(bbox)

    G = await graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )

    msg = f"graph_from_bbox returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)
    return G


async def graph_from_point(
    center_point: tuple[float, float],
    dist: float,
    *,
    dist_type: str = "bbox",
    network_type: str = "all",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = False,
    custom_filter: str | list[str] | None = None,
) -> nx.MultiDiGraph:
    """Download and create a graph within some distance of a lat-lon point.

    Async equivalent of ``osmnx.graph.graph_from_point``.

    Parameters
    ----------
    center_point
        The ``(lat, lon)`` center point around which to construct the graph.
    dist
        Retain only those nodes within this many meters of ``center_point``.
    dist_type
        If "bbox", retain only those nodes within a bounding box. If
        "network", retain only those nodes within network distance.
    network_type
        What type of street network to retrieve if ``custom_filter`` is None.
    simplify
        If True, simplify graph topology.
    retain_all
        If True, return the entire graph even if it is not connected.
    truncate_by_edge
        If True, retain nodes outside bounding box if at least one neighbor
        lies within it.
    custom_filter
        A custom ways filter to be used instead of ``network_type`` presets.

    Returns
    -------
    G
        The resulting MultiDiGraph.
    """
    if dist_type not in {"bbox", "network"}:  # pragma: no cover
        msg = "`dist_type` must be 'bbox' or 'network'."
        raise ValueError(msg)

    bbox = utils_geo.bbox_from_point(center_point, dist)

    G = await graph_from_bbox(
        bbox,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )

    if dist_type == "network":
        node = distance.nearest_nodes(G, X=center_point[1], Y=center_point[0])
        G = truncate.truncate_graph_dist(G, node, dist)

    msg = f"graph_from_point returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)
    return G


async def graph_from_address(
    address: str,
    dist: float,
    *,
    dist_type: str = "bbox",
    network_type: str = "all",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = False,
    custom_filter: str | list[str] | None = None,
) -> nx.MultiDiGraph:
    """Download and create a graph within some distance of an address.

    Async equivalent of ``osmnx.graph.graph_from_address``.

    Parameters
    ----------
    address
        The address to geocode and use as the central point.
    dist
        Retain only those nodes within this many meters.
    dist_type
        If "bbox" or "network".
    network_type
        What type of street network to retrieve if ``custom_filter`` is None.
    simplify
        If True, simplify graph topology.
    retain_all
        If True, return the entire graph even if it is not connected.
    truncate_by_edge
        If True, retain nodes outside bounding box if at least one neighbor
        lies within it.
    custom_filter
        A custom ways filter to be used instead of ``network_type`` presets.

    Returns
    -------
    G
        The resulting MultiDiGraph.
    """
    point = await geocoder.geocode(address)

    G = await graph_from_point(
        point,
        dist,
        dist_type=dist_type,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )

    msg = f"graph_from_address returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)
    return G


async def graph_from_place(
    query: str | dict[str, str] | list[str | dict[str, str]],
    *,
    network_type: str = "all",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = False,
    which_result: int | None | list[int | None] = None,
    custom_filter: str | list[str] | None = None,
) -> nx.MultiDiGraph:
    """Download and create a graph within the boundaries of some place(s).

    Async equivalent of ``osmnx.graph.graph_from_place``.

    Parameters
    ----------
    query
        The query or queries to geocode to retrieve place boundary polygon(s).
    network_type
        What type of street network to retrieve if ``custom_filter`` is None.
    simplify
        If True, simplify graph topology.
    retain_all
        If True, return the entire graph even if it is not connected.
    truncate_by_edge
        If True, retain nodes outside the place boundary polygon(s) if at
        least one of the node's neighbors lies within the polygon(s).
    which_result
        Which geocoding result to use.
    custom_filter
        A custom ways filter to be used instead of ``network_type`` presets.

    Returns
    -------
    G
        The resulting MultiDiGraph.
    """
    gdf = await geocoder.geocode_to_gdf(query, which_result=which_result)
    polygon = gdf.union_all()
    msg = "Constructed place geometry polygon(s) to query Overpass"
    utils.log(msg, level=lg.INFO)

    G = await graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )

    msg = f"graph_from_place returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)
    return G


async def graph_from_polygon(
    polygon: Polygon | MultiPolygon,
    *,
    network_type: str = "all",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = False,
    custom_filter: str | list[str] | None = None,
) -> nx.MultiDiGraph:
    """Download and create a graph within the boundaries of a (Multi)Polygon.

    Async equivalent of ``osmnx.graph.graph_from_polygon``. The HTTP
    download is performed asynchronously; the CPU-bound graph construction
    reuses the sync ``osmnx.graph._create_graph`` function.

    Parameters
    ----------
    polygon
        The geometry within which to construct the graph.
    network_type
        What type of street network to retrieve if ``custom_filter`` is None.
    simplify
        If True, simplify graph topology.
    retain_all
        If True, return the entire graph even if it is not connected.
    truncate_by_edge
        If True, retain nodes outside ``polygon`` if at least one neighbor
        lies within ``polygon``.
    custom_filter
        A custom ways filter to be used instead of ``network_type`` presets.

    Returns
    -------
    G
        The resulting MultiDiGraph.
    """
    if not polygon.is_valid:  # pragma: no cover
        msg = "The geometry of `polygon` is invalid."
        raise ValueError(msg)
    if not isinstance(polygon, (Polygon, MultiPolygon)):  # pragma: no cover
        msg = (
            "Geometry must be a shapely Polygon or MultiPolygon. If you "
            "requested graph from place name, make sure your query resolves "
            "to a Polygon or MultiPolygon, and not some other geometry, like "
            "a Point. See OSMnx documentation for details."
        )
        raise TypeError(msg)

    # create a buffered polygon 0.5km around the desired one
    poly_proj, crs_utm = projection.project_geometry(polygon)
    poly_proj_buff = poly_proj.buffer(500)
    poly_buff, _ = projection.project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)

    # download the network data asynchronously, collecting all responses
    response_jsons = [
        rj
        async for rj in _overpass._download_overpass_network(
            poly_buff, network_type, custom_filter,
        )
    ]

    # reuse sync CPU-bound graph construction
    bidirectional = network_type in settings._get("bidirectional_network_types")
    G_buff = _graph_sync._create_graph(iter(response_jsons), bidirectional)

    # truncate, simplify, etc. — all CPU-bound, no IO
    G_buff = truncate.truncate_graph_polygon(
        G_buff, poly_buff, truncate_by_edge=truncate_by_edge,
    )

    if not retain_all:
        G_buff = truncate.largest_component(G_buff, strongly=False)

    if simplify:
        G_buff = simplification.simplify_graph(G_buff)

    G = truncate.truncate_graph_polygon(
        G_buff, polygon, truncate_by_edge=truncate_by_edge,
    )

    if not retain_all:
        G = truncate.largest_component(G, strongly=False)

    spn = stats.count_streets_per_node(G_buff, nodes=G.nodes)
    nx.set_node_attributes(G, values=spn, name="street_count")

    msg = f"graph_from_polygon returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)
    return G
