"""Async alternatives for OSMnx functions that perform HTTP IO.

This subpackage provides ``async`` versions of all OSMnx functions that
perform network requests. It uses ``httpx`` as its HTTP transport library
and requires the ``async`` optional-dependency extra::

    pip install osmnx[async]

Usage example::

    import asyncio
    from osmnx.aio import graph

    async def main():
        G = await graph.graph_from_place("Piedmont, CA, USA")

    asyncio.run(main())
"""

from __future__ import annotations

try:
    import httpx  # noqa: F401
except ImportError as e:
    msg = "Install the 'async' extra to use osmnx.aio: pip install osmnx[async]"
    raise ImportError(msg) from e

from . import elevation as elevation
from . import features as features
from . import geocoder as geocoder
from . import graph as graph
