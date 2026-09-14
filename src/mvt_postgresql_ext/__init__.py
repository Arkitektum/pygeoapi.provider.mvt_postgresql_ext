"""Custom MVT PostgreSQL provider for pygeoapi.

Extends :class:`pygeoapi.provider.mvt_postgresql.MVTPostgreSQLProvider` with:

* an in-process TTL cache for rendered tiles and for the layer extent, so
  repeated requests for the same tile do not hit PostgreSQL again,
* a configurable layer name (the base provider always uses the table name),
* TileJSON 3.0.0 vendor metadata, including ``bounds``, ``center`` and
  ``vector_layers``, which the base provider does not implement.
"""

from os import getenv
import logging
from pathlib import Path
from pygeoapi.util import url_join
from typing import Any, Dict, List, Callable
from cachetools import cached, TTLCache, keys
from geoalchemy2.functions import (ST_Transform, ST_AsMVTGeom, ST_AsMVT,
                                   ST_CurveToLine, ST_Extent, ST_XMax, ST_YMax, ST_XMin, ST_YMin, ST_SetSRID)
from sqlalchemy import (Engine, Label, String, Integer,
                        Float, Boolean, Numeric, Date, DateTime)
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
from sqlalchemy.sql import select
from sqlalchemy.orm import Session
from pyproj import CRS
from pygeoapi.crs import get_crs
from pygeoapi.provider.mvt_postgresql import MVTPostgreSQLProvider as MVTPostgreSQLProviderBase

# Cache lifetime in days, overridable per deployment. Tiles are only evicted
# on TTL expiry, so this is also the maximum staleness of served tiles.
_MEM_CACHE_DAYS = getenv('MVT_CACHE_DAYS')

_logger = logging.getLogger(__name__)

_mem_cache_days = int(_MEM_CACHE_DAYS) if _MEM_CACHE_DAYS else 1

# Memory budget for the tile cache. `getsizeof=len` makes maxsize count the
# bytes of the cached tiles rather than the number of entries, which is the
# figure that actually matters: tile sizes vary by orders of magnitude, so an
# entry count is a poor proxy for memory use. Per worker process; the few
# hundred bytes of key and dict overhead per entry are not counted.
_MVT_CACHE_BYTES = 256 * 1024 * 1024

# The caches live at module level rather than on the provider instance: pygeoapi
# builds a new provider object per request, so an instance attribute would be
# thrown away before it could ever be hit. Both are keyed by a string that
# identifies the database, schema and table (see _get_cache_key), which keeps
# collections in the same process from sharing entries.
_mvt_cache = TTLCache(
    maxsize=_MVT_CACHE_BYTES, ttl=_mem_cache_days * 86400, getsizeof=len
)

# Entry count, one per collection. Extents are four floats, so the limit is
# about how many collections a process serves, not about memory.
_bounds_cache = TTLCache(maxsize=250, ttl=_mem_cache_days * 86400)


class MVTPostgreSQLProvider(MVTPostgreSQLProviderBase):
    def __init__(self, provider_def: Dict):
        MVTPostgreSQLProviderBase.__init__(self, provider_def)

        # The base provider hardcodes the table name as the layer name. Allow
        # the config to override it, since the layer name is what styles in the
        # client refer to ("source-layer").
        self._layer: str = provider_def.get("layer", self.table)

    def get_layer(self) -> str:
        return self._layer

    def get_tiles(
        self,
        layer=None,
        tileset=None,
        z=None,
        y=None,
        x=None,
        format_=None
    ) -> bytes:
        # Tile indices arrive as strings from the URL path.
        z, y, x = map(int, [str(z), str(y), str(x)])

        [tileset_schema] = [
            schema
            for schema in self.get_tiling_schemes()
            if tileset == schema.tileMatrixSet
        ]

        # Unlike the base provider, which raises ProviderTileNotFoundError, an
        # out-of-range tile is answered with an empty tile. Clients request
        # tiles speculatively and a 404 shows up as an error in the console.
        if not self.is_in_limits(tileset_schema, z, x, y):
            return bytes()

        # Path-shaped key, one entry per tile per collection. The layout mirrors
        # how the tiles would be laid out on disk, which makes cache keys
        # readable in logs.
        cache_key = str(
            Path(self._get_cache_key()).joinpath(
                f"{tileset}/{z}/{y}/{x}.pbf")
        )

        # The actual query lives in a module-level function so that the cache
        # survives this (per-request) provider instance.
        result = _get_tiles(
            str(layer),
            tileset_schema.tileMatrixSet,
            z,
            y,
            x,
            self.storage_crs,
            tileset_schema.crs,
            self._engine,
            self.table_model,
            self.geom,
            self.fields,
            self.get_envelope,
            cache_key
        )

        return result

    def get_vendor_metadata(
        self,
        dataset,
        server_url,
        layer,
        tileset,
        title,
        description,
        keywords,
        **kwargs
    ) -> Dict[str, Any]:
        """Build TileJSON 3.0.0 metadata for the collection.

        Served from ``/collections/{dataset}/tiles/{tileset}/metadata?f=tilejson``
        and consumed directly by MapLibre/Mapbox GL as a vector source.
        """
        # The placeholders stay unexpanded on purpose: they are part of the
        # TileJSON tile URL template the client fills in per tile.
        service_url = url_join(
            server_url,
            f"collections/{dataset}/tiles/{tileset}",
            "{tileMatrix}/{tileRow}/{tileCol}?f=mvt"
        )

        tilejson = {
            "tilejson": "3.0.0",
            "name": title or dataset,
            "description": description,
            "version": "1.0.0",
            "scheme": "tms",
            "tiles": [service_url],
            "minzoom": self.options["zoom"]["min"],
            "maxzoom": self.options["zoom"]["max"],
        }

        bounds = _get_bounds(self._engine, self.table_model,
                             self.geom, self._get_cache_key())

        # bounds/center are optional in TileJSON; omit them for an empty table
        # rather than emitting nulls the client would have to handle.
        if bounds:
            tilejson["bounds"] = bounds

            tilejson["center"] = [
                (bounds[0] + bounds[2]) / 2,
                (bounds[1] + bounds[3]) / 2,
                self.options["zoom"]["min"]
            ]

        # Declaring the attributes lets clients do data-driven styling and
        # filtering without first inspecting a tile.
        tilejson["vector_layers"] = [
            {
                "id": layer,
                "description": description,
                "fields": {
                    name: self._map_field_type(column.type)
                    for name, column in self.get_fields().items()
                }
            }
        ]

        return tilejson

    def _map_field_type(self, field_type) -> str:
        """Map a SQLAlchemy column type to a TileJSON field type.

        TileJSON only knows ``String``, ``Number`` and ``Boolean``, so anything
        without a numeric or boolean equivalent is reported as ``String``.
        """
        if isinstance(field_type, (String)):
            return "String"
        elif isinstance(field_type, (Integer, Float, Numeric)):
            return "Number"
        elif isinstance(field_type, Boolean):
            return "Boolean"
        elif isinstance(field_type, (Date, DateTime)):
            return "String"
        elif isinstance(field_type, (UUID, JSON, JSONB)):
            return "String"

        return "String"

    def _get_cache_key(self) -> str:
        """Cache key prefix identifying the source table: db/schema/table."""
        return f"{self.db_name}/{self.db_search_path[0]}/{self.table}" # type: ignore


# Only cache_key takes part in the lookup key. The remaining arguments are
# engines, models and CRS objects that are either unhashable or recreated per
# request, and they are all derived from the same table the key already names.
@cached(
    cache=_mvt_cache,
    key=lambda layer,
    tileset,
    z,
    y,
    x,
    storage_crs,
    tileset_schema_crs,
    engine,
    table_model,
    geom,
    fields,
    get_envelope_func,
    cache_key: keys.hashkey(cache_key)
)
def _get_tiles(
    layer: str,
    tileset: str,
    z: int,
    y: int,
    x: int,
    storage_crs: CRS,
    tileset_schema_crs: str,
    engine: Engine,
    table_model: Any,
    geom: Any,
    fields: Dict,
    get_envelope_func: Callable[[int, int, int, str], Label],
    cache_key: str
) -> bytes:
    """Render a single tile in PostGIS and return it as MVT bytes.

    Mirrors the base provider's query; it exists as a free function so the
    result can be memoized across provider instances.
    """
    storage_srid = get_crs(storage_crs).to_string()
    out_srid = get_crs(tileset_schema_crs).to_string()
    envelope = get_envelope_func(z, y, x, tileset)

    geom_column = getattr(table_model, geom)

    # Filter in the storage CRS so an existing spatial index on the column can
    # be used; transforming the column instead would make the index useless.
    geom_filter = geom_column.intersects(
        ST_Transform(envelope, storage_srid)  # type: ignore
    )

    # ST_AsMVTGeom clips to the tile and converts to tile-local coordinates.
    # ST_CurveToLine first, because MVT cannot represent curved geometries.
    mvtgeom = ST_AsMVTGeom(
        ST_Transform(ST_CurveToLine(geom_column), out_srid),
        ST_Transform(envelope, out_srid),
    ).label("mvtgeom")

    mvtrow = select(mvtgeom, *fields.values()
                    ).filter(geom_filter).cte("mvtrow")

    mvtquery = select(ST_AsMVT(mvtrow.table_valued(), layer))

    with Session(engine) as session:
        memview: Any = session.execute(mvtquery).scalar()
        result = bytes(memview) or None

    # Always bytes, never None: an empty tile is a valid answer and gets cached
    # like any other, so empty areas are not re-queried on every request.
    return result or bytes()


# Keyed on the table identifier only, for the same reason as _get_tiles above.
@cached(
    cache=_bounds_cache,
    key=lambda engine,
    table_model,
    geom,
    cache_key: keys.hashkey(cache_key)
)
def _get_bounds(
    engine: Engine,
    table_model: Any,
    geom: Any,
    cache_key: str
) -> List[float] | None:
    """Return the layer extent as ``[minx, miny, maxx, maxy]`` in WGS 84.

    Returns ``None`` when the table holds no geometries. The full-table scan
    this implies is the reason the result is cached.
    """
    geom_column = getattr(table_model, geom)
    # ST_Extent drops the SRID, so set it back before reading the corners.
    extent = ST_SetSRID(ST_Extent(ST_Transform(geom_column, 4326)), 4326)

    stmt = select(
        ST_XMin(extent),
        ST_YMin(extent),
        ST_XMax(extent),
        ST_YMax(extent)
    )

    with Session(engine) as session:
        row = session.execute(stmt).first()

    # ST_Extent over an empty table yields a single NULL row.
    if not row or row[0] is None:
        return None

    return [row[0], row[1], row[2], row[3]]
