import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable
from cachetools import cached, TTLCache, keys
from geoalchemy2.functions import ST_Transform, ST_AsMVTGeom, ST_AsMVT, ST_CurveToLine
from sqlalchemy import Engine, Label
from sqlalchemy.sql import select
from sqlalchemy.orm import Session
from pygeoapi.util import get_crs_from_uri
from pygeoapi.provider.mvt_postgresql import MVTPostgreSQLProvider

LOGGER = logging.getLogger(__name__)
_MEM_CACHE_DAYS = 1


class MVTPostgreSQLExtendedProvider(MVTPostgreSQLProvider):
    def __init__(self, provider_def: Dict):
        MVTPostgreSQLProvider.__init__(self, provider_def)

        self._layer: str = provider_def.get('layer', self.table)
        self._setup_caching(provider_def.get('cache'))

    def get_layer(self) -> str:
        return self._layer

    def get_tiles(self, layer=None, tileset=None, z=None, y=None, x=None, format_=None) -> bytes:
        z, y, x = map(int, [str(z), str(y), str(x)])

        [tileset_schema] = [
            schema for schema in self.get_tiling_schemes()
            if tileset == schema.tileMatrixSet
        ]

        if not self.is_in_limits(tileset_schema, z, x, y):
            return bytes()

        layer = layer or self.get_layer()
        tile_path = f'{tileset}/{z}/{y}/{x}.pbf'

        result = _get_tiles(layer, tileset_schema.tileMatrixSet, z, y, x, self.storage_crs, tileset_schema.crs, self._engine,
                            self.table_model, self.geom, self.fields, self.get_envelope, tile_path, self._cache_options)

        return result

    def _setup_caching(self, cache_options: Optional[Dict[str, Any]]) -> None:
        if not cache_options:
            self._cache_options = None
            return

        base_path: str = cache_options['path']
        instance_path = f'{self.db_name}/{self.db_search_path[0]}/{self.table}'

        self._cache_options = {
            'base_path': Path(base_path).joinpath(instance_path),
            'max_age_days': cache_options.get('max_age_days', 7)
        }

    def __repr__(self):
        return f'<MVTPostgreSQLExtendedProvider> {self.data}'


_mem_cache = TTLCache(maxsize=640*1024, ttl=_MEM_CACHE_DAYS * 86400)


@cached(cache=_mem_cache, key=lambda layer, tileset, z, y, x, storage_crs, tileset_schema_crs, engine, table_model, geom, fields, get_envelope_func, tile_path, cache_options: keys.hashkey(tile_path))
def _get_tiles(
    layer: str,
    tileset: str,
    z: int,
    y: int,
    x: int,
    storage_crs: str,
    tileset_schema_crs: str,
    engine: Engine,
    table_model: Any,
    geom: Any,
    fields: Dict,
    get_envelope_func: Callable[[int, int, int, str], Label],
    tile_path: str,
    cache_options: Optional[Dict[str, Any]] = None
) -> bytes:
    if cache_options:
        base_path: Path = cache_options['base_path']
        tile_cache_path = base_path.joinpath(tile_path)
        max_age = cache_options['max_age_days']

        if tile_cache_path.exists() and not _should_refresh_cache(tile_cache_path, max_age):
            with open(tile_cache_path, 'rb') as file:
                return file.read()

    storage_srid = get_crs_from_uri(storage_crs).to_string()
    out_srid = get_crs_from_uri(tileset_schema_crs).to_string()
    envelope = get_envelope_func(z, y, x, tileset)

    geom_column = getattr(table_model, geom)

    geom_filter = geom_column.intersects(
        ST_Transform(envelope, storage_srid) # type: ignore
    )

    mvtgeom = (
        ST_AsMVTGeom(
            ST_Transform(ST_CurveToLine(geom_column), out_srid),
            ST_Transform(envelope, out_srid))
        .label('mvtgeom')
    )

    mvtrow = (
        select(mvtgeom, *fields.values())
        .filter(geom_filter)
        .cte('mvtrow')
    )

    mvtquery = select(
        ST_AsMVT(mvtrow.table_valued(), layer)
    )

    with Session(engine) as session:
        memview: Any = session.execute(mvtquery).scalar()
        result = bytes(memview) or None

    if result and cache_options:
        base_path: Path = cache_options['base_path']
        tile_cache_path = base_path.joinpath(tile_path)
        tile_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tile_cache_path, 'wb') as file:
            file.write(result)

    return result or bytes()


def _should_refresh_cache(file_path: Path, cache_days: int) -> bool:
    timestamp = file_path.stat().st_mtime
    modified = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    diff = datetime.now(tz=timezone.utc) - modified

    return diff.days > cache_days
