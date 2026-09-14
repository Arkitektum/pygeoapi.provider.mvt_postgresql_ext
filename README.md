# mvt_postgresql_dibk

A custom [pygeoapi](https://pygeoapi.io/) tile provider that extends the built-in
`MVT-postgresql` provider with **tile caching** and **TileJSON metadata**.

Tiles are still rendered on the fly by PostGIS (`ST_AsMVT`); this provider adds an
in-process cache in front of that, serves a configurable layer name, and implements
the TileJSON output that the base provider leaves unimplemented.

## What it adds over `pygeoapi.provider.mvt_postgresql.MVTPostgreSQLProvider`

| | Base provider | This provider |
| --- | --- | --- |
| Tile rendering | One PostGIS query per request | Cached in memory, TTL-based |
| Layer extent | Not exposed | `ST_Extent` over the table, cached |
| Layer name | Always the table name | `layer` config option, defaults to the table name |
| `f=tilejson` metadata | `NotImplementedError` | TileJSON 3.0.0 with `bounds`, `center` and `vector_layers` |
| Tile outside zoom/extent limits | `ProviderTileNotFoundError` (404) | Empty tile (200) |

## Requirements

* Python >= 3.12
* pygeoapi >= 0.24.0
* PostgreSQL with PostGIS

## Installation

```bash
uv sync
```

or, into an existing pygeoapi environment:

```bash
pip install .
```

## Configuration

Register the provider by its fully qualified class name in the pygeoapi config.
All options of the built-in `MVT-postgresql` provider apply unchanged; the only
addition is `layer`.

```yaml
resources:
  waterways:
    type: collection
    title: Waterways
    description: Waterways as vector tiles
    extents:
      spatial:
        bbox: [4.0, 57.9, 31.1, 71.2]
        crs: http://www.opengis.net/def/crs/OGC/1.3/CRS84
    providers:
      - type: tile
        name: mvt_postgresql_dibk.MVTPostgreSQLProvider
        data:
          host: localhost
          port: 5432
          dbname: gis
          user: pygeoapi
          password: ${POSTGRES_PASSWORD}
          search_path: [public]
        id_field: id
        table: waterways
        geom_field: geom
        layer: waterways          # optional, defaults to `table`
        storage_crs: http://www.opengis.net/def/crs/EPSG/0/25833
        options:
          zoom:
            min: 0
            max: 15
        format:
          name: pbf
          mimetype: application/vnd.mapbox-vector-tile
```

### `layer`

The layer name written into the MVT and reported in TileJSON `vector_layers`. This
is what a client style refers to as `source-layer`. The base provider hardcodes the
table name, which makes the style depend on the physical table; setting `layer`
decouples the two.

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MVT_CACHE_DAYS` | `1` | Time-to-live, in days, for cached tiles and for the cached layer extent. See [Expiry](#expiry). |

## Caching

Two `cachetools.TTLCache` instances live at module level, one for tiles and one for
the layer extent. Module level is deliberate: pygeoapi constructs a new provider
instance per request, so a cache held on the instance would never be hit.

* **Tiles** are keyed by `<database>/<schema>/<table>/<tileset>/<z>/<y>/<x>.pbf`,
  so collections sharing a process never share entries. Empty tiles are cached too,
  which keeps empty areas of the map from re-querying PostGIS on every request.
* **The layer extent** is keyed by `<database>/<schema>/<table>` and backs the
  TileJSON `bounds` and `center`. It requires a full-table `ST_Extent`, so it is
  worth not repeating per metadata request.

### Size limits

The tile cache is bounded by **memory**, not by entry count: it is built with
`getsizeof=len`, so `maxsize` counts the bytes of the cached tiles. The budget is
`_MVT_CACHE_BYTES` in `src/mvt_postgresql_dibk/__init__.py`, currently **256 MiB**.
Once it is full, the least recently used tiles are evicted to make room. Key and
dict overhead — a few hundred bytes per entry — is not counted, so actual resident
memory is somewhat higher.

The extent cache is bounded by entry count, **250**. Each entry is four floats, so
this is a limit on how many collections a process can keep extents for, not a
memory limit.

Both limits are per worker process. The cache is not shared between workers: with
*N* pygeoapi workers a tile may be rendered up to *N* times before every worker has
it, and each worker carries its own 256 MiB budget. Restarting pygeoapi empties it.

### Expiry

Entries are only evicted on TTL expiry or, for tiles, on memory pressure — there is
no explicit invalidation hook. `MVT_CACHE_DAYS` is therefore the maximum staleness
of what is served.

Note that the TTL runs from when each tile was *inserted*, not from a wall-clock
boundary. If the underlying tables are refreshed nightly, a tile cached shortly
before the refresh is served for nearly a full day afterwards, while a tile cached
shortly after it expires promptly. With the default of one day, worst-case
staleness is just under 24 hours. Lower `MVT_CACHE_DAYS` if that matters.

## TileJSON metadata

`GET /collections/{collection}/tiles/{tileMatrixSetId}/metadata?f=tilejson` returns
a [TileJSON 3.0.0](https://github.com/mapbox/tilejson-spec/tree/master/3.0.0)
document that MapLibre GL and Mapbox GL can consume directly as a vector source:

```json
{
  "tilejson": "3.0.0",
  "name": "Waterways",
  "description": "Waterways as vector tiles",
  "version": "1.0.0",
  "scheme": "tms",
  "tiles": ["http://localhost:5000/collections/waterways/tiles/WebMercatorQuad/{tileMatrix}/{tileRow}/{tileCol}?f=mvt"],
  "minzoom": 0,
  "maxzoom": 15,
  "bounds": [4.0, 57.9, 31.1, 71.2],
  "center": [17.55, 64.55, 0],
  "vector_layers": [
    {
      "id": "waterways",
      "description": "Waterways as vector tiles",
      "fields": {"id": "Number", "name": "String", "navigable": "Boolean"}
    }
  ]
}
```

`bounds` come from `ST_Extent` over the geometry column, transformed to WGS 84, and
`center` is its midpoint at the minimum zoom. Both are omitted when the table holds
no geometries.

Column types are mapped to the three types TileJSON allows:

| PostgreSQL / SQLAlchemy type | TileJSON |
| --- | --- |
| `Integer`, `Float`, `Numeric` | `Number` |
| `Boolean` | `Boolean` |
| `String`, `Date`, `DateTime`, `UUID`, `JSON`, `JSONB`, anything else | `String` |

The other metadata formats (`f=json`, `f=html`, `f=jsonld`) are inherited from the
base provider and behave as documented by pygeoapi.

## Tiling schemes

`WebMercatorQuad` and `WorldCRS84Quad`, inherited from the base provider.
Geometries are filtered in the storage CRS so an existing spatial index on the
geometry column is used, then transformed to the tileset CRS for encoding.

## Layout

```
src/mvt_postgresql_dibk/__init__.py   the provider, the two cached query functions
```

## License

See `LICENSE`.
