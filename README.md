# aef-mosaic

Mosaic [AEF](https://source.coop/tge-labs/aef/README.md) embeddings into a contiguous Zarr array.

## Quick Start

```bash
# Install dependencies and build
pixi install
pixi run build

# Generate a sample config
pixi run mosaic generate-config -o config.yaml

# Edit config.yaml, then run
pixi run mosaic run -c config.yaml
```

## Commands

```bash
# Run the full pipeline
pixi run mosaic run -c config.yaml

# Run with custom concurrency
pixi run mosaic run -c config.yaml --concurrency 512

# Analyze input data and output grid (without processing)
pixi run analyze -c config.yaml

# Validate configuration
pixi run validate -c config.yaml

# Generate a sample configuration file
pixi run mosaic generate-config -o config.yaml
```

## Development

```bash
pixi run check   # Type check
pixi run test    # Run tests
pixi run build   # Build release binary
```

## Configuration

Generate a sample config with `aef-mosaic generate-config -o config.yaml`. Full reference below:

```yaml
# === INPUT: Where to read COG tiles from ===
# Defaults to AEF v1 on source.coop (public, no credentials needed)
input:
  index_path: "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
  cog_bucket: "us-west-2.opendata.source.coop"

# === OUTPUT: Where to write the Zarr array ===
# Choose ONE of: local_path OR bucket+prefix
output:
  # Option 1: Local filesystem
  local_path: "/tmp/aef-mosaic.zarr"

  # Option 2: S3 (comment out local_path, uncomment these)
  # bucket: "output-bucket"
  # prefix: "zarr/aef-mosaic"

  crs: "EPSG:4326"           # WGS84 geographic (pixels shrink E-W toward poles)
  resolution: 0.00009        # degrees (~10m at equator)
  num_years: 1               # time dimension size
  start_year: 2024           # first year
  num_bands: 64              # embedding dimensions

  # chunk_shape defines the SHARD size when sharding is enabled (default)
  chunk_shape:
    time: 1         # One year per chunk
    embedding: 64   # Full embedding dimension
    height: 4096    # Shard size (~40km at 10m resolution)
    width: 4096

  # Sharding is enabled by default for better I/O performance
  sharding:
    enabled: true              # Default: true
    subchunk_shape: [256, 256] # Inner chunk dimensions

  compression_level: 3       # zstd 0-22

# === PROCESSING: Performance tuning ===
processing:
  fetch_concurrency: 8       # concurrent COG fetches (network I/O)
  mosaic_concurrency: 8      # concurrent mosaic/reproject (CPU)
  write_concurrency: 8       # concurrent Zarr writes (I/O)
  tile_cache_gb: 32.0        # decoded tile cache in GB
  metadata_cache_entries: 10000
  enable_metrics: true
  metrics_interval_secs: 10

  retry:
    max_retries: 3
    initial_backoff_ms: 100
    max_backoff_ms: 10000

# === FILTER: Limit processing area (optional) ===
# filter:
#   bounds: [-122.6, 37.2, -121.8, 37.9]  # [min_lon, min_lat, max_lon, max_lat]
#   years: [2024]
```

### Minimal Config

Input defaults to AEF on source.coop. For local output, you only need:

```yaml
output:
  local_path: "/tmp/aef-mosaic.zarr"

aws:
  region: "us-west-2"
```

For S3 output:

```yaml
output:
  bucket: "your-output-bucket"
  prefix: "aef-mosaic"

aws:
  region: "us-west-2"
```

To disable sharding (legacy mode with smaller chunks):

```yaml
output:
  local_path: "/tmp/aef-mosaic.zarr"
  chunk_shape:
    height: 1024
    width: 1024
  sharding:
    enabled: false
```

## Output Format

The pipeline produces a Zarr V3 array with shape `(time, band, y, x)`:

- **Data type**: Int8 embeddings (-128 = NoData)
- **Compression**: Zstd
- **Sharding**: Enabled by default (4096x4096 shards with 256x256 inner chunks)
- **Coordinate arrays**: `/x`, `/y`, `/time` for xarray compatibility

Geospatial attributes follow both CF Conventions and GeoZarr `proj:` namespace:

```python
import xarray as xr

ds = xr.open_zarr("/tmp/aef-mosaic.zarr")
print(ds.embeddings.attrs['crs'])        # "EPSG:4326"
print(ds.embeddings.attrs['proj:code'])  # "EPSG:4326" (GeoZarr)
```