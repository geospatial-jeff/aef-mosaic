# geoembeddings-mosaic

Mosaic geo-embedding Cloud-Optimized GeoTIFFs into a contiguous Zarr V3 array.
Supports multiple embedding datasets behind a shared core:

- **[AEF](https://source.coop/tge-labs/aef/README.md)** — AlphaEarth Foundations, `int8` (quantized), discovered from a precomputed parquet index.
- **[Spheer](https://huggingface.co/datasets/spheer/spheer-fm-embeddings)** — Spheer FM, `float32`, discovered by scanning a folder of COGs.

The dataset is selected by the `dataset:` config field, which determines the element
type, tile discovery method, and output metadata. The COG-read → mosaic → GeoZarr-write
core is shared across datasets.

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

`cargo test -- --ignored` runs the network/gated integration tests (e.g. the Spheer
HuggingFace read), which require an `HF_TOKEN`.

## Datasets

The `dataset:` field selects dataset-specific behavior:

| dataset  | dtype   | bands | default CRS  | discovery                         |
|----------|---------|-------|--------------|-----------------------------------|
| `aef`    | int8    | 64    | EPSG:4326    | parquet index (`input.index_path`) |
| `spheer` | float32 | 100   | EPSG:32631   | COG folder scan (`input.index_path` prefix) |

Tiles are read from the store named by `input.cog_bucket`:

- an S3 bucket name (e.g. `us-west-2.opendata.source.coop`) → anonymous S3 read, or
- `hf://owner/name` (e.g. `hf://spheer/spheer-fm-embeddings`) → authenticated HuggingFace
  read using the `HF_TOKEN` environment variable.

For `aef`, `input.index_path` is the parquet index (`s3://…` or a local path). For
`spheer`, `input.index_path` is the key **prefix** to scan for `*.tif`/`*.tiff` under the
COG store (empty string scans the whole store); each COG header supplies its dimensions,
geotransform, and CRS, and the year is parsed from the file path.

## Configuration

Generate a sample config with `pixi run mosaic generate-config -o config.yaml`. Full
reference below (values shown are the defaults):

```yaml
# Which dataset to mosaic: "aef" (int8, parquet index) or "spheer" (float32, folder scan)
dataset: aef

# === INPUT: where to read COG tiles from ===
input:
  # aef: parquet index path (s3:// or local).
  # spheer: key prefix to scan for COGs within the cog_bucket store ("" scans all).
  index_path: "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
  # S3 bucket, or "hf://owner/name" for a (gated) HuggingFace dataset repo.
  cog_bucket: "us-west-2.opendata.source.coop"

# === OUTPUT: where to write the Zarr array ===
# Choose ONE of: local_path OR bucket+prefix
output:
  # Option 1: local filesystem
  local_path: "/tmp/mosaic.zarr"

  # Option 2: S3 (comment out local_path, uncomment these)
  # bucket: "output-bucket"
  # prefix: "zarr/mosaic"

  crs: "EPSG:4326"           # aef: EPSG:4326; spheer: EPSG:32631
  resolution: 0.0000898      # CRS units (degrees for EPSG:4326 ~10m; meters for projected)
  num_bands: 64              # embedding dimensions (aef: 64, spheer: 100); must equal chunk_shape.embedding

  # chunk_shape is the INNER chunk. When sharding is enabled the shard size is
  # chunk_shape * sharding.shard_shape (so 256 * 16 = 4096 px shards by default).
  chunk_shape:
    time: 1                  # must be 1 (one year per chunk)
    embedding: 64            # must equal num_bands (chunks always span all bands)
    height: 256
    width: 256

  # Sharding is enabled by default (Zarr V3 sharding codec).
  sharding:
    enabled: true
    shard_shape: [16, 16]    # chunks per shard [rows, cols]

  compression_level: 3       # zstd 0-22

  # Optional: explicit output years for the time dimension. Lets multiple VMs write
  # different time slices of the same array. If omitted, derived from discovery/filter.
  # years: [2024]

# === PROCESSING: performance tuning ===
processing:
  fetch_concurrency: 8       # concurrent COG fetches (network I/O)
  mosaic_concurrency: 8      # concurrent mosaic/reproject (CPU)
  write_concurrency: 8       # concurrent Zarr writes (I/O)
  max_concurrent_http: 128   # cap on concurrent HTTP requests across all workers
  tile_cache_enabled: true
  tile_cache_gb: 32.0        # decoded tile cache in GB
  metadata_cache_entries: 50000
  enable_metrics: true
  metrics_interval_secs: 10
  # metrics_output_path: "metrics.json"
  checkpoint:
    enabled: true            # resumable processing
    interval_secs: 60
    # prefix: "2024"         # e.g. per-year checkpoint file for multi-VM runs

# === FILTER: limit processing area and years (optional) ===
# filter:
#   bounds: [-122.6, 37.2, -121.8, 37.9]  # WGS84 [min_lon, min_lat, max_lon, max_lat]
#   years: [2024]
```

### Minimal AEF config

Input defaults to AEF on source.coop. For local output you only need:

```yaml
output:
  local_path: "/tmp/aef-mosaic.zarr"
```

### Minimal Spheer config

```yaml
dataset: spheer
input:
  index_path: ""                              # scan the whole repo for COGs
  cog_bucket: "hf://spheer/spheer-fm-embeddings"   # needs HF_TOKEN
output:
  bucket: "your-output-bucket"
  prefix: "spheer-mosaic"
  crs: "EPSG:32631"
  resolution: 10.0
  num_bands: 100
  chunk_shape:
    time: 1
    embedding: 100
    width: 128
    height: 128
```

To disable sharding (legacy mode with smaller chunks):

```yaml
output:
  local_path: "/tmp/mosaic.zarr"
  chunk_shape:
    height: 1024
    width: 1024
  sharding:
    enabled: false
```

## Output Format

The pipeline produces a Zarr V3 array with shape `(time, band, y, x)`:

- **Data type**: dataset-dependent — `int8` for AEF (`-128` = NoData), `float32` for Spheer (`NaN` = NoData)
- **Compression**: Zstd
- **Sharding**: enabled by default (4096×4096 shards with 256×256 inner chunks for AEF)
- **Coordinate arrays**: `/x`, `/y`, `/time` for xarray compatibility

Geospatial attributes follow both CF Conventions and the GeoZarr `proj:`, `spatial:`, and
`geoemb:` namespaces:

```python
import xarray as xr

ds = xr.open_zarr("/tmp/mosaic.zarr")
print(ds.attrs["proj:code"])          # e.g. "EPSG:4326" (AEF) or "EPSG:32631" (Spheer)
print(ds.attrs["spatial:transform"])  # affine transform
print(ds.embeddings.dtype)            # int8 (AEF) or float32 (Spheer)
```

## async-tiff patch

COG decoding uses [`async-tiff`](https://github.com/developmentseed/async-tiff), vendored
under `vendor-async-tiff/` and wired in via a `[patch]` in `Cargo.toml`. The vendored copy
carries a fix to the floating-point predictor for tiles whose width is not a multiple of
the internal tile size (needed for the Spheer `float32`, predictor-3 COGs).
