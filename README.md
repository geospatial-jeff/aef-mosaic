# aef-mosaic

Mosaic AEF[https://source.coop/tge-labs/aef/README.md] embeddings into a contiguous Zarr array.

## Quick Start

```bash
# Install dependencies and build
pixi install
pixi run build

# Generate a sample config
./target/release/aef-mosaic generate-config -o config.yaml

# Edit config.yaml, then run
./target/release/aef-mosaic run -c config.yaml
```

## Commands

```bash
# Run the full pipeline
aef-mosaic run -c config.yaml

# Run with custom concurrency
aef-mosaic run -c config.yaml --concurrency 512

# Analyze input data and output grid (without processing)
aef-mosaic analyze -c config.yaml

# Validate configuration
aef-mosaic validate -c config.yaml

# Generate a sample configuration file
aef-mosaic generate-config -o config.yaml
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

  crs: "EPSG:6933"           # EASE-Grid 2.0 (equal-area)
  resolution: 10.0           # meters
  num_years: 1               # time dimension size
  start_year: 2024           # first year
  num_bands: 64              # embedding dimensions

  chunk_shape:
    time: 1
    embedding: 64
    height: 1024
    width: 1024

  compression_level: 3       # zstd 0-22
  use_sharding: false
  shard_shape: [8, 8]

# === PROCESSING: Performance tuning ===
processing:
  concurrency: 256           # concurrent chunks
  cog_fetch_concurrency: 8   # COG fetches per chunk
  metatile_size: 32          # spatial locality grouping
  tile_cache_gb: 32.0        # decoded tile cache
  metadata_cache_entries: 10000
  enable_metrics: true
  metrics_interval_secs: 10
  # worker_threads: 64       # Tokio threads (default: num CPUs)
  # rayon_threads: 64        # Rayon threads (default: num CPUs)

  retry:
    max_retries: 3
    initial_backoff_ms: 100
    max_backoff_ms: 10000

# === AWS: S3 connection settings ===
aws:
  region: "us-west-2"
  use_express: false
  use_instance_profile: true
  # endpoint_url: "http://localhost:4566"  # for LocalStack/MinIO

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