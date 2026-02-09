//! AEF Mosaic Pipeline CLI
//!
//! High-performance pipeline to mosaic AEF COG files into Zarr arrays.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aef_mosaic::{build_runtime, init_rayon, run_pipeline, Config};

#[derive(Parser)]
#[command(name = "aef-mosaic")]
#[command(about = "Mosaic AEF COG files into Zarr arrays", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.yaml", global = true)]
    config: PathBuf,

    /// Override concurrency level
    #[arg(long, global = true)]
    concurrency: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the mosaic pipeline (default if no command specified)
    Run,

    /// Analyze the work without processing
    Analyze,

    /// Validate configuration
    Validate,

    /// Generate a sample configuration file
    GenerateConfig {
        /// Output path for configuration file
        #[arg(short, long, default_value = "config.yaml")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Run) => {
            run_command(cli.config, cli.concurrency, false)?;
        }

        Some(Commands::Analyze) => {
            analyze_command(cli.config)?;
        }

        Some(Commands::Validate) => {
            validate_command(cli.config)?;
        }

        Some(Commands::GenerateConfig { output }) => {
            generate_config_command(output)?;
        }
    }

    Ok(())
}

fn run_command(config_path: PathBuf, concurrency: Option<usize>, dry_run: bool) -> Result<()> {
    let mut config = Config::from_file(&config_path)?;

    // Apply overrides
    if let Some(c) = concurrency {
        config.processing.concurrency = c;
    }

    config.validate()?;

    if dry_run {
        tracing::info!("Dry run mode - analyzing work without processing");
        return analyze_work(&config);
    }

    // Initialize Rayon
    init_rayon(config.processing.rayon_threads)?;

    // Build and run Tokio runtime
    let runtime = build_runtime(config.processing.worker_threads)?;
    runtime.block_on(async { run_pipeline(config).await })?;

    Ok(())
}

fn analyze_command(config_path: PathBuf) -> Result<()> {
    let config = Config::from_file(&config_path)?;
    config.validate()?;
    analyze_work(&config)
}

fn analyze_work(config: &Config) -> Result<()> {
    let runtime = build_runtime(None)?;

    runtime.block_on(async {
        use aef_mosaic::{io, InputIndex, OutputGrid, SpatialLookup};
        use std::sync::Arc;

        // Load index
        tracing::info!("Loading tile index from {}", config.input.index_path);
        let input_index = if config.input.index_path.starts_with("s3://") {
            let (bucket, key) = io::parse_s3_uri(&config.input.index_path)?;
            let store = io::create_object_store(bucket, &config.aws)?;
            let path = object_store::path::Path::from(key);
            InputIndex::from_s3(store, &path).await?
        } else {
            InputIndex::from_local_parquet(&config.input.index_path)?
        };
        tracing::info!("Loaded {} tiles", input_index.len());

        // Apply filter if specified
        let input_index = if let Some(filter) = &config.filter {
            input_index.filter(
                filter.bounds.as_ref(),
                filter.years.as_deref(),
            )
        } else {
            input_index
        };

        if input_index.is_empty() {
            anyhow::bail!("No tiles match the filter criteria");
        }

        // Get bounds from filtered tiles (or use filter bounds if specified)
        let bounds = match config.filter.as_ref().and_then(|f| f.bounds) {
            Some(filter_bounds) => filter_bounds,
            None => input_index.bounds_wgs84()
                .ok_or_else(|| anyhow::anyhow!("No tiles in index"))?,
        };

        let input_index = Arc::new(input_index);

        // Create output grid
        let output_grid = Arc::new(OutputGrid::new(
            bounds,
            config.output.crs.clone(),
            config.output.resolution,
            1,
            2024,
            64,
            config.output.chunk_shape.clone(),
        )?);

        // Create spatial lookup
        let spatial_lookup = Arc::new(SpatialLookup::new(input_index, output_grid.clone())?);

        // Print analysis
        println!("\n=== Work Analysis ===");
        println!("Input tiles: {}", spatial_lookup.input_index().len());
        println!(
            "Input bounds: [{:.4}, {:.4}, {:.4}, {:.4}]",
            bounds[0], bounds[1], bounds[2], bounds[3]
        );
        println!(
            "Output grid: {}x{} pixels",
            output_grid.width, output_grid.height
        );
        println!("Output chunks: {}", output_grid.num_chunks());
        println!(
            "Chunk shape: {}x{}x{}x{}",
            config.output.chunk_shape.time,
            config.output.chunk_shape.embedding,
            config.output.chunk_shape.height,
            config.output.chunk_shape.width
        );

        let coverage = spatial_lookup.coverage_stats();
        println!("\n=== Coverage ===");
        println!("{}", coverage);

        // Estimate data size
        let chunk_size_bytes = config.output.chunk_shape.time
            * config.output.chunk_shape.embedding
            * config.output.chunk_shape.height
            * config.output.chunk_shape.width;
        let estimated_output_size =
            coverage.chunks_with_data * chunk_size_bytes;
        let estimated_input_size =
            (coverage.chunks_with_data as f64 * coverage.avg_tiles_per_chunk * chunk_size_bytes as f64) as usize;

        println!("\n=== Size Estimates ===");
        println!(
            "Chunk size: {:.1} MB (uncompressed)",
            chunk_size_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Estimated input read: {:.1} GB",
            estimated_input_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "Estimated output size: {:.1} GB (uncompressed)",
            estimated_output_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "Estimated output size: {:.1} GB (compressed ~3:1)",
            estimated_output_size as f64 / (1024.0 * 1024.0 * 1024.0) / 3.0
        );

        // Estimate time
        let target_throughput_gbps = 25.0;
        let estimated_time_secs =
            estimated_input_size as f64 / (target_throughput_gbps * 1024.0 * 1024.0 * 1024.0);
        println!("\n=== Time Estimate (at {} GB/s) ===", target_throughput_gbps);
        println!("Estimated time: {:.0} seconds ({:.1} minutes)", estimated_time_secs, estimated_time_secs / 60.0);

        println!("=====================\n");

        Ok(())
    })
}

fn validate_command(config_path: PathBuf) -> Result<()> {
    let config = Config::from_file(&config_path)?;
    config.validate()?;
    println!("Configuration is valid");
    Ok(())
}

fn generate_config_command(output: PathBuf) -> Result<()> {
    // Generate a commented YAML config
    let yaml = r#"# AEF Mosaic Pipeline Configuration

# === INPUT: Where to read COG tiles from ===
input:
  # Path to parquet index file
  # Default: AEF v1 index on source.coop (public, no credentials needed)
  index_path: "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"

  # S3 bucket containing the COG files
  cog_bucket: "us-west-2.opendata.source.coop"

# === OUTPUT: Where to write the Zarr array ===
# Choose ONE of: local_path (local disk) OR bucket+prefix (S3)
output:
  # Option 1: Write to local filesystem
  local_path: "/tmp/aef-mosaic.zarr"

  # Option 2: Write to S3 (comment out local_path and uncomment these)
  # bucket: "output-bucket"
  # prefix: "zarr/aef-mosaic"

  # Output CRS - EPSG:6933 is EASE-Grid 2.0 (equal-area, global)
  # Every pixel represents the same area regardless of latitude
  crs: "EPSG:6933"

  # Pixel size in CRS units (meters for EPSG:6933)
  resolution: 10.0

  # Zarr chunk dimensions: (time, embedding, height, width)
  chunk_shape:
    time: 1         # One year per chunk (enables incremental updates)
    embedding: 64   # Full embedding dimension
    height: 1024    # ~10km at 10m resolution
    width: 1024

  # Zstd compression level (0-22, higher = smaller but slower)
  compression_level: 3

  # Enable Zarr V3 sharding (reduces S3 object count)
  use_sharding: false

  # Chunks per shard [height, width] if sharding enabled
  shard_shape: [8, 8]

# === PROCESSING: Performance tuning ===
processing:
  # Number of output chunks to process concurrently
  concurrency: 256

  # Max concurrent COG fetches per chunk
  cog_fetch_concurrency: 8

  # Tokio async worker threads (null = num CPUs)
  # worker_threads: 64

  # Rayon thread pool size for CPU work (null = num CPUs)
  # rayon_threads: 64

  # Print throughput metrics during processing
  enable_metrics: true

  # Metrics reporting interval in seconds
  metrics_interval_secs: 10

  # Retry configuration for transient S3 failures
  retry:
    max_retries: 3
    initial_backoff_ms: 100
    max_backoff_ms: 10000

# === AWS: S3 connection settings ===
aws:
  # AWS region (source.coop is in us-west-2)
  region: "us-west-2"

  # Use S3 Express One Zone (lower latency, if available)
  use_express: false

  # Custom S3 endpoint (for LocalStack, MinIO, etc.)
  # endpoint_url: "http://localhost:4566"

  # Use EC2 instance profile for credentials (for output bucket)
  # Note: source.coop input bucket is public, no credentials needed
  use_instance_profile: true

# === FILTER: Limit processing to a subset (optional) ===
# Uncomment to process only a specific area or time range.
# Useful for testing on a small area before full production runs.

# filter:
#   # Bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat]
#   # Example: San Francisco Bay Area
#   bounds: [-122.6, 37.2, -121.8, 37.9]
#
#   # Years to process (empty or omit = all years)
#   years: [2024]
"#;

    std::fs::write(&output, yaml)?;
    println!("Generated sample configuration at: {}", output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_default() {
        // No subcommand - should default to Run
        let cli = Cli::try_parse_from(["aef-mosaic"]);
        assert!(cli.is_ok());
        assert!(cli.unwrap().command.is_none());
    }

    #[test]
    fn test_cli_parse_with_config() {
        let cli = Cli::try_parse_from(["aef-mosaic", "-c", "other.yaml"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_validate() {
        let cli = Cli::try_parse_from(["aef-mosaic", "validate", "-c", "test.json"]);
        assert!(cli.is_ok());
    }
}
