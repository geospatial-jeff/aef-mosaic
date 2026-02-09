//! AEF Mosaic Pipeline
//!
//! High-performance pipeline to mosaic 235K+ AEF COG files into a contiguous Zarr array,
//! targeting 25-30+ GB/s throughput on a single EC2 node.
//!
//! # Architecture
//!
//! The pipeline consists of:
//!
//! - **Index**: Tile index management with R-tree spatial queries
//! - **I/O**: Async COG reading and Zarr writing using object_store
//! - **Transform**: UTM→WGS84 reprojection and mean mosaicing
//! - **Pipeline**: Concurrent chunk processing with metrics
//!
//! # Usage
//!
//! ```no_run
//! use aef_mosaic::{Config, run_pipeline};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = Config::from_file(&"config.json".into())?;
//!     run_pipeline(config).await?;
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod crs;
pub mod index;
pub mod io;
pub mod pipeline;
pub mod transform;

pub use config::{Config, FilterConfig};
pub use index::{InputIndex, OutputGrid, SpatialLookup};
pub use io::{CogReader, ZarrWriter};
pub use pipeline::{ChunkProcessor, Metrics, Scheduler, SchedulerConfig};
pub use transform::MosaicAccumulator;

use anyhow::Result;
use std::sync::Arc;

/// Run the full mosaic pipeline with the given configuration.
pub async fn run_pipeline(config: Config) -> Result<pipeline::SchedulerStats> {
    // Validate configuration
    config.validate()?;

    let config = Arc::new(config);

    // Initialize tracing
    tracing::info!("Starting AEF Mosaic Pipeline");
    tracing::info!("Configuration loaded");

    // Create object stores
    let cog_store = io::create_cog_store(&config)?;
    let output_store = io::create_output_store(&config)?;

    // Load input index
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

    tracing::info!(
        "Processing bounds: [{:.4}, {:.4}, {:.4}, {:.4}]",
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3]
    );

    // Create output grid
    let output_grid = Arc::new(OutputGrid::new(
        bounds,
        config.output.crs.clone(),
        config.output.resolution,
        config.output.num_years,
        config.output.start_year,
        config.output.num_bands,
        config.output.chunk_shape.clone(),
    )?);

    tracing::info!(
        "Output grid: {}x{} pixels, {} chunks",
        output_grid.width,
        output_grid.height,
        output_grid.num_chunks()
    );

    // Create spatial lookup
    let spatial_lookup = Arc::new(SpatialLookup::new(input_index.clone(), output_grid.clone())?);

    // Print coverage stats
    let coverage = spatial_lookup.coverage_stats();
    tracing::info!("{}", coverage);

    // Create Zarr writer
    tracing::info!("Writing Zarr output to: {}", config.output.path_display());

    let output_prefix = io::get_output_prefix(&config);
    let zarr_writer = Arc::new(
        ZarrWriter::create(output_store.clone(), output_prefix, output_grid.clone(), &config).await?,
    );

    // Create metrics
    let metrics = Metrics::new();

    // Create COG reader with metrics for cache tracking
    let cog_reader = Arc::new(CogReader::with_metrics(cog_store, metrics.clone()));

    // Create chunk processor
    // Note: Reprojector is created per-task inside spawn_blocking (Proj is not Send)
    let processor = Arc::new(ChunkProcessor::new(
        cog_reader,
        zarr_writer.clone(),
        spatial_lookup.clone(),
        metrics.clone(),
        config.clone(),
    ));

    // Create scheduler
    let scheduler_config = SchedulerConfig {
        concurrency: config.processing.concurrency,
        skip_empty: true,
        enable_metrics: config.processing.enable_metrics,
        metrics_interval_secs: config.processing.metrics_interval_secs,
        spatial_ordering: true,
        enable_metatiling: true,
        metatile_size: config.processing.metatile_size,
        enable_prefetch: config.processing.enable_prefetch,
        metrics_output_path: config.processing.metrics_output_path.clone(),
    };

    let scheduler = Scheduler::new(
        processor,
        spatial_lookup,
        output_grid,
        metrics,
        scheduler_config,
    );

    // Print work estimate
    let estimate = scheduler.estimate_total_work();
    tracing::info!("Work estimate: {}", estimate);

    // Run the pipeline
    tracing::info!("Starting chunk processing...");
    let stats = scheduler.run().await?;

    // Finalize
    zarr_writer.finalize()?;

    tracing::info!("Pipeline complete: {}", stats);

    Ok(stats)
}

/// Build a Tokio runtime with the specified configuration.
pub fn build_runtime(worker_threads: Option<usize>) -> Result<tokio::runtime::Runtime> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();

    if let Some(threads) = worker_threads {
        builder.worker_threads(threads);
    }

    builder.enable_all();

    Ok(builder.build()?)
}

/// Initialize the Rayon thread pool.
pub fn init_rayon(threads: Option<usize>) -> Result<()> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    Ok(())
}
