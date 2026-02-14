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
pub use pipeline::{ChunkProcessor, Metrics, MetricsReporter, Pipeline, PipelineConfig, Scheduler, SchedulerConfig};
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

    // Load input index (reuse cog_store if loading from same bucket)
    tracing::info!("Loading tile index from {}", config.input.index_path);
    let input_index = if config.input.index_path.starts_with("s3://") {
        let (_bucket, key) = io::parse_s3_uri(&config.input.index_path)?;
        // Reuse the cog_store instead of creating a new connection pool
        let path = object_store::path::Path::from(key);
        InputIndex::from_s3(cog_store.clone(), &path).await?
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

    // Collect chunks to process (pre-filter empty chunks)
    let mut chunks: Vec<_> = output_grid
        .enumerate_chunks()
        .filter(|chunk| spatial_lookup.chunk_has_data(chunk))
        .collect();

    // Sort by row first, then column (row-major order) for cache locality
    chunks.sort_by_key(|c| (c.row_idx, c.col_idx));

    let total_chunks = chunks.len();
    tracing::info!(
        "Work estimate: {} chunks with data (filtered from {} total)",
        total_chunks,
        output_grid.num_chunks()
    );

    // Create pipeline config
    let pipeline_config = PipelineConfig {
        fetch_concurrency: config.processing.concurrency,
        mosaic_concurrency: 8.min(config.processing.concurrency / 2).max(1),
        write_concurrency: 16.min(config.processing.concurrency / 2).max(1),
        fetch_buffer: 16,
        mosaic_buffer: 8,
    };

    tracing::info!(
        "Pipeline config: fetch={}, mosaic={}, write={}",
        pipeline_config.fetch_concurrency,
        pipeline_config.mosaic_concurrency,
        pipeline_config.write_concurrency
    );

    // Create decoupled pipeline
    let pipeline = Pipeline::new(
        cog_reader,
        zarr_writer.clone(),
        spatial_lookup,
        output_grid,
        metrics.clone(),
        pipeline_config,
    );

    // Start metrics reporter if enabled
    let (shutdown_tx, shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let reporter_handle = if config.processing.enable_metrics {
        let reporter = MetricsReporter::new(
            metrics.clone(),
            config.processing.metrics_interval_secs,
            total_chunks as u64,
        );
        Some(tokio::spawn(reporter.run(shutdown_rx)))
    } else {
        drop(shutdown_rx);
        None
    };

    // Run the pipeline
    tracing::info!("Starting chunk processing...");
    let pipeline_stats = pipeline.run(chunks).await?;

    // Shutdown metrics reporter
    let _ = shutdown_tx.send(()).await;
    if let Some(handle) = reporter_handle {
        let _ = handle.await;
    }

    // Print final summary
    if config.processing.enable_metrics {
        let reporter = MetricsReporter::new(
            metrics.clone(),
            config.processing.metrics_interval_secs,
            total_chunks as u64,
        );
        reporter.print_summary();

        if let Some(ref path) = config.processing.metrics_output_path {
            let snapshot = metrics.snapshot();
            if let Err(e) = snapshot.save_to_file(path) {
                tracing::warn!("Failed to save metrics to {}: {}", path, e);
            }
        }
    }

    // Finalize
    zarr_writer.finalize()?;

    // Convert pipeline stats to scheduler stats for backward compatibility
    let stats = pipeline::SchedulerStats {
        total_chunks: pipeline_stats.total_chunks,
        chunks_processed: pipeline_stats.chunks_processed,
        chunks_skipped: pipeline_stats.chunks_skipped,
        chunks_failed: 0, // Pipeline doesn't track failures separately
    };

    tracing::info!("Pipeline complete: {}", stats);

    Ok(stats)
}
