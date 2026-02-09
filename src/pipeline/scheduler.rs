//! Work distribution and scheduling for chunk processing.
//!
//! The scheduler distributes output chunks across async tasks with configurable concurrency.
//! It supports two processing modes:
//! 1. Standard: Process all chunks with buffer_unordered for maximum throughput
//! 2. Meta-tiled: Group chunks into spatial windows and process one window at a time
//!    for better cache locality

use crate::index::{OutputChunk, OutputGrid, SpatialLookup};
use crate::pipeline::{ChunkProcessor, ChunkResult, Metrics, MetricsReporter};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of concurrent chunk processors
    pub concurrency: usize,

    /// Whether to skip empty chunks upfront
    pub skip_empty: bool,

    /// Enable progress reporting
    pub enable_metrics: bool,

    /// Metrics reporting interval in seconds
    pub metrics_interval_secs: u64,

    /// Enable spatial chunk ordering for better cache locality
    pub spatial_ordering: bool,

    /// Enable meta-tiling for improved cache locality
    /// When enabled, chunks are grouped into spatial windows and processed together
    pub enable_metatiling: bool,

    /// Size of meta-tiles (NxN output chunks per meta-tile)
    /// Only used if enable_metatiling is true
    pub metatile_size: usize,

    /// Enable prefetching all tiles for a metatile before processing
    /// When true: fewer, larger S3 requests but processing waits for prefetch
    /// When false: more, smaller requests but I/O overlaps with compute
    pub enable_prefetch: bool,

    /// Optional path to save metrics JSON after run completes
    pub metrics_output_path: Option<String>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            concurrency: 256,
            skip_empty: true,
            enable_metrics: true,
            metrics_interval_secs: 10,
            spatial_ordering: true,
            enable_metatiling: true,
            metatile_size: 32,
            enable_prefetch: false,
            metrics_output_path: None,
        }
    }
}

/// A spatial window of output chunks (meta-tile).
///
/// Meta-tiles group adjacent chunks together for processing, ensuring that
/// tiles needed by chunks in the same window are likely to be cached.
#[derive(Debug)]
pub struct MetaTile {
    /// Starting row index of this meta-tile
    pub row_start: usize,
    /// Starting column index of this meta-tile
    pub col_start: usize,
    /// Size of this meta-tile (NxN chunks)
    pub size: usize,
    /// Chunks in this meta-tile
    pub chunks: Vec<OutputChunk>,
}

/// Scheduler for distributing chunk processing across async tasks.
pub struct Scheduler {
    /// Chunk processor
    processor: Arc<ChunkProcessor>,

    /// Spatial lookup for pre-filtering
    spatial_lookup: Arc<SpatialLookup>,

    /// Output grid
    output_grid: Arc<OutputGrid>,

    /// Metrics
    metrics: Arc<Metrics>,

    /// Configuration
    config: SchedulerConfig,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(
        processor: Arc<ChunkProcessor>,
        spatial_lookup: Arc<SpatialLookup>,
        output_grid: Arc<OutputGrid>,
        metrics: Arc<Metrics>,
        config: SchedulerConfig,
    ) -> Self {
        Self {
            processor,
            spatial_lookup,
            output_grid,
            metrics,
            config,
        }
    }

    /// Run the scheduler to process all chunks.
    ///
    /// If meta-tiling is enabled, chunks are grouped into spatial windows
    /// and processed together for better cache locality.
    pub async fn run(&self) -> Result<SchedulerStats> {
        // Collect chunks to process
        let mut chunks: Vec<OutputChunk> = if self.config.skip_empty {
            // Pre-filter to only chunks with data
            self.output_grid
                .enumerate_chunks()
                .filter(|chunk| self.spatial_lookup.chunk_has_data(chunk))
                .collect()
        } else {
            self.output_grid.enumerate_chunks().collect()
        };

        // Apply spatial ordering for better COG cache locality
        // Sort by row first, then column (row-major order)
        if self.config.spatial_ordering {
            chunks.sort_by_key(|c| (c.row_idx, c.col_idx));
        }

        let total_chunks = chunks.len();

        // Use meta-tiling if enabled
        if self.config.enable_metatiling {
            tracing::info!(
                "Scheduling {} chunks for processing with meta-tiling ({}x{} metatiles, {} concurrent per metatile)",
                total_chunks,
                self.config.metatile_size,
                self.config.metatile_size,
                self.config.concurrency
            );
            return self.run_metatiled(chunks, total_chunks).await;
        }

        tracing::info!(
            "Scheduling {} chunks for processing ({} concurrent)",
            total_chunks,
            self.config.concurrency
        );

        // Start metrics reporter if enabled
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        let reporter_handle = if self.config.enable_metrics {
            let reporter = MetricsReporter::new(
                self.metrics.clone(),
                self.config.metrics_interval_secs,
                total_chunks as u64,
            );
            Some(tokio::spawn(reporter.run(shutdown_rx)))
        } else {
            drop(shutdown_rx);
            None
        };

        // Process chunks with bounded concurrency
        let processor = self.processor.clone();

        let results: Vec<Result<ChunkResult>> = stream::iter(chunks)
            .map(|chunk| {
                let processor = processor.clone();
                async move { processor.process_chunk_with_retry(chunk).await }
            })
            .buffer_unordered(self.config.concurrency)
            .collect()
            .await;

        // Shutdown metrics reporter
        let _ = shutdown_tx.send(()).await;
        if let Some(handle) = reporter_handle {
            let _ = handle.await;
        }

        // Collect statistics
        let mut stats = SchedulerStats::default();
        for result in results {
            match result {
                Ok(ChunkResult::Processed { .. }) => stats.chunks_processed += 1,
                Ok(ChunkResult::Skipped) => stats.chunks_skipped += 1,
                Err(_) => stats.chunks_failed += 1,
            }
        }

        stats.total_chunks = total_chunks;

        // Print final summary and optionally save to file
        if self.config.enable_metrics {
            let reporter = MetricsReporter::new(
                self.metrics.clone(),
                self.config.metrics_interval_secs,
                total_chunks as u64,
            );
            reporter.print_summary();

            if let Some(ref path) = self.config.metrics_output_path {
                let snapshot = self.metrics.snapshot();
                if let Err(e) = snapshot.save_to_file(path) {
                    tracing::warn!("Failed to save metrics to {}: {}", path, e);
                }
            }
        }

        Ok(stats)
    }

    /// Run the scheduler with meta-tiling for improved cache locality.
    ///
    /// Chunks are grouped into spatial windows (meta-tiles) and processed
    /// one meta-tile at a time. Within each meta-tile, chunks are processed
    /// with high concurrency. This ensures that tiles used by adjacent chunks
    /// are likely to still be in cache.
    async fn run_metatiled(
        &self,
        chunks: Vec<OutputChunk>,
        total_chunks: usize,
    ) -> Result<SchedulerStats> {
        // Create meta-tiles from chunks
        let metatiles = self.create_metatiles(chunks);
        let num_metatiles = metatiles.len();

        tracing::info!(
            "Created {} meta-tiles from {} chunks",
            num_metatiles,
            total_chunks
        );

        // Start metrics reporter if enabled
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        let reporter_handle = if self.config.enable_metrics {
            let reporter = MetricsReporter::new(
                self.metrics.clone(),
                self.config.metrics_interval_secs,
                total_chunks as u64,
            );
            Some(tokio::spawn(reporter.run(shutdown_rx)))
        } else {
            drop(shutdown_rx);
            None
        };

        let mut stats = SchedulerStats::default();
        stats.total_chunks = total_chunks;

        // Process meta-tiles sequentially (or with low concurrency)
        // Within each meta-tile, use high concurrency
        for (idx, metatile) in metatiles.into_iter().enumerate() {
            tracing::debug!(
                "Processing meta-tile {}/{} at ({}, {}) with {} chunks",
                idx + 1,
                num_metatiles,
                metatile.row_start,
                metatile.col_start,
                metatile.chunks.len()
            );

            // Optionally prefetch all tiles needed for this metatile
            // This fetches all internal COG tiles in batch requests (one per COG file)
            if self.config.enable_prefetch {
                let prefetch_start = std::time::Instant::now();
                if let Err(e) = self.processor.prefetch_for_chunks(&metatile.chunks).await {
                    tracing::warn!("Prefetch failed for metatile {}: {}", idx + 1, e);
                    // Continue anyway - individual chunk processing will fetch on demand
                }
                self.metrics.add_prefetch_time(prefetch_start.elapsed());
            }

            let processor = self.processor.clone();

            let results: Vec<Result<ChunkResult>> = stream::iter(metatile.chunks)
                .map(|chunk| {
                    let processor = processor.clone();
                    async move { processor.process_chunk_with_retry(chunk).await }
                })
                .buffer_unordered(self.config.concurrency)
                .collect()
                .await;

            // Accumulate statistics
            for result in results {
                match result {
                    Ok(ChunkResult::Processed { .. }) => stats.chunks_processed += 1,
                    Ok(ChunkResult::Skipped) => stats.chunks_skipped += 1,
                    Err(_) => stats.chunks_failed += 1,
                }
            }
        }

        // Shutdown metrics reporter
        let _ = shutdown_tx.send(()).await;
        if let Some(handle) = reporter_handle {
            let _ = handle.await;
        }

        // Print final summary and optionally save to file
        if self.config.enable_metrics {
            let reporter = MetricsReporter::new(
                self.metrics.clone(),
                self.config.metrics_interval_secs,
                total_chunks as u64,
            );
            reporter.print_summary();

            if let Some(ref path) = self.config.metrics_output_path {
                let snapshot = self.metrics.snapshot();
                if let Err(e) = snapshot.save_to_file(path) {
                    tracing::warn!("Failed to save metrics to {}: {}", path, e);
                }
            }
        }

        Ok(stats)
    }

    /// Group chunks into meta-tiles based on their spatial location.
    ///
    /// Meta-tiles are NxN groups of output chunks that are processed together.
    /// This ensures that tiles needed by adjacent chunks are likely to be cached.
    fn create_metatiles(&self, chunks: Vec<OutputChunk>) -> Vec<MetaTile> {
        use std::collections::HashMap;

        let metatile_size = self.config.metatile_size;

        // Group chunks by their meta-tile
        let mut metatile_map: HashMap<(usize, usize), Vec<OutputChunk>> = HashMap::new();

        for chunk in chunks {
            // Calculate which meta-tile this chunk belongs to
            let meta_row = chunk.row_idx / metatile_size;
            let meta_col = chunk.col_idx / metatile_size;
            let key = (meta_row, meta_col);

            metatile_map.entry(key).or_default().push(chunk);
        }

        // Convert to MetaTile structs and sort by position (row-major order)
        let mut metatiles: Vec<MetaTile> = metatile_map
            .into_iter()
            .map(|((meta_row, meta_col), mut chunks)| {
                // Sort chunks within meta-tile by row-major order for best locality
                chunks.sort_by_key(|c| (c.row_idx, c.col_idx));

                MetaTile {
                    row_start: meta_row * metatile_size,
                    col_start: meta_col * metatile_size,
                    size: metatile_size,
                    chunks,
                }
            })
            .collect();

        // Sort meta-tiles by position (row-major order)
        metatiles.sort_by_key(|m| (m.row_start, m.col_start));

        metatiles
    }

    /// Run with a subset of chunks (for testing or partial processing).
    pub async fn run_subset(&self, chunks: Vec<OutputChunk>) -> Result<SchedulerStats> {
        let total_chunks = chunks.len();
        tracing::info!(
            "Processing subset of {} chunks ({} concurrent)",
            total_chunks,
            self.config.concurrency
        );

        let processor = self.processor.clone();

        let results: Vec<Result<ChunkResult>> = stream::iter(chunks)
            .map(|chunk| {
                let processor = processor.clone();
                async move { processor.process_chunk_with_retry(chunk).await }
            })
            .buffer_unordered(self.config.concurrency)
            .collect()
            .await;

        let mut stats = SchedulerStats::default();
        for result in results {
            match result {
                Ok(ChunkResult::Processed { .. }) => stats.chunks_processed += 1,
                Ok(ChunkResult::Skipped) => stats.chunks_skipped += 1,
                Err(_) => stats.chunks_failed += 1,
            }
        }

        stats.total_chunks = total_chunks;
        Ok(stats)
    }

    /// Estimate the total work for progress reporting.
    pub fn estimate_total_work(&self) -> WorkEstimate {
        let total_chunks = self.output_grid.num_chunks();
        let chunks_with_data: usize = self
            .output_grid
            .enumerate_chunks()
            .filter(|chunk| self.spatial_lookup.chunk_has_data(chunk))
            .count();

        let coverage = self.spatial_lookup.coverage_stats();

        WorkEstimate {
            total_chunks,
            chunks_with_data,
            empty_chunks: total_chunks - chunks_with_data,
            estimated_tiles: (chunks_with_data as f64 * coverage.avg_tiles_per_chunk) as usize,
            max_overlap: coverage.max_tiles_per_chunk,
        }
    }
}

/// Statistics from a scheduler run.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    /// Total chunks attempted
    pub total_chunks: usize,

    /// Chunks successfully processed
    pub chunks_processed: usize,

    /// Chunks skipped (no data)
    pub chunks_skipped: usize,

    /// Chunks that failed
    pub chunks_failed: usize,
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Processed: {}, Skipped: {}, Failed: {}, Total: {}",
            self.chunks_processed, self.chunks_skipped, self.chunks_failed, self.total_chunks
        )
    }
}

/// Work estimate for progress reporting.
#[derive(Debug)]
pub struct WorkEstimate {
    /// Total output chunks
    pub total_chunks: usize,

    /// Chunks with input data
    pub chunks_with_data: usize,

    /// Empty chunks (no input data)
    pub empty_chunks: usize,

    /// Estimated total tiles to read
    pub estimated_tiles: usize,

    /// Maximum tile overlap
    pub max_overlap: usize,
}

impl std::fmt::Display for WorkEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total chunks: {}, With data: {} ({:.1}%), Estimated tiles: {}, Max overlap: {}",
            self.total_chunks,
            self.chunks_with_data,
            self.chunks_with_data as f64 / self.total_chunks as f64 * 100.0,
            self.estimated_tiles,
            self.max_overlap
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.concurrency, 256);
        assert!(config.skip_empty);
        assert!(config.enable_metrics);
        assert!(config.spatial_ordering);
        assert!(config.enable_metatiling);
        assert_eq!(config.metatile_size, 32);
        assert!(!config.enable_prefetch);
        assert!(config.metrics_output_path.is_none());
    }

    #[test]
    fn test_scheduler_stats_display() {
        let stats = SchedulerStats {
            total_chunks: 100,
            chunks_processed: 80,
            chunks_skipped: 15,
            chunks_failed: 5,
        };

        let display = format!("{}", stats);
        assert!(display.contains("80"));
        assert!(display.contains("15"));
        assert!(display.contains("5"));
    }

    #[test]
    fn test_metatile_grouping() {
        // Create test chunks spanning multiple meta-tiles
        let chunks: Vec<OutputChunk> = (0..100)
            .flat_map(|row| {
                (0..100).map(move |col| OutputChunk {
                    row_idx: row,
                    col_idx: col,
                    time_idx: 0,
                })
            })
            .collect();

        // Simulate create_metatiles logic
        let metatile_size = 32;
        let mut metatile_map: std::collections::HashMap<(usize, usize), Vec<OutputChunk>> =
            std::collections::HashMap::new();

        for chunk in chunks {
            let meta_row = chunk.row_idx / metatile_size;
            let meta_col = chunk.col_idx / metatile_size;
            metatile_map.entry((meta_row, meta_col)).or_default().push(chunk);
        }

        // Should have 4x4 = 16 meta-tiles (100/32 = 3.125, so 4 meta-tiles per dimension)
        assert_eq!(metatile_map.len(), 16);

        // First meta-tile should have 32x32 = 1024 chunks
        assert_eq!(metatile_map.get(&(0, 0)).map(|v| v.len()), Some(1024));

        // Last meta-tile should have (100-96)x(100-96) = 4x4 = 16 chunks
        assert_eq!(metatile_map.get(&(3, 3)).map(|v| v.len()), Some(16));
    }
}
