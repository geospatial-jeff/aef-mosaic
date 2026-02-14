//! Work distribution and scheduling for chunk processing.
//!
//! The scheduler distributes output chunks across async tasks with configurable concurrency.

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

    /// Optional path to save metrics JSON after run completes
    pub metrics_output_path: Option<String>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            concurrency: 16,
            skip_empty: true,
            enable_metrics: true,
            metrics_interval_secs: 10,
            metrics_output_path: None,
        }
    }
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

        // Sort by row first, then column (row-major order) for cache locality
        chunks.sort_by_key(|c| (c.row_idx, c.col_idx));

        let total_chunks = chunks.len();

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
        assert_eq!(config.concurrency, 16);
        assert!(config.skip_empty);
        assert!(config.enable_metrics);
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
}
