//! Throughput monitoring and metrics collection.

use serde::{Serialize, Serializer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;

fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f64(duration.as_secs_f64())
}

/// Metrics for the pipeline.
#[derive(Debug, Default)]
pub struct Metrics {
    /// Total bytes read from S3
    pub bytes_read: AtomicU64,

    /// Total bytes written to Zarr
    pub bytes_written: AtomicU64,

    /// Number of chunks processed
    pub chunks_processed: AtomicU64,

    /// Number of chunks skipped (no data)
    pub chunks_skipped: AtomicU64,

    /// Number of tiles read
    pub tiles_read: AtomicU64,

    /// Number of failed operations
    pub failures: AtomicU64,

    /// Start time
    start_time: Option<Instant>,

    // Per-component timing (in microseconds for precision)
    /// Time spent reading COGs (microseconds)
    pub cog_read_us: AtomicU64,

    /// Time spent reprojecting (microseconds)
    pub reproject_us: AtomicU64,

    /// Time spent mosaicing (microseconds)
    pub mosaic_us: AtomicU64,

    /// Time spent writing to Zarr (microseconds)
    pub zarr_write_us: AtomicU64,

    // Cache metrics
    /// TIFF metadata cache hits
    pub metadata_cache_hits: AtomicU64,

    /// TIFF metadata cache misses
    pub metadata_cache_misses: AtomicU64,

    /// Tile data cache hits
    pub tile_cache_hits: AtomicU64,

    /// Tile data cache misses
    pub tile_cache_misses: AtomicU64,

    /// Tile cache coalesced requests (single-flight: requests that waited on in-flight fetch)
    pub tile_cache_coalesced: AtomicU64,

    /// Current tile cache size in bytes
    pub tile_cache_bytes: AtomicU64,
}

impl Metrics {
    /// Create new metrics.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            chunks_processed: AtomicU64::new(0),
            chunks_skipped: AtomicU64::new(0),
            tiles_read: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            start_time: Some(Instant::now()),
            cog_read_us: AtomicU64::new(0),
            reproject_us: AtomicU64::new(0),
            mosaic_us: AtomicU64::new(0),
            zarr_write_us: AtomicU64::new(0),
            metadata_cache_hits: AtomicU64::new(0),
            metadata_cache_misses: AtomicU64::new(0),
            tile_cache_hits: AtomicU64::new(0),
            tile_cache_misses: AtomicU64::new(0),
            tile_cache_coalesced: AtomicU64::new(0),
            tile_cache_bytes: AtomicU64::new(0),
        })
    }

    /// Record bytes read.
    pub fn add_bytes_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes written.
    pub fn add_bytes_written(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a processed chunk.
    pub fn add_chunk_processed(&self) {
        self.chunks_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a skipped chunk.
    pub fn add_chunk_skipped(&self) {
        self.chunks_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record tiles read.
    pub fn add_tiles_read(&self, count: u64) {
        self.tiles_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a failure.
    pub fn add_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record time spent reading COGs (in microseconds).
    pub fn add_cog_read_time(&self, duration: Duration) {
        self.cog_read_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record time spent reprojecting (in microseconds).
    pub fn add_reproject_time(&self, duration: Duration) {
        self.reproject_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record time spent mosaicing (in microseconds).
    pub fn add_mosaic_time(&self, duration: Duration) {
        self.mosaic_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record time spent writing to Zarr (in microseconds).
    pub fn add_zarr_write_time(&self, duration: Duration) {
        self.zarr_write_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a metadata cache hit.
    pub fn add_metadata_cache_hit(&self) {
        self.metadata_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a metadata cache miss.
    pub fn add_metadata_cache_miss(&self) {
        self.metadata_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tile cache hit.
    pub fn add_tile_cache_hit(&self) {
        self.tile_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tile cache miss.
    pub fn add_tile_cache_miss(&self) {
        self.tile_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a coalesced tile cache request (waited on in-flight fetch).
    pub fn add_tile_cache_coalesced(&self) {
        self.tile_cache_coalesced.fetch_add(1, Ordering::Relaxed);
    }

    /// Set the current tile cache size in bytes.
    pub fn set_tile_cache_bytes(&self, bytes: u64) {
        self.tile_cache_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Get throughput in GB/s for reads.
    pub fn read_throughput_gbps(&self) -> f64 {
        let bytes = self.bytes_read.load(Ordering::Relaxed);
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            (bytes as f64) / (1024.0 * 1024.0 * 1024.0) / elapsed
        } else {
            0.0
        }
    }

    /// Get throughput in GB/s for writes.
    pub fn write_throughput_gbps(&self) -> f64 {
        let bytes = self.bytes_written.load(Ordering::Relaxed);
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            (bytes as f64) / (1024.0 * 1024.0 * 1024.0) / elapsed
        } else {
            0.0
        }
    }

    /// Get chunks per second.
    pub fn chunks_per_second(&self) -> f64 {
        let chunks = self.chunks_processed.load(Ordering::Relaxed);
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            chunks as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let cog_read_us = self.cog_read_us.load(Ordering::Relaxed);
        let reproject_us = self.reproject_us.load(Ordering::Relaxed);
        let mosaic_us = self.mosaic_us.load(Ordering::Relaxed);
        let zarr_write_us = self.zarr_write_us.load(Ordering::Relaxed);

        MetricsSnapshot {
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            chunks_processed: self.chunks_processed.load(Ordering::Relaxed),
            chunks_skipped: self.chunks_skipped.load(Ordering::Relaxed),
            tiles_read: self.tiles_read.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            elapsed: self.elapsed(),
            read_throughput_gbps: self.read_throughput_gbps(),
            write_throughput_gbps: self.write_throughput_gbps(),
            chunks_per_second: self.chunks_per_second(),
            cog_read_secs: cog_read_us as f64 / 1_000_000.0,
            reproject_secs: reproject_us as f64 / 1_000_000.0,
            mosaic_secs: mosaic_us as f64 / 1_000_000.0,
            zarr_write_secs: zarr_write_us as f64 / 1_000_000.0,
            metadata_cache_hits: self.metadata_cache_hits.load(Ordering::Relaxed),
            metadata_cache_misses: self.metadata_cache_misses.load(Ordering::Relaxed),
            tile_cache_hits: self.tile_cache_hits.load(Ordering::Relaxed),
            tile_cache_misses: self.tile_cache_misses.load(Ordering::Relaxed),
            tile_cache_coalesced: self.tile_cache_coalesced.load(Ordering::Relaxed),
            tile_cache_bytes: self.tile_cache_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub chunks_processed: u64,
    pub chunks_skipped: u64,
    pub tiles_read: u64,
    pub failures: u64,
    #[serde(serialize_with = "serialize_duration")]
    pub elapsed: Duration,
    pub read_throughput_gbps: f64,
    pub write_throughput_gbps: f64,
    pub chunks_per_second: f64,
    /// Total CPU time spent reading COGs (seconds, summed across threads)
    pub cog_read_secs: f64,
    /// Total CPU time spent reprojecting (seconds, summed across threads)
    pub reproject_secs: f64,
    /// Total CPU time spent mosaicing (seconds, summed across threads)
    pub mosaic_secs: f64,
    /// Total CPU time spent writing to Zarr (seconds, summed across threads)
    pub zarr_write_secs: f64,
    /// Metadata cache hits
    pub metadata_cache_hits: u64,
    /// Metadata cache misses
    pub metadata_cache_misses: u64,
    /// Tile cache hits
    pub tile_cache_hits: u64,
    /// Tile cache misses
    pub tile_cache_misses: u64,
    /// Tile cache coalesced (single-flight deduplication)
    pub tile_cache_coalesced: u64,
    /// Tile cache size in bytes
    pub tile_cache_bytes: u64,
}

impl MetricsSnapshot {
    /// Save metrics to a JSON file.
    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        tracing::info!("Metrics saved to {}", path);
        Ok(())
    }
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Calculate percentage of time in each component
        let total_component_time = self.cog_read_secs + self.reproject_secs + self.mosaic_secs + self.zarr_write_secs;
        let (read_pct, reproj_pct, mosaic_pct, write_pct) = if total_component_time > 0.0 {
            (
                self.cog_read_secs / total_component_time * 100.0,
                self.reproject_secs / total_component_time * 100.0,
                self.mosaic_secs / total_component_time * 100.0,
                self.zarr_write_secs / total_component_time * 100.0,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // Cache hit rates
        let metadata_total = self.metadata_cache_hits + self.metadata_cache_misses;
        let metadata_hit_rate = if metadata_total > 0 {
            self.metadata_cache_hits as f64 / metadata_total as f64 * 100.0
        } else {
            0.0
        };

        let tile_total = self.tile_cache_hits + self.tile_cache_misses + self.tile_cache_coalesced;
        let tile_hit_rate = if tile_total > 0 {
            (self.tile_cache_hits + self.tile_cache_coalesced) as f64 / tile_total as f64 * 100.0
        } else {
            0.0
        };

        write!(
            f,
            "Chunks: {} processed, {} skipped | Tiles: {} | \
             Read: {:.2} GB @ {:.2} GB/s | Write: {:.2} GB @ {:.2} GB/s | \
             Rate: {:.1} chunks/s | Failures: {} | Elapsed: {:.1}s | \
             Time: COG {:.0}% | Reproj {:.0}% | Mosaic {:.0}% | Zarr {:.0}% | \
             Cache: meta {:.0}% tile {:.0}%",
            self.chunks_processed,
            self.chunks_skipped,
            self.tiles_read,
            self.bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
            self.read_throughput_gbps,
            self.bytes_written as f64 / (1024.0 * 1024.0 * 1024.0),
            self.write_throughput_gbps,
            self.chunks_per_second,
            self.failures,
            self.elapsed.as_secs_f64(),
            read_pct,
            reproj_pct,
            mosaic_pct,
            write_pct,
            metadata_hit_rate,
            tile_hit_rate,
        )
    }
}

/// Periodic metrics reporter.
pub struct MetricsReporter {
    metrics: Arc<Metrics>,
    interval_secs: u64,
    total_chunks: u64,
}

impl MetricsReporter {
    /// Create a new metrics reporter.
    pub fn new(metrics: Arc<Metrics>, interval_secs: u64, total_chunks: u64) -> Self {
        Self {
            metrics,
            interval_secs,
            total_chunks,
        }
    }

    /// Start the periodic reporter.
    pub async fn run(self, mut shutdown: mpsc::Receiver<()>) {
        let mut ticker = interval(Duration::from_secs(self.interval_secs));

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let snapshot = self.metrics.snapshot();
                    let progress = if self.total_chunks > 0 {
                        (snapshot.chunks_processed + snapshot.chunks_skipped) as f64
                            / self.total_chunks as f64
                            * 100.0
                    } else {
                        0.0
                    };

                    tracing::info!(
                        "[{:.1}%] {}",
                        progress,
                        snapshot
                    );
                }
                _ = shutdown.recv() => {
                    // Final report
                    let snapshot = self.metrics.snapshot();
                    tracing::info!("Final: {}", snapshot);
                    break;
                }
            }
        }
    }

    /// Print a final summary.
    pub fn print_summary(&self) {
        let snapshot = self.metrics.snapshot();

        println!("\n=== Pipeline Summary ===");
        println!("Total time: {:.1}s", snapshot.elapsed.as_secs_f64());
        println!("Chunks processed: {}", snapshot.chunks_processed);
        println!("Chunks skipped: {}", snapshot.chunks_skipped);
        println!("Tiles read: {}", snapshot.tiles_read);
        println!(
            "Data read: {:.2} GB",
            snapshot.bytes_read as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "Data written: {:.2} GB",
            snapshot.bytes_written as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("Read throughput: {:.2} GB/s", snapshot.read_throughput_gbps);
        println!("Write throughput: {:.2} GB/s", snapshot.write_throughput_gbps);
        println!("Processing rate: {:.1} chunks/s", snapshot.chunks_per_second);
        println!("Failures: {}", snapshot.failures);

        // Component time breakdown
        let total_component = snapshot.cog_read_secs + snapshot.reproject_secs + snapshot.mosaic_secs + snapshot.zarr_write_secs;
        if total_component > 0.0 {
            println!("\n--- Component Time Breakdown ---");
            println!("COG read:    {:>7.1}s ({:>5.1}%)", snapshot.cog_read_secs, snapshot.cog_read_secs / total_component * 100.0);
            println!("Reproject:   {:>7.1}s ({:>5.1}%)", snapshot.reproject_secs, snapshot.reproject_secs / total_component * 100.0);
            println!("Mosaic:      {:>7.1}s ({:>5.1}%)", snapshot.mosaic_secs, snapshot.mosaic_secs / total_component * 100.0);
            println!("Zarr write:  {:>7.1}s ({:>5.1}%)", snapshot.zarr_write_secs, snapshot.zarr_write_secs / total_component * 100.0);
        }

        // Cache statistics
        let metadata_total = snapshot.metadata_cache_hits + snapshot.metadata_cache_misses;
        let tile_total = snapshot.tile_cache_hits + snapshot.tile_cache_misses + snapshot.tile_cache_coalesced;

        if metadata_total > 0 || tile_total > 0 {
            println!("\n--- Cache Statistics ---");
            if metadata_total > 0 {
                let hit_rate = snapshot.metadata_cache_hits as f64 / metadata_total as f64 * 100.0;
                println!(
                    "Metadata cache: {} hits, {} misses ({:.1}% hit rate)",
                    snapshot.metadata_cache_hits, snapshot.metadata_cache_misses, hit_rate
                );
            }
            if tile_total > 0 {
                let hit_rate = (snapshot.tile_cache_hits + snapshot.tile_cache_coalesced) as f64 / tile_total as f64 * 100.0;
                println!(
                    "Tile cache: {} hits, {} misses, {} coalesced ({:.1}% effective hit rate)",
                    snapshot.tile_cache_hits, snapshot.tile_cache_misses, snapshot.tile_cache_coalesced, hit_rate
                );
                println!(
                    "Tile cache size: {:.2} GB",
                    snapshot.tile_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        println!("========================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_increment() {
        let metrics = Metrics::new();

        metrics.add_bytes_read(1000);
        metrics.add_bytes_read(500);

        assert_eq!(metrics.bytes_read.load(Ordering::Relaxed), 1500);
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = Metrics::new();

        metrics.add_chunk_processed();
        metrics.add_chunk_processed();
        metrics.add_chunk_skipped();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.chunks_processed, 2);
        assert_eq!(snapshot.chunks_skipped, 1);
    }

    #[test]
    fn test_all_counters() {
        let metrics = Metrics::new();

        metrics.add_bytes_read(1024);
        metrics.add_bytes_written(2048);
        metrics.add_chunk_processed();
        metrics.add_chunk_skipped();
        metrics.add_tiles_read(5);
        metrics.add_failure();
        metrics.add_metadata_cache_hit();
        metrics.add_metadata_cache_miss();
        metrics.add_tile_cache_hit();
        metrics.add_tile_cache_miss();
        metrics.add_tile_cache_coalesced();
        metrics.set_tile_cache_bytes(1_000_000);

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.bytes_read, 1024);
        assert_eq!(snapshot.bytes_written, 2048);
        assert_eq!(snapshot.chunks_processed, 1);
        assert_eq!(snapshot.chunks_skipped, 1);
        assert_eq!(snapshot.tiles_read, 5);
        assert_eq!(snapshot.failures, 1);
        assert_eq!(snapshot.metadata_cache_hits, 1);
        assert_eq!(snapshot.metadata_cache_misses, 1);
        assert_eq!(snapshot.tile_cache_hits, 1);
        assert_eq!(snapshot.tile_cache_misses, 1);
        assert_eq!(snapshot.tile_cache_coalesced, 1);
        assert_eq!(snapshot.tile_cache_bytes, 1_000_000);
    }

    #[test]
    fn test_timing_metrics() {
        let metrics = Metrics::new();

        metrics.add_cog_read_time(Duration::from_millis(100));
        metrics.add_reproject_time(Duration::from_millis(50));
        metrics.add_mosaic_time(Duration::from_millis(25));
        metrics.add_zarr_write_time(Duration::from_millis(75));

        let snapshot = metrics.snapshot();

        assert!((snapshot.cog_read_secs - 0.1).abs() < 0.001);
        assert!((snapshot.reproject_secs - 0.05).abs() < 0.001);
        assert!((snapshot.mosaic_secs - 0.025).abs() < 0.001);
        assert!((snapshot.zarr_write_secs - 0.075).abs() < 0.001);
    }

    #[test]
    fn test_snapshot_display() {
        let snapshot = MetricsSnapshot {
            bytes_read: 1024 * 1024 * 1024, // 1 GB
            bytes_written: 512 * 1024 * 1024, // 0.5 GB
            chunks_processed: 100,
            chunks_skipped: 10,
            tiles_read: 500,
            failures: 2,
            elapsed: Duration::from_secs(10),
            read_throughput_gbps: 0.1,
            write_throughput_gbps: 0.05,
            chunks_per_second: 10.0,
            cog_read_secs: 5.0,
            reproject_secs: 2.0,
            mosaic_secs: 1.0,
            zarr_write_secs: 2.0,
            metadata_cache_hits: 400,
            metadata_cache_misses: 100,
            tile_cache_hits: 300,
            tile_cache_misses: 150,
            tile_cache_coalesced: 50,
            tile_cache_bytes: 100 * 1024 * 1024,
        };

        let display = format!("{}", snapshot);

        // Verify key parts are present
        assert!(display.contains("100 processed"));
        assert!(display.contains("10 skipped"));
        assert!(display.contains("500")); // tiles
        assert!(display.contains("Failures: 2"));
    }

    #[test]
    fn test_cache_hit_rates_in_display() {
        // 80% metadata hit rate (80 hits, 20 misses)
        // 70% tile hit rate (50 hits + 20 coalesced = 70, 30 misses)
        let snapshot = MetricsSnapshot {
            bytes_read: 0,
            bytes_written: 0,
            chunks_processed: 0,
            chunks_skipped: 0,
            tiles_read: 0,
            failures: 0,
            elapsed: Duration::from_secs(1),
            read_throughput_gbps: 0.0,
            write_throughput_gbps: 0.0,
            chunks_per_second: 0.0,
            cog_read_secs: 0.0,
            reproject_secs: 0.0,
            mosaic_secs: 0.0,
            zarr_write_secs: 0.0,
            metadata_cache_hits: 80,
            metadata_cache_misses: 20,
            tile_cache_hits: 50,
            tile_cache_misses: 30,
            tile_cache_coalesced: 20,
            tile_cache_bytes: 0,
        };

        let display = format!("{}", snapshot);

        // Should show 80% metadata hit rate and 70% tile hit rate
        assert!(display.contains("meta 80%"));
        assert!(display.contains("tile 70%"));
    }

    #[test]
    fn test_zero_elapsed_no_panic() {
        // Create metrics without start_time to test zero elapsed case
        let metrics = Metrics {
            start_time: None,
            ..Default::default()
        };

        metrics.add_bytes_read(1000);

        // Should not panic, should return 0.0
        assert_eq!(metrics.read_throughput_gbps(), 0.0);
        assert_eq!(metrics.write_throughput_gbps(), 0.0);
        assert_eq!(metrics.chunks_per_second(), 0.0);
    }

    #[test]
    fn test_metrics_reporter_new() {
        let metrics = Metrics::new();
        let reporter = MetricsReporter::new(metrics, 10, 1000);

        assert_eq!(reporter.interval_secs, 10);
        assert_eq!(reporter.total_chunks, 1000);
    }
}
