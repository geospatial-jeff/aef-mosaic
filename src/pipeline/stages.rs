//! Decoupled pipeline stages for high-throughput processing.
//!
//! The pipeline is split into three stages connected by bounded channels:
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │ COG Fetcher │────▶│   Mosaic    │────▶│ Zarr Writer │
//! │   Stage     │     │   Stage     │     │   Stage     │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!        │                   │                   │
//!     fetch_rx            mosaic_rx           write_rx
//! ```
//!
//! Benefits:
//! - Network stays busy while CPU does mosaicing
//! - CPU stays busy while I/O writes to Zarr
//! - Backpressure via bounded channels prevents memory explosion
//!
//! Chunk ordering uses a two-level Hilbert curve sort:
//! 1. Group chunks by their primary COG tile (Hilbert order of COG centroids)
//! 2. Within each COG group, order chunks by their own Hilbert index
//! This maximizes cache locality by processing all chunks needing a COG together.

use crate::checkpoint::CheckpointManager;
use crate::crs::{self, ProjCache};
use crate::index::{CogTile, InputIndex, OutputChunk, OutputGrid, SpatialLookup};
use crate::io::{CogReader, GeoTransform, PixelWindow, WindowData, ZarrWriter};
use crate::pipeline::Metrics;
use crate::transform::{mosaic_tiles, ReprojectConfig, Reprojector};
use anyhow::Result;
use dashmap::DashSet;
use futures::stream::{self, StreamExt};
use ndarray::Array4;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

// Thread-local ProjCache to cache Proj objects per-thread.
// Note: Proj objects contain raw pointers and are not Send/Sync, so we cannot
// share them across threads. The thread-local approach is necessary.
// Memory usage: ~10KB per UTM zone × ~60 zones × N threads.
// With the blocking pool default of 512 threads, worst case is ~300MB.
thread_local! {
    static PROJ_CACHE: RefCell<ProjCache> = RefCell::new(ProjCache::new());
}

/// Compute Hilbert curve index for WGS84 coordinates.
///
/// Maps longitude [-180, 180] and latitude [-90, 90] to a 16-bit grid,
/// then computes the Hilbert index. This keeps spatially adjacent
/// locations close in the 1D index.
fn wgs84_hilbert_index(lon: f64, lat: f64) -> u64 {
    // Map to [0, 65535] grid (16-bit resolution)
    let x = (((lon + 180.0) / 360.0) * 65536.0).clamp(0.0, 65535.0) as usize;
    let y = (((lat + 90.0) / 180.0) * 65536.0).clamp(0.0, 65535.0) as usize;
    hilbert_index(x, y, 16)
}

/// Compute Hilbert curve index for (x, y) coordinates.
fn hilbert_index(x: usize, y: usize, order: u32) -> u64 {
    let mut x = x as i64;
    let mut y = y as i64;
    let mut d: u64 = 0;

    let mut s: i64 = (1i64 << order) / 2;
    while s > 0 {
        let rx = if (x & s) > 0 { 1i64 } else { 0 };
        let ry = if (y & s) > 0 { 1i64 } else { 0 };
        d += (s * s) as u64 * ((3 * rx) ^ ry) as u64;

        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Compute the centroid of WGS84 bounds.
fn bounds_centroid(bounds: &[f64; 4]) -> (f64, f64) {
    let lon = (bounds[0] + bounds[2]) / 2.0;
    let lat = (bounds[1] + bounds[3]) / 2.0;
    (lon, lat)
}

/// A pre-computed fetch request (pixel window already calculated).
struct FetchRequest {
    tile: Arc<CogTile>,
    window: PixelWindow,
    intersection_bounds: [f64; 4],
}

/// Pre-computed work item for a chunk (computed before spawning workers).
struct ChunkWorkItem {
    chunk: OutputChunk,
    tiles: Vec<Arc<CogTile>>,
    chunk_bounds: [f64; 4],
    chunk_bounds_wgs84: [f64; 4],
    /// Chunk Hilbert index for secondary sorting within COG groups.
    chunk_hilbert: u64,
}

/// Streams work items to fetch workers, processing one COG at a time.
///
/// This inverts the typical lookup: instead of pre-computing all chunk->COG mappings,
/// we iterate COGs in Hilbert order and compute intersecting chunks on-the-fly.
/// This reduces memory from O(all_chunks) to O(chunks_per_COG).
struct StreamingWorkPublisher {
    /// COG tiles sorted by Hilbert index of their centroids
    cog_tiles: Vec<Arc<CogTile>>,
    /// Output grid for computing chunk intersections
    output_grid: Arc<OutputGrid>,
    /// Spatial lookup for finding tiles per chunk
    spatial_lookup: Arc<SpatialLookup>,
    /// Work channel to send items to fetch workers
    work_tx: async_channel::Sender<ChunkWorkItem>,
    /// Chunks remaining to process (from checkpoint filter)
    pending_chunks: Arc<DashSet<(usize, usize, usize)>>,
    /// Chunks already published (for deduplication across COGs)
    processed_chunks: Arc<DashSet<(usize, usize, usize)>>,
    /// Metrics for tracking skipped chunks
    metrics: Arc<Metrics>,
}

impl StreamingWorkPublisher {
    /// Create a new streaming publisher.
    fn new(
        input_index: &InputIndex,
        output_grid: Arc<OutputGrid>,
        spatial_lookup: Arc<SpatialLookup>,
        pending_chunks: Vec<OutputChunk>,
        work_tx: async_channel::Sender<ChunkWorkItem>,
        metrics: Arc<Metrics>,
    ) -> Self {
        // Sort COG tiles by Hilbert index of their centroids
        let mut cog_tiles = input_index.all_tiles().to_vec();
        cog_tiles.sort_by_key(|t| {
            let (lon, lat) = bounds_centroid(&t.bounds_wgs84);
            wgs84_hilbert_index(lon, lat)
        });

        // Build pending chunks set from input
        let pending_set = Arc::new(DashSet::with_capacity(pending_chunks.len()));
        for chunk in pending_chunks {
            pending_set.insert((chunk.time_idx, chunk.row_idx, chunk.col_idx));
        }

        Self {
            cog_tiles,
            output_grid,
            spatial_lookup,
            work_tx,
            pending_chunks: pending_set,
            processed_chunks: Arc::new(DashSet::new()),
            metrics,
        }
    }

    /// Publish all work items by iterating COGs in Hilbert order.
    async fn publish_all(self) -> Result<()> {
        for cog in &self.cog_tiles {
            // Find chunks that intersect this COG
            let chunks = self.output_grid.chunks_for_bounds_wgs84(
                &cog.bounds_wgs84,
                cog.year,
            )?;

            // Filter to pending chunks, deduplicate, and prepare work items
            let mut work_items: Vec<ChunkWorkItem> = Vec::new();
            for chunk in chunks {
                let key = (chunk.time_idx, chunk.row_idx, chunk.col_idx);

                // Skip if not pending or already processed
                if !self.pending_chunks.contains(&key) {
                    continue;
                }
                if !self.processed_chunks.insert(key) {
                    continue; // Already published by another COG
                }

                // Look up all tiles for this chunk (may include other COGs)
                let tiles = match self.spatial_lookup.tiles_for_chunk(&chunk) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("Failed to find tiles for chunk {:?}: {}", chunk.chunk_indices(), e);
                        self.metrics.add_failure();
                        continue;
                    }
                };

                if tiles.is_empty() {
                    self.metrics.add_chunk_skipped();
                    continue;
                }

                let chunk_bounds = self.output_grid.chunk_bounds(&chunk);
                let chunk_bounds_wgs84 = match self.output_grid.chunk_bounds_wgs84(&chunk) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("Failed to transform chunk bounds: {}", e);
                        self.metrics.add_failure();
                        continue;
                    }
                };

                // Compute chunk Hilbert index for sorting within this COG batch
                let (chunk_lon, chunk_lat) = bounds_centroid(&chunk_bounds_wgs84);
                let chunk_hilbert = wgs84_hilbert_index(chunk_lon, chunk_lat);

                work_items.push(ChunkWorkItem {
                    chunk,
                    tiles,
                    chunk_bounds,
                    chunk_bounds_wgs84,
                    chunk_hilbert,
                });
            }

            // Sort by chunk Hilbert index for spatial locality within this COG group
            work_items.sort_by_key(|w| w.chunk_hilbert);

            // Send work items to fetch workers
            for item in work_items {
                if self.work_tx.send(item).await.is_err() {
                    // Receiver dropped, stop publishing
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}

/// Data passed from COG fetcher to mosaic worker.
pub struct FetchedChunk {
    pub chunk: OutputChunk,
    pub window_data: Vec<WindowData>,
    pub chunk_bounds: [f64; 4],
}

/// Data passed from mosaic worker to Zarr writer.
pub struct MosaicedChunk {
    pub chunk: OutputChunk,
    pub data: Array4<i8>,
}

/// Configuration for the decoupled pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of concurrent COG fetch tasks
    pub fetch_concurrency: usize,
    /// Number of concurrent mosaic tasks (CPU-bound, uses spawn_blocking)
    pub mosaic_concurrency: usize,
    /// Number of concurrent Zarr write tasks
    pub write_concurrency: usize,
    /// HTTP concurrency per chunk (for tile fetches within a single chunk)
    pub http_concurrency_per_chunk: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fetch_concurrency: 8,
            mosaic_concurrency: 8,
            write_concurrency: 8,
            http_concurrency_per_chunk: 32,
        }
    }
}

/// Decoupled pipeline executor.
pub struct Pipeline {
    cog_reader: Arc<CogReader>,
    zarr_writer: Arc<ZarrWriter>,
    spatial_lookup: Arc<SpatialLookup>,
    output_grid: Arc<OutputGrid>,
    metrics: Arc<Metrics>,
    config: PipelineConfig,
    checkpoint_manager: Option<Arc<CheckpointManager>>,
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(
        cog_reader: Arc<CogReader>,
        zarr_writer: Arc<ZarrWriter>,
        spatial_lookup: Arc<SpatialLookup>,
        output_grid: Arc<OutputGrid>,
        metrics: Arc<Metrics>,
        config: PipelineConfig,
        checkpoint_manager: Option<Arc<CheckpointManager>>,
    ) -> Self {
        Self {
            cog_reader,
            zarr_writer,
            spatial_lookup,
            output_grid,
            metrics,
            config,
            checkpoint_manager,
        }
    }

    /// Run the pipeline on the given chunks.
    pub async fn run(&self, chunks: Vec<OutputChunk>) -> Result<PipelineStats> {
        let total_chunks = chunks.len();

        // Create channels between stages
        // Buffer = downstream worker count (1:1 to limit memory usage)
        let fetch_buffer = self.config.mosaic_concurrency;
        let write_buffer = self.config.write_concurrency;
        let (mosaic_tx, mosaic_rx) = async_channel::bounded::<FetchedChunk>(fetch_buffer);
        let (write_tx, write_rx) = async_channel::bounded::<MosaicedChunk>(write_buffer);

        // Spawn queue monitor for debugging backpressure
        let mosaic_tx_monitor = mosaic_tx.clone();
        let write_tx_monitor = write_tx.clone();
        let queue_monitor = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                let fetch_queue = mosaic_tx_monitor.len();
                let fetch_capacity = mosaic_tx_monitor.capacity().unwrap_or(0);
                let mosaic_queue = write_tx_monitor.len();
                let mosaic_capacity = write_tx_monitor.capacity().unwrap_or(0);
                tracing::info!(
                    "Queue backlog: fetch→mosaic {}/{}, mosaic→write {}/{}",
                    fetch_queue, fetch_capacity,
                    mosaic_queue, mosaic_capacity
                );
            }
        });

        // Spawn mosaic stage (multiple workers for concurrent mosaicing)
        let mosaic_handles = self.spawn_mosaic_stage(mosaic_rx, write_tx);

        // Spawn write stage (multiple workers for concurrent shard writes)
        let write_handles = self.spawn_write_stage(write_rx, self.checkpoint_manager.clone());

        // Run fetch stage (this is the main work - processes all chunks)
        self.run_fetch_stage(chunks, mosaic_tx).await;

        // Stop queue monitor BEFORE waiting for downstream stages.
        // The monitor holds clones of the channel senders, which would prevent
        // channels from closing and cause downstream workers to hang forever.
        queue_monitor.abort();

        // Wait for downstream stages to complete
        for handle in mosaic_handles {
            handle.await?;
        }
        for handle in write_handles {
            handle.await?;
        }

        Ok(PipelineStats {
            total_chunks,
            chunks_processed: self.metrics.chunks_processed.load(std::sync::atomic::Ordering::Relaxed) as usize,
            chunks_skipped: self.metrics.chunks_skipped.load(std::sync::atomic::Ordering::Relaxed) as usize,
        })
    }

    /// Run the fetch stage using streaming work publisher.
    ///
    /// Instead of pre-computing all chunk→COG mappings upfront, this iterates
    /// COGs in Hilbert order and computes intersecting chunks on-the-fly.
    /// This reduces memory from O(all_chunks) to O(chunks_per_COG).
    async fn run_fetch_stage(
        &self,
        chunks: Vec<OutputChunk>,
        mosaic_tx: async_channel::Sender<FetchedChunk>,
    ) {
        let fetch_concurrency = self.config.fetch_concurrency;
        let total_pending = chunks.len();

        tracing::info!(
            "Starting streaming fetch stage with {} pending chunks, {} COG tiles",
            total_pending,
            self.spatial_lookup.input_index().len()
        );

        // Create a bounded work queue - small buffer to limit memory usage
        // Workers will pull from this, and backpressure will slow the publisher
        let channel_capacity = (fetch_concurrency * 4).max(64);
        let (work_tx, work_rx) = async_channel::bounded::<ChunkWorkItem>(channel_capacity);

        // Spawn fetch workers FIRST (before starting publisher)
        // This allows backpressure to work - publisher blocks when channel is full
        let mut handles = Vec::with_capacity(fetch_concurrency);
        let http_concurrency_per_chunk = self.config.http_concurrency_per_chunk;
        for worker_id in 0..fetch_concurrency {
            let reader = self.cog_reader.clone();
            let metrics = self.metrics.clone();
            let mosaic_tx = mosaic_tx.clone();
            let work_rx = work_rx.clone();

            let handle = tokio::spawn(async move {
                // Fixed-rate startup to avoid thundering herd on connection pool.
                // Each worker starts 0.5s after the previous one, providing consistent
                // spacing regardless of worker count.
                let startup_delay_secs = worker_id as f64 * 0.5;
                tokio::time::sleep(std::time::Duration::from_secs_f64(startup_delay_secs)).await;

                // Process chunks sequentially from the shared queue
                while let Ok(work) = work_rx.recv().await {
                    let chunk_id = work.chunk.chunk_indices();

                    // Fetch all tile windows for this chunk
                    let cog_start = Instant::now();
                    let window_data = fetch_tile_windows(
                        &reader,
                        &work.tiles,
                        work.chunk_bounds_wgs84,
                        &metrics,
                        http_concurrency_per_chunk,
                    ).await;
                    let fetch_duration = cog_start.elapsed();
                    metrics.add_cog_read_time(fetch_duration);

                    if window_data.is_empty() {
                        metrics.add_chunk_skipped();
                        tracing::info!(
                            chunk = ?chunk_id,
                            fetch_ms = fetch_duration.as_millis(),
                            "fetch worker completed (skipped)"
                        );
                        continue;
                    }

                    metrics.add_tiles_read(window_data.len() as u64);

                    // Send to mosaic stage
                    let fetched = FetchedChunk {
                        chunk: work.chunk,
                        window_data,
                        chunk_bounds: work.chunk_bounds,
                    };

                    let send_start = Instant::now();
                    if mosaic_tx.send(fetched).await.is_err() {
                        tracing::debug!("Mosaic receiver dropped, stopping fetch worker");
                        break;
                    }
                    let channel_wait = send_start.elapsed();

                    tracing::info!(
                        chunk = ?chunk_id,
                        fetch_ms = fetch_duration.as_millis(),
                        channel_wait_ms = channel_wait.as_millis(),
                        "fetch worker completed"
                    );
                }
            });

            handles.push(handle);
        }

        // Create and run the streaming publisher in the current task
        // (SpatialLookup contains ProjCache which isn't Send-safe)
        let publisher = StreamingWorkPublisher::new(
            self.spatial_lookup.input_index(),
            self.output_grid.clone(),
            self.spatial_lookup.clone(),
            chunks,
            work_tx.clone(),
            self.metrics.clone(),
        );

        // Run publisher in current task (not spawned)
        if let Err(e) = publisher.publish_all().await {
            tracing::warn!("Publisher error: {}", e);
        }
        work_tx.close();

        // Wait for all workers to complete
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Spawn the mosaic stage - multiple workers receive fetched data, mosaic, send to write stage.
    ///
    /// Each worker independently pulls from the shared channel and processes work.
    /// This avoids serialization issues where a single task's control flow blocks reception.
    fn spawn_mosaic_stage(
        &self,
        mosaic_rx: async_channel::Receiver<FetchedChunk>,
        write_tx: async_channel::Sender<MosaicedChunk>,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let mosaic_concurrency = self.config.mosaic_concurrency;
        let mut handles = Vec::with_capacity(mosaic_concurrency);

        for _ in 0..mosaic_concurrency {
            let output_grid = self.output_grid.clone();
            let metrics = self.metrics.clone();
            let mosaic_rx = mosaic_rx.clone();
            let write_tx = write_tx.clone();

            let handle = tokio::spawn(async move {
                while let Ok(fetched) = mosaic_rx.recv().await {
                    let chunk_id = fetched.chunk.chunk_indices();
                    let num_tiles = fetched.window_data.len();

                    let pixel_bounds = output_grid.chunk_pixel_bounds(&fetched.chunk);
                    let height = pixel_bounds[2] - pixel_bounds[0];
                    let width = pixel_bounds[3] - pixel_bounds[1];

                    let reproject_config = ReprojectConfig {
                        target_crs: output_grid.crs.clone(),
                        target_resolution: output_grid.resolution,
                        target_bounds: fetched.chunk_bounds,
                        target_shape: (height, width),
                        num_bands: output_grid.num_bands,
                    };

                    let target_crs = reproject_config.target_crs.clone();
                    let window_data = fetched.window_data;
                    let metrics_clone = metrics.clone();

                    // CPU-bound work in spawn_blocking
                    let mosaic_start = Instant::now();
                    let mosaic_result = tokio::task::spawn_blocking(move || {
                        let start = Instant::now();
                        let reprojector = Reprojector::new(&target_crs);
                        let result = mosaic_tiles(&window_data, &reprojector, &reproject_config);
                        metrics_clone.add_reproject_time(start.elapsed());
                        result
                    })
                    .await;
                    let mosaic_duration = mosaic_start.elapsed();

                    match mosaic_result {
                        Ok(Ok(mosaic)) => {
                            let mosaiced = MosaicedChunk {
                                chunk: fetched.chunk,
                                data: mosaic,
                            };
                            let send_start = Instant::now();
                            if write_tx.send(mosaiced).await.is_err() {
                                tracing::debug!("Write receiver dropped, stopping mosaic worker");
                                break;
                            }
                            let channel_wait = send_start.elapsed();

                            tracing::info!(
                                chunk = ?chunk_id,
                                num_tiles = num_tiles,
                                mosaic_ms = mosaic_duration.as_millis(),
                                channel_wait_ms = channel_wait.as_millis(),
                                "mosaic worker completed"
                            );
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("Mosaic failed: {}", e);
                            metrics.add_failure();
                        }
                        Err(e) => {
                            tracing::warn!("Mosaic task panicked: {}", e);
                            metrics.add_failure();
                        }
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }

    /// Spawn the write stage - multiple workers receive mosaiced data, write to Zarr.
    fn spawn_write_stage(
        &self,
        write_rx: async_channel::Receiver<MosaicedChunk>,
        checkpoint_manager: Option<Arc<CheckpointManager>>,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let write_concurrency = self.config.write_concurrency;
        let mut handles = Vec::with_capacity(write_concurrency);

        for _ in 0..write_concurrency {
            let writer = self.zarr_writer.clone();
            let metrics = self.metrics.clone();
            let write_rx = write_rx.clone();
            let checkpoint = checkpoint_manager.clone();

            let handle = tokio::spawn(async move {
                while let Ok(mosaiced) = write_rx.recv().await {
                    // Use block_in_place for sync zarrs API (enables parallel compression)
                    // Note: Must use block_in_place (not spawn_blocking) because the zarrs
                    // AsyncToSyncStorageAdapter uses handle.block_on() internally, which
                    // requires running on a tokio runtime thread.
                    //
                    // Each worker can write to a different shard concurrently, and rayon
                    // handles parallel compression within each shard.
                    let bytes_written = mosaiced.data.len() as u64;
                    let chunk = mosaiced.chunk.clone();
                    let chunk_id = chunk.chunk_indices();

                    let write_start = Instant::now();
                    let write_result = tokio::task::block_in_place(|| {
                        writer.write_chunk_sync(&mosaiced.chunk, mosaiced.data)
                    });
                    let write_duration = write_start.elapsed();

                    if let Err(e) = write_result {
                        tracing::warn!("Zarr write failed: {}", e);
                        metrics.add_failure();
                    } else {
                        metrics.add_zarr_write_time(write_duration);
                        metrics.add_bytes_written(bytes_written);
                        metrics.add_chunk_processed();

                        // Mark chunk as completed in checkpoint
                        if let Some(ref checkpoint) = checkpoint {
                            checkpoint.mark_completed(&chunk);
                        }

                        tracing::info!(
                            chunk = ?chunk_id,
                            write_ms = write_duration.as_millis(),
                            bytes = bytes_written,
                            "write worker completed"
                        );
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }
}

/// Fetch tile windows for a single chunk.
/// HTTP requests within a chunk are concurrent (up to http_concurrency).
async fn fetch_tile_windows(
    reader: &CogReader,
    tiles: &[Arc<CogTile>],
    chunk_bounds_wgs84: [f64; 4],
    metrics: &Metrics,
    http_concurrency: usize,
) -> Vec<WindowData> {

    // Step 1: Fetch geotransforms concurrently (async)
    // Clone Arc refs during iteration (cheap) instead of pre-cloning entire Vec
    let geo_transforms: Vec<_> = stream::iter(tiles.iter().cloned())
        .map(|tile| {
            let reader = reader;
            async move {
                let gt = reader.get_geo_transform(&tile).await;
                (tile, gt)
            }
        })
        .buffer_unordered(http_concurrency)
        .collect()
        .await;

    // Step 2: Compute pixel windows (sync - ProjCache created here, not held across await)
    let fetch_requests = compute_fetch_requests(geo_transforms, &chunk_bounds_wgs84);

    // Step 3: Fetch tile data concurrently (async)
    let results: Vec<_> = stream::iter(fetch_requests)
        .map(|req| {
            let reader = reader;
            async move {
                reader.read_window(&req.tile, req.window, req.intersection_bounds).await
            }
        })
        .buffer_unordered(http_concurrency)
        .collect()
        .await;

    // Collect successful results
    let mut window_data = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(data) => {
                let (bands, h, w) = data.data.dim();
                metrics.add_bytes_read((bands * h * w) as u64);
                window_data.push(data);
            }
            Err(e) => {
                tracing::warn!("Failed to read tile window: {}", e);
                metrics.add_failure();
            }
        }
    }

    window_data
}

/// Compute fetch requests from geotransforms (sync, uses thread-local ProjCache).
fn compute_fetch_requests(
    geo_transforms: Vec<(Arc<CogTile>, Result<Option<GeoTransform>, anyhow::Error>)>,
    chunk_bounds_wgs84: &[f64; 4],
) -> Vec<FetchRequest> {
    // Use thread-local ProjCache - Proj objects are not thread-safe
    PROJ_CACHE.with(|cache| {
        let proj_cache = cache.borrow();

        geo_transforms
            .into_iter()
            .filter_map(|(tile, gt_result)| {
                let geo_transform = match gt_result {
                    Ok(Some(gt)) => gt,
                    Ok(None) => {
                        tracing::warn!("No geotransform for tile {}", tile.tile_id);
                        return None;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to get geotransform for {}: {}", tile.tile_id, e);
                        return None;
                    }
                };

                match compute_pixel_window(&tile, chunk_bounds_wgs84, &proj_cache, &geo_transform) {
                    Ok((window, intersection)) => Some(FetchRequest {
                        tile,
                        window,
                        intersection_bounds: intersection,
                    }),
                    Err(e) => {
                        tracing::debug!("No intersection for tile: {}", e);
                        None
                    }
                }
            })
            .collect()
    })
}

/// Compute pixel window for a tile intersection.
fn compute_pixel_window(
    tile: &CogTile,
    chunk_bounds_wgs84: &[f64; 4],
    proj_cache: &ProjCache,
    geo_transform: &GeoTransform,
) -> Result<(PixelWindow, [f64; 4])> {
    let tile_bounds = tile.bounds_native;

    // Transform chunk bounds to tile's native CRS
    let chunk_bounds_native = crs::transform_bounds(
        chunk_bounds_wgs84,
        crs::codes::WGS84,
        &tile.crs,
        proj_cache,
    )?;

    // Compute intersection
    let intersection = crs::intersect_bounds(&chunk_bounds_native, &tile_bounds)
        .ok_or_else(|| anyhow::anyhow!("No intersection"))?;

    // Convert to pixel coordinates
    let (col_min, row_min) = geo_transform.world_to_pixel(intersection[0], intersection[1]);
    let (col_max, row_max) = geo_transform.world_to_pixel(intersection[2], intersection[3]);

    let (row_start, row_end) = if row_min < row_max {
        (row_min, row_max)
    } else {
        (row_max, row_min)
    };

    let x = col_min.floor().max(0.0) as usize;
    let y = row_start.floor().max(0.0) as usize;
    let x_end = col_max.ceil().max(0.0) as usize;
    let y_end = row_end.ceil().max(0.0) as usize;

    let width = (x_end - x).max(1);
    let height = (y_end - y).max(1);

    Ok((PixelWindow::new(x, y, width, height), intersection))
}

/// Statistics from a pipeline run.
#[derive(Debug, Default)]
pub struct PipelineStats {
    pub total_chunks: usize,
    pub chunks_processed: usize,
    pub chunks_skipped: usize,
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Processed: {}, Skipped: {}, Total: {}",
            self.chunks_processed, self.chunks_skipped, self.total_chunks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.fetch_concurrency, 8);
        assert_eq!(config.mosaic_concurrency, 8);
        assert_eq!(config.write_concurrency, 8);
        assert_eq!(config.http_concurrency_per_chunk, 32);
    }

    #[test]
    fn test_pipeline_stats_default() {
        let stats = PipelineStats::default();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.chunks_skipped, 0);
    }

    #[test]
    fn test_pipeline_stats_display() {
        let stats = PipelineStats {
            total_chunks: 100,
            chunks_processed: 90,
            chunks_skipped: 10,
        };
        let display = format!("{}", stats);
        assert!(display.contains("90"));
        assert!(display.contains("10"));
        assert!(display.contains("100"));
    }
}
