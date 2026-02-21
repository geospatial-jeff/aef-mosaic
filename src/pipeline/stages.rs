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

use crate::crs::{self, ProjCache};
use crate::index::{CogTile, OutputChunk, OutputGrid, SpatialLookup};
use crate::io::{CogReader, GeoTransform, PixelWindow, WindowData, ZarrWriter};
use crate::pipeline::Metrics;
use crate::transform::{mosaic_tiles, ReprojectConfig, Reprojector};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use ndarray::Array4;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

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
    /// Two-level sort key: (primary_cog_hilbert, chunk_hilbert)
    /// Groups chunks by their primary COG, then by spatial locality within.
    sort_key: (u64, u64),
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
    /// Channel buffer size between fetch and mosaic stages
    pub fetch_buffer: usize,
    /// Channel buffer size between mosaic and write stages
    pub mosaic_buffer: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fetch_concurrency: 8,
            mosaic_concurrency: 8,
            write_concurrency: 8,
            fetch_buffer: 16,
            mosaic_buffer: 8,
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
    ) -> Self {
        Self {
            cog_reader,
            zarr_writer,
            spatial_lookup,
            output_grid,
            metrics,
            config,
        }
    }

    /// Run the pipeline on the given chunks.
    pub async fn run(&self, chunks: Vec<OutputChunk>) -> Result<PipelineStats> {
        let total_chunks = chunks.len();

        // Create channels between stages
        let (mosaic_tx, mosaic_rx) = mpsc::channel::<FetchedChunk>(self.config.fetch_buffer);
        let (write_tx, write_rx) = mpsc::channel::<MosaicedChunk>(self.config.mosaic_buffer);

        // Spawn mosaic stage
        let mosaic_handle = self.spawn_mosaic_stage(mosaic_rx, write_tx);

        // Spawn write stage
        let write_handle = self.spawn_write_stage(write_rx);

        // Run fetch stage (this is the main work - processes all chunks)
        self.run_fetch_stage(chunks, mosaic_tx).await;

        // Wait for downstream stages to complete
        mosaic_handle.await?;
        write_handle.await?;

        Ok(PipelineStats {
            total_chunks,
            chunks_processed: self.metrics.chunks_processed.load(std::sync::atomic::Ordering::Relaxed) as usize,
            chunks_skipped: self.metrics.chunks_skipped.load(std::sync::atomic::Ordering::Relaxed) as usize,
        })
    }

    /// Run the fetch stage - spawns fetch_concurrency workers that each process chunks sequentially.
    async fn run_fetch_stage(
        &self,
        chunks: Vec<OutputChunk>,
        mosaic_tx: mpsc::Sender<FetchedChunk>,
    ) {
        let fetch_concurrency = self.config.fetch_concurrency;

        // Pre-compute tile lookups and bounds (uses SpatialLookup which is not Send)
        let mut work_items = Vec::with_capacity(chunks.len());
        for chunk in chunks {
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

            // Compute two-level sort key for cache-optimal ordering:
            // 1. Primary COG tile's Hilbert index (groups chunks by COG)
            // 2. Chunk's own Hilbert index (orders within COG group)
            let primary_cog = &tiles[0]; // Use first (often largest overlap) COG
            let (cog_lon, cog_lat) = bounds_centroid(&primary_cog.bounds_wgs84);
            let cog_hilbert = wgs84_hilbert_index(cog_lon, cog_lat);

            let (chunk_lon, chunk_lat) = bounds_centroid(&chunk_bounds_wgs84);
            let chunk_hilbert = wgs84_hilbert_index(chunk_lon, chunk_lat);

            let tile_refs: Vec<_> = tiles.iter().map(|t| Arc::clone(t)).collect();

            work_items.push(ChunkWorkItem {
                chunk,
                tiles: tile_refs,
                chunk_bounds,
                chunk_bounds_wgs84,
                sort_key: (cog_hilbert, chunk_hilbert),
            });
        }

        // Sort work items by two-level Hilbert key for optimal cache locality:
        // - Chunks sharing the same COG are processed together
        // - Within each COG group, spatially adjacent chunks are processed together
        work_items.sort_by_key(|item| item.sort_key);

        // Count unique COG groups for logging
        let unique_cogs: std::collections::HashSet<u64> = work_items.iter()
            .map(|item| item.sort_key.0)
            .collect();

        tracing::info!(
            "Sorted {} chunks into {} COG groups (two-level Hilbert ordering)",
            work_items.len(),
            unique_cogs.len()
        );

        // Create a shared work queue
        let (work_tx, work_rx) = async_channel::bounded::<ChunkWorkItem>(work_items.len().max(1));

        // Send all work items to the queue (in sorted order)
        for item in work_items {
            let _ = work_tx.send(item).await;
        }
        work_tx.close();

        // Spawn fetch_concurrency workers
        let mut handles = Vec::with_capacity(fetch_concurrency);
        for _ in 0..fetch_concurrency {
            let reader = self.cog_reader.clone();
            let metrics = self.metrics.clone();
            let mosaic_tx = mosaic_tx.clone();
            let work_rx = work_rx.clone();

            let handle = tokio::spawn(async move {
                // Process chunks sequentially from the shared queue
                while let Ok(work) = work_rx.recv().await {
                    // Fetch all tile windows for this chunk
                    let cog_start = Instant::now();
                    let window_data = fetch_tile_windows(
                        &reader,
                        &work.tiles,
                        work.chunk_bounds_wgs84,
                        &metrics,
                    ).await;
                    metrics.add_cog_read_time(cog_start.elapsed());

                    if window_data.is_empty() {
                        metrics.add_chunk_skipped();
                        continue;
                    }

                    metrics.add_tiles_read(window_data.len() as u64);

                    // Send to mosaic stage
                    let fetched = FetchedChunk {
                        chunk: work.chunk,
                        window_data,
                        chunk_bounds: work.chunk_bounds,
                    };

                    if mosaic_tx.send(fetched).await.is_err() {
                        tracing::debug!("Mosaic receiver dropped, stopping fetch worker");
                        break;
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Spawn the mosaic stage - receives fetched data, mosaics, sends to write stage.
    fn spawn_mosaic_stage(
        &self,
        mut mosaic_rx: mpsc::Receiver<FetchedChunk>,
        write_tx: mpsc::Sender<MosaicedChunk>,
    ) -> tokio::task::JoinHandle<()> {
        let output_grid = self.output_grid.clone();
        let metrics = self.metrics.clone();
        let mosaic_concurrency = self.config.mosaic_concurrency;

        tokio::spawn(async move {
            // Collect incoming fetched chunks and process them concurrently
            let mut pending_futures = Vec::new();

            loop {
                tokio::select! {
                    // Check for new work
                    fetched = mosaic_rx.recv() => {
                        match fetched {
                            Some(fetched) => {
                                let output_grid = output_grid.clone();
                                let metrics = metrics.clone();
                                let write_tx = write_tx.clone();

                                // Spawn mosaic task
                                let future = async move {
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
                                    let mosaic_result = tokio::task::spawn_blocking(move || {
                                        let start = Instant::now();
                                        let reprojector = Reprojector::new(&target_crs);
                                        let result = mosaic_tiles(&window_data, &reprojector, &reproject_config);
                                        metrics_clone.add_reproject_time(start.elapsed());
                                        result
                                    }).await;

                                    match mosaic_result {
                                        Ok(Ok(mosaic)) => {
                                            let mosaiced = MosaicedChunk {
                                                chunk: fetched.chunk,
                                                data: mosaic,
                                            };
                                            let _ = write_tx.send(mosaiced).await;
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
                                };

                                pending_futures.push(tokio::spawn(future));

                                // Limit concurrency
                                while pending_futures.len() >= mosaic_concurrency {
                                    // Wait for at least one to complete
                                    let (result, _idx, remaining) = futures::future::select_all(pending_futures).await;
                                    let _ = result; // Ignore JoinHandle result
                                    pending_futures = remaining;
                                }
                            }
                            None => {
                                // Input channel closed, wait for remaining work
                                for handle in pending_futures {
                                    let _ = handle.await;
                                }
                                return;
                            }
                        }
                    }
                }
            }
        })
    }

    /// Spawn the write stage - receives mosaiced data, writes to Zarr.
    fn spawn_write_stage(
        &self,
        mut write_rx: mpsc::Receiver<MosaicedChunk>,
    ) -> tokio::task::JoinHandle<()> {
        let writer = self.zarr_writer.clone();
        let metrics = self.metrics.clone();
        let write_concurrency = self.config.write_concurrency;

        tokio::spawn(async move {
            let mut pending_futures = Vec::new();

            loop {
                tokio::select! {
                    mosaiced = write_rx.recv() => {
                        match mosaiced {
                            Some(mosaiced) => {
                                let writer = writer.clone();
                                let metrics = metrics.clone();

                                let future = async move {
                                    let bytes_written = mosaiced.data.len() as u64;

                                    let start = Instant::now();
                                    if let Err(e) = writer.write_chunk_async(&mosaiced.chunk, mosaiced.data).await {
                                        tracing::warn!("Zarr write failed: {}", e);
                                        metrics.add_failure();
                                        return;
                                    }
                                    metrics.add_zarr_write_time(start.elapsed());
                                    metrics.add_bytes_written(bytes_written);
                                    metrics.add_chunk_processed();
                                };

                                pending_futures.push(tokio::spawn(future));

                                // Limit concurrency
                                while pending_futures.len() >= write_concurrency {
                                    let (result, _idx, remaining) = futures::future::select_all(pending_futures).await;
                                    let _ = result;
                                    pending_futures = remaining;
                                }
                            }
                            None => {
                                // Input channel closed, wait for remaining work
                                for handle in pending_futures {
                                    let _ = handle.await;
                                }
                                return;
                            }
                        }
                    }
                }
            }
        })
    }
}

/// Fetch tile windows for a single chunk.
/// HTTP requests within a chunk are concurrent (up to 32).
async fn fetch_tile_windows(
    reader: &CogReader,
    tiles: &[Arc<CogTile>],
    chunk_bounds_wgs84: [f64; 4],
    metrics: &Metrics,
) -> Vec<WindowData> {
    // Fixed HTTP concurrency within a chunk
    const HTTP_CONCURRENCY: usize = 32;

    // Clone tiles for owned iteration
    let tiles_owned: Vec<_> = tiles.iter().cloned().collect();

    // Step 1: Fetch geotransforms concurrently (async)
    let geo_transforms: Vec<_> = stream::iter(tiles_owned)
        .map(|tile| {
            let reader = reader;
            async move {
                let gt = reader.get_geo_transform(&tile).await;
                (tile, gt)
            }
        })
        .buffer_unordered(HTTP_CONCURRENCY)
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
        .buffer_unordered(HTTP_CONCURRENCY)
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

/// Compute fetch requests from geotransforms (sync, creates ProjCache internally).
fn compute_fetch_requests(
    geo_transforms: Vec<(Arc<CogTile>, Result<Option<GeoTransform>, anyhow::Error>)>,
    chunk_bounds_wgs84: &[f64; 4],
) -> Vec<FetchRequest> {
    // Create ProjCache here - not held across any await
    let proj_cache = ProjCache::new();

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
        assert_eq!(config.fetch_buffer, 16);
        assert_eq!(config.mosaic_buffer, 8);
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
