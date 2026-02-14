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

/// A pre-computed fetch request (pixel window already calculated).
struct FetchRequest {
    tile: Arc<CogTile>,
    window: PixelWindow,
    intersection_bounds: [f64; 4],
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
            fetch_concurrency: 32,
            mosaic_concurrency: 8,
            write_concurrency: 16,
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

    /// Run the fetch stage - processes chunks and sends to mosaic stage.
    async fn run_fetch_stage(
        &self,
        chunks: Vec<OutputChunk>,
        mosaic_tx: mpsc::Sender<FetchedChunk>,
    ) {
        // Process chunks with bounded concurrency using buffer_unordered
        let reader = self.cog_reader.clone();
        let spatial_lookup = self.spatial_lookup.clone();
        let output_grid = self.output_grid.clone();
        let metrics = self.metrics.clone();
        let fetch_concurrency = self.config.fetch_concurrency;

        // Create a ProjCache for this stage (not Send, so kept in main task)
        let proj_cache = ProjCache::new();

        // Process each chunk
        for chunk in chunks {
            // Find intersecting tiles
            let tiles = match spatial_lookup.tiles_for_chunk(&chunk) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("Failed to find tiles for chunk {:?}: {}", chunk.chunk_indices(), e);
                    metrics.add_failure();
                    continue;
                }
            };

            if tiles.is_empty() {
                metrics.add_chunk_skipped();
                continue;
            }

            let chunk_bounds = output_grid.chunk_bounds(&chunk);
            let chunk_bounds_wgs84 = match output_grid.chunk_bounds_wgs84(&chunk) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("Failed to transform chunk bounds: {}", e);
                    metrics.add_failure();
                    continue;
                }
            };

            // Pre-compute fetch requests (pixel windows) - this uses proj_cache
            // but happens in the main task, not spawned
            let fetch_requests: Vec<_> = tiles
                .iter()
                .filter_map(|tile| {
                    // Get geotransform - we need to fetch this async, but for pre-computing
                    // windows we'll defer to fetch time and do the transform there
                    // Actually, we need the geotransform to compute pixel window...
                    // For now, pass tile with bounds and compute window at fetch time
                    Some(Arc::clone(tile))
                })
                .collect();

            // Fetch all tile windows concurrently
            let cog_start = Instant::now();
            let window_data = fetch_tile_windows_concurrent(
                &reader,
                &fetch_requests,
                &chunk_bounds_wgs84,
                &proj_cache,
                &metrics,
                fetch_concurrency,
            ).await;
            metrics.add_cog_read_time(cog_start.elapsed());

            if window_data.is_empty() {
                metrics.add_chunk_skipped();
                continue;
            }

            metrics.add_tiles_read(window_data.len() as u64);

            // Send to mosaic stage
            let fetched = FetchedChunk {
                chunk,
                window_data,
                chunk_bounds,
            };

            if mosaic_tx.send(fetched).await.is_err() {
                tracing::debug!("Mosaic receiver dropped, stopping fetch stage");
                break;
            }
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

/// Fetch tile windows concurrently using buffer_unordered.
async fn fetch_tile_windows_concurrent(
    reader: &CogReader,
    tiles: &[Arc<CogTile>],
    chunk_bounds_wgs84: &[f64; 4],
    proj_cache: &ProjCache,
    metrics: &Metrics,
    concurrency: usize,
) -> Vec<WindowData> {
    // Pre-compute pixel windows in the current task (uses proj_cache which is not Send)
    // We need to fetch geotransform first, so we'll structure this as:
    // 1. First fetch all geotransforms concurrently
    // 2. Then compute pixel windows (sync, uses proj_cache)
    // 3. Then fetch tile data concurrently

    // Step 1: Fetch geotransforms
    let geo_transforms: Vec<_> = stream::iter(tiles.iter())
        .map(|tile| {
            let reader = reader;
            let tile = Arc::clone(tile);
            async move {
                let gt = reader.get_geo_transform(&tile).await;
                (tile, gt)
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Step 2: Compute pixel windows (sync)
    let fetch_requests: Vec<_> = geo_transforms
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

            match compute_pixel_window(&tile, chunk_bounds_wgs84, proj_cache, &geo_transform) {
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
        .collect();

    // Step 3: Fetch tile data concurrently
    let results: Vec<_> = stream::iter(fetch_requests)
        .map(|req| {
            let reader = reader;
            async move {
                reader.read_window(&req.tile, req.window, req.intersection_bounds).await
            }
        })
        .buffer_unordered(concurrency)
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
        assert_eq!(config.fetch_concurrency, 32);
        assert_eq!(config.mosaic_concurrency, 8);
        assert_eq!(config.write_concurrency, 16);
    }
}
