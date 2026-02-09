//! Per-chunk processing pipeline.
//!
//! ## Coordinate Systems
//!
//! This module handles three coordinate systems:
//!
//! - **WGS84 (EPSG:4326)**: Geographic coordinates used for spatial lookup (R-tree queries).
//! - **Output CRS (e.g., EPSG:6933)**: The target projection for the output Zarr array.
//! - **Tile native CRS (e.g., EPSG:32610)**: The source projection for each input tile.
//!
//! The key transformation flow is:
//! 1. Get chunk bounds in output CRS
//! 2. Transform to WGS84 for R-tree spatial lookup
//! 3. For each intersecting tile, transform chunk bounds to tile's native CRS
//! 4. Compute pixel window in native CRS coordinates
//! 5. Read pixel data and reproject to output CRS

use crate::config::Config;
use crate::crs::{self, ProjCache};
use crate::index::{CogTile, OutputChunk, SpatialLookup};
use crate::io::{CogReader, GeoTransform, PixelWindow, WindowData, ZarrWriter};
use crate::pipeline::Metrics;
use crate::transform::{mosaic_tiles, ReprojectConfig, Reprojector};
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

/// Processor for individual output chunks.
pub struct ChunkProcessor {
    /// COG reader for fetching tiles
    cog_reader: Arc<CogReader>,

    /// Zarr writer for output
    zarr_writer: Arc<ZarrWriter>,

    /// Spatial lookup for finding input tiles
    spatial_lookup: Arc<SpatialLookup>,

    /// Metrics collector
    metrics: Arc<Metrics>,

    /// Configuration
    config: Arc<Config>,

    /// Maximum concurrent COG fetches per chunk
    cog_fetch_concurrency: usize,

    /// CRS transformation cache
    proj_cache: Arc<ProjCache>,
}

impl ChunkProcessor {
    /// Create a new chunk processor.
    pub fn new(
        cog_reader: Arc<CogReader>,
        zarr_writer: Arc<ZarrWriter>,
        spatial_lookup: Arc<SpatialLookup>,
        metrics: Arc<Metrics>,
        config: Arc<Config>,
    ) -> Self {
        Self {
            cog_reader,
            zarr_writer,
            spatial_lookup,
            metrics,
            cog_fetch_concurrency: config.processing.cog_fetch_concurrency,
            config,
            proj_cache: Arc::new(ProjCache::new()),
        }
    }

    /// Process a single output chunk.
    pub async fn process_chunk(&self, chunk: OutputChunk) -> Result<ChunkResult> {
        // Find intersecting input tiles
        let tiles = self.spatial_lookup.tiles_for_chunk(&chunk)?;

        if tiles.is_empty() {
            self.metrics.add_chunk_skipped();
            return Ok(ChunkResult::Skipped);
        }

        // Get chunk bounds in both output CRS (for reprojection) and WGS84 (for tile intersection)
        let output_grid = self.zarr_writer.output_grid();
        let chunk_bounds = output_grid.chunk_bounds(&chunk);
        let chunk_bounds_wgs84 = output_grid.chunk_bounds_wgs84(&chunk)?;

        // Fetch window from each COG tile (only the overlapping region)
        let cog_start = Instant::now();
        let window_data = self.fetch_tile_windows(&tiles, &chunk_bounds_wgs84).await?;
        let cog_elapsed = cog_start.elapsed();
        self.metrics.add_cog_read_time(cog_elapsed);

        let tiles_read = window_data.len() as u64;
        self.metrics.add_tiles_read(tiles_read);

        // Calculate bytes read
        let bytes_read: u64 = window_data
            .iter()
            .map(|w| {
                let (bands, h, width) = w.data.dim();
                (bands * h * width) as u64
            })
            .sum();
        self.metrics.add_bytes_read(bytes_read);

        // Get pixel bounds for output
        let pixel_bounds = output_grid.chunk_pixel_bounds(&chunk);
        let height = pixel_bounds[2] - pixel_bounds[0];
        let width = pixel_bounds[3] - pixel_bounds[1];

        let reproject_config = ReprojectConfig {
            target_crs: output_grid.crs.clone(),
            target_resolution: output_grid.resolution,
            target_bounds: chunk_bounds,
            target_shape: (height, width),
            num_bands: output_grid.num_bands,
        };

        // Mosaic tiles in blocking task (reprojection is CPU-bound)
        // Note: mosaic_tiles includes both reprojection and mosaicing time
        let target_crs = reproject_config.target_crs.clone();
        let metrics_clone = self.metrics.clone();
        let mosaic = tokio::task::spawn_blocking(move || {
            let reproj_start = Instant::now();
            let reprojector = Reprojector::new(&target_crs);
            let result = mosaic_tiles(&window_data, &reprojector, &reproject_config);
            // Record combined reproject+mosaic time (they're interleaved in mosaic_tiles)
            let reproj_elapsed = reproj_start.elapsed();
            metrics_clone.add_reproject_time(reproj_elapsed);
            result
        })
        .await
        .map_err(|e| anyhow::anyhow!("Mosaic task panicked: {}", e))??;

        // Write to Zarr
        let bytes_written = (mosaic.len()) as u64;

        // Log if data is all zeros (fill value) - zarrs may skip writing fill-value chunks
        let non_zero_count = mosaic.iter().filter(|&&v| v != 0).count();
        if non_zero_count == 0 {
            tracing::debug!(
                "Chunk {:?} mosaic is all zeros (fill value), {} tiles read",
                chunk.chunk_indices(), tiles_read
            );
        }

        let zarr_start = Instant::now();
        self.zarr_writer.write_chunk_async(&chunk, mosaic).await?;
        let zarr_elapsed = zarr_start.elapsed();
        self.metrics.add_zarr_write_time(zarr_elapsed);
        self.metrics.add_bytes_written(bytes_written);

        self.metrics.add_chunk_processed();

        Ok(ChunkResult::Processed {
            tiles_read: tiles_read as usize,
            bytes_read,
            bytes_written,
        })
    }

    /// Fetch windows from tiles that overlap with the chunk bounds.
    ///
    /// # Arguments
    /// * `tiles` - List of tiles that intersect the chunk (from R-tree query)
    /// * `chunk_bounds_wgs84` - Chunk bounds in WGS84 for intersection calculation
    async fn fetch_tile_windows(
        &self,
        tiles: &[&CogTile],
        chunk_bounds_wgs84: &[f64; 4],
    ) -> Result<Vec<WindowData>> {
        use futures::stream::{self, StreamExt};

        let proj_cache = self.proj_cache.clone();
        let results: Vec<_> = stream::iter(tiles.iter())
            .map(|tile| {
                let reader = self.cog_reader.clone();
                let bounds = *chunk_bounds_wgs84;
                let cache = proj_cache.clone();
                let tile = (*tile).clone();
                async move {
                    // Get the geotransform from the TIFF header
                    let geo_transform = reader.get_geo_transform(&tile).await?
                        .ok_or_else(|| anyhow::anyhow!(
                            "No geotransform found in TIFF for tile {}",
                            tile.tile_id
                        ))?;

                    // Calculate the pixel window within this tile that overlaps with chunk
                    let (window, intersection_bounds) = compute_pixel_window(&tile, &bounds, &cache, &geo_transform)?;
                    reader.read_window(&tile, window, intersection_bounds).await
                }
            })
            .buffer_unordered(self.cog_fetch_concurrency)
            .collect()
            .await;

        // Collect successful reads, log failures
        let mut window_data = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(data) => window_data.push(data),
                Err(e) => {
                    tracing::warn!("Failed to read tile window: {}", e);
                    self.metrics.add_failure();
                }
            }
        }

        Ok(window_data)
    }

    /// Prefetch all COG tiles needed for a set of output chunks.
    ///
    /// This method analyzes all the chunks, finds which COG files they need,
    /// calculates the pixel windows, and prefetches all internal tiles from
    /// each COG in a single batch request per COG.
    ///
    /// # Arguments
    /// * `chunks` - The output chunks to prefetch tiles for
    ///
    /// # Returns
    /// The number of tiles prefetched (not counting cache hits)
    pub async fn prefetch_for_chunks(&self, chunks: &[OutputChunk]) -> Result<usize> {
        use std::collections::HashMap;

        if chunks.is_empty() {
            return Ok(0);
        }

        // Group chunks by the COG tiles they need
        // Map: COG s3_path -> (CogTile, Vec<PixelWindow>)
        let mut cog_windows: HashMap<String, (CogTile, Vec<PixelWindow>)> = HashMap::new();

        let output_grid = self.zarr_writer.output_grid();

        for chunk in chunks {
            // Find intersecting COG tiles
            let tiles = match self.spatial_lookup.tiles_for_chunk(chunk) {
                Ok(t) => t,
                Err(_) => continue,
            };

            if tiles.is_empty() {
                continue;
            }

            // Get chunk bounds in WGS84 for pixel window calculation
            let chunk_bounds_wgs84 = match output_grid.chunk_bounds_wgs84(chunk) {
                Ok(b) => b,
                Err(_) => continue,
            };

            // For each COG tile, calculate the pixel window
            for tile in tiles {
                // Get the geotransform (this will be cached after first fetch)
                let geo_transform = match self.cog_reader.get_geo_transform(tile).await {
                    Ok(Some(gt)) => gt,
                    Ok(None) => {
                        tracing::warn!("No geotransform for tile {}, skipping", tile.tile_id);
                        continue;
                    }
                    Err(_) => continue,
                };

                let (window, _bounds) = match compute_pixel_window(tile, &chunk_bounds_wgs84, &self.proj_cache, &geo_transform) {
                    Ok(w) => w,
                    Err(_) => continue,
                };

                cog_windows
                    .entry(tile.s3_path.clone())
                    .or_insert_with(|| (tile.clone(), Vec::new()))
                    .1
                    .push(window);
            }
        }

        if cog_windows.is_empty() {
            return Ok(0);
        }

        tracing::info!(
            "Prefetching tiles for {} chunks from {} COG files",
            chunks.len(),
            cog_windows.len()
        );

        // Prefetch from each COG concurrently
        let reader = self.cog_reader.clone();
        let mut prefetch_futures = Vec::new();

        for (cog_path, (tile, windows)) in cog_windows {
            let reader = reader.clone();
            prefetch_futures.push(async move {
                let result = reader.prefetch_tiles(&tile, &windows).await;
                match &result {
                    Ok(count) => {
                        if *count > 0 {
                            tracing::debug!(
                                "Prefetched {} tiles from {} ({} windows)",
                                count,
                                cog_path,
                                windows.len()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to prefetch from {}: {}", cog_path, e);
                    }
                }
                result.unwrap_or(0)
            });
        }

        // Wait for all prefetches to complete
        let results: Vec<usize> = futures::future::join_all(prefetch_futures).await;
        let total_prefetched: usize = results.into_iter().sum();

        tracing::info!("Prefetch complete: {} tiles fetched", total_prefetched);

        Ok(total_prefetched)
    }

    /// Process a chunk with retry logic.
    pub async fn process_chunk_with_retry(&self, chunk: OutputChunk) -> Result<ChunkResult> {
        let max_retries = self.config.processing.retry.max_retries;
        let initial_backoff = self.config.processing.retry.initial_backoff_ms;
        let max_backoff = self.config.processing.retry.max_backoff_ms;

        let mut attempt = 0;
        let mut backoff = initial_backoff;

        loop {
            match self.process_chunk(chunk).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= max_retries {
                        tracing::error!(
                            "Chunk {:?} failed after {} attempts: {}",
                            chunk.chunk_indices(),
                            attempt,
                            e
                        );
                        return Err(e);
                    }

                    tracing::warn!(
                        "Chunk {:?} attempt {} failed: {}, retrying in {}ms",
                        chunk.chunk_indices(),
                        attempt,
                        e,
                        backoff
                    );

                    tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                    backoff = (backoff * 2).min(max_backoff);
                }
            }
        }
    }
}

/// Compute the pixel window within a tile that overlaps with the given bounds.
///
/// This function:
/// 1. Transforms chunk bounds from WGS84 to tile's native CRS
/// 2. Computes intersection in native CRS
/// 3. Uses the TIFF's geotransform to convert world coordinates to pixel coordinates
///
/// # Arguments
/// * `tile` - The input tile with bounds in both native CRS and WGS84
/// * `chunk_bounds_wgs84` - Chunk bounds in WGS84 [min_lon, min_lat, max_lon, max_lat]
/// * `proj_cache` - Cache for CRS transformations
/// * `geo_transform` - Affine transform from the TIFF header for world-to-pixel conversion
///
/// # Returns
/// A tuple of (PixelWindow, intersection_bounds_native)
fn compute_pixel_window(
    tile: &CogTile,
    chunk_bounds_wgs84: &[f64; 4],
    proj_cache: &ProjCache,
    geo_transform: &GeoTransform,
) -> Result<(PixelWindow, [f64; 4])> {
    let tile_bounds = tile.bounds_native;

    // Transform chunk bounds from WGS84 to tile's native CRS
    let chunk_bounds_native = crs::transform_bounds(
        chunk_bounds_wgs84,
        crs::codes::WGS84,
        &tile.crs,
        proj_cache,
    )?;

    // Compute intersection in tile's native CRS
    let intersection = crs::intersect_bounds(&chunk_bounds_native, &tile_bounds)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No intersection between chunk (native: {:?}, wgs84: {:?}) and tile {:?}",
                chunk_bounds_native,
                chunk_bounds_wgs84,
                tile_bounds
            )
        })?;

    // Use geotransform to convert intersection bounds to pixel coordinates
    // intersection = [min_x, min_y, max_x, max_y]
    let (col_min, row_min) = geo_transform.world_to_pixel(intersection[0], intersection[1]);
    let (col_max, row_max) = geo_transform.world_to_pixel(intersection[2], intersection[3]);

    // Handle both top-down (negative y scale) and bottom-up (positive y scale) images
    let (row_start, row_end) = if row_min < row_max {
        (row_min, row_max)
    } else {
        (row_max, row_min)
    };

    // Convert to integer pixel coordinates with proper rounding
    let x = col_min.floor().max(0.0) as usize;
    let y = row_start.floor().max(0.0) as usize;
    let x_end = col_max.ceil().max(0.0) as usize;
    let y_end = row_end.ceil().max(0.0) as usize;

    let width = (x_end - x).max(1);
    let height = (y_end - y).max(1);

    tracing::trace!(
        "Pixel window for tile {}: ({}, {}, {}x{}) from intersection {:?} (geotransform: scale_y={})",
        tile.tile_id, x, y, width, height, intersection,
        geo_transform.e
    );

    Ok((PixelWindow::new(x, y, width, height), intersection))
}

/// Result of processing a chunk.
#[derive(Debug)]
pub enum ChunkResult {
    /// Chunk was processed successfully.
    Processed {
        /// Number of input COG tiles that were read and mosaiced.
        tiles_read: usize,
        /// Total bytes read from input tiles (uncompressed pixel data).
        bytes_read: u64,
        /// Total bytes written to output Zarr (uncompressed pixel data).
        bytes_written: u64,
    },

    /// Chunk was skipped because no input tiles intersect this chunk.
    Skipped,
}

impl ChunkResult {
    /// Check if the chunk was processed.
    pub fn is_processed(&self) -> bool {
        matches!(self, ChunkResult::Processed { .. })
    }

    /// Check if the chunk was skipped.
    pub fn is_skipped(&self) -> bool {
        matches!(self, ChunkResult::Skipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tile() -> CogTile {
        CogTile {
            tile_id: "test".to_string(),
            s3_path: "s3://bucket/tile.tif".to_string(),
            crs: "EPSG:32610".to_string(),
            bounds_native: [500000.0, 4000000.0, 581920.0, 4081920.0],
            bounds_wgs84: [-123.0, 36.0, -122.0, 37.0],
            resolution: 10.0,
            year: 2024,
        }
    }

    /// Create a test geotransform matching the test tile.
    /// AEF COGs are bottom-up (origin at bottom-left, positive y scale).
    fn make_test_geotransform() -> GeoTransform {
        // For the test tile:
        // - bounds_native: [500000.0, 4000000.0, 581920.0, 4081920.0]
        // - resolution: 10.0
        // - This is an 8192 x 8192 pixel tile (81920m / 10m = 8192)
        //
        // Bottom-up geotransform:
        // - origin at (500000.0, 4000000.0) which is bottom-left
        // - scale_y = +10.0 (positive for bottom-up)
        GeoTransform {
            a: 10.0,       // pixel width (x scale)
            b: 0.0,        // rotation (typically 0)
            c: 500000.0,   // x origin (min x)
            d: 0.0,        // rotation (typically 0)
            e: 10.0,       // pixel height (positive for bottom-up)
            f: 4000000.0,  // y origin (min y, bottom of image)
        }
    }

    #[test]
    fn test_chunk_result() {
        let processed = ChunkResult::Processed {
            tiles_read: 5,
            bytes_read: 1000,
            bytes_written: 500,
        };
        assert!(processed.is_processed());
        assert!(!processed.is_skipped());

        let skipped = ChunkResult::Skipped;
        assert!(skipped.is_skipped());
        assert!(!skipped.is_processed());
    }

    #[test]
    fn test_compute_pixel_window_full_overlap() {
        let tile = make_test_tile();
        let proj_cache = ProjCache::new();
        let geo_transform = make_test_geotransform();

        // Chunk fully inside tile (in WGS84)
        let chunk_bounds = [-122.8, 36.2, -122.2, 36.8];
        let (window, bounds) = compute_pixel_window(&tile, &chunk_bounds, &proj_cache, &geo_transform).unwrap();

        // Window should be roughly in the middle of the tile
        assert!(window.x > 0, "x should be > 0, got {}", window.x);
        assert!(window.y > 0, "y should be > 0, got {}", window.y);
        assert!(window.width > 0, "width should be > 0, got {}", window.width);
        assert!(window.height > 0, "height should be > 0, got {}", window.height);

        // Window should not be the full tile
        assert!(window.width < 8192, "width should be < 8192, got {}", window.width);
        assert!(window.height < 8192, "height should be < 8192, got {}", window.height);

        // Bounds should be valid
        assert!(bounds[0] < bounds[2], "min_x should be less than max_x");
        assert!(bounds[1] < bounds[3], "min_y should be less than max_y");
    }

    #[test]
    fn test_compute_pixel_window_partial_overlap() {
        let tile = make_test_tile();
        let proj_cache = ProjCache::new();
        let geo_transform = make_test_geotransform();

        // Chunk extends beyond tile on the east
        let chunk_bounds = [-122.5, 36.3, -121.5, 36.7];
        let (window, _bounds) = compute_pixel_window(&tile, &chunk_bounds, &proj_cache, &geo_transform).unwrap();

        // Window should have valid dimensions
        assert!(window.width > 0);
        assert!(window.height > 0);
    }

    #[test]
    fn test_compute_pixel_window_no_overlap() {
        let tile = make_test_tile();
        let proj_cache = ProjCache::new();
        let geo_transform = make_test_geotransform();

        // Chunk completely outside tile (in WGS84)
        let chunk_bounds = [-121.0, 38.0, -120.0, 39.0];
        let result = compute_pixel_window(&tile, &chunk_bounds, &proj_cache, &geo_transform);

        assert!(result.is_err());
    }

    #[test]
    fn test_compute_pixel_window_accuracy() {
        // Create a tile with known bounds for easier verification
        let tile = CogTile {
            tile_id: "test".to_string(),
            s3_path: "s3://bucket/tile.tif".to_string(),
            crs: "EPSG:32610".to_string(),
            // 81920m x 81920m tile = 8192 pixels at 10m resolution
            bounds_native: [500000.0, 4000000.0, 581920.0, 4081920.0],
            bounds_wgs84: [-123.0, 36.0, -122.0, 37.0],
            resolution: 10.0,
            year: 2024,
        };
        let proj_cache = ProjCache::new();
        let geo_transform = make_test_geotransform();

        // Request the center 50% of the tile
        // This is approximate since WGS84->UTM is non-linear
        let chunk_bounds = [-122.75, 36.25, -122.25, 36.75];
        let (window, _bounds) = compute_pixel_window(&tile, &chunk_bounds, &proj_cache, &geo_transform).unwrap();

        // The window should be roughly centered and cover about 25-50% of the tile
        // (exact values depend on UTM projection distortion)
        let tile_pixels = 8192;
        let window_area = window.width * window.height;
        let tile_area = tile_pixels * tile_pixels;
        let coverage = window_area as f64 / tile_area as f64;

        assert!(
            coverage > 0.1 && coverage < 0.6,
            "Expected ~25% coverage, got {:.1}% (window: {}x{} at ({}, {}))",
            coverage * 100.0,
            window.width,
            window.height,
            window.x,
            window.y
        );
    }
}
