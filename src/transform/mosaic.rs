//! Mosaicing overlapping tiles using forward mapping and mean aggregation.
//!
//! This module uses forward mapping (source → dest) instead of inverse mapping
//! (dest → source) to achieve sequential source reads, which are cache-friendly.
//! The tradeoff is scattered destination writes, but writes are faster than
//! random reads due to write combining in modern CPUs.
//!
//! Tiles are processed in parallel using Rayon, with an atomic accumulator
//! to handle concurrent writes to overlapping destination pixels.

use crate::io::WindowData;
use crate::transform::ReprojectConfig;
use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Array4, Axis, Zip, s};
use ndarray::parallel::prelude::*;
use proj::Proj;
use std::sync::atomic::{AtomicI32, AtomicU16, Ordering};
use std::time::Instant;

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Grid cell size for forward mapping interpolation.
/// Smaller = more Proj calls but better accuracy.
/// 32x32 is a good balance for typical reprojection scenarios.
const FORWARD_GRID_SIZE: usize = 32;

/// Thread-safe accumulator for parallel mosaicing using atomic operations.
///
/// Uses AtomicI32 for sum to handle many overlapping tiles without overflow.
/// Uses AtomicU16 for count (max 65535 overlapping tiles per pixel).
pub struct AtomicAccumulator {
    /// Sum of values for each pixel: flat array [band * height * width + row * width + col]
    sum: Vec<AtomicI32>,

    /// Count of contributions for each pixel: flat array [row * width + col]
    count: Vec<AtomicU16>,

    /// Number of bands
    bands: usize,

    /// Height in pixels
    height: usize,

    /// Width in pixels
    width: usize,
}

impl AtomicAccumulator {
    /// Create a new atomic accumulator for the given dimensions.
    pub fn new(bands: usize, height: usize, width: usize) -> Self {
        let pixel_count = height * width;
        let band_pixel_count = bands * pixel_count;

        // Initialize with zeros using Vec and then convert
        let sum: Vec<AtomicI32> = (0..band_pixel_count)
            .map(|_| AtomicI32::new(0))
            .collect();
        let count: Vec<AtomicU16> = (0..pixel_count)
            .map(|_| AtomicU16::new(0))
            .collect();

        Self {
            sum,
            count,
            bands,
            height,
            width,
        }
    }

    /// Accumulate a single pixel with band values.
    /// Thread-safe using atomic operations.
    #[inline]
    pub fn accumulate(&self, bands_data: &[i8], dst_row: usize, dst_col: usize) {
        debug_assert!(dst_row < self.height);
        debug_assert!(dst_col < self.width);
        debug_assert_eq!(bands_data.len(), self.bands);

        // Check if any band has valid data
        let has_data = bands_data.iter().any(|&v| v != NODATA);
        if !has_data {
            return;
        }

        let pixel_idx = dst_row * self.width + dst_col;

        // Increment count atomically
        self.count[pixel_idx].fetch_add(1, Ordering::Relaxed);

        // Add each band value atomically
        for (b, &val) in bands_data.iter().enumerate() {
            let band_idx = b * self.height * self.width + pixel_idx;
            self.sum[band_idx].fetch_add(val as i32, Ordering::Relaxed);
        }
    }

    /// Finalize the mosaic by computing the mean.
    ///
    /// Returns an int8 array: (1, bands, height, width)
    /// Pixels with no data are set to NODATA (-128).
    ///
    /// Uses ndarray's parallel Zip for efficient array operations.
    pub fn finalize(self) -> Array4<i8> {
        let width = self.width;
        let height = self.height;
        let bands = self.bands;

        // Convert atomics to regular values (just reads inner values, no sync overhead)
        let sum_data: Vec<i32> = self.sum.into_iter()
            .map(|a| a.into_inner())
            .collect();
        let count_data: Vec<u16> = self.count.into_iter()
            .map(|a| a.into_inner())
            .collect();

        // Create ndarray views - sum is [bands, height, width], count is [height, width]
        let sum_arr = Array3::<i32>::from_shape_vec((bands, height, width), sum_data)
            .expect("sum shape mismatch");
        let count_arr = Array2::<u16>::from_shape_vec((height, width), count_data)
            .expect("count shape mismatch");

        // Create result array initialized to NODATA
        let mut result = Array4::<i8>::from_elem((1, bands, height, width), NODATA);

        // Process each band using parallel Zip (broadcasts count across pixels)
        for b in 0..bands {
            let sum_band = sum_arr.index_axis(Axis(0), b);
            let mut result_band = result.slice_mut(s![0, b, .., ..]);

            Zip::from(&mut result_band)
                .and(&sum_band)
                .and(&count_arr)
                .par_for_each(|r, &s, &c| {
                    if c > 0 {
                        let c = c as i32;
                        let half_c = c / 2;
                        *r = if s >= 0 {
                            ((s + half_c) / c) as i8
                        } else {
                            ((s - half_c) / c) as i8
                        };
                    }
                });
        }

        result
    }

    /// Get the maximum overlap count.
    pub fn max_overlap(&self) -> u16 {
        self.count
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }
}

// SAFETY: AtomicAccumulator only contains atomic types which are Sync
unsafe impl Sync for AtomicAccumulator {}

/// A grid cell for forward mapping interpolation.
/// Contains pre-projected corner coordinates to enable fast bilinear interpolation.
#[derive(Debug, Clone, Copy)]
struct ForwardCell {
    /// Source pixel bounds [x0, y0, x1, y1]
    src_x0: usize,
    src_y0: usize,
    src_x1: usize,
    src_y1: usize,
    /// Destination pixel coordinates at corners (after projection)
    /// Order: [top-left, top-right, bottom-left, bottom-right]
    /// Each is (dst_col, dst_row) in floating point
    dst_corners: [(f64, f64); 4],
    /// Whether all corners projected successfully
    valid: bool,
}

impl ForwardCell {
    /// Interpolate destination coordinates for a source pixel within this cell.
    #[inline]
    fn interpolate(&self, src_col: usize, src_row: usize) -> (f64, f64) {
        let cell_width = (self.src_x1 - self.src_x0) as f64;
        let cell_height = (self.src_y1 - self.src_y0) as f64;

        // Normalized position within cell [0, 1]
        let t_col = if cell_width > 0.0 {
            (src_col - self.src_x0) as f64 / cell_width
        } else {
            0.0
        };
        let t_row = if cell_height > 0.0 {
            (src_row - self.src_y0) as f64 / cell_height
        } else {
            0.0
        };

        let [(x00, y00), (x01, y01), (x10, y10), (x11, y11)] = self.dst_corners;

        // Bilinear interpolation
        let dst_col = x00 * (1.0 - t_row) * (1.0 - t_col)
            + x01 * (1.0 - t_row) * t_col
            + x10 * t_row * (1.0 - t_col)
            + x11 * t_row * t_col;

        let dst_row = y00 * (1.0 - t_row) * (1.0 - t_col)
            + y01 * (1.0 - t_row) * t_col
            + y10 * t_row * (1.0 - t_col)
            + y11 * t_row * t_col;

        (dst_col, dst_row)
    }
}

/// Build a grid of forward-projected cells for a source image.
fn build_forward_grid(
    src_width: usize,
    src_height: usize,
    source_bounds: &[f64; 4],
    fwd_proj: &Proj,
    target_bounds: &[f64; 4],
    target_resolution: f64,
) -> Vec<ForwardCell> {
    let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
    let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

    let inv_target_res = 1.0 / target_resolution;
    let target_min_x = target_bounds[0];
    let target_max_y = target_bounds[3]; // Top of image (row 0)

    let mut cells = Vec::new();

    let mut src_y = 0;
    while src_y < src_height {
        let src_y1 = (src_y + FORWARD_GRID_SIZE).min(src_height);

        let mut src_x = 0;
        while src_x < src_width {
            let src_x1 = (src_x + FORWARD_GRID_SIZE).min(src_width);

            // Project the 4 corners of this cell
            let corners_src = [
                (src_x, src_y),         // top-left (in source, which is bottom-up)
                (src_x1, src_y),        // top-right
                (src_x, src_y1),        // bottom-left
                (src_x1, src_y1),       // bottom-right
            ];

            let mut dst_corners = [(0.0, 0.0); 4];
            let mut valid = true;

            for (i, &(px, py)) in corners_src.iter().enumerate() {
                // Source pixel to world coordinates
                // Source is bottom-up: row 0 is at min_y
                let world_x = source_bounds[0] + (px as f64 + 0.5) * src_pixel_x;
                let world_y = source_bounds[1] + (py as f64 + 0.5) * src_pixel_y;

                // Project to target CRS
                match fwd_proj.convert((world_x, world_y)) {
                    Ok((target_x, target_y)) => {
                        // Convert to target pixel coordinates (target is top-down)
                        let dst_col = (target_x - target_min_x) * inv_target_res;
                        let dst_row = (target_max_y - target_y) * inv_target_res;
                        dst_corners[i] = (dst_col, dst_row);
                    }
                    Err(_) => {
                        valid = false;
                        break;
                    }
                }
            }

            cells.push(ForwardCell {
                src_x0: src_x,
                src_y0: src_y,
                src_x1,
                src_y1,
                dst_corners,
                valid,
            });

            src_x = src_x1;
        }
        src_y = src_y1;
    }

    cells
}

/// Forward-map a single tile directly to the accumulator.
///
/// This uses forward mapping (source → dest) instead of inverse mapping.
/// Source pixels are read sequentially (cache-friendly), destination writes
/// are scattered but use atomic operations for thread safety.
fn forward_map_tile(
    window: &WindowData,
    config: &ReprojectConfig,
    accumulator: &AtomicAccumulator,
) -> Result<usize> {
    let data = &window.data;
    let (bands, src_height, src_width) = data.dim();
    let (dst_height, dst_width) = config.target_shape;

    let source_bounds = &window.bounds_native;
    let source_crs = &window.tile.crs;

    // Create forward projection: source CRS → target CRS
    let fwd_proj = Proj::new_known_crs(source_crs, &config.target_crs, None)
        .with_context(|| format!("Failed to create Proj for {} -> {}", source_crs, config.target_crs))?;

    // Build the forward grid
    let grid = build_forward_grid(
        src_width,
        src_height,
        source_bounds,
        &fwd_proj,
        &config.target_bounds,
        config.target_resolution,
    );

    let mut pixels_written = 0usize;

    // Pre-allocate a buffer for reading bands (avoids repeated allocations)
    let mut band_buf: Vec<i8> = vec![0; bands];

    // Process each cell
    for cell in &grid {
        if !cell.valid {
            continue;
        }

        // Process each source pixel in this cell (sequential reads!)
        for src_row in cell.src_y0..cell.src_y1 {
            for src_col in cell.src_x0..cell.src_x1 {
                // Interpolate destination coordinates
                let (dst_col_f, dst_row_f) = cell.interpolate(src_col, src_row);

                // Round to nearest pixel
                let dst_col = dst_col_f.round() as isize;
                let dst_row = dst_row_f.round() as isize;

                // Bounds check
                if dst_col < 0 || dst_col >= dst_width as isize
                    || dst_row < 0 || dst_row >= dst_height as isize
                {
                    continue;
                }

                let dst_col = dst_col as usize;
                let dst_row = dst_row as usize;

                // Read all bands from source (sequential memory access)
                let mut has_data = false;
                for b in 0..bands {
                    let val = data[[b, src_row, src_col]];
                    band_buf[b] = val;
                    if val != NODATA {
                        has_data = true;
                    }
                }

                if !has_data {
                    continue;
                }

                // Accumulate to destination (atomic, thread-safe)
                accumulator.accumulate(&band_buf, dst_row, dst_col);
                pixels_written += 1;
            }
        }
    }

    Ok(pixels_written)
}

/// Mosaic multiple tile windows into a single output chunk using parallel forward mapping.
///
/// This function uses forward mapping (source → dest) for cache-friendly source reads.
/// Tiles are processed in parallel using Rayon, with an atomic accumulator
/// handling concurrent writes to overlapping destination pixels.
pub fn mosaic_tiles(
    windows: &[WindowData],
    _reprojector: &crate::transform::Reprojector, // Unused, kept for API compatibility
    reproject_config: &ReprojectConfig,
) -> Result<Array4<i8>> {
    if windows.is_empty() {
        return Ok(Array4::zeros((
            1,
            reproject_config.num_bands,
            reproject_config.target_shape.0,
            reproject_config.target_shape.1,
        )));
    }

    let bands = windows[0].data.dim().0;
    let (height, width) = reproject_config.target_shape;

    // Log input sizes
    let total_input_pixels: usize = windows.iter()
        .map(|w| w.data.dim().1 * w.data.dim().2)
        .sum();
    let output_pixels = height * width;

    tracing::info!(
        num_tiles = windows.len(),
        total_input_pixels = total_input_pixels,
        output_pixels = output_pixels,
        input_output_ratio = format!("{:.2}", total_input_pixels as f64 / output_pixels as f64),
        "mosaic_tiles starting (parallel forward mapping)"
    );

    // Create atomic accumulator (thread-safe)
    let accumulator = AtomicAccumulator::new(bands, height, width);

    // Process tiles in parallel using Rayon
    let map_start = Instant::now();

    let results: Vec<Result<usize>> = windows
        .par_iter()
        .enumerate()
        .map(|(i, window)| {
            let tile_start = Instant::now();
            let (_, src_h, src_w) = window.data.dim();

            let pixels_written = forward_map_tile(window, reproject_config, &accumulator)?;

            tracing::info!(
                tile = i,
                input_shape = ?(src_h, src_w),
                pixels_written = pixels_written,
                tile_ms = tile_start.elapsed().as_millis(),
                "forward_map_tile completed"
            );

            Ok(pixels_written)
        })
        .collect();

    let map_time = map_start.elapsed();

    // Sum up pixels written, propagate any errors
    let mut total_pixels_written = 0usize;
    for result in results {
        total_pixels_written += result?;
    }

    tracing::info!(
        total_pixels_written = total_pixels_written,
        map_ms = map_time.as_millis(),
        pixels_per_ms = if map_time.as_millis() > 0 {
            total_pixels_written as f64 / map_time.as_millis() as f64
        } else {
            0.0
        },
        "parallel forward mapping completed"
    );

    // Finalize (compute means) - this is also parallelized internally
    let finalize_start = Instant::now();
    let result = accumulator.finalize();
    let finalize_time = finalize_start.elapsed();

    tracing::info!(
        finalize_ms = finalize_time.as_millis(),
        "accumulator finalized"
    );

    Ok(result)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_accumulator_single_pixel() {
        let acc = AtomicAccumulator::new(2, 4, 4);

        acc.accumulate(&[10, 20], 1, 1);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 1, 1]], 10);
        assert_eq!(result[[0, 1, 1, 1]], 20);
    }

    #[test]
    fn test_atomic_accumulator_mean() {
        let acc = AtomicAccumulator::new(1, 2, 2);

        // Accumulate two values at same location
        acc.accumulate(&[10], 0, 0);
        acc.accumulate(&[20], 0, 0);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 0, 0]], 15); // Mean of 10 and 20
    }

    #[test]
    fn test_atomic_accumulator_nodata_skip() {
        let acc = AtomicAccumulator::new(2, 2, 2);

        // All nodata should not increment count
        acc.accumulate(&[NODATA, NODATA], 0, 0);

        assert_eq!(acc.count[0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_atomic_accumulator_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let acc = Arc::new(AtomicAccumulator::new(1, 10, 10));

        // Spawn multiple threads writing to same pixel
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let acc = Arc::clone(&acc);
                thread::spawn(move || {
                    acc.accumulate(&[10], 5, 5);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Count should be 10, sum should be 100, mean should be 10
        assert_eq!(acc.count[5 * 10 + 5].load(Ordering::Relaxed), 10);

        // Can't easily test finalize without consuming, but count is correct
    }

    #[test]
    fn test_forward_cell_interpolation() {
        let cell = ForwardCell {
            src_x0: 0,
            src_y0: 0,
            src_x1: 10,
            src_y1: 10,
            dst_corners: [
                (0.0, 0.0),   // top-left
                (10.0, 0.0),  // top-right
                (0.0, 10.0),  // bottom-left
                (10.0, 10.0), // bottom-right
            ],
            valid: true,
        };

        // Center should interpolate to (5, 5)
        let (x, y) = cell.interpolate(5, 5);
        assert!((x - 5.0).abs() < 0.01);
        assert!((y - 5.0).abs() < 0.01);

        // Top-left corner
        let (x, y) = cell.interpolate(0, 0);
        assert!((x - 0.0).abs() < 0.01);
        assert!((y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_max_overlap() {
        let acc = AtomicAccumulator::new(1, 2, 2);

        acc.accumulate(&[10], 0, 0);
        acc.accumulate(&[20], 0, 0);
        acc.accumulate(&[30], 0, 0);

        assert_eq!(acc.max_overlap(), 3);
    }
}
