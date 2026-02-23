//! Mosaicing overlapping tiles using forward mapping and mean aggregation.
//!
//! This module uses forward mapping (source → dest) instead of inverse mapping
//! (dest → source) to achieve sequential source reads, which are cache-friendly.
//! The tradeoff is scattered destination writes, but writes are faster than
//! random reads due to write combining in modern CPUs.

use crate::io::WindowData;
use crate::transform::ReprojectConfig;
use anyhow::{Context, Result};
use ndarray::{s, Array2, Array3, Array4, Zip};
use proj::Proj;
use std::time::Instant;

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Grid cell size for forward mapping interpolation.
/// Smaller = more Proj calls but better accuracy.
/// 32x32 is a good balance for typical reprojection scenarios.
const FORWARD_GRID_SIZE: usize = 32;

/// Accumulator for mosaicing multiple tiles using mean aggregation.
///
/// Uses i16 accumulators which can handle up to ~250 overlapping tiles without overflow.
/// (i8 range is -128 to 127; worst case 127 * 250 = 31,750 < i16 max of 32,767)
/// This halves memory usage compared to i32 accumulators.
#[derive(Debug)]
pub struct MosaicAccumulator {
    /// Sum of values for each pixel: (bands, height, width)
    /// Using 3D instead of 4D for simpler indexing in pixel-level operations
    sum: Array3<i16>,

    /// Count of contributions for each pixel: (height, width)
    count: Array2<u16>,

    /// Number of bands
    bands: usize,

    /// Height in pixels
    height: usize,

    /// Width in pixels
    width: usize,
}

impl MosaicAccumulator {
    /// Create a new accumulator for the given dimensions.
    pub fn new(bands: usize, height: usize, width: usize) -> Self {
        Self {
            sum: Array3::zeros((bands, height, width)),
            count: Array2::zeros((height, width)),
            bands,
            height,
            width,
        }
    }

    /// Accumulate a single pixel from source data directly.
    ///
    /// This is the core operation for forward mapping - we read source pixels
    /// sequentially and write to scattered destination locations.
    #[inline]
    pub fn accumulate_pixel(
        &mut self,
        src_data: &Array3<i8>,
        src_row: usize,
        src_col: usize,
        dst_row: usize,
        dst_col: usize,
    ) {
        // Check if source pixel has valid data (any band non-nodata)
        let mut has_data = false;
        for b in 0..self.bands {
            if src_data[[b, src_row, src_col]] != NODATA {
                has_data = true;
                break;
            }
        }

        if !has_data {
            return;
        }

        // Accumulate all bands
        for b in 0..self.bands {
            let val = src_data[[b, src_row, src_col]];
            self.sum[[b, dst_row, dst_col]] += val as i16;
        }
        self.count[[dst_row, dst_col]] += 1;
    }

    /// Accumulate a single pixel with pre-fetched band values.
    /// Used when we've already read the source pixel.
    #[inline]
    pub fn accumulate_bands(&mut self, bands: &[i8], dst_row: usize, dst_col: usize) {
        // Check if any band has valid data
        let has_data = bands.iter().any(|&v| v != NODATA);
        if !has_data {
            return;
        }

        // Accumulate all bands
        for (b, &val) in bands.iter().enumerate() {
            self.sum[[b, dst_row, dst_col]] += val as i16;
        }
        self.count[[dst_row, dst_col]] += 1;
    }

    /// Finalize the mosaic by computing the mean.
    ///
    /// Returns an int8 array: (1, bands, height, width)
    /// Pixels with no data are set to NODATA (-128).
    pub fn finalize(self) -> Array4<i8> {
        // Initialize with NODATA
        let mut result = Array4::<i8>::from_elem((1, self.bands, self.height, self.width), NODATA);

        // Process each band using Zip
        for b in 0..self.bands {
            let sum_band = self.sum.slice(s![b, .., ..]);
            let mut result_band = result.slice_mut(s![0, b, .., ..]);

            Zip::from(&mut result_band)
                .and(&sum_band)
                .and(&self.count)
                .for_each(|result, &sum, &count| {
                    if count > 0 {
                        let c = count as i16;
                        let half_c = c / 2;
                        // Rounded integer division
                        *result = if sum >= 0 {
                            ((sum + half_c) / c) as i8
                        } else {
                            ((sum - half_c) / c) as i8
                        };
                    }
                });
        }

        result
    }

    /// Get the maximum overlap count.
    pub fn max_overlap(&self) -> u16 {
        *self.count.iter().max().unwrap_or(&0)
    }

    /// Check if any pixels have data.
    pub fn has_data(&self) -> bool {
        self.count.iter().any(|&c| c > 0)
    }
}

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
/// are scattered but benefit from write combining.
fn forward_map_tile(
    window: &WindowData,
    config: &ReprojectConfig,
    accumulator: &mut MosaicAccumulator,
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

                // Accumulate to destination
                accumulator.accumulate_bands(&band_buf, dst_row, dst_col);
                pixels_written += 1;
            }
        }
    }

    Ok(pixels_written)
}

/// Mosaic multiple tile windows into a single output chunk using forward mapping.
///
/// This function uses forward mapping (source → dest) for cache-friendly source reads.
/// Each source tile is processed sequentially, with pixels accumulated directly
/// to the output without creating intermediate reprojected arrays.
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
        "mosaic_tiles starting (forward mapping)"
    );

    // Create accumulator
    let mut accumulator = MosaicAccumulator::new(bands, height, width);

    // Process each tile with forward mapping
    // Sequential processing avoids synchronization overhead
    let map_start = Instant::now();
    let mut total_pixels_written = 0usize;

    for (i, window) in windows.iter().enumerate() {
        let tile_start = Instant::now();
        let (_, src_h, src_w) = window.data.dim();

        let pixels_written = forward_map_tile(window, reproject_config, &mut accumulator)?;
        total_pixels_written += pixels_written;

        tracing::info!(
            tile = i,
            input_shape = ?(src_h, src_w),
            pixels_written = pixels_written,
            tile_ms = tile_start.elapsed().as_millis(),
            "forward_map_tile completed"
        );
    }

    let map_time = map_start.elapsed();

    tracing::info!(
        total_pixels_written = total_pixels_written,
        map_ms = map_time.as_millis(),
        pixels_per_ms = total_pixels_written as f64 / map_time.as_millis() as f64,
        "forward mapping completed"
    );

    Ok(accumulator.finalize())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_accumulator_single_pixel() {
        let mut acc = MosaicAccumulator::new(2, 4, 4);

        // Create source data with one valid pixel
        let mut data = Array3::from_elem((2, 2, 2), NODATA);
        data[[0, 0, 0]] = 10;
        data[[1, 0, 0]] = 20;

        acc.accumulate_pixel(&data, 0, 0, 1, 1);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 1, 1]], 10);
        assert_eq!(result[[0, 1, 1, 1]], 20);
    }

    #[test]
    fn test_accumulator_mean_pixels() {
        let mut acc = MosaicAccumulator::new(1, 2, 2);

        // Accumulate two values at same location
        acc.accumulate_bands(&[10], 0, 0);
        acc.accumulate_bands(&[20], 0, 0);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 0, 0]], 15); // Mean of 10 and 20
    }

    #[test]
    fn test_accumulator_nodata_skip() {
        let mut acc = MosaicAccumulator::new(2, 2, 2);

        // All nodata should not increment count
        acc.accumulate_bands(&[NODATA, NODATA], 0, 0);

        assert_eq!(acc.count[[0, 0]], 0);
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
        let mut acc = MosaicAccumulator::new(1, 2, 2);

        acc.accumulate_bands(&[10], 0, 0);
        acc.accumulate_bands(&[20], 0, 0);
        acc.accumulate_bands(&[30], 0, 0);

        assert_eq!(acc.max_overlap(), 3);
    }
}
