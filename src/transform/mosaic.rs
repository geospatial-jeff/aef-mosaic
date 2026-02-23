//! Mosaicing overlapping tiles using block-parallel inverse mapping.
//!
//! This module processes output blocks in parallel, using inverse mapping
//! (destination → source) to sample from overlapping input tiles. Each block
//! maintains its own local accumulator, eliminating the need for atomic operations.
//! Mean computation uses ndarray's vectorized Zip operations.

use crate::io::WindowData;
use crate::transform::ReprojectConfig;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, Axis, Zip, s};
use proj::Proj;
use rayon::prelude::*;
use std::time::Instant;

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Block size for parallel processing.
/// Each output block is processed independently with its own accumulator.
const BLOCK_SIZE: usize = 256;

/// Grid cell size for inverse mapping interpolation.
/// Pre-project grid corners to reduce Proj calls.
const GRID_SIZE: usize = 32;

/// A grid cell for inverse mapping interpolation.
/// Contains pre-projected corner coordinates for bilinear interpolation.
#[derive(Debug, Clone)]
struct InverseGridCell {
    /// Output pixel bounds in the block [row0, col0, row1, col1]
    out_row0: usize,
    out_col0: usize,
    out_row1: usize,
    out_col1: usize,
    /// Source coordinates at corners (after inverse projection)
    /// Order: [top-left, top-right, bottom-left, bottom-right]
    /// Each is (src_col, src_row) in floating point
    src_corners: [(f64, f64); 4],
    /// Whether all corners projected successfully
    valid: bool,
}

impl InverseGridCell {
    /// Interpolate source coordinates for an output pixel within this cell.
    #[inline]
    fn interpolate(&self, out_row: usize, out_col: usize) -> (f64, f64) {
        let cell_height = (self.out_row1 - self.out_row0) as f64;
        let cell_width = (self.out_col1 - self.out_col0) as f64;

        // Normalized position within cell [0, 1]
        let t_row = if cell_height > 0.0 {
            (out_row - self.out_row0) as f64 / cell_height
        } else {
            0.0
        };
        let t_col = if cell_width > 0.0 {
            (out_col - self.out_col0) as f64 / cell_width
        } else {
            0.0
        };

        let [(x00, y00), (x01, y01), (x10, y10), (x11, y11)] = self.src_corners;

        // Bilinear interpolation
        let src_col = x00 * (1.0 - t_row) * (1.0 - t_col)
            + x01 * (1.0 - t_row) * t_col
            + x10 * t_row * (1.0 - t_col)
            + x11 * t_row * t_col;

        let src_row = y00 * (1.0 - t_row) * (1.0 - t_col)
            + y01 * (1.0 - t_row) * t_col
            + y10 * t_row * (1.0 - t_col)
            + y11 * t_row * t_col;

        (src_col, src_row)
    }
}

/// Build an inverse grid for mapping output block pixels to source tile pixels.
fn build_inverse_grid(
    block_row_start: usize,
    block_col_start: usize,
    block_height: usize,
    block_width: usize,
    target_bounds: &[f64; 4],
    target_resolution: f64,
    source_bounds: &[f64; 4],
    src_height: usize,
    src_width: usize,
    inv_proj: &Proj,
) -> Vec<InverseGridCell> {
    let target_min_x = target_bounds[0];
    let target_max_y = target_bounds[3];

    let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
    let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;
    let src_min_x = source_bounds[0];
    let src_min_y = source_bounds[1];

    let mut cells = Vec::new();

    let mut out_row = 0;
    while out_row < block_height {
        let out_row1 = (out_row + GRID_SIZE).min(block_height);

        let mut out_col = 0;
        while out_col < block_width {
            let out_col1 = (out_col + GRID_SIZE).min(block_width);

            // Project the 4 corners of this cell from output to source
            let corners_out = [
                (out_row, out_col),     // top-left
                (out_row, out_col1),    // top-right
                (out_row1, out_col),    // bottom-left
                (out_row1, out_col1),   // bottom-right
            ];

            let mut src_corners = [(0.0, 0.0); 4];
            let mut valid = true;

            for (i, &(row, col)) in corners_out.iter().enumerate() {
                // Output pixel to world coordinates (target CRS)
                let global_row = block_row_start + row;
                let global_col = block_col_start + col;
                let world_x = target_min_x + (global_col as f64 + 0.5) * target_resolution;
                let world_y = target_max_y - (global_row as f64 + 0.5) * target_resolution;

                // Inverse project to source CRS
                match inv_proj.convert((world_x, world_y)) {
                    Ok((src_world_x, src_world_y)) => {
                        // Convert to source pixel coordinates
                        let src_col_f = (src_world_x - src_min_x) / src_pixel_x - 0.5;
                        let src_row_f = (src_world_y - src_min_y) / src_pixel_y - 0.5;
                        src_corners[i] = (src_col_f, src_row_f);
                    }
                    Err(_) => {
                        valid = false;
                        break;
                    }
                }
            }

            cells.push(InverseGridCell {
                out_row0: out_row,
                out_col0: out_col,
                out_row1,
                out_col1,
                src_corners,
                valid,
            });

            out_col = out_col1;
        }
        out_row = out_row1;
    }

    cells
}

/// Local block accumulator using ndarray (no atomics needed).
struct BlockAccumulator {
    /// Sum of values: [bands, block_height, block_width]
    sum: Array3<i32>,
    /// Count of contributions: [block_height, block_width]
    count: Array2<u16>,
    bands: usize,
}

impl BlockAccumulator {
    fn new(bands: usize, height: usize, width: usize) -> Self {
        Self {
            sum: Array3::zeros((bands, height, width)),
            count: Array2::zeros((height, width)),
            bands,
        }
    }

    /// Accumulate a pixel value at the given local coordinates.
    #[inline]
    fn accumulate(&mut self, data: &Array3<i8>, src_row: usize, src_col: usize, out_row: usize, out_col: usize) {
        let mut has_data = false;

        for b in 0..self.bands {
            let val = data[[b, src_row, src_col]];
            if val != NODATA {
                has_data = true;
                self.sum[[b, out_row, out_col]] += val as i32;
            }
        }

        if has_data {
            self.count[[out_row, out_col]] += 1;
        }
    }

    /// Finalize using ndarray Zip for vectorized mean computation.
    fn finalize(self) -> Array3<i8> {
        let (bands, height, width) = self.sum.dim();
        let mut result = Array3::<i8>::from_elem((bands, height, width), NODATA);

        // Compute means using ndarray Zip (vectorized, efficient)
        for b in 0..bands {
            let sum_band = self.sum.index_axis(Axis(0), b);
            let mut result_band = result.index_axis_mut(Axis(0), b);

            Zip::from(&mut result_band)
                .and(&sum_band)
                .and(&self.count)
                .for_each(|r, &s, &c| {
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
}

/// Process a single output block using inverse mapping.
fn process_block(
    block_row_start: usize,
    block_col_start: usize,
    block_height: usize,
    block_width: usize,
    bands: usize,
    windows: &[WindowData],
    config: &ReprojectConfig,
) -> Result<Array3<i8>> {
    let mut accumulator = BlockAccumulator::new(bands, block_height, block_width);

    // Process each source tile
    for window in windows {
        let data = &window.data;
        let (_, src_height, src_width) = data.dim();
        let source_bounds = &window.bounds_native;
        let source_crs = &window.tile.crs;

        // Create inverse projection: target CRS → source CRS
        let inv_proj = match Proj::new_known_crs(&config.target_crs, source_crs, None) {
            Ok(p) => p,
            Err(_) => continue, // Skip tiles we can't project
        };

        // Build the inverse grid for this tile
        let grid = build_inverse_grid(
            block_row_start,
            block_col_start,
            block_height,
            block_width,
            &config.target_bounds,
            config.target_resolution,
            source_bounds,
            src_height,
            src_width,
            &inv_proj,
        );

        // Process each cell in the grid
        for cell in &grid {
            if !cell.valid {
                continue;
            }

            // Process each output pixel in this cell
            for out_row in cell.out_row0..cell.out_row1 {
                for out_col in cell.out_col0..cell.out_col1 {
                    // Interpolate source coordinates
                    let (src_col_f, src_row_f) = cell.interpolate(out_row, out_col);

                    // Nearest neighbor sampling with bounds check
                    let src_col = src_col_f.round() as isize;
                    let src_row = src_row_f.round() as isize;

                    if src_col < 0 || src_col >= src_width as isize
                        || src_row < 0 || src_row >= src_height as isize
                    {
                        continue;
                    }

                    accumulator.accumulate(
                        data,
                        src_row as usize,
                        src_col as usize,
                        out_row,
                        out_col,
                    );
                }
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Mosaic multiple tile windows into a single output chunk using block-parallel inverse mapping.
///
/// This function splits the output into blocks and processes them in parallel.
/// Each block uses inverse mapping (dest → source) with local accumulators,
/// then computes means using ndarray's vectorized operations.
pub fn mosaic_tiles(
    windows: &[WindowData],
    _reprojector: &crate::transform::Reprojector,
    reproject_config: &ReprojectConfig,
) -> Result<Array4<i8>> {
    let (height, width) = reproject_config.target_shape;
    let bands = reproject_config.num_bands;

    if windows.is_empty() {
        return Ok(Array4::from_elem((1, bands, height, width), NODATA));
    }

    // Log input sizes
    let total_input_pixels: usize = windows
        .iter()
        .map(|w| w.data.dim().1 * w.data.dim().2)
        .sum();
    let output_pixels = height * width;

    let num_block_rows = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let num_block_cols = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_blocks = num_block_rows * num_block_cols;

    tracing::info!(
        num_tiles = windows.len(),
        total_input_pixels = total_input_pixels,
        output_pixels = output_pixels,
        num_blocks = total_blocks,
        block_size = BLOCK_SIZE,
        "mosaic_tiles starting (block-parallel inverse mapping)"
    );

    let process_start = Instant::now();

    // Generate all block coordinates
    let blocks: Vec<(usize, usize, usize, usize)> = (0..num_block_rows)
        .flat_map(|br| {
            (0..num_block_cols).map(move |bc| {
                let row_start = br * BLOCK_SIZE;
                let col_start = bc * BLOCK_SIZE;
                let block_height = BLOCK_SIZE.min(height - row_start);
                let block_width = BLOCK_SIZE.min(width - col_start);
                (row_start, col_start, block_height, block_width)
            })
        })
        .collect();

    // Process blocks in parallel
    let block_results: Vec<Result<((usize, usize, usize, usize), Array3<i8>)>> = blocks
        .par_iter()
        .map(|&(row_start, col_start, block_height, block_width)| {
            let block_data = process_block(
                row_start,
                col_start,
                block_height,
                block_width,
                bands,
                windows,
                reproject_config,
            )?;
            Ok(((row_start, col_start, block_height, block_width), block_data))
        })
        .collect();

    let process_time = process_start.elapsed();

    // Assemble blocks into final output using ndarray slicing
    let assemble_start = Instant::now();
    let mut result = Array4::<i8>::from_elem((1, bands, height, width), NODATA);

    for block_result in block_results {
        let ((row_start, col_start, block_height, block_width), block_data) = block_result?;

        // Use ndarray slice assignment for efficient copy
        let mut target_slice = result.slice_mut(s![
            0,
            ..,
            row_start..row_start + block_height,
            col_start..col_start + block_width
        ]);
        target_slice.assign(&block_data);
    }

    let assemble_time = assemble_start.elapsed();

    tracing::info!(
        process_ms = process_time.as_millis(),
        assemble_ms = assemble_time.as_millis(),
        pixels_per_ms = if process_time.as_millis() > 0 {
            output_pixels as f64 / process_time.as_millis() as f64
        } else {
            0.0
        },
        "block-parallel inverse mapping completed"
    );

    Ok(result)
}

/// Thread-safe accumulator for parallel mosaicing using atomic operations.
/// Kept for backwards compatibility and potential future use.
pub struct AtomicAccumulator {
    bands: usize,
    height: usize,
    width: usize,
}

impl AtomicAccumulator {
    /// Create a new atomic accumulator (stub for API compatibility).
    pub fn new(bands: usize, height: usize, width: usize) -> Self {
        Self { bands, height, width }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_accumulator_single_pixel() {
        let mut acc = BlockAccumulator::new(2, 4, 4);

        // Create a small test array
        let data = Array3::from_shape_vec((2, 2, 2), vec![10, 20, 30, 40, 50, 60, 70, 80])
            .unwrap();

        acc.accumulate(&data, 0, 0, 1, 1);

        let result = acc.finalize();
        assert_eq!(result[[0, 1, 1]], 10);
        assert_eq!(result[[1, 1, 1]], 50);
    }

    #[test]
    fn test_block_accumulator_mean() {
        let mut acc = BlockAccumulator::new(1, 2, 2);

        let data1 = Array3::from_shape_vec((1, 1, 1), vec![10]).unwrap();
        let data2 = Array3::from_shape_vec((1, 1, 1), vec![20]).unwrap();

        acc.accumulate(&data1, 0, 0, 0, 0);
        acc.accumulate(&data2, 0, 0, 0, 0);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 0]], 15); // Mean of 10 and 20
    }

    #[test]
    fn test_block_accumulator_nodata_skip() {
        let mut acc = BlockAccumulator::new(2, 2, 2);

        // All nodata
        let data = Array3::from_shape_vec((2, 1, 1), vec![NODATA, NODATA]).unwrap();

        acc.accumulate(&data, 0, 0, 0, 0);

        assert_eq!(acc.count[[0, 0]], 0);
    }

    #[test]
    fn test_inverse_grid_cell_interpolation() {
        let cell = InverseGridCell {
            out_row0: 0,
            out_col0: 0,
            out_row1: 10,
            out_col1: 10,
            src_corners: [
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
}
