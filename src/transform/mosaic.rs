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
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Block size for parallel processing.
/// Each output block is processed independently with its own accumulator.
const BLOCK_SIZE: usize = 256;

/// Grid cell size for inverse mapping interpolation.
/// Pre-project grid corners to reduce Proj calls.
const GRID_SIZE: usize = 32;

// Thread-local cache for Proj objects keyed by (target_crs, source_crs).
// Uses Rc since Proj doesn't implement Clone.
thread_local! {
    static PROJ_CACHE: RefCell<HashMap<(String, String), Rc<Proj>>> = RefCell::new(HashMap::new());
}

/// Get or create a cached Proj object for the given CRS pair.
/// Returns Rc<Proj> since Proj doesn't implement Clone.
fn get_cached_proj(target_crs: &str, source_crs: &str) -> Option<Rc<Proj>> {
    PROJ_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = (target_crs.to_string(), source_crs.to_string());

        if let Some(proj) = cache.get(&key) {
            Some(Rc::clone(proj))
        } else {
            match Proj::new_known_crs(target_crs, source_crs, None) {
                Ok(proj) => {
                    let rc_proj = Rc::new(proj);
                    cache.insert(key, Rc::clone(&rc_proj));
                    Some(rc_proj)
                }
                Err(_) => None,
            }
        }
    })
}

/// Pre-computed tile info with bounds in target CRS.
struct TileInfo<'a> {
    window: &'a WindowData,
    /// Bounds in target CRS [min_x, min_y, max_x, max_y]
    bounds_target: [f64; 4],
}

/// Transform tile bounds from native CRS to target CRS.
fn transform_tile_bounds(
    source_bounds: &[f64; 4],
    source_crs: &str,
    target_crs: &str,
) -> Option<[f64; 4]> {
    // Use forward projection: source → target
    let fwd_proj = Proj::new_known_crs(source_crs, target_crs, None).ok()?;

    // Transform corners and compute bounding box
    let corners = [
        (source_bounds[0], source_bounds[1]), // min_x, min_y
        (source_bounds[2], source_bounds[1]), // max_x, min_y
        (source_bounds[0], source_bounds[3]), // min_x, max_y
        (source_bounds[2], source_bounds[3]), // max_x, max_y
    ];

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (x, y) in corners {
        if let Ok((tx, ty)) = fwd_proj.convert((x, y)) {
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }
    }

    if min_x < max_x && min_y < max_y {
        Some([min_x, min_y, max_x, max_y])
    } else {
        None
    }
}

/// Check if two bounding boxes intersect.
#[inline]
fn bounds_intersect(a: &[f64; 4], b: &[f64; 4]) -> bool {
    // a and b are [min_x, min_y, max_x, max_y]
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

/// A grid cell for inverse mapping interpolation.
#[derive(Debug, Clone)]
struct InverseGridCell {
    out_row0: usize,
    out_col0: usize,
    out_row1: usize,
    out_col1: usize,
    /// Source coordinates at corners (after inverse projection)
    src_corners: [(f64, f64); 4],
    valid: bool,
}

impl InverseGridCell {
    #[inline]
    fn interpolate(&self, out_row: usize, out_col: usize) -> (f64, f64) {
        let cell_height = (self.out_row1 - self.out_row0) as f64;
        let cell_width = (self.out_col1 - self.out_col0) as f64;

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

            let corners_out = [
                (out_row, out_col),
                (out_row, out_col1),
                (out_row1, out_col),
                (out_row1, out_col1),
            ];

            let mut src_corners = [(0.0, 0.0); 4];
            let mut valid = true;

            for (i, &(row, col)) in corners_out.iter().enumerate() {
                let global_row = block_row_start + row;
                let global_col = block_col_start + col;
                let world_x = target_min_x + (global_col as f64 + 0.5) * target_resolution;
                let world_y = target_max_y - (global_row as f64 + 0.5) * target_resolution;

                match inv_proj.convert((world_x, world_y)) {
                    Ok((src_world_x, src_world_y)) => {
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

/// Local block accumulator using ndarray.
struct BlockAccumulator {
    sum: Array3<i32>,
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

    fn finalize(self) -> Array3<i8> {
        let (bands, height, width) = self.sum.dim();
        let mut result = Array3::<i8>::from_elem((bands, height, width), NODATA);

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
/// Only processes tiles that overlap this block's bounds.
fn process_block(
    block_row_start: usize,
    block_col_start: usize,
    block_height: usize,
    block_width: usize,
    block_bounds: &[f64; 4],
    bands: usize,
    tiles: &[TileInfo],
    config: &ReprojectConfig,
) -> Result<Array3<i8>> {
    let mut accumulator = BlockAccumulator::new(bands, block_height, block_width);

    // Only process tiles that intersect this block
    for tile_info in tiles {
        // Early exit: skip tiles that don't overlap this block
        if !bounds_intersect(block_bounds, &tile_info.bounds_target) {
            continue;
        }

        let window = tile_info.window;
        let data = &window.data;
        let (_, src_height, src_width) = data.dim();
        let source_bounds = &window.bounds_native;
        let source_crs = &window.tile.crs;

        // Get cached inverse projection: target CRS → source CRS
        let inv_proj = match get_cached_proj(&config.target_crs, source_crs) {
            Some(p) => p,
            None => continue,
        };

        // Build the inverse grid for this tile
        // Use &* to deref Rc<Proj> to &Proj
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
            &*inv_proj,
        );

        // Process each cell in the grid
        for cell in &grid {
            if !cell.valid {
                continue;
            }

            for out_row in cell.out_row0..cell.out_row1 {
                for out_col in cell.out_col0..cell.out_col1 {
                    let (src_col_f, src_row_f) = cell.interpolate(out_row, out_col);

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

/// Compute block bounds in target CRS.
#[inline]
fn compute_block_bounds(
    block_row_start: usize,
    block_col_start: usize,
    block_height: usize,
    block_width: usize,
    target_bounds: &[f64; 4],
    target_resolution: f64,
) -> [f64; 4] {
    let min_x = target_bounds[0] + block_col_start as f64 * target_resolution;
    let max_x = min_x + block_width as f64 * target_resolution;
    let max_y = target_bounds[3] - block_row_start as f64 * target_resolution;
    let min_y = max_y - block_height as f64 * target_resolution;
    [min_x, min_y, max_x, max_y]
}

/// Mosaic multiple tile windows into a single output chunk using block-parallel inverse mapping.
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

    let total_input_pixels: usize = windows
        .iter()
        .map(|w| w.data.dim().1 * w.data.dim().2)
        .sum();
    let output_pixels = height * width;

    let num_block_rows = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let num_block_cols = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_blocks = num_block_rows * num_block_cols;

    // Pre-compute tile bounds in target CRS (once per tile, not per block)
    let prep_start = Instant::now();
    let tiles: Vec<TileInfo> = windows
        .iter()
        .filter_map(|window| {
            let bounds_target = transform_tile_bounds(
                &window.bounds_native,
                &window.tile.crs,
                &reproject_config.target_crs,
            )?;
            Some(TileInfo { window, bounds_target })
        })
        .collect();
    let prep_time = prep_start.elapsed();

    tracing::info!(
        num_tiles = tiles.len(),
        total_input_pixels = total_input_pixels,
        output_pixels = output_pixels,
        num_blocks = total_blocks,
        block_size = BLOCK_SIZE,
        prep_ms = prep_time.as_millis(),
        "mosaic_tiles starting (block-parallel inverse mapping)"
    );

    let process_start = Instant::now();

    // Generate all block coordinates
    let blocks: Vec<(usize, usize, usize, usize, [f64; 4])> = (0..num_block_rows)
        .flat_map(|br| {
            (0..num_block_cols).map(move |bc| {
                let row_start = br * BLOCK_SIZE;
                let col_start = bc * BLOCK_SIZE;
                let block_height = BLOCK_SIZE.min(height - row_start);
                let block_width = BLOCK_SIZE.min(width - col_start);
                let block_bounds = compute_block_bounds(
                    row_start,
                    col_start,
                    block_height,
                    block_width,
                    &reproject_config.target_bounds,
                    reproject_config.target_resolution,
                );
                (row_start, col_start, block_height, block_width, block_bounds)
            })
        })
        .collect();

    // Process blocks in parallel
    let block_results: Vec<Result<((usize, usize, usize, usize), Array3<i8>)>> = blocks
        .par_iter()
        .map(|&(row_start, col_start, block_height, block_width, ref block_bounds)| {
            let block_data = process_block(
                row_start,
                col_start,
                block_height,
                block_width,
                block_bounds,
                bands,
                &tiles,
                reproject_config,
            )?;
            Ok(((row_start, col_start, block_height, block_width), block_data))
        })
        .collect();

    let process_time = process_start.elapsed();

    // Assemble blocks into final output
    let assemble_start = Instant::now();
    let mut result = Array4::<i8>::from_elem((1, bands, height, width), NODATA);

    for block_result in block_results {
        let ((row_start, col_start, block_height, block_width), block_data) = block_result?;

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

/// Thread-safe accumulator stub for backwards compatibility.
#[allow(dead_code)]
pub struct AtomicAccumulator {
    bands: usize,
    height: usize,
    width: usize,
}

impl AtomicAccumulator {
    pub fn new(bands: usize, height: usize, width: usize) -> Self {
        Self { bands, height, width }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_intersect() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        assert!(bounds_intersect(&a, &b));

        let c = [20.0, 20.0, 30.0, 30.0];
        assert!(!bounds_intersect(&a, &c));
    }

    #[test]
    fn test_block_accumulator_single_pixel() {
        let mut acc = BlockAccumulator::new(2, 4, 4);
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
        assert_eq!(result[[0, 0, 0]], 15);
    }

    #[test]
    fn test_block_accumulator_nodata_skip() {
        let mut acc = BlockAccumulator::new(2, 2, 2);
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
                (0.0, 0.0),
                (10.0, 0.0),
                (0.0, 10.0),
                (10.0, 10.0),
            ],
            valid: true,
        };

        let (x, y) = cell.interpolate(5, 5);
        assert!((x - 5.0).abs() < 0.01);
        assert!((y - 5.0).abs() < 0.01);
    }
}
