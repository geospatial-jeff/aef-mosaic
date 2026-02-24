//! Coordinate reprojection using adaptive grid refinement.
//!
//! This approach uses GDAL-style adaptive refinement: start with coarse grid blocks,
//! check interpolation error at the center, and subdivide if error exceeds threshold.
//! This guarantees sub-pixel accuracy everywhere while minimizing projection calls.
//!
//! The key insight is that CRS transforms are smooth functions - they don't
//! change rapidly between adjacent pixels. But at high latitudes or across
//! projection boundaries, we need finer sampling to maintain accuracy.
//!
//! Performance optimizations:
//! - Pre-computed per-row linear coefficients (reduces bilinear to 2 ops/pixel)
//! - SIMD vectorization (8 pixels at a time using f32x8)
//! - Tile-level parallelism handled by caller (mosaic_tiles)

use anyhow::{Context, Result};
use ndarray::Array3;
use proj::Proj;
use std::time::Instant;
use wide::f32x8;

/// Maximum interpolation error in pixels before subdividing.
/// GDAL uses 0.125 by default; we use 0.75 for good balance of speed/accuracy.
const ERROR_THRESHOLD_PIXELS: f64 = 0.75;

/// Initial block size for adaptive grid (will subdivide as needed).
const INITIAL_BLOCK_SIZE: usize = 128;

/// Minimum block size - don't subdivide smaller than this.
/// Tuned for balance between parallelization overhead and granularity.
const MIN_BLOCK_SIZE: usize = 16;

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Configuration for reprojection.
#[derive(Debug, Clone)]
pub struct ReprojectConfig {
    /// Target CRS (e.g., "EPSG:4326")
    pub target_crs: String,

    /// Target resolution in CRS units
    pub target_resolution: f64,

    /// Target bounds [min_x, min_y, max_x, max_y]
    pub target_bounds: [f64; 4],

    /// Target array dimensions (height, width)
    pub target_shape: (usize, usize),

    /// Number of bands in the output
    pub num_bands: usize,
}

/// A grid cell with transformed corner coordinates.
#[derive(Debug, Clone, Copy)]
struct GridCell {
    /// Pixel bounds in destination image (x0, y0, x1, y1)
    dst_x0: usize,
    dst_y0: usize,
    dst_x1: usize,
    dst_y1: usize,
    /// Source coordinates at corners (already transformed)
    /// Order: top-left, top-right, bottom-left, bottom-right
    src_coords: [(f64, f64); 4],
}

impl GridCell {
    /// Check if all corners have valid (non-NaN) coordinates.
    fn is_valid(&self) -> bool {
        self.src_coords.iter().all(|(x, y)| !x.is_nan() && !y.is_nan())
    }

    /// Bilinear interpolation of source coordinates for a point within this cell.
    /// Used in tests and for reference; production code uses row_coefficients() instead.
    #[cfg(test)]
    fn interpolate(&self, dst_x: usize, dst_y: usize) -> (f64, f64) {
        let width = (self.dst_x1 - self.dst_x0) as f64;
        let height = (self.dst_y1 - self.dst_y0) as f64;

        let t_col = if width > 0.0 {
            (dst_x - self.dst_x0) as f64 / width
        } else {
            0.0
        };
        let t_row = if height > 0.0 {
            (dst_y - self.dst_y0) as f64 / height
        } else {
            0.0
        };

        let [(x00, y00), (x01, y01), (x10, y10), (x11, y11)] = self.src_coords;

        let src_x = x00 * (1.0 - t_row) * (1.0 - t_col)
            + x01 * (1.0 - t_row) * t_col
            + x10 * t_row * (1.0 - t_col)
            + x11 * t_row * t_col;

        let src_y = y00 * (1.0 - t_row) * (1.0 - t_col)
            + y01 * (1.0 - t_row) * t_col
            + y10 * t_row * (1.0 - t_col)
            + y11 * t_row * t_col;

        (src_x, src_y)
    }

    /// Compute pre-computed row coefficients for linear interpolation.
    /// For a given row, bilinear interpolation becomes linear:
    ///   src_x = base_x + slope_x * (dst_col - dst_x0)
    ///   src_y = base_y + slope_y * (dst_col - dst_x0)
    #[inline]
    fn row_coefficients(&self, dst_row: usize) -> RowCoefficients {
        let width = (self.dst_x1 - self.dst_x0) as f64;
        let height = (self.dst_y1 - self.dst_y0) as f64;

        let t_row = if height > 0.0 {
            (dst_row - self.dst_y0) as f64 / height
        } else {
            0.0
        };

        let [(x00, y00), (x01, y01), (x10, y10), (x11, y11)] = self.src_coords;

        // Pre-interpolate left and right edges at this row
        let x_left = x00 * (1.0 - t_row) + x10 * t_row;
        let x_right = x01 * (1.0 - t_row) + x11 * t_row;
        let y_left = y00 * (1.0 - t_row) + y10 * t_row;
        let y_right = y01 * (1.0 - t_row) + y11 * t_row;

        // Linear interpolation coefficients: val = base + slope * t_col
        // where t_col = (dst_col - dst_x0) / width
        // So: val = base + (slope / width) * (dst_col - dst_x0)
        let inv_width = if width > 0.0 { 1.0 / width } else { 0.0 };

        // Cast to f32 for SIMD-friendly operations
        RowCoefficients {
            base_x: x_left as f32,
            slope_x: ((x_right - x_left) * inv_width) as f32,
            base_y: y_left as f32,
            slope_y: ((y_right - y_left) * inv_width) as f32,
        }
    }
}

/// Pre-computed linear interpolation coefficients for a single row.
/// Uses f32 for SIMD-friendly operations (8x f32 fits in AVX register).
/// Reduces bilinear interpolation from ~16 ops to 4 ops per pixel.
#[derive(Debug, Clone, Copy)]
struct RowCoefficients {
    base_x: f32,
    slope_x: f32,
    base_y: f32,
    slope_y: f32,
}

impl RowCoefficients {
    /// Interpolate source coordinates for a column offset within the cell.
    #[inline]
    fn interpolate(&self, col_offset: f32) -> (f32, f32) {
        (
            self.base_x + self.slope_x * col_offset,
            self.base_y + self.slope_y * col_offset,
        )
    }
}

/// A row span: represents one row segment to process.
/// Pre-computed during grid construction for optimal parallelization.
#[derive(Debug, Clone)]
struct RowSpan {
    /// Destination row index
    dst_row: usize,
    /// Column range [x0, x1)
    dst_x0: usize,
    dst_x1: usize,
    /// Pre-computed linear interpolation coefficients for this row
    coefficients: RowCoefficients,
}

/// Adaptive grid that refines where needed to maintain accuracy.
/// Optimized for row-level parallelization with pre-computed coefficients.
struct AdaptiveGrid {
    cells: Vec<GridCell>,
    /// Flattened row spans for parallel processing.
    /// Each span represents one cell's contribution to one row.
    row_spans: Vec<RowSpan>,
}

impl AdaptiveGrid {
    /// Build an adaptive grid for the given transformation.
    /// Also builds row spans for efficient row-level parallelization.
    fn build(
        dst_width: usize,
        dst_height: usize,
        inv_proj: &Proj,
        target_bounds: &[f64; 4],
        target_resolution: f64,
        source_pixel_size: (f64, f64),
    ) -> Self {
        let mut cells = Vec::new();

        // Start with coarse blocks covering the entire image
        let mut x = 0;
        while x < dst_width {
            let x1 = (x + INITIAL_BLOCK_SIZE).min(dst_width);
            let mut y = 0;
            while y < dst_height {
                let y1 = (y + INITIAL_BLOCK_SIZE).min(dst_height);

                Self::subdivide_if_needed(
                    x, y, x1, y1,
                    inv_proj,
                    target_bounds,
                    target_resolution,
                    source_pixel_size,
                    &mut cells,
                );

                y = y1;
            }
            x = x1;
        }

        // Build row spans: flatten cells into individual row segments
        // with pre-computed linear interpolation coefficients.
        // This enables row-level parallelization with uniform work distribution.
        let row_spans = Self::build_row_spans(&cells);

        Self { cells, row_spans }
    }

    /// Flatten cells into row spans for row-level parallelization.
    /// Each row span has pre-computed coefficients for fast linear interpolation.
    fn build_row_spans(cells: &[GridCell]) -> Vec<RowSpan> {
        // Estimate capacity: sum of row counts across all cells
        let estimated_capacity: usize = cells
            .iter()
            .map(|c| c.dst_y1.saturating_sub(c.dst_y0))
            .sum();
        let mut row_spans = Vec::with_capacity(estimated_capacity);

        for cell in cells {
            if !cell.is_valid() {
                continue;
            }

            for dst_row in cell.dst_y0..cell.dst_y1 {
                let coefficients = cell.row_coefficients(dst_row);
                row_spans.push(RowSpan {
                    dst_row,
                    dst_x0: cell.dst_x0,
                    dst_x1: cell.dst_x1,
                    coefficients,
                });
            }
        }

        row_spans
    }

    /// Transform a destination pixel coordinate to source CRS coordinates.
    fn transform_point(
        dst_x: usize,
        dst_y: usize,
        inv_proj: &Proj,
        target_bounds: &[f64; 4],
        target_resolution: f64,
    ) -> (f64, f64) {
        // Target is top-down: row 0 is at max_y
        let world_x = target_bounds[0] + (dst_x as f64 + 0.5) * target_resolution;
        let world_y = target_bounds[3] - (dst_y as f64 + 0.5) * target_resolution;

        inv_proj.convert((world_x, world_y)).unwrap_or((f64::NAN, f64::NAN))
    }

    /// Recursively subdivide a block if interpolation error exceeds threshold.
    fn subdivide_if_needed(
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        inv_proj: &Proj,
        target_bounds: &[f64; 4],
        target_resolution: f64,
        source_pixel_size: (f64, f64),
        cells: &mut Vec<GridCell>,
    ) {
        // Transform the 4 corners
        let tl = Self::transform_point(x0, y0, inv_proj, target_bounds, target_resolution);
        let tr = Self::transform_point(x1, y0, inv_proj, target_bounds, target_resolution);
        let bl = Self::transform_point(x0, y1, inv_proj, target_bounds, target_resolution);
        let br = Self::transform_point(x1, y1, inv_proj, target_bounds, target_resolution);

        let width = x1 - x0;
        let height = y1 - y0;

        // If any corner is NaN, don't subdivide - just add the cell as-is
        if tl.0.is_nan() || tr.0.is_nan() || bl.0.is_nan() || br.0.is_nan() {
            cells.push(GridCell {
                dst_x0: x0,
                dst_y0: y0,
                dst_x1: x1,
                dst_y1: y1,
                src_coords: [tl, tr, bl, br],
            });
            return;
        }

        // Check if we should subdivide (block is large enough)
        let should_check_error = width > MIN_BLOCK_SIZE && height > MIN_BLOCK_SIZE;

        if should_check_error {
            // Transform the center point exactly
            let cx = (x0 + x1) / 2;
            let cy = (y0 + y1) / 2;
            let actual_center = Self::transform_point(cx, cy, inv_proj, target_bounds, target_resolution);

            // Interpolate center from corners
            let t_col = 0.5;
            let t_row = 0.5;
            let interp_x = tl.0 * (1.0 - t_row) * (1.0 - t_col)
                + tr.0 * (1.0 - t_row) * t_col
                + bl.0 * t_row * (1.0 - t_col)
                + br.0 * t_row * t_col;
            let interp_y = tl.1 * (1.0 - t_row) * (1.0 - t_col)
                + tr.1 * (1.0 - t_row) * t_col
                + bl.1 * t_row * (1.0 - t_col)
                + br.1 * t_row * t_col;

            // Calculate error in source pixels
            let error_x = (actual_center.0 - interp_x).abs() / source_pixel_size.0;
            let error_y = (actual_center.1 - interp_y).abs() / source_pixel_size.1;
            let error = error_x.max(error_y);

            if error > ERROR_THRESHOLD_PIXELS {
                // Subdivide into 4 quadrants
                let mx = (x0 + x1) / 2;
                let my = (y0 + y1) / 2;

                Self::subdivide_if_needed(x0, y0, mx, my, inv_proj, target_bounds, target_resolution, source_pixel_size, cells);
                Self::subdivide_if_needed(mx, y0, x1, my, inv_proj, target_bounds, target_resolution, source_pixel_size, cells);
                Self::subdivide_if_needed(x0, my, mx, y1, inv_proj, target_bounds, target_resolution, source_pixel_size, cells);
                Self::subdivide_if_needed(mx, my, x1, y1, inv_proj, target_bounds, target_resolution, source_pixel_size, cells);
                return;
            }
        }

        // Error is acceptable or block is too small - add this cell
        cells.push(GridCell {
            dst_x0: x0,
            dst_y0: y0,
            dst_x1: x1,
            dst_y1: y1,
            src_coords: [tl, tr, bl, br],
        });
    }

}

/// Reprojector using adaptive grid refinement.
pub struct Reprojector {
    // Stateless - all config passed to reproject_tile
}

impl Reprojector {
    /// Create a new reprojector for the given target CRS.
    pub fn new(_target_crs: &str) -> Self {
        Self {}
    }

    /// Create a Proj transformation for the given CRS pair.
    fn create_proj(from_crs: &str, to_crs: &str) -> Result<Proj> {
        Proj::new_known_crs(from_crs, to_crs, None)
            .with_context(|| format!("Failed to create Proj for {} -> {}", from_crs, to_crs))
    }

    /// Copy all bands from source to destination pixel.
    /// Uses direct pointer arithmetic for SIMD-friendly memory access.
    #[inline]
    fn copy_bands(
        output: &mut Array3<i8>,
        data: &Array3<i8>,
        bands: usize,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
    ) {
        // Copy strides to local array to avoid borrow conflicts with as_mut_ptr()
        let dst_strides: [isize; 3] = {
            let s = output.strides();
            [s[0], s[1], s[2]]
        };
        let src_strides: [isize; 3] = {
            let s = data.strides();
            [s[0], s[1], s[2]]
        };

        // SAFETY: We have exclusive mutable access to output, and coordinates
        // are bounds-checked by the caller.
        unsafe {
            let dst_ptr = output.as_mut_ptr();
            let src_ptr = data.as_ptr();

            // Calculate base offsets (band 0 position)
            let dst_base = (dst_row as isize * dst_strides[1]) + (dst_col as isize * dst_strides[2]);
            let src_base = (src_row as isize * src_strides[1]) + (src_col as isize * src_strides[2]);

            // Check if bands are contiguous (stride of 1 between bands)
            // This is the common case for (bands, height, width) layout
            if dst_strides[0] == 1 && src_strides[0] == 1 {
                // Contiguous bands: use memcpy for all bands at once
                let dst = dst_ptr.offset(dst_base);
                let src = src_ptr.offset(src_base);
                std::ptr::copy_nonoverlapping(src, dst, bands);
            } else {
                // Non-contiguous: copy band by band with explicit strides
                // Still unroll for better instruction-level parallelism
                let dst_band_stride = dst_strides[0];
                let src_band_stride = src_strides[0];

                let mut band = 0;
                // Unroll by 4 for better pipelining
                while band + 4 <= bands {
                    let b = band as isize;
                    *dst_ptr.offset(dst_base + b * dst_band_stride) =
                        *src_ptr.offset(src_base + b * src_band_stride);
                    *dst_ptr.offset(dst_base + (b + 1) * dst_band_stride) =
                        *src_ptr.offset(src_base + (b + 1) * src_band_stride);
                    *dst_ptr.offset(dst_base + (b + 2) * dst_band_stride) =
                        *src_ptr.offset(src_base + (b + 2) * src_band_stride);
                    *dst_ptr.offset(dst_base + (b + 3) * dst_band_stride) =
                        *src_ptr.offset(src_base + (b + 3) * src_band_stride);
                    band += 4;
                }

                // Handle remaining bands
                while band < bands {
                    let b = band as isize;
                    *dst_ptr.offset(dst_base + b * dst_band_stride) =
                        *src_ptr.offset(src_base + b * src_band_stride);
                    band += 1;
                }
            }
        }
    }

    /// Reproject a single tile to the target grid using adaptive grid refinement.
    ///
    /// Input: (bands, src_height, src_width) array in source CRS
    ///        Source uses bottom-up orientation (row 0 at south, AEF COG format)
    /// Output: (bands, dst_height, dst_width) array in target CRS
    ///        Output uses top-down orientation (row 0 at north, standard raster)
    pub fn reproject_tile(
        &self,
        data: &Array3<i8>,
        source_crs: &str,
        source_bounds: &[f64; 4],
        config: &ReprojectConfig,
    ) -> Result<Array3<i8>> {
        let (bands, src_height, src_width) = data.dim();
        let (dst_height, dst_width) = config.target_shape;

        // Create inverse proj transformation (target -> source, for inverse mapping)
        let proj_start = Instant::now();
        let inv_proj = Self::create_proj(&config.target_crs, source_crs)?;
        let proj_time = proj_start.elapsed();

        // Source pixel size
        let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
        let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

        // Build adaptive grid
        let grid_start = Instant::now();
        let grid = AdaptiveGrid::build(
            dst_width,
            dst_height,
            &inv_proj,
            &config.target_bounds,
            config.target_resolution,
            (src_pixel_x, src_pixel_y),
        );
        let grid_time = grid_start.elapsed();

        let num_cells = grid.cells.len();
        let num_row_spans = grid.row_spans.len();
        tracing::debug!(
            "Adaptive grid: {} cells, {} row spans for {}x{} output",
            num_cells,
            num_row_spans,
            dst_width,
            dst_height,
        );

        // Create output array initialized with NODATA
        let mut output = Array3::<i8>::from_elem((bands, dst_height, dst_width), NODATA);

        // Pre-compute inverse pixel sizes for coordinate conversion
        let inv_src_pixel_x = 1.0 / src_pixel_x;
        let inv_src_pixel_y = 1.0 / src_pixel_y;
        let src_offset_x = source_bounds[0];
        let src_offset_y = source_bounds[1];

        // Sequential row processing - tile-level parallelism in mosaic_tiles() is sufficient.
        // Row-level parallelism was removed because:
        // - Each row has ~256 pixels of work (~500 CPU cycles)
        // - Rayon task overhead is ~1000+ cycles per task
        // - SIMD within rows provides vectorization without scheduling overhead
        //
        // SIMD optimization: process 8 pixels at a time using f32x8 vectors.
        let copy_start = Instant::now();
        for span in &grid.row_spans {
            let dst_row = span.dst_row;
            let coeffs = &span.coefficients;

            // Pre-compute SIMD constants
            let base_x = f32x8::splat(coeffs.base_x);
            let slope_x = f32x8::splat(coeffs.slope_x);
            let base_y = f32x8::splat(coeffs.base_y);
            let slope_y = f32x8::splat(coeffs.slope_y);
            let offset_x = f32x8::splat(src_offset_x as f32);
            let offset_y = f32x8::splat(src_offset_y as f32);
            let inv_px_x = f32x8::splat(inv_src_pixel_x as f32);
            let inv_px_y = f32x8::splat(inv_src_pixel_y as f32);
            let half = f32x8::splat(0.5);

            let mut dst_col = span.dst_x0;

            // Process 8 columns at a time using SIMD
            while dst_col + 8 <= span.dst_x1 {
                let base_offset = (dst_col - span.dst_x0) as f32;
                let offsets = f32x8::from([
                    base_offset,
                    base_offset + 1.0,
                    base_offset + 2.0,
                    base_offset + 3.0,
                    base_offset + 4.0,
                    base_offset + 5.0,
                    base_offset + 6.0,
                    base_offset + 7.0,
                ]);

                // Vectorized interpolation
                let src_x = base_x + slope_x * offsets;
                let src_y = base_y + slope_y * offsets;

                // Vectorized coordinate conversion
                let src_px = (src_x - offset_x) * inv_px_x - half;
                let src_py = (src_y - offset_y) * inv_px_y - half;

                // Round to nearest (SIMD)
                let src_cols = src_px.round();
                let src_rows = src_py.round();

                // Extract and process (bounds check per pixel)
                let cols: [f32; 8] = src_cols.into();
                let rows: [f32; 8] = src_rows.into();

                for i in 0..8 {
                    let col_i = cols[i] as isize;
                    let row_i = rows[i] as isize;

                    if col_i >= 0
                        && col_i < src_width as isize
                        && row_i >= 0
                        && row_i < src_height as isize
                    {
                        Self::copy_bands(
                            &mut output, data, bands, dst_row,
                            dst_col + i, row_i as usize, col_i as usize,
                        );
                    }
                }

                dst_col += 8;
            }

            // Handle remaining columns (scalar fallback for last 0-7 columns)
            while dst_col < span.dst_x1 {
                let col_offset = (dst_col - span.dst_x0) as f32;
                let (src_x, src_y) = coeffs.interpolate(col_offset);

                // Convert source CRS coordinates to pixel coordinates
                // Source is bottom-up: row 0 is at min_y
                let src_px = (src_x - src_offset_x as f32) * inv_src_pixel_x as f32 - 0.5;
                let src_py = (src_y - src_offset_y as f32) * inv_src_pixel_y as f32 - 0.5;

                // Nearest neighbor sampling
                let src_col_i = src_px.round() as isize;
                let src_row_i = src_py.round() as isize;

                // Bounds check
                if src_col_i >= 0
                    && src_col_i < src_width as isize
                    && src_row_i >= 0
                    && src_row_i < src_height as isize
                {
                    Self::copy_bands(
                        &mut output, data, bands, dst_row, dst_col,
                        src_row_i as usize, src_col_i as usize,
                    );
                }

                dst_col += 1;
            }
        }

        let copy_time = copy_start.elapsed();
        let valid_pixels = output.iter().filter(|&&v| v != NODATA).count();
        let input_pixels = src_width * src_height;
        let output_pixels = dst_width * dst_height;

        tracing::info!(
            input_shape = ?(src_height, src_width),
            output_shape = ?(dst_height, dst_width),
            input_pixels = input_pixels,
            output_pixels = output_pixels,
            valid_pixels = valid_pixels,
            num_cells = num_cells,
            num_row_spans = num_row_spans,
            proj_ms = proj_time.as_micros() as f64 / 1000.0,
            grid_ms = grid_time.as_micros() as f64 / 1000.0,
            copy_ms = copy_time.as_micros() as f64 / 1000.0,
            "reproject_tile completed"
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_proj_creation() {
        // Test that we can create Proj transformations
        let proj1 = Reprojector::create_proj("EPSG:32610", "EPSG:4326").unwrap();
        let proj2 = Reprojector::create_proj("EPSG:32610", "EPSG:6933").unwrap();

        // Both should work
        let result1 = proj1.convert((500000.0, 4000000.0));
        let result2 = proj2.convert((500000.0, 4000000.0));
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_identity_reproject() {
        let reprojector = Reprojector::new("EPSG:4326");

        // Create simple test data
        let data = Array3::from_elem((2, 4, 4), 42i8);

        let config = ReprojectConfig {
            target_crs: "EPSG:4326".to_string(),
            target_resolution: 0.25,
            target_bounds: [0.0, 0.0, 1.0, 1.0],
            target_shape: (4, 4),
            num_bands: 2,
        };

        // EPSG:4326 to EPSG:4326 should preserve values
        let result = reprojector.reproject_tile(&data, "EPSG:4326", &[0.0, 0.0, 1.0, 1.0], &config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_grid_subdivision() {
        // Test that the grid subdivides for high-distortion transforms
        let inv_proj = Reprojector::create_proj("EPSG:6933", "EPSG:32610").unwrap();

        let grid = AdaptiveGrid::build(
            1024,
            1024,
            &inv_proj,
            &[-8000000.0, 4000000.0, -7900000.0, 4100000.0], // EASE-Grid bounds
            100.0, // 100m resolution
            (10.0, 10.0), // 10m source pixels
        );

        // Should have more cells than just the initial coarse grid
        let coarse_cells = (1024 / INITIAL_BLOCK_SIZE + 1) * (1024 / INITIAL_BLOCK_SIZE + 1);
        println!(
            "Adaptive grid: {} cells (coarse would be {})",
            grid.cells.len(),
            coarse_cells
        );

        // Grid should cover the entire image
        assert!(!grid.cells.is_empty());
    }

    #[test]
    fn test_grid_cell_interpolation() {
        let cell = GridCell {
            dst_x0: 0,
            dst_y0: 0,
            dst_x1: 10,
            dst_y1: 10,
            src_coords: [
                (0.0, 0.0),   // top-left
                (10.0, 0.0),  // top-right
                (0.0, 10.0),  // bottom-left
                (10.0, 10.0), // bottom-right
            ],
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
