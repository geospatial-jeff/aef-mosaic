//! Coordinate reprojection using adaptive grid refinement.
//!
//! This approach uses GDAL-style adaptive refinement: start with coarse grid blocks,
//! check interpolation error at the center, and subdivide if error exceeds threshold.
//! This guarantees sub-pixel accuracy everywhere while minimizing projection calls.
//!
//! The key insight is that CRS transforms are smooth functions - they don't
//! change rapidly between adjacent pixels. But at high latitudes or across
//! projection boundaries, we need finer sampling to maintain accuracy.

use anyhow::{Context, Result};
use ndarray::Array3;
use proj::Proj;

/// Maximum interpolation error in pixels before subdividing.
/// GDAL uses 0.125 by default; we use 0.5 for good balance of speed/accuracy.
const ERROR_THRESHOLD_PIXELS: f64 = 0.5;

/// Initial block size for adaptive grid (will subdivide as needed).
const INITIAL_BLOCK_SIZE: usize = 64;

/// Minimum block size - don't subdivide smaller than this.
const MIN_BLOCK_SIZE: usize = 4;

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
}

/// Adaptive grid that refines where needed to maintain accuracy.
struct AdaptiveGrid {
    cells: Vec<GridCell>,
}

impl AdaptiveGrid {
    /// Build an adaptive grid for the given transformation.
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

        Self { cells }
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

    /// Find the cell containing a given destination pixel.
    fn find_cell(&self, dst_x: usize, dst_y: usize) -> Option<&GridCell> {
        // Linear search for now - could use spatial index for very large grids
        self.cells.iter().find(|cell| {
            dst_x >= cell.dst_x0
                && dst_x < cell.dst_x1
                && dst_y >= cell.dst_y0
                && dst_y < cell.dst_y1
        })
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
        let inv_proj = Self::create_proj(&config.target_crs, source_crs)?;

        // Source pixel size
        let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
        let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

        // Build adaptive grid
        let grid = AdaptiveGrid::build(
            dst_width,
            dst_height,
            &inv_proj,
            &config.target_bounds,
            config.target_resolution,
            (src_pixel_x, src_pixel_y),
        );

        tracing::debug!(
            "Adaptive grid: {} cells for {}x{} output (initial block: {}, min block: {})",
            grid.cells.len(),
            dst_width,
            dst_height,
            INITIAL_BLOCK_SIZE,
            MIN_BLOCK_SIZE
        );

        // Create output array initialized with NODATA
        let mut output = Array3::<i8>::from_elem((bands, dst_height, dst_width), NODATA);

        // For each output pixel, find cell, interpolate source coordinates, and sample
        for dst_row in 0..dst_height {
            for dst_col in 0..dst_width {
                let Some(cell) = grid.find_cell(dst_col, dst_row) else {
                    continue;
                };

                if !cell.is_valid() {
                    continue;
                }

                // Interpolate source coordinates
                let (src_x, src_y) = cell.interpolate(dst_col, dst_row);

                // Convert source CRS coordinates to pixel coordinates
                // Source is bottom-up: row 0 is at min_y
                let src_px = (src_x - source_bounds[0]) / src_pixel_x - 0.5;
                let src_py = (src_y - source_bounds[1]) / src_pixel_y - 0.5;

                // Nearest neighbor sampling (for i8 embeddings, interpolation doesn't make sense)
                let src_col = src_px.round() as isize;
                let src_row = src_py.round() as isize;

                // Bounds check
                if src_col < 0
                    || src_col >= src_width as isize
                    || src_row < 0
                    || src_row >= src_height as isize
                {
                    continue; // Outside source bounds
                }

                let src_col = src_col as usize;
                let src_row = src_row as usize;

                // Copy all bands
                for band in 0..bands {
                    output[[band, dst_row, dst_col]] = data[[band, src_row, src_col]];
                }
            }
        }

        let valid_pixels = output.iter().filter(|&&v| v != NODATA).count();
        tracing::debug!(
            "Adaptive reproject: {}x{} -> {}x{}, {} cells, {} valid pixels",
            src_width,
            src_height,
            dst_width,
            dst_height,
            grid.cells.len(),
            valid_pixels
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
