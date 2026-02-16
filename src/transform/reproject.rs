//! Coordinate reprojection using sparse grid + bilinear interpolation.
//!
//! This approach samples coordinates at sparse grid points and interpolates
//! for pixels in between. This is ~1000x faster than projecting every pixel
//! while maintaining sufficient accuracy for 10m resolution data.
//!
//! The key insight is that CRS transforms are smooth functions - they don't
//! change rapidly between adjacent pixels.

use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use proj::Proj;

/// Grid spacing for sparse coordinate sampling.
/// At 32-pixel spacing on a 2048x2048 tile, we transform ~4K points instead of 4M.
const GRID_SPACING: usize = 32;

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

/// Reprojector using sparse grid + bilinear interpolation.
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

    /// Reproject a single tile to the target grid using sparse grid interpolation.
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

        // Build sparse grid of target coordinates and transform to source
        let grid_rows = (dst_height + GRID_SPACING - 1) / GRID_SPACING + 1;
        let grid_cols = (dst_width + GRID_SPACING - 1) / GRID_SPACING + 1;

        // Grid of source coordinates (x, y) for each sparse grid point
        let mut src_x_grid = Array2::<f64>::zeros((grid_rows, grid_cols));
        let mut src_y_grid = Array2::<f64>::zeros((grid_rows, grid_cols));

        // Transform sparse grid points from target CRS to source CRS
        for grid_row in 0..grid_rows {
            let dst_row = (grid_row * GRID_SPACING).min(dst_height.saturating_sub(1));
            // Target is top-down: row 0 is at max_y
            let dst_y = config.target_bounds[3] - (dst_row as f64 + 0.5) * config.target_resolution;

            for grid_col in 0..grid_cols {
                let dst_col = (grid_col * GRID_SPACING).min(dst_width.saturating_sub(1));
                let dst_x = config.target_bounds[0] + (dst_col as f64 + 0.5) * config.target_resolution;

                // Transform to source CRS
                let (src_x, src_y) = inv_proj.convert((dst_x, dst_y))
                    .unwrap_or((f64::NAN, f64::NAN));

                src_x_grid[[grid_row, grid_col]] = src_x;
                src_y_grid[[grid_row, grid_col]] = src_y;
            }
        }

        // Create output array initialized with NODATA
        let mut output = Array3::<i8>::from_elem((bands, dst_height, dst_width), NODATA);

        // For each output pixel, interpolate source coordinates and sample
        for dst_row in 0..dst_height {
            let grid_row = dst_row / GRID_SPACING;
            let grid_row_next = (grid_row + 1).min(grid_rows - 1);
            let t_row = if grid_row_next > grid_row {
                (dst_row % GRID_SPACING) as f64 / GRID_SPACING as f64
            } else {
                0.0
            };

            for dst_col in 0..dst_width {
                let grid_col = dst_col / GRID_SPACING;
                let grid_col_next = (grid_col + 1).min(grid_cols - 1);
                let t_col = if grid_col_next > grid_col {
                    (dst_col % GRID_SPACING) as f64 / GRID_SPACING as f64
                } else {
                    0.0
                };

                // Bilinear interpolation of source coordinates
                let x00 = src_x_grid[[grid_row, grid_col]];
                let x01 = src_x_grid[[grid_row, grid_col_next]];
                let x10 = src_x_grid[[grid_row_next, grid_col]];
                let x11 = src_x_grid[[grid_row_next, grid_col_next]];

                let y00 = src_y_grid[[grid_row, grid_col]];
                let y01 = src_y_grid[[grid_row, grid_col_next]];
                let y10 = src_y_grid[[grid_row_next, grid_col]];
                let y11 = src_y_grid[[grid_row_next, grid_col_next]];

                // Check for NaN (transformation failed)
                if x00.is_nan() || x01.is_nan() || x10.is_nan() || x11.is_nan() {
                    continue; // Leave as zero (nodata)
                }

                let src_x = x00 * (1.0 - t_row) * (1.0 - t_col)
                    + x01 * (1.0 - t_row) * t_col
                    + x10 * t_row * (1.0 - t_col)
                    + x11 * t_row * t_col;

                let src_y = y00 * (1.0 - t_row) * (1.0 - t_col)
                    + y01 * (1.0 - t_row) * t_col
                    + y10 * t_row * (1.0 - t_col)
                    + y11 * t_row * t_col;

                // Convert source CRS coordinates to pixel coordinates
                // Source is bottom-up: row 0 is at min_y
                let src_px = (src_x - source_bounds[0]) / src_pixel_x - 0.5;
                let src_py = (src_y - source_bounds[1]) / src_pixel_y - 0.5;

                // Nearest neighbor sampling (for i8 embeddings, interpolation doesn't make sense)
                let src_col = src_px.round() as isize;
                let src_row = src_py.round() as isize;

                // Bounds check
                if src_col < 0 || src_col >= src_width as isize
                    || src_row < 0 || src_row >= src_height as isize
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
            "Sparse grid reproject: {}x{} -> {}x{}, {} valid pixels",
            src_width, src_height, dst_width, dst_height, valid_pixels
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
        let result = reprojector.reproject_tile(
            &data,
            "EPSG:4326",
            &[0.0, 0.0, 1.0, 1.0],
            &config,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_sparse_grid_spacing() {
        // Verify grid dimensions
        let dst_height = 1024;
        let dst_width = 1024;
        let grid_rows = (dst_height + GRID_SPACING - 1) / GRID_SPACING + 1;
        let grid_cols = (dst_width + GRID_SPACING - 1) / GRID_SPACING + 1;

        // With 32-pixel spacing, 1024 pixels needs 33 grid points
        assert_eq!(grid_rows, 33);
        assert_eq!(grid_cols, 33);

        // Total grid points: 33 * 33 = 1089 instead of 1024 * 1024 = 1M
        assert!(grid_rows * grid_cols < 2000);
    }
}
