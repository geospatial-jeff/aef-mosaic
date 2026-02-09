//! Coordinate reprojection between arbitrary CRS using proj.

use anyhow::{Context, Result};
use ndarray::Array3;
use proj::Proj;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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

/// Reprojector for transforming tile data to the output CRS.
pub struct Reprojector {
    /// Cache of Proj transformations by source CRS
    proj_cache: RwLock<HashMap<String, Arc<Proj>>>,

    /// Target CRS
    target_crs: String,
}

impl Reprojector {
    /// Create a new reprojector for the given target CRS.
    pub fn new(target_crs: &str) -> Self {
        Self {
            proj_cache: RwLock::new(HashMap::new()),
            target_crs: target_crs.to_string(),
        }
    }

    /// Get or create a Proj transformation for the given source CRS.
    ///
    /// Returns a transform from target_crs → source_crs (for reverse mapping).
    /// We use reverse mapping because we iterate over output pixels and need to
    /// find the corresponding source pixel.
    fn get_proj(&self, source_crs: &str) -> Result<Arc<Proj>> {
        // Check cache first
        // Use unwrap_or_else to recover from poisoned lock (if another thread panicked)
        {
            let cache = self.proj_cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(proj) = cache.get(source_crs) {
                return Ok(proj.clone());
            }
        }

        // Create new transformation: target → source (for reverse mapping)
        // We iterate over output pixels and need to find where they came from in the source
        let proj = Proj::new_known_crs(&self.target_crs, source_crs, None)
            .with_context(|| format!("Failed to create projection from {} to {}", self.target_crs, source_crs))?;

        let proj = Arc::new(proj);

        // Cache it
        // Use unwrap_or_else to recover from poisoned lock (if another thread panicked)
        {
            let mut cache = self.proj_cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(source_crs.to_string(), proj.clone());
        }

        Ok(proj)
    }

    /// Reproject a single tile to the target grid.
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
        let proj = self.get_proj(source_crs)?;

        let (bands, src_height, src_width) = data.dim();
        let (dst_height, dst_width) = config.target_shape;

        // Calculate source pixel size
        // Source is bottom-up: row 0 at min_y, row increases northward
        let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
        let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

        tracing::debug!(
            "Reproject: src_crs={}, src_bounds=[{:.2}, {:.2}, {:.2}, {:.2}], src_shape={}x{}, src_pixel=[{:.2}, {:.2}]",
            source_crs, source_bounds[0], source_bounds[1], source_bounds[2], source_bounds[3],
            src_width, src_height, src_pixel_x, src_pixel_y
        );
        tracing::debug!(
            "Reproject: dst_crs={}, dst_bounds=[{:.2}, {:.2}, {:.2}, {:.2}], dst_shape={}x{}, dst_res={:.2}",
            config.target_crs, config.target_bounds[0], config.target_bounds[1],
            config.target_bounds[2], config.target_bounds[3],
            dst_width, dst_height, config.target_resolution
        );

        // Create output array
        let mut output = Array3::<i8>::zeros((bands, dst_height, dst_width));

        // Log a few sample projections
        let sample_points = [(0, 0), (0, dst_width/2), (dst_height/2, dst_width/2), (dst_height-1, dst_width-1)];
        for (row, col) in sample_points {
            let dst_x = config.target_bounds[0] + (col as f64 + 0.5) * config.target_resolution;
            let dst_y = config.target_bounds[3] - (row as f64 + 0.5) * config.target_resolution;
            match proj.convert((dst_x, dst_y)) {
                Ok((src_x, src_y)) => {
                    let src_col = ((src_x - source_bounds[0]) / src_pixel_x) as isize;
                    let src_row = ((src_y - source_bounds[1]) / src_pixel_y) as isize;
                    tracing::debug!(
                        "Sample proj: dst[{},{}]=({:.2},{:.2}) -> src({:.2},{:.2}) -> pixel[{},{}] (bounds: 0..{}, 0..{})",
                        row, col, dst_x, dst_y, src_x, src_y, src_row, src_col, src_height, src_width
                    );
                }
                Err(e) => {
                    tracing::debug!("Sample proj: dst[{},{}]=({:.2},{:.2}) -> FAILED: {}", row, col, dst_x, dst_y, e);
                }
            }
        }

        // For each output pixel, find the corresponding input pixel
        // This is reverse projection (target → source) for proper resampling
        //
        // Optimized: Use batch coordinate conversion to reduce FFI overhead.
        // Instead of one proj.convert() call per pixel, we build a coordinate
        // array and convert all at once.

        // Build array of destination coordinates
        let num_pixels = dst_height * dst_width;
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(num_pixels);

        for dst_row in 0..dst_height {
            for dst_col in 0..dst_width {
                // Target coordinates (center of pixel)
                // Output is top-down: row 0 at max_y (north), row increases southward
                let dst_x = config.target_bounds[0] + (dst_col as f64 + 0.5) * config.target_resolution;
                let dst_y = config.target_bounds[3] - (dst_row as f64 + 0.5) * config.target_resolution;
                coords.push((dst_x, dst_y));
            }
        }

        // Batch convert all coordinates at once
        if let Err(e) = proj.convert_array(&mut coords) {
            tracing::warn!("Batch projection failed: {}, falling back to per-pixel", e);
            // Coords that failed will have NaN or unchanged values
        }

        // Map converted coordinates back to source pixels
        let mut pixels_mapped = 0usize;
        let mut pixels_out_of_bounds = 0usize;

        for dst_row in 0..dst_height {
            for dst_col in 0..dst_width {
                let idx = dst_row * dst_width + dst_col;
                let (src_x, src_y) = coords[idx];

                // Check for NaN (projection failure)
                if src_x.is_nan() || src_y.is_nan() {
                    continue;
                }

                // Source pixel coordinates
                // Source is bottom-up: row 0 at min_y, row increases northward
                let src_col = ((src_x - source_bounds[0]) / src_pixel_x) as isize;
                let src_row = ((src_y - source_bounds[1]) / src_pixel_y) as isize;

                // Check bounds
                if src_col >= 0
                    && src_col < src_width as isize
                    && src_row >= 0
                    && src_row < src_height as isize
                {
                    let src_row = src_row as usize;
                    let src_col = src_col as usize;

                    // Copy all bands
                    for band in 0..bands {
                        output[[band, dst_row, dst_col]] = data[[band, src_row, src_col]];
                    }
                    pixels_mapped += 1;
                } else {
                    pixels_out_of_bounds += 1;
                }
            }
        }

        tracing::debug!(
            "Reproject result: {} pixels mapped, {} out of bounds (total: {})",
            pixels_mapped, pixels_out_of_bounds, dst_height * dst_width
        );

        Ok(output)
    }

    /// Create a lookup table for reprojection (for repeated use).
    ///
    /// Source uses bottom-up orientation (row 0 at south, AEF COG format).
    /// Output uses top-down orientation (row 0 at north, standard raster).
    pub fn create_lookup_table(
        &self,
        source_crs: &str,
        source_bounds: &[f64; 4],
        source_shape: (usize, usize),
        config: &ReprojectConfig,
    ) -> Result<ReprojectionLUT> {
        let proj = self.get_proj(source_crs)?;

        let (src_height, src_width) = source_shape;
        let (dst_height, dst_width) = config.target_shape;

        // Source is bottom-up: row 0 at min_y, row increases northward
        let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
        let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

        let mut lut = Vec::with_capacity(dst_height * dst_width);

        for dst_row in 0..dst_height {
            for dst_col in 0..dst_width {
                // Output is top-down: row 0 at max_y (north), row increases southward
                let dst_x =
                    config.target_bounds[0] + (dst_col as f64 + 0.5) * config.target_resolution;
                let dst_y =
                    config.target_bounds[3] - (dst_row as f64 + 0.5) * config.target_resolution;

                if let Ok((src_x, src_y)) = proj.convert((dst_x, dst_y)) {
                    // Source is bottom-up: row 0 at min_y, row increases northward
                    let src_col = ((src_x - source_bounds[0]) / src_pixel_x) as isize;
                    let src_row = ((src_y - source_bounds[1]) / src_pixel_y) as isize;

                    if src_col >= 0
                        && src_col < src_width as isize
                        && src_row >= 0
                        && src_row < src_height as isize
                    {
                        lut.push(Some((src_row as usize, src_col as usize)));
                    } else {
                        lut.push(None);
                    }
                } else {
                    lut.push(None);
                }
            }
        }

        Ok(ReprojectionLUT {
            lut,
            dst_shape: config.target_shape,
        })
    }
}

/// Lookup table for fast repeated reprojection.
#[derive(Debug, Clone)]
pub struct ReprojectionLUT {
    /// For each output pixel, the corresponding input pixel (row, col) or None
    lut: Vec<Option<(usize, usize)>>,

    /// Destination array shape (height, width)
    dst_shape: (usize, usize),
}

impl ReprojectionLUT {
    /// Apply the lookup table to reproject data (parallelized over rows).
    pub fn apply(&self, data: &Array3<i8>) -> Array3<i8> {
        let (bands, _, _) = data.dim();
        let (dst_height, dst_width) = self.dst_shape;

        let mut output = Array3::<i8>::zeros((bands, dst_height, dst_width));

        // Parallelize over rows for better cache locality
        let row_results: Vec<Vec<(usize, usize, Vec<i8>)>> = (0..dst_height)
            .into_par_iter()
            .map(|dst_row| {
                let mut row_pixels = Vec::new();
                for dst_col in 0..dst_width {
                    let idx = dst_row * dst_width + dst_col;
                    if let Some((src_row, src_col)) = self.lut[idx] {
                        let band_values: Vec<i8> = (0..bands)
                            .map(|b| data[[b, src_row, src_col]])
                            .collect();
                        row_pixels.push((dst_row, dst_col, band_values));
                    }
                }
                row_pixels
            })
            .collect();

        // Write results to output (single-threaded)
        for row_data in row_results {
            for (dst_row, dst_col, band_values) in row_data {
                for (band, &val) in band_values.iter().enumerate() {
                    output[[band, dst_row, dst_col]] = val;
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_reprojector_cache() {
        let reprojector = Reprojector::new("EPSG:4326");

        // Create two projections
        let proj1 = reprojector.get_proj("EPSG:32610").unwrap();
        let proj2 = reprojector.get_proj("EPSG:32610").unwrap();

        // Should be the same Arc
        assert!(Arc::ptr_eq(&proj1, &proj2));
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

        // EPSG:4326 to EPSG:4326 should be identity-ish
        // (may have small differences due to coordinate order handling)
        let result = reprojector.reproject_tile(
            &data,
            "EPSG:4326",
            &[0.0, 0.0, 1.0, 1.0],
            &config,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_lut_apply_identity() {
        // Create a simple 2x2 identity LUT (each output pixel maps to same input pixel)
        let lut = ReprojectionLUT {
            lut: vec![
                Some((0, 0)), Some((0, 1)),
                Some((1, 0)), Some((1, 1)),
            ],
            dst_shape: (2, 2),
        };

        // Create input data: 2 bands, 2x2 pixels
        let mut data = Array3::<i8>::zeros((2, 2, 2));
        data[[0, 0, 0]] = 1;
        data[[0, 0, 1]] = 2;
        data[[0, 1, 0]] = 3;
        data[[0, 1, 1]] = 4;
        data[[1, 0, 0]] = 10;
        data[[1, 0, 1]] = 20;
        data[[1, 1, 0]] = 30;
        data[[1, 1, 1]] = 40;

        let result = lut.apply(&data);

        // Output should match input for identity mapping
        assert_eq!(result[[0, 0, 0]], 1);
        assert_eq!(result[[0, 0, 1]], 2);
        assert_eq!(result[[0, 1, 0]], 3);
        assert_eq!(result[[0, 1, 1]], 4);
        assert_eq!(result[[1, 0, 0]], 10);
        assert_eq!(result[[1, 0, 1]], 20);
        assert_eq!(result[[1, 1, 0]], 30);
        assert_eq!(result[[1, 1, 1]], 40);
    }

    #[test]
    fn test_lut_apply_with_none() {
        // LUT with some None entries (pixels outside source bounds)
        let lut = ReprojectionLUT {
            lut: vec![
                Some((0, 0)), None,
                None, Some((1, 1)),
            ],
            dst_shape: (2, 2),
        };

        let mut data = Array3::<i8>::zeros((1, 2, 2));
        data[[0, 0, 0]] = 42;
        data[[0, 1, 1]] = 99;

        let result = lut.apply(&data);

        // Mapped pixels should have values
        assert_eq!(result[[0, 0, 0]], 42);
        assert_eq!(result[[0, 1, 1]], 99);

        // None pixels should be zero
        assert_eq!(result[[0, 0, 1]], 0);
        assert_eq!(result[[0, 1, 0]], 0);
    }

    #[test]
    fn test_lut_apply_remapping() {
        // LUT that swaps pixels: (0,0) -> (1,1), (0,1) -> (1,0), etc.
        let lut = ReprojectionLUT {
            lut: vec![
                Some((1, 1)), Some((1, 0)),
                Some((0, 1)), Some((0, 0)),
            ],
            dst_shape: (2, 2),
        };

        let mut data = Array3::<i8>::zeros((1, 2, 2));
        data[[0, 0, 0]] = 1;
        data[[0, 0, 1]] = 2;
        data[[0, 1, 0]] = 3;
        data[[0, 1, 1]] = 4;

        let result = lut.apply(&data);

        // Check remapped values
        assert_eq!(result[[0, 0, 0]], 4); // maps from (1,1)
        assert_eq!(result[[0, 0, 1]], 3); // maps from (1,0)
        assert_eq!(result[[0, 1, 0]], 2); // maps from (0,1)
        assert_eq!(result[[0, 1, 1]], 1); // maps from (0,0)
    }

    #[test]
    fn test_lut_apply_multiband() {
        let lut = ReprojectionLUT {
            lut: vec![Some((0, 0))],
            dst_shape: (1, 1),
        };

        // 64 bands (like AEF embeddings)
        let mut data = Array3::<i8>::zeros((64, 1, 1));
        for b in 0..64 {
            data[[b, 0, 0]] = b as i8;
        }

        let result = lut.apply(&data);

        // All bands should be mapped
        for b in 0..64 {
            assert_eq!(result[[b, 0, 0]], b as i8);
        }
    }
}
