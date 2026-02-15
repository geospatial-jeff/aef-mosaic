//! Coordinate reprojection using GDAL.
//!
//! Uses GDAL's optimized warp implementation which includes:
//! - Approximate transformers (sparse grid + interpolation)
//! - Efficient C implementation
//! - Proper handling of edge cases

use anyhow::{Context, Result};
use gdal::spatial_ref::SpatialRef;
use gdal::DriverManager;
use ndarray::Array3;
use std::collections::HashMap;
use std::sync::RwLock;

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

/// Reprojector for transforming tile data to the output CRS using GDAL.
pub struct Reprojector {
    /// Cache of SpatialRef by CRS string
    srs_cache: RwLock<HashMap<String, SpatialRef>>,

    /// Target SpatialRef (cached)
    target_srs: SpatialRef,
}

impl Reprojector {
    /// Create a new reprojector for the given target CRS.
    pub fn new(target_crs: &str) -> Self {
        let target_srs = SpatialRef::from_definition(target_crs)
            .expect("Failed to create target SpatialRef");

        Self {
            srs_cache: RwLock::new(HashMap::new()),
            target_srs,
        }
    }

    /// Get or create a SpatialRef for the given CRS.
    fn get_srs(&self, crs: &str) -> Result<SpatialRef> {
        // Check cache first
        {
            let cache = self.srs_cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(srs) = cache.get(crs) {
                return Ok(srs.clone());
            }
        }

        // Create new SpatialRef
        let srs = SpatialRef::from_definition(crs)
            .with_context(|| format!("Failed to create SpatialRef for {}", crs))?;

        // Cache it
        {
            let mut cache = self.srs_cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(crs.to_string(), srs.clone());
        }

        Ok(srs)
    }

    /// Reproject a single tile to the target grid using GDAL.
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

        // Calculate source geotransform
        // Source is bottom-up: origin at (min_x, min_y), positive y scale
        let src_pixel_x = (source_bounds[2] - source_bounds[0]) / src_width as f64;
        let src_pixel_y = (source_bounds[3] - source_bounds[1]) / src_height as f64;

        // For bottom-up data, we need to flip vertically when creating the GDAL dataset
        // GDAL expects top-down (origin at top-left, negative y scale)
        // GeoTransform: [origin_x, pixel_width, rotation, origin_y, rotation, pixel_height]
        let src_geotransform = [
            source_bounds[0],           // origin x (min_x)
            src_pixel_x,                // pixel width
            0.0,                        // rotation
            source_bounds[3],           // origin y (max_y for top-down)
            0.0,                        // rotation
            -src_pixel_y,               // pixel height (negative for top-down)
        ];

        // Target geotransform (top-down)
        let dst_geotransform = [
            config.target_bounds[0],    // origin x (min_x)
            config.target_resolution,   // pixel width
            0.0,                        // rotation
            config.target_bounds[3],    // origin y (max_y)
            0.0,                        // rotation
            -config.target_resolution,  // pixel height (negative)
        ];

        tracing::debug!(
            "GDAL reproject: src_crs={}, src_shape={}x{}, dst_crs={}, dst_shape={}x{}",
            source_crs, src_width, src_height,
            config.target_crs, dst_width, dst_height
        );

        // Get MEM driver
        let mem_driver = DriverManager::get_driver_by_name("MEM")
            .context("Failed to get MEM driver")?;

        // Create source dataset
        let mut src_dataset = mem_driver
            .create_with_band_type::<i8, _>(
                "",
                src_width,
                src_height,
                bands,
            )
            .context("Failed to create source dataset")?;

        // Set source geotransform and CRS
        src_dataset.set_geo_transform(&src_geotransform)
            .context("Failed to set source geotransform")?;

        let src_srs = self.get_srs(source_crs)?;
        src_dataset.set_spatial_ref(&src_srs)
            .context("Failed to set source CRS")?;

        // Write data to source dataset (flip vertically for GDAL's top-down expectation)
        for band_idx in 0..bands {
            let mut band = src_dataset.rasterband(band_idx + 1)
                .context("Failed to get source band")?;

            // Flip data vertically: row 0 in our bottom-up data becomes row (height-1) in GDAL
            let mut flipped_band: Vec<i8> = Vec::with_capacity(src_height * src_width);
            for row in (0..src_height).rev() {
                for col in 0..src_width {
                    flipped_band.push(data[[band_idx, row, col]]);
                }
            }

            band.write(
                (0, 0),
                (src_width, src_height),
                &mut gdal::raster::Buffer::new((src_width, src_height), flipped_band),
            ).context("Failed to write source band")?;
        }

        // Create destination dataset
        let mut dst_dataset = mem_driver
            .create_with_band_type::<i8, _>(
                "",
                dst_width,
                dst_height,
                bands,
            )
            .context("Failed to create destination dataset")?;

        // Set destination geotransform and CRS
        dst_dataset.set_geo_transform(&dst_geotransform)
            .context("Failed to set destination geotransform")?;

        dst_dataset.set_spatial_ref(&self.target_srs)
            .context("Failed to set destination CRS")?;

        // Perform reprojection
        gdal::raster::reproject(&src_dataset, &dst_dataset)
            .context("GDAL reproject failed")?;

        // Read result into output array
        let mut output = Array3::<i8>::zeros((bands, dst_height, dst_width));

        for band_idx in 0..bands {
            let band = dst_dataset.rasterband(band_idx + 1)
                .context("Failed to get destination band")?;

            let buffer = band.read_as::<i8>(
                (0, 0),
                (dst_width, dst_height),
                (dst_width, dst_height),
                None,
            ).context("Failed to read destination band")?;

            // Copy to output array
            for row in 0..dst_height {
                for col in 0..dst_width {
                    output[[band_idx, row, col]] = buffer.data()[row * dst_width + col];
                }
            }
        }

        let non_zero = output.iter().filter(|&&v| v != 0).count();
        tracing::debug!(
            "GDAL reproject result: {} non-zero pixels out of {}",
            non_zero, dst_height * dst_width * bands
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_reprojector_cache() {
        let reprojector = Reprojector::new("EPSG:4326");

        // Create two SRS for the same CRS
        let srs1 = reprojector.get_srs("EPSG:32610").unwrap();
        let srs2 = reprojector.get_srs("EPSG:32610").unwrap();

        // Should both be valid (can't easily check if same object with SpatialRef)
        assert!(srs1.to_wkt().is_ok());
        assert!(srs2.to_wkt().is_ok());
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
}
