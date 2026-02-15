//! Coordinate Reference System utilities.
//!
//! This module provides utilities for transforming coordinates between CRS.
//!
//! ## CRS used in this pipeline:
//!
//! - **WGS84 (EPSG:4326)**: Geographic coordinates (lon, lat in degrees).
//!   Used for: R-tree spatial index, user-facing bounds, tile intersection queries.
//!
//! - **Output CRS (e.g., EPSG:6933)**: Equal-area projection (x, y in meters).
//!   Used for: Output Zarr grid, chunk bounds for reprojection.
//!
//! - **Tile native CRS (e.g., EPSG:32610 UTM 10N)**: Projected coordinates (x, y in meters).
//!   Used for: Pixel coordinate calculations within input tiles.
//!
//! ## Coordinate order convention:
//!
//! - Bounds arrays: `[min_x, min_y, max_x, max_y]` = `[west, south, east, north]`
//! - For WGS84: `[min_lon, min_lat, max_lon, max_lat]`

/// Common CRS codes used throughout the pipeline.
pub mod codes {
    /// WGS84 geographic coordinate system (lon/lat in degrees).
    /// Used for R-tree spatial indexing and user-facing bounds.
    pub const WGS84: &str = "EPSG:4326";

    /// Cylindrical Equal Area projection (x/y in meters).
    /// Default output CRS for global mosaics.
    pub const CEA: &str = "EPSG:6933";
}

use anyhow::{Context, Result};
use proj::Proj;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Bounds in a specific CRS: [min_x, min_y, max_x, max_y]
pub type Bounds = [f64; 4];

/// Thread-safe cache for Proj transformations.
///
/// Creating Proj objects is expensive, so we cache them by (source, target) CRS pair.
#[derive(Default)]
pub struct ProjCache {
    cache: RwLock<HashMap<(String, String), Arc<Proj>>>,
}

impl ProjCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a Proj transformation between two CRS.
    pub fn get(&self, from_crs: &str, to_crs: &str) -> Result<Arc<Proj>> {
        let key = (from_crs.to_string(), to_crs.to_string());

        // Check cache first
        {
            let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(proj) = cache.get(&key) {
                return Ok(proj.clone());
            }
        }

        // Create new transformation
        let proj = Proj::new_known_crs(from_crs, to_crs, None)
            .with_context(|| format!("Failed to create projection from {} to {}", from_crs, to_crs))?;

        let proj = Arc::new(proj);

        // Cache it
        {
            let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
            cache.insert(key, proj.clone());
        }

        Ok(proj)
    }
}

/// Transform a single point between CRS.
pub fn transform_point(
    x: f64,
    y: f64,
    from_crs: &str,
    to_crs: &str,
    cache: &ProjCache,
) -> Result<(f64, f64)> {
    if from_crs == to_crs {
        return Ok((x, y));
    }

    let proj = cache.get(from_crs, to_crs)?;
    proj.convert((x, y))
        .with_context(|| format!("Failed to transform point ({}, {}) from {} to {}", x, y, from_crs, to_crs))
}

/// Transform bounds between CRS.
///
/// Transforms all 4 corners and returns the bounding box of the result.
/// This handles projection distortion properly.
pub fn transform_bounds(
    bounds: &Bounds,
    from_crs: &str,
    to_crs: &str,
    cache: &ProjCache,
) -> Result<Bounds> {
    if from_crs == to_crs {
        return Ok(*bounds);
    }

    let proj = cache.get(from_crs, to_crs)?;

    // Transform all 4 corners
    let corners = [
        (bounds[0], bounds[1]), // min_x, min_y (SW)
        (bounds[2], bounds[1]), // max_x, min_y (SE)
        (bounds[2], bounds[3]), // max_x, max_y (NE)
        (bounds[0], bounds[3]), // min_x, max_y (NW)
    ];

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (x, y) in corners {
        let (tx, ty) = proj.convert((x, y))
            .with_context(|| format!("Failed to transform corner ({}, {})", x, y))?;
        min_x = min_x.min(tx);
        min_y = min_y.min(ty);
        max_x = max_x.max(tx);
        max_y = max_y.max(ty);
    }

    Ok([min_x, min_y, max_x, max_y])
}

/// Convert an EPSG code to PROJ definition string.
///
/// Returns the PROJ definition string (e.g., "+proj=cea +lon_0=0 ..."),
/// which can be used by PROJ/GDAL/rasterio to interpret the CRS.
/// This is useful alongside the EPSG code for CRS identification.
///
/// Returns None if the definition is not available (can happen with some CRS).
pub fn epsg_to_proj_definition(epsg_code: &str) -> Option<String> {
    let proj = Proj::new(epsg_code).ok()?;
    let def = proj.def().ok()?;
    if def.is_empty() {
        None
    } else {
        Some(def)
    }
}

/// Transform bounds with edge sampling for better accuracy.
///
/// Samples points along edges to handle non-linear projections.
pub fn transform_bounds_with_densification(
    bounds: &Bounds,
    from_crs: &str,
    to_crs: &str,
    cache: &ProjCache,
    n_samples: usize,
) -> Result<Bounds> {
    if from_crs == to_crs {
        return Ok(*bounds);
    }

    let proj = cache.get(from_crs, to_crs)?;

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    // Sample along all 4 edges
    for i in 0..=n_samples {
        let t = i as f64 / n_samples as f64;

        // Bottom edge (min_y)
        let x = bounds[0] + t * (bounds[2] - bounds[0]);
        if let Ok((tx, ty)) = proj.convert((x, bounds[1])) {
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }

        // Top edge (max_y)
        if let Ok((tx, ty)) = proj.convert((x, bounds[3])) {
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }

        // Left edge (min_x)
        let y = bounds[1] + t * (bounds[3] - bounds[1]);
        if let Ok((tx, ty)) = proj.convert((bounds[0], y)) {
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }

        // Right edge (max_x)
        if let Ok((tx, ty)) = proj.convert((bounds[2], y)) {
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }
    }

    Ok([min_x, min_y, max_x, max_y])
}

/// Compute the intersection of two bounds.
///
/// Returns None if there's no intersection.
pub fn intersect_bounds(a: &Bounds, b: &Bounds) -> Option<Bounds> {
    let min_x = a[0].max(b[0]);
    let min_y = a[1].max(b[1]);
    let max_x = a[2].min(b[2]);
    let max_y = a[3].min(b[3]);

    if min_x < max_x && min_y < max_y {
        Some([min_x, min_y, max_x, max_y])
    } else {
        None
    }
}

/// Convert geographic bounds to pixel coordinates within a tile.
///
/// The tile has bounds in its native CRS and a known resolution.
/// Returns pixel window: (x, y, width, height) where (x, y) is top-left corner.
///
/// # Arguments
/// * `bounds_native` - Bounds to convert, in the tile's native CRS
/// * `tile_bounds` - The tile's full bounds in native CRS
/// * `_tile_resolution` - Pixel size in native CRS units (unused, derived from tile_bounds/tile_size)
/// * `tile_size` - Tile dimensions in pixels (width, height)
pub fn bounds_to_pixel_window(
    bounds_native: &Bounds,
    tile_bounds: &Bounds,
    _tile_resolution: f64,
    tile_size: (usize, usize),
) -> (usize, usize, usize, usize) {
    let (tile_width_px, tile_height_px) = tile_size;

    // Calculate tile dimensions in CRS units
    let tile_width_crs = tile_bounds[2] - tile_bounds[0];
    let tile_height_crs = tile_bounds[3] - tile_bounds[1];

    // Convert bounds to relative position within tile (0-1 range)
    let rel_west = (bounds_native[0] - tile_bounds[0]) / tile_width_crs;
    let rel_east = (bounds_native[2] - tile_bounds[0]) / tile_width_crs;
    let rel_south = (bounds_native[1] - tile_bounds[1]) / tile_height_crs;
    let rel_north = (bounds_native[3] - tile_bounds[1]) / tile_height_crs;

    // Clamp to valid range
    let rel_west = rel_west.max(0.0).min(1.0);
    let rel_east = rel_east.max(0.0).min(1.0);
    let rel_south = rel_south.max(0.0).min(1.0);
    let rel_north = rel_north.max(0.0).min(1.0);

    // Convert to pixel coordinates
    // Note: In image coordinates, y=0 is at the top (north), so we flip
    let x = (rel_west * tile_width_px as f64).floor() as usize;
    let y = ((1.0 - rel_north) * tile_height_px as f64).floor() as usize;
    let x_end = (rel_east * tile_width_px as f64).ceil() as usize;
    let y_end = ((1.0 - rel_south) * tile_height_px as f64).ceil() as usize;

    // Clamp to valid pixel range
    let x = x.min(tile_width_px);
    let y = y.min(tile_height_px);
    let width = (x_end - x).min(tile_width_px - x).max(1);
    let height = (y_end - y).min(tile_height_px - y).max(1);

    (x, y, width, height)
}

/// Convert pixel window to geographic bounds within a tile.
///
/// This is the inverse of `bounds_to_pixel_window`.
pub fn pixel_window_to_bounds(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    tile_bounds: &Bounds,
    tile_size: (usize, usize),
) -> Bounds {
    let (tile_width_px, tile_height_px) = tile_size;
    let tile_width_crs = tile_bounds[2] - tile_bounds[0];
    let tile_height_crs = tile_bounds[3] - tile_bounds[1];

    // Convert pixel coordinates to relative positions
    let rel_west = x as f64 / tile_width_px as f64;
    let rel_east = (x + width) as f64 / tile_width_px as f64;
    // Note: y=0 is at top (north), so we flip
    let rel_north = 1.0 - (y as f64 / tile_height_px as f64);
    let rel_south = 1.0 - ((y + height) as f64 / tile_height_px as f64);

    // Convert to geographic coordinates
    let west = tile_bounds[0] + rel_west * tile_width_crs;
    let east = tile_bounds[0] + rel_east * tile_width_crs;
    let south = tile_bounds[1] + rel_south * tile_height_crs;
    let north = tile_bounds[1] + rel_north * tile_height_crs;

    [west, south, east, north]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crs_codes() {
        assert_eq!(codes::WGS84, "EPSG:4326");
        assert_eq!(codes::CEA, "EPSG:6933");
    }

    #[test]
    fn test_epsg_to_proj_definition() {
        // Test that the function returns a result (None is acceptable if PROJ can't resolve)
        let def = epsg_to_proj_definition(codes::WGS84);
        // If available, check it's well-formed
        if let Some(d) = def {
            assert!(!d.is_empty());
        }

        // Test EPSG:6933 (NSIDC EASE-Grid 2.0 Global)
        let def = epsg_to_proj_definition(codes::CEA);
        if let Some(d) = def {
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_transform_bounds_identity() {
        let cache = ProjCache::new();
        let bounds = [-122.0, 37.0, -121.0, 38.0];
        let result = transform_bounds(&bounds, codes::WGS84, codes::WGS84, &cache).unwrap();
        assert_eq!(result, bounds);
    }

    #[test]
    fn test_transform_bounds_wgs84_to_utm() {
        let cache = ProjCache::new();
        let bounds_wgs84 = [-122.5, 37.5, -122.0, 38.0];
        let result = transform_bounds(&bounds_wgs84, codes::WGS84, "EPSG:32610", &cache).unwrap();

        // UTM zone 10N should give positive easting and northing
        assert!(result[0] > 0.0); // min_x (easting)
        assert!(result[1] > 0.0); // min_y (northing)
        assert!(result[2] > result[0]); // max_x > min_x
        assert!(result[3] > result[1]); // max_y > min_y

        // Approximate expected values for SF area
        assert!(result[0] > 500000.0 && result[0] < 600000.0);
        assert!(result[1] > 4100000.0 && result[1] < 4300000.0);
    }

    #[test]
    fn test_densification_expands_global_bounds() {
        // For large areas (especially near poles), densification should capture
        // the full extent better than just transforming corners
        let cache = ProjCache::new();

        // Transform a large WGS84 region to CEA (cylindrical equal area)
        let bounds_wgs84 = [-180.0, -60.0, 180.0, 60.0];

        let corners_only = transform_bounds(&bounds_wgs84, codes::WGS84, codes::CEA, &cache).unwrap();
        let with_densification = transform_bounds_with_densification(
            &bounds_wgs84,
            codes::WGS84,
            codes::CEA,
            &cache,
            20,
        ).unwrap();

        // Densified bounds should be at least as large as corners-only
        assert!(
            with_densification[0] <= corners_only[0],
            "min_x: densified {} should be <= corners {}",
            with_densification[0], corners_only[0]
        );
        assert!(
            with_densification[1] <= corners_only[1],
            "min_y: densified {} should be <= corners {}",
            with_densification[1], corners_only[1]
        );
        assert!(
            with_densification[2] >= corners_only[2],
            "max_x: densified {} should be >= corners {}",
            with_densification[2], corners_only[2]
        );
        assert!(
            with_densification[3] >= corners_only[3],
            "max_y: densified {} should be >= corners {}",
            with_densification[3], corners_only[3]
        );
    }

    #[test]
    fn test_densification_for_small_area() {
        // For small areas, densification should give similar results to corners-only
        let cache = ProjCache::new();

        let bounds_wgs84 = [-122.5, 37.5, -122.0, 38.0];

        let corners_only = transform_bounds(&bounds_wgs84, codes::WGS84, "EPSG:32610", &cache).unwrap();
        let with_densification = transform_bounds_with_densification(
            &bounds_wgs84,
            codes::WGS84,
            "EPSG:32610",
            &cache,
            10,
        ).unwrap();

        // For small areas, the difference should be minimal (< 1% of extent)
        let width = corners_only[2] - corners_only[0];
        let height = corners_only[3] - corners_only[1];

        assert!(
            (with_densification[0] - corners_only[0]).abs() < width * 0.01,
            "min_x difference too large"
        );
        assert!(
            (with_densification[2] - corners_only[2]).abs() < width * 0.01,
            "max_x difference too large"
        );
        assert!(
            (with_densification[1] - corners_only[1]).abs() < height * 0.01,
            "min_y difference too large"
        );
        assert!(
            (with_densification[3] - corners_only[3]).abs() < height * 0.01,
            "max_y difference too large"
        );
    }

    #[test]
    fn test_intersect_bounds() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let result = intersect_bounds(&a, &b);
        assert_eq!(result, Some([5.0, 5.0, 10.0, 10.0]));
    }

    #[test]
    fn test_intersect_bounds_no_overlap() {
        let a = [0.0, 0.0, 5.0, 5.0];
        let b = [10.0, 10.0, 15.0, 15.0];
        let result = intersect_bounds(&a, &b);
        assert_eq!(result, None);
    }

    #[test]
    fn test_bounds_to_pixel_window() {
        // Tile covers [0, 0, 1000, 1000] in native CRS
        let tile_bounds = [0.0, 0.0, 1000.0, 1000.0];
        let tile_size = (100, 100); // 10m resolution

        // Query the center of the tile
        let query_bounds = [250.0, 250.0, 750.0, 750.0];
        let (x, y, w, h) = bounds_to_pixel_window(&query_bounds, &tile_bounds, 10.0, tile_size);

        // Should be centered with 50% width/height
        assert_eq!(x, 25);
        assert_eq!(y, 25); // y is flipped: (1 - 0.75) * 100 = 25
        assert_eq!(w, 50);
        assert_eq!(h, 50);
    }

    #[test]
    fn test_pixel_window_to_bounds_roundtrip() {
        let tile_bounds = [500000.0, 4000000.0, 581920.0, 4081920.0];
        let tile_size = (8192, 8192);

        // Original window
        let (x, y, w, h) = (1024, 2048, 512, 512);

        // Convert to bounds and back
        let bounds = pixel_window_to_bounds(x, y, w, h, &tile_bounds, tile_size);
        let (x2, y2, w2, h2) = bounds_to_pixel_window(&bounds, &tile_bounds, 10.0, tile_size);

        assert_eq!(x, x2);
        assert_eq!(y, y2);
        assert_eq!(w, w2);
        assert_eq!(h, h2);
    }
}
