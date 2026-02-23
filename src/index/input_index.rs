//! Load AEF parquet index and build R-tree for spatial queries.

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use geo::{Coord, Intersects, LineString, Polygon, Rect};
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rstar::{RTree, RTreeObject, AABB};
use std::sync::Arc;

use crate::crs::ProjCache;

/// Reference wrapper for float arrays to avoid allocation during row-wise access.
/// Handles both Float64Array and Float32Array transparently.
enum FloatArrayRef<'a> {
    F64(&'a Float64Array),
    F32(&'a arrow::array::Float32Array),
}

impl<'a> FloatArrayRef<'a> {
    /// Get value at index as f64 (converts f32 if needed).
    #[inline]
    fn get_f64(&self, i: usize) -> f64 {
        match self {
            FloatArrayRef::F64(arr) => arr.value(i),
            FloatArrayRef::F32(arr) => arr.value(i) as f64,
        }
    }
}

/// A single COG tile from the AEF index.
#[derive(Debug, Clone)]
pub struct CogTile {
    /// Unique tile identifier
    pub tile_id: String,

    /// S3 path to the COG file
    pub s3_path: String,

    /// Source CRS (e.g., "EPSG:32610" for UTM 10N)
    pub crs: String,

    /// Bounding box in source CRS [min_x, min_y, max_x, max_y]
    pub bounds_native: [f64; 4],

    /// Bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat]
    pub bounds_wgs84: [f64; 4],

    /// Exact footprint polygon in WGS84 (4 corners transformed from native CRS).
    /// Used for precise intersection tests after R-tree AABB filtering.
    pub footprint_wgs84: Polygon<f64>,

    /// Resolution in source CRS units (meters for UTM)
    pub resolution: f64,

    /// Year of the tile data
    pub year: i32,
}

impl CogTile {
    /// Compute WGS84 footprint by transforming 4 native CRS corners.
    ///
    /// This creates an exact polygon representation of the tile in WGS84,
    /// which handles the distortion that occurs when transforming from
    /// projected CRS (like UTM) to geographic coordinates.
    pub fn compute_footprint(
        bounds_native: &[f64; 4],
        crs: &str,
        proj_cache: &ProjCache,
    ) -> Result<Polygon<f64>> {
        let corners = [
            (bounds_native[0], bounds_native[1]), // SW
            (bounds_native[2], bounds_native[1]), // SE
            (bounds_native[2], bounds_native[3]), // NE
            (bounds_native[0], bounds_native[3]), // NW
        ];

        let mut coords: Vec<Coord<f64>> = Vec::with_capacity(5);
        for (x, y) in corners {
            let (lon, lat) = crate::crs::transform_point(x, y, crs, "EPSG:4326", proj_cache)?;
            coords.push(Coord { x: lon, y: lat });
        }
        coords.push(coords[0]); // Close ring

        Ok(Polygon::new(LineString::new(coords), vec![]))
    }

    /// Create a simple rectangular footprint from WGS84 bounds.
    /// Used when the tile is already in WGS84 or for test data.
    pub fn footprint_from_wgs84_bounds(bounds: &[f64; 4]) -> Polygon<f64> {
        Rect::new(
            Coord {
                x: bounds[0],
                y: bounds[1],
            },
            Coord {
                x: bounds[2],
                y: bounds[3],
            },
        )
        .to_polygon()
    }
}

impl RTreeObject for CogTile {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.bounds_wgs84[0], self.bounds_wgs84[1]],
            [self.bounds_wgs84[2], self.bounds_wgs84[3]],
        )
    }
}

/// Newtype wrapper for Arc<CogTile> to implement RTreeObject.
/// This avoids cloning the underlying CogTile data when building the R-tree.
#[derive(Debug, Clone)]
pub struct ArcTile(pub Arc<CogTile>);

impl std::ops::Deref for ArcTile {
    type Target = CogTile;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RTreeObject for ArcTile {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.bounds_wgs84[0], self.bounds_wgs84[1]],
            [self.bounds_wgs84[2], self.bounds_wgs84[3]],
        )
    }
}

/// Index of all input COG tiles with spatial query support.
pub struct InputIndex {
    /// All tiles loaded from the parquet index (wrapped in Arc for cheap cloning)
    tiles: Vec<Arc<CogTile>>,

    /// R-tree for efficient spatial queries (uses ArcTile wrapper)
    rtree: RTree<ArcTile>,
}

impl InputIndex {
    /// Load index from a local parquet file.
    pub fn from_local_parquet(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open parquet file: {}", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let proj_cache = ProjCache::new();
        // Pre-allocate with Arc directly - avoids double allocation
        let mut tiles: Vec<Arc<CogTile>> = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            Self::extract_tiles_from_batch(&batch, &mut tiles, &proj_cache)?;
        }

        // Build RTree using ArcTile wrapper (cloning Arc is cheap)
        let rtree_tiles: Vec<ArcTile> = tiles.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);

        tracing::info!("Loaded {} tiles from parquet index", tiles.len());

        Ok(Self { tiles, rtree })
    }

    /// Load index from S3 using object_store.
    pub async fn from_s3(store: Arc<dyn ObjectStore>, path: &object_store::path::Path) -> Result<Self> {
        let bytes = store
            .get(path)
            .await?
            .bytes()
            .await?;

        Self::from_parquet_bytes(bytes)
    }

    /// Load index from parquet bytes.
    pub fn from_parquet_bytes(bytes: Bytes) -> Result<Self> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let reader = builder.build()?;

        let proj_cache = ProjCache::new();
        // Pre-allocate with Arc directly - avoids double allocation
        let mut tiles: Vec<Arc<CogTile>> = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            Self::extract_tiles_from_batch(&batch, &mut tiles, &proj_cache)?;
        }

        // Build RTree using ArcTile wrapper (cloning Arc is cheap)
        let rtree_tiles: Vec<ArcTile> = tiles.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);

        tracing::info!("Loaded {} tiles from parquet index", tiles.len());

        Ok(Self { tiles, rtree })
    }

    /// Extract tiles from a record batch.
    ///
    /// Memory-optimized: processes row-by-row and wraps in Arc immediately
    /// to avoid intermediate column Vecs and double allocation.
    fn extract_tiles_from_batch(
        batch: &RecordBatch,
        tiles: &mut Vec<Arc<CogTile>>,
        proj_cache: &ProjCache,
    ) -> Result<()> {
        let schema = batch.schema();
        let num_rows = batch.num_rows();

        // Get column indices - flexible to handle different column naming conventions
        let tile_id_col = Self::find_column(&schema, &["tile_id", "id", "name", "fid"])?;
        let s3_path_col = Self::find_column(&schema, &["s3_path", "path", "uri", "url", "location"])?;
        let crs_col = Self::find_column(&schema, &["crs", "epsg", "srs"])?;

        // Native CRS bounds columns
        let min_x_col = Self::find_column(&schema, &["min_x", "xmin", "left", "utm_west"])?;
        let min_y_col = Self::find_column(&schema, &["min_y", "ymin", "bottom", "utm_south"])?;
        let max_x_col = Self::find_column(&schema, &["max_x", "xmax", "right", "utm_east"])?;
        let max_y_col = Self::find_column(&schema, &["max_y", "ymax", "top", "utm_north"])?;

        // WGS84 bounds
        let min_lon_col = Self::find_column(&schema, &["min_lon", "lon_min", "west", "wgs84_west"])?;
        let min_lat_col = Self::find_column(&schema, &["min_lat", "lat_min", "south", "wgs84_south"])?;
        let max_lon_col = Self::find_column(&schema, &["max_lon", "lon_max", "east", "wgs84_east"])?;
        let max_lat_col = Self::find_column(&schema, &["max_lat", "lat_max", "north", "wgs84_north"])?;

        // Resolution is optional for AEF (fixed at 10m)
        let resolution_col = Self::find_column(&schema, &["resolution", "res", "pixel_size"]).ok();
        let year_col = Self::find_column(&schema, &["year", "date_year"]).ok();

        // Get column arrays as references (no allocation) for row-wise processing
        let tile_id_arr = batch.column(tile_id_col);
        let s3_path_arr = batch.column(s3_path_col).as_any().downcast_ref::<StringArray>()
            .context("s3_path/location column must be string")?;
        let crs_arr = batch.column(crs_col).as_any().downcast_ref::<StringArray>()
            .context("crs column must be string")?;

        // Get float arrays as references
        let min_x_arr = Self::get_f64_array_ref(batch.column(min_x_col))?;
        let min_y_arr = Self::get_f64_array_ref(batch.column(min_y_col))?;
        let max_x_arr = Self::get_f64_array_ref(batch.column(max_x_col))?;
        let max_y_arr = Self::get_f64_array_ref(batch.column(max_y_col))?;
        let min_lon_arr = Self::get_f64_array_ref(batch.column(min_lon_col))?;
        let min_lat_arr = Self::get_f64_array_ref(batch.column(min_lat_col))?;
        let max_lon_arr = Self::get_f64_array_ref(batch.column(max_lon_col))?;
        let max_lat_arr = Self::get_f64_array_ref(batch.column(max_lat_col))?;

        // Optional arrays - get references if present
        let resolution_arr = resolution_col.and_then(|col| Self::get_f64_array_ref(batch.column(col)).ok());
        let year_arr = year_col.map(|col| batch.column(col));

        // Reserve capacity to avoid reallocations during row iteration
        tiles.reserve(num_rows);

        // Process row-by-row, wrapping in Arc immediately
        for i in 0..num_rows {
            let bounds_native = [
                min_x_arr.get_f64(i),
                min_y_arr.get_f64(i),
                max_x_arr.get_f64(i),
                max_y_arr.get_f64(i),
            ];
            let bounds_wgs84 = [
                min_lon_arr.get_f64(i),
                min_lat_arr.get_f64(i),
                max_lon_arr.get_f64(i),
                max_lat_arr.get_f64(i),
            ];

            // Get tile_id - handle string or integer
            let tile_id = Self::get_tile_id(tile_id_arr, i)?;

            let crs = crs_arr.value(i).to_string();

            // Compute exact footprint by transforming 4 corners from native CRS to WGS84
            let footprint_wgs84 = CogTile::compute_footprint(&bounds_native, &crs, proj_cache)
                .unwrap_or_else(|_| {
                    // Fall back to rectangular footprint from WGS84 bounds if transform fails
                    CogTile::footprint_from_wgs84_bounds(&bounds_wgs84)
                });

            // Get resolution (default 10.0 for AEF)
            let resolution = resolution_arr
                .as_ref()
                .map(|arr| arr.get_f64(i))
                .unwrap_or(10.0);

            // Get year (default 2024)
            let year = Self::get_year(year_arr, i);

            // Wrap in Arc immediately to avoid double allocation
            tiles.push(Arc::new(CogTile {
                tile_id,
                s3_path: s3_path_arr.value(i).to_string(),
                crs,
                bounds_native,
                bounds_wgs84,
                footprint_wgs84,
                resolution,
                year,
            }));
        }

        Ok(())
    }

    /// Get tile_id from array at index, handling string or integer types.
    fn get_tile_id(arr: &Arc<dyn Array>, i: usize) -> Result<String> {
        if let Some(str_arr) = arr.as_any().downcast_ref::<StringArray>() {
            Ok(str_arr.value(i).to_string())
        } else if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
            Ok(i64_arr.value(i).to_string())
        } else if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
            Ok(i32_arr.value(i).to_string())
        } else {
            anyhow::bail!("tile_id column must be string or integer")
        }
    }

    /// Get year from optional array at index (default 2024).
    fn get_year(arr: Option<&Arc<dyn Array>>, i: usize) -> i32 {
        match arr {
            Some(arr) => {
                if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
                    i32_arr.value(i)
                } else if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    i64_arr.value(i) as i32
                } else {
                    2024
                }
            }
            None => 2024,
        }
    }

    /// Find a column by checking multiple possible names.
    fn find_column(schema: &SchemaRef, names: &[&str]) -> Result<usize> {
        for name in names {
            if let Some((idx, _)) = schema.column_with_name(name) {
                return Ok(idx);
            }
        }
        anyhow::bail!(
            "Could not find column with any of these names: {:?}",
            names
        )
    }

    /// Reference to a float array that supports f64 access without allocation.
    /// Handles both Float64Array and Float32Array.
    fn get_f64_array_ref(array: &Arc<dyn Array>) -> Result<FloatArrayRef<'_>> {
        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            Ok(FloatArrayRef::F64(arr))
        } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Float32Array>() {
            Ok(FloatArrayRef::F32(arr))
        } else {
            anyhow::bail!("Expected float array")
        }
    }

    /// Query tiles that intersect the given WGS84 bounding box.
    ///
    /// Uses a two-stage approach:
    /// 1. R-tree query with axis-aligned bounding boxes (fast candidate filter)
    /// 2. Exact polygon intersection test (eliminates false positives from AABB)
    ///
    /// Returns Arc clones (cheap reference count increment).
    pub fn query_intersecting(&self, bounds: &[f64; 4]) -> Vec<Arc<CogTile>> {
        let envelope = AABB::from_corners([bounds[0], bounds[1]], [bounds[2], bounds[3]]);

        // Create query rectangle as polygon for exact intersection test
        let query_rect = Rect::new(
            Coord {
                x: bounds[0],
                y: bounds[1],
            },
            Coord {
                x: bounds[2],
                y: bounds[3],
            },
        );
        let query_poly = query_rect.to_polygon();

        // R-tree candidate query + polygon refinement
        self.rtree
            .locate_in_envelope_intersecting(&envelope)
            .filter(|arc_tile| arc_tile.footprint_wgs84.intersects(&query_poly))
            .map(|arc_tile| Arc::clone(&arc_tile.0))
            .collect()
    }

    /// Get all tiles.
    pub fn all_tiles(&self) -> &[Arc<CogTile>] {
        &self.tiles
    }

    /// Get the number of tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Get the overall bounding box in WGS84.
    pub fn bounds_wgs84(&self) -> Option<[f64; 4]> {
        if self.tiles.is_empty() {
            return None;
        }

        let mut min_lon = f64::MAX;
        let mut min_lat = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut max_lat = f64::MIN;

        for tile in &self.tiles {
            min_lon = min_lon.min(tile.bounds_wgs84[0]);
            min_lat = min_lat.min(tile.bounds_wgs84[1]);
            max_lon = max_lon.max(tile.bounds_wgs84[2]);
            max_lat = max_lat.max(tile.bounds_wgs84[3]);
        }

        Some([min_lon, min_lat, max_lon, max_lat])
    }

    /// Filter tiles by bounding box and/or years.
    ///
    /// Returns a new InputIndex containing only tiles that:
    /// - Intersect the given bounds (if provided)
    /// - Match one of the given years (if provided)
    ///
    /// If both bounds and years are None, returns a clone of self.
    /// Note: This clones Arc references (cheap), not the underlying CogTile data.
    pub fn filter(&self, bounds: Option<&[f64; 4]>, years: Option<&[i32]>) -> Self {
        let filtered: Vec<Arc<CogTile>> = self.tiles.iter()
            .filter(|tile| {
                // Filter by bounds if specified
                if let Some(b) = bounds {
                    // Check if tile intersects the filter bounds
                    let intersects = tile.bounds_wgs84[0] < b[2]  // tile min_lon < filter max_lon
                        && tile.bounds_wgs84[2] > b[0]            // tile max_lon > filter min_lon
                        && tile.bounds_wgs84[1] < b[3]            // tile min_lat < filter max_lat
                        && tile.bounds_wgs84[3] > b[1];           // tile max_lat > filter min_lat
                    if !intersects {
                        return false;
                    }
                }

                // Filter by year if specified
                if let Some(y) = years {
                    if !y.is_empty() && !y.contains(&tile.year) {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect();

        // Build RTree using ArcTile wrapper (cloning Arc is cheap)
        let rtree_tiles: Vec<ArcTile> = filtered.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);

        tracing::info!(
            "Filtered {} -> {} tiles (bounds: {:?}, years: {:?})",
            self.tiles.len(),
            filtered.len(),
            bounds,
            years
        );

        Self { tiles: filtered, rtree }
    }

    /// Get unique years present in the index.
    pub fn unique_years(&self) -> Vec<i32> {
        let mut years: Vec<i32> = self.tiles.iter().map(|t| t.year).collect();
        years.sort();
        years.dedup();
        years
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tile(id: &str, bounds: [f64; 4], year: i32) -> CogTile {
        CogTile {
            tile_id: id.to_string(),
            s3_path: format!("s3://bucket/{}.tif", id),
            crs: "EPSG:32610".to_string(),
            bounds_native: [0.0, 0.0, 10000.0, 10000.0],
            bounds_wgs84: bounds,
            footprint_wgs84: CogTile::footprint_from_wgs84_bounds(&bounds),
            resolution: 10.0,
            year,
        }
    }

    fn create_test_index() -> InputIndex {
        let tiles: Vec<Arc<CogTile>> = vec![
            Arc::new(create_test_tile("t1", [-122.5, 37.5, -122.0, 38.0], 2023)),
            Arc::new(create_test_tile("t2", [-122.0, 37.5, -121.5, 38.0], 2023)),
            Arc::new(create_test_tile("t3", [-123.0, 38.0, -122.5, 38.5], 2024)),
            Arc::new(create_test_tile("t4", [-121.5, 37.0, -121.0, 37.5], 2024)),
        ];
        let rtree_tiles: Vec<ArcTile> = tiles.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);
        InputIndex { tiles, rtree }
    }

    #[test]
    fn test_rtree_query() {
        let tiles: Vec<Arc<CogTile>> = vec![
            Arc::new(create_test_tile("t1", [-122.5, 37.5, -122.0, 38.0], 2024)),
            Arc::new(create_test_tile("t2", [-122.0, 37.5, -121.5, 38.0], 2024)),
            Arc::new(create_test_tile("t3", [-123.0, 38.0, -122.5, 38.5], 2024)),
        ];

        let rtree_tiles: Vec<ArcTile> = tiles.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);

        // Query overlapping t1 and t2
        let envelope = AABB::from_corners([-122.3, 37.6], [-121.8, 37.9]);
        let results: Vec<_> = rtree.locate_in_envelope_intersecting(&envelope).collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bounds_wgs84() {
        let index = create_test_index();
        let bounds = index.bounds_wgs84().unwrap();

        // Should encompass all tiles
        assert_eq!(bounds[0], -123.0); // min_lon
        assert_eq!(bounds[1], 37.0);   // min_lat
        assert_eq!(bounds[2], -121.0); // max_lon
        assert_eq!(bounds[3], 38.5);   // max_lat
    }

    #[test]
    fn test_bounds_wgs84_empty() {
        let index = InputIndex {
            tiles: Vec::<Arc<CogTile>>::new(),
            rtree: RTree::<ArcTile>::new(),
        };
        assert!(index.bounds_wgs84().is_none());
    }

    #[test]
    fn test_filter_by_bounds() {
        let index = create_test_index();

        // Filter to a narrow band that only hits t1 and t2
        // t1: [-122.5, 37.5, -122.0, 38.0]
        // t2: [-122.0, 37.5, -121.5, 38.0]
        // t3: [-123.0, 38.0, -122.5, 38.5] - excluded (north of 38.0)
        // t4: [-121.5, 37.0, -121.0, 37.5] - excluded (south of 37.5)
        let bounds = [-122.6, 37.6, -121.4, 37.9];
        let filtered = index.filter(Some(&bounds), None);

        assert_eq!(filtered.len(), 2);
        let ids: Vec<_> = filtered.all_tiles().iter().map(|t| &t.tile_id).collect();
        assert!(ids.contains(&&"t1".to_string()));
        assert!(ids.contains(&&"t2".to_string()));
    }

    #[test]
    fn test_filter_by_year() {
        let index = create_test_index();

        // Filter to 2024 only - should match t3, t4
        let filtered = index.filter(None, Some(&[2024]));

        assert_eq!(filtered.len(), 2);
        for tile in filtered.all_tiles() {
            assert_eq!(tile.year, 2024);
        }
    }

    #[test]
    fn test_filter_by_bounds_and_year() {
        let index = create_test_index();

        // Filter to 2023 and SF area - should match t1, t2
        let bounds = [-123.0, 37.0, -121.0, 39.0];
        let filtered = index.filter(Some(&bounds), Some(&[2023]));

        assert_eq!(filtered.len(), 2);
        for tile in filtered.all_tiles() {
            assert_eq!(tile.year, 2023);
        }
    }

    #[test]
    fn test_filter_no_match() {
        let index = create_test_index();

        // Filter to year that doesn't exist
        let filtered = index.filter(None, Some(&[2020]));
        assert!(filtered.is_empty());

        // Filter to bounds that don't intersect
        let bounds = [0.0, 0.0, 1.0, 1.0];
        let filtered = index.filter(Some(&bounds), None);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_empty_years_matches_all() {
        let index = create_test_index();

        // Empty years slice should match all
        let filtered = index.filter(None, Some(&[]));
        assert_eq!(filtered.len(), index.len());
    }

    #[test]
    fn test_unique_years() {
        let index = create_test_index();
        let years = index.unique_years();

        assert_eq!(years, vec![2023, 2024]);
    }

    #[test]
    fn test_query_intersecting() {
        let index = create_test_index();

        // Query for tiles intersecting a small area
        let results = index.query_intersecting(&[-122.3, 37.6, -121.8, 37.9]);
        assert_eq!(results.len(), 2);

        // Query for tiles in a non-overlapping area
        let results = index.query_intersecting(&[0.0, 0.0, 1.0, 1.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);
        assert!(!index.is_empty());

        let empty = InputIndex {
            tiles: Vec::<Arc<CogTile>>::new(),
            rtree: RTree::<ArcTile>::new(),
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}
