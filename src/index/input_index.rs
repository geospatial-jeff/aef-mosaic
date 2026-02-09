//! Load AEF parquet index and build R-tree for spatial queries.

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rstar::{RTree, RTreeObject, AABB};
use std::sync::Arc;

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

    /// Resolution in source CRS units (meters for UTM)
    pub resolution: f64,

    /// Year of the tile data
    pub year: i32,
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

/// Index of all input COG tiles with spatial query support.
pub struct InputIndex {
    /// All tiles loaded from the parquet index
    tiles: Vec<CogTile>,

    /// R-tree for efficient spatial queries
    rtree: RTree<CogTile>,
}

impl InputIndex {
    /// Load index from a local parquet file.
    pub fn from_local_parquet(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open parquet file: {}", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut tiles = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            Self::extract_tiles_from_batch(&batch, &mut tiles)?;
        }

        let rtree = RTree::bulk_load(tiles.clone());

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

        let mut tiles = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            Self::extract_tiles_from_batch(&batch, &mut tiles)?;
        }

        let rtree = RTree::bulk_load(tiles.clone());

        tracing::info!("Loaded {} tiles from parquet index", tiles.len());

        Ok(Self { tiles, rtree })
    }

    /// Extract tiles from a record batch.
    fn extract_tiles_from_batch(batch: &RecordBatch, tiles: &mut Vec<CogTile>) -> Result<()> {
        let schema = batch.schema();

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
        let resolution_col = Self::find_column(&schema, &["resolution", "res", "pixel_size"]);
        let year_col = Self::find_column(&schema, &["year", "date_year"]);

        // Extract arrays - handle both string and int tile IDs
        let tile_id_arr = batch.column(tile_id_col);
        let tile_ids: Vec<String> = if let Some(str_arr) = tile_id_arr.as_any().downcast_ref::<StringArray>() {
            (0..str_arr.len()).map(|i| str_arr.value(i).to_string()).collect()
        } else if let Some(i64_arr) = tile_id_arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
            (0..i64_arr.len()).map(|i| i64_arr.value(i).to_string()).collect()
        } else if let Some(i32_arr) = tile_id_arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
            (0..i32_arr.len()).map(|i| i32_arr.value(i).to_string()).collect()
        } else {
            anyhow::bail!("tile_id column must be string or integer");
        };

        let s3_paths = batch.column(s3_path_col).as_any().downcast_ref::<StringArray>()
            .context("s3_path/location column must be string")?;
        let crs_values = batch.column(crs_col).as_any().downcast_ref::<StringArray>()
            .context("crs column must be string")?;

        let min_x = Self::get_f64_array(batch.column(min_x_col))?;
        let min_y = Self::get_f64_array(batch.column(min_y_col))?;
        let max_x = Self::get_f64_array(batch.column(max_x_col))?;
        let max_y = Self::get_f64_array(batch.column(max_y_col))?;

        let min_lon = Self::get_f64_array(batch.column(min_lon_col))?;
        let min_lat = Self::get_f64_array(batch.column(min_lat_col))?;
        let max_lon = Self::get_f64_array(batch.column(max_lon_col))?;
        let max_lat = Self::get_f64_array(batch.column(max_lat_col))?;

        // Resolution is optional (default to 10m for AEF)
        let resolution: Vec<f64> = resolution_col.ok()
            .and_then(|col| Self::get_f64_array(batch.column(col)).ok())
            .unwrap_or_else(|| vec![10.0; batch.num_rows()]);

        // Year is optional
        let years: Vec<i32> = year_col.ok().map(|col| {
            let arr = batch.column(col);
            if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
                (0..arr.len()).map(|i| i32_arr.value(i)).collect()
            } else if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow::array::Int64Array>() {
                (0..arr.len()).map(|i| i64_arr.value(i) as i32).collect()
            } else {
                vec![2024; arr.len()]
            }
        }).unwrap_or_else(|| vec![2024; batch.num_rows()]);

        for i in 0..batch.num_rows() {
            let tile = CogTile {
                tile_id: tile_ids[i].clone(),
                s3_path: s3_paths.value(i).to_string(),
                crs: crs_values.value(i).to_string(),
                bounds_native: [min_x[i], min_y[i], max_x[i], max_y[i]],
                bounds_wgs84: [min_lon[i], min_lat[i], max_lon[i], max_lat[i]],
                resolution: resolution[i],
                year: years[i],
            };
            tiles.push(tile);
        }

        Ok(())
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

    /// Get f64 values from an array (handles f32 and f64).
    fn get_f64_array(array: &Arc<dyn Array>) -> Result<Vec<f64>> {
        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            Ok((0..arr.len()).map(|i| arr.value(i)).collect())
        } else if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Float32Array>() {
            Ok((0..arr.len()).map(|i| arr.value(i) as f64).collect())
        } else {
            anyhow::bail!("Expected float array")
        }
    }

    /// Query tiles that intersect the given WGS84 bounding box.
    pub fn query_intersecting(&self, bounds: &[f64; 4]) -> Vec<&CogTile> {
        let envelope = AABB::from_corners(
            [bounds[0], bounds[1]],
            [bounds[2], bounds[3]],
        );
        self.rtree.locate_in_envelope_intersecting(&envelope).collect()
    }

    /// Get all tiles.
    pub fn all_tiles(&self) -> &[CogTile] {
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
    pub fn filter(&self, bounds: Option<&[f64; 4]>, years: Option<&[i32]>) -> Self {
        let filtered: Vec<CogTile> = self.tiles.iter()
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

        let rtree = RTree::bulk_load(filtered.clone());

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
            resolution: 10.0,
            year,
        }
    }

    fn create_test_index() -> InputIndex {
        let tiles = vec![
            create_test_tile("t1", [-122.5, 37.5, -122.0, 38.0], 2023),
            create_test_tile("t2", [-122.0, 37.5, -121.5, 38.0], 2023),
            create_test_tile("t3", [-123.0, 38.0, -122.5, 38.5], 2024),
            create_test_tile("t4", [-121.5, 37.0, -121.0, 37.5], 2024),
        ];
        let rtree = RTree::bulk_load(tiles.clone());
        InputIndex { tiles, rtree }
    }

    #[test]
    fn test_rtree_query() {
        let tiles = vec![
            create_test_tile("t1", [-122.5, 37.5, -122.0, 38.0], 2024),
            create_test_tile("t2", [-122.0, 37.5, -121.5, 38.0], 2024),
            create_test_tile("t3", [-123.0, 38.0, -122.5, 38.5], 2024),
        ];

        let rtree = RTree::bulk_load(tiles.clone());

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
            tiles: vec![],
            rtree: RTree::new(),
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
            tiles: vec![],
            rtree: RTree::new(),
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}
