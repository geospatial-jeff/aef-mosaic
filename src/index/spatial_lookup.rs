//! Spatial lookup: Output tile → input COG mapping.

use super::{CogTile, InputIndex, OutputChunk, OutputGrid};
use crate::crs;
use anyhow::Result;
use std::sync::Arc;

/// Provides spatial lookup from output chunks to input COG tiles.
pub struct SpatialLookup {
    /// The input tile index with R-tree
    input_index: Arc<InputIndex>,

    /// The output grid definition
    output_grid: Arc<OutputGrid>,

    /// CRS transformation cache (for output CRS → WGS84)
    proj_cache: crs::ProjCache,

    /// Whether output CRS is WGS84 (no transformation needed)
    is_wgs84: bool,
}

impl SpatialLookup {
    /// Create a new spatial lookup.
    ///
    /// Returns an error if the projection from output CRS to WGS84 cannot be created.
    pub fn new(input_index: Arc<InputIndex>, output_grid: Arc<OutputGrid>) -> Result<Self> {
        let is_wgs84 = output_grid.crs == crs::codes::WGS84;

        // Verify the projection can be created (fail fast)
        let proj_cache = crs::ProjCache::new();
        if !is_wgs84 {
            proj_cache.get(&output_grid.crs, crs::codes::WGS84)?;
        }

        Ok(Self {
            input_index,
            output_grid,
            proj_cache,
            is_wgs84,
        })
    }

    /// Transform bounds from output CRS to WGS84.
    fn transform_to_wgs84(&self, bounds: &[f64; 4]) -> Result<[f64; 4]> {
        if self.is_wgs84 {
            return Ok(*bounds);
        }

        // Use the centralized transform_bounds function
        crs::transform_bounds(bounds, &self.output_grid.crs, crs::codes::WGS84, &self.proj_cache)
    }

    /// Find all input COG tiles that intersect the given output chunk.
    /// Returns Arc clones (cheap reference count increment).
    ///
    /// Returns an error if coordinate transformation fails.
    pub fn tiles_for_chunk(&self, chunk: &OutputChunk) -> Result<Vec<Arc<CogTile>>> {
        let bounds_crs = self.output_grid.chunk_bounds(chunk);
        let bounds_wgs84 = self.transform_to_wgs84(&bounds_crs)?;
        Ok(self.input_index.query_intersecting(&bounds_wgs84))
    }

    /// Find all input COG tiles that intersect the given WGS84 bounds.
    /// Returns Arc clones (cheap reference count increment).
    pub fn tiles_for_bounds(&self, bounds: &[f64; 4]) -> Vec<Arc<CogTile>> {
        self.input_index.query_intersecting(bounds)
    }

    /// Check if a chunk has any intersecting input tiles.
    ///
    /// Returns false if coordinate transformation fails.
    pub fn chunk_has_data(&self, chunk: &OutputChunk) -> bool {
        self.tiles_for_chunk(chunk)
            .map(|tiles| !tiles.is_empty())
            .unwrap_or(false)
    }

    /// Get statistics about tile coverage.
    pub fn coverage_stats(&self) -> CoverageStats {
        let total_chunks = self.output_grid.num_spatial_chunks();
        let mut chunks_with_data = 0;
        let mut max_tiles_per_chunk = 0;
        let mut total_tile_refs = 0;

        for chunk in self.output_grid.enumerate_chunks() {
            if let Ok(tiles) = self.tiles_for_chunk(&chunk) {
                if !tiles.is_empty() {
                    chunks_with_data += 1;
                    max_tiles_per_chunk = max_tiles_per_chunk.max(tiles.len());
                    total_tile_refs += tiles.len();
                }
            }
        }

        let avg_tiles_per_chunk = if chunks_with_data > 0 {
            total_tile_refs as f64 / chunks_with_data as f64
        } else {
            0.0
        };

        CoverageStats {
            total_chunks,
            chunks_with_data,
            empty_chunks: total_chunks - chunks_with_data,
            max_tiles_per_chunk,
            avg_tiles_per_chunk,
            total_input_tiles: self.input_index.len(),
        }
    }

    /// Get the output grid.
    pub fn output_grid(&self) -> &OutputGrid {
        &self.output_grid
    }

    /// Get the input index.
    pub fn input_index(&self) -> &InputIndex {
        &self.input_index
    }
}

/// Statistics about tile coverage.
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Total number of output chunks
    pub total_chunks: usize,

    /// Number of chunks that have input data
    pub chunks_with_data: usize,

    /// Number of empty chunks (no input data)
    pub empty_chunks: usize,

    /// Maximum number of input tiles overlapping a single chunk
    pub max_tiles_per_chunk: usize,

    /// Average number of input tiles per non-empty chunk
    pub avg_tiles_per_chunk: f64,

    /// Total number of input tiles in the index
    pub total_input_tiles: usize,
}

impl std::fmt::Display for CoverageStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coverage: {}/{} chunks ({:.1}%), max overlap: {}, avg overlap: {:.1}, input tiles: {}",
            self.chunks_with_data,
            self.total_chunks,
            self.chunks_with_data as f64 / self.total_chunks as f64 * 100.0,
            self.max_tiles_per_chunk,
            self.avg_tiles_per_chunk,
            self.total_input_tiles
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::ArcTile;
    use rstar::RTree;

    fn create_test_tile(id: &str, bounds: [f64; 4]) -> CogTile {
        CogTile {
            tile_id: id.to_string(),
            s3_path: format!("s3://bucket/{}.tif", id),
            crs: "EPSG:32610".to_string(),
            bounds_native: [0.0, 0.0, 10000.0, 10000.0],
            bounds_wgs84: bounds,
            footprint_wgs84: CogTile::footprint_from_wgs84_bounds(&bounds),
            resolution: 10.0,
            year: 2024,
        }
    }

    #[test]
    fn test_spatial_lookup() {
        // Create test tiles wrapped in Arc
        let tiles: Vec<Arc<CogTile>> = vec![
            Arc::new(create_test_tile("t1", [-1.0, -1.0, 0.0, 0.0])),
            Arc::new(create_test_tile("t2", [0.0, 0.0, 1.0, 1.0])),
            Arc::new(create_test_tile("t3", [1.0, 1.0, 2.0, 2.0])),
        ];

        // Build index manually for testing using ArcTile wrapper
        let rtree_tiles: Vec<ArcTile> = tiles.iter().map(|t| ArcTile(Arc::clone(t))).collect();
        let rtree = RTree::bulk_load(rtree_tiles);

        // Query overlapping t2
        let bounds = [0.5, 0.5, 0.6, 0.6];
        let envelope = rstar::AABB::from_corners([bounds[0], bounds[1]], [bounds[2], bounds[3]]);
        let results: Vec<_> = rtree.locate_in_envelope_intersecting(&envelope).collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tile_id, "t2");
    }

    #[test]
    fn test_coverage_stats_display() {
        let stats = CoverageStats {
            total_chunks: 100,
            chunks_with_data: 75,
            empty_chunks: 25,
            max_tiles_per_chunk: 5,
            avg_tiles_per_chunk: 2.5,
            total_input_tiles: 50,
        };

        let display = format!("{}", stats);

        assert!(display.contains("75/100"));
        assert!(display.contains("75.0%"));
        assert!(display.contains("max overlap: 5"));
        assert!(display.contains("avg overlap: 2.5"));
        assert!(display.contains("input tiles: 50"));
    }

    #[test]
    fn test_coverage_stats_zero_chunks() {
        let stats = CoverageStats {
            total_chunks: 0,
            chunks_with_data: 0,
            empty_chunks: 0,
            max_tiles_per_chunk: 0,
            avg_tiles_per_chunk: 0.0,
            total_input_tiles: 0,
        };

        // Should not panic on division by zero
        let display = format!("{}", stats);
        assert!(display.contains("0/0"));
    }

    #[test]
    fn test_coverage_stats_full_coverage() {
        let stats = CoverageStats {
            total_chunks: 50,
            chunks_with_data: 50,
            empty_chunks: 0,
            max_tiles_per_chunk: 3,
            avg_tiles_per_chunk: 1.5,
            total_input_tiles: 75,
        };

        let display = format!("{}", stats);
        assert!(display.contains("100.0%"));
    }
}
