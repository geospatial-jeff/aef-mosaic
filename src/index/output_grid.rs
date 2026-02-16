//! Define the output Zarr grid and enumerate chunks.

use crate::config::ChunkShape;
use crate::crs;
use anyhow::Result;

/// A single output chunk in the Zarr array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputChunk {
    /// Time index (typically 0 for single year)
    pub time_idx: usize,

    /// Row index in the spatial grid
    pub row_idx: usize,

    /// Column index in the spatial grid
    pub col_idx: usize,
}

impl OutputChunk {
    /// Get the chunk indices as a 4D array index [time, band, row, col].
    pub fn chunk_indices(&self) -> [usize; 4] {
        // Bands are always chunk 0 since we store all 64 bands in one chunk
        [self.time_idx, 0, self.row_idx, self.col_idx]
    }
}

/// Output grid definition for the Zarr array.
#[derive(Debug, Clone)]
pub struct OutputGrid {
    /// Bounding box in output CRS [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],

    /// Output CRS (e.g., "EPSG:4326")
    pub crs: String,

    /// Resolution in output CRS units
    pub resolution: f64,

    /// Number of years (time dimension)
    pub num_years: usize,

    /// Starting year
    pub start_year: i32,

    /// Number of embedding dimensions
    pub num_bands: usize,

    /// Total height in pixels
    pub height: usize,

    /// Total width in pixels
    pub width: usize,

    /// Chunk shape
    pub chunk_shape: ChunkShape,

    /// Number of chunks in each dimension
    pub chunk_counts: [usize; 4], // [time, band, row, col]
}

impl OutputGrid {
    /// Create a new output grid from WGS84 bounds and configuration.
    ///
    /// The bounds_wgs84 are transformed to the output CRS before calculating dimensions.
    pub fn new(
        bounds_wgs84: [f64; 4],
        crs: String,
        resolution: f64,
        num_years: usize,
        start_year: i32,
        num_bands: usize,
        chunk_shape: ChunkShape,
    ) -> Result<Self> {
        // Transform WGS84 bounds to output CRS
        let bounds = Self::transform_bounds_to_crs(&bounds_wgs84, &crs)?;

        tracing::info!(
            "Transformed bounds: WGS84 [{:.2}, {:.2}, {:.2}, {:.2}] -> {} [{:.0}, {:.0}, {:.0}, {:.0}]",
            bounds_wgs84[0], bounds_wgs84[1], bounds_wgs84[2], bounds_wgs84[3],
            crs,
            bounds[0], bounds[1], bounds[2], bounds[3]
        );

        // Calculate total dimensions in output CRS units, rounded UP to chunk boundaries.
        // This ensures all chunks are always full-sized (no partial edge chunks).
        let raw_width = ((bounds[2] - bounds[0]) / resolution).ceil() as usize;
        let raw_height = ((bounds[3] - bounds[1]) / resolution).ceil() as usize;

        // Round up to nearest chunk multiple
        let col_chunks = raw_width.div_ceil(chunk_shape.width);
        let row_chunks = raw_height.div_ceil(chunk_shape.height);
        let width = col_chunks * chunk_shape.width;
        let height = row_chunks * chunk_shape.height;

        // Expand bounds to match the chunk-aligned dimensions
        let bounds = [
            bounds[0],
            bounds[1],
            bounds[0] + width as f64 * resolution,
            bounds[1] + height as f64 * resolution,
        ];

        // Calculate chunk counts for time and bands
        let time_chunks = num_years.div_ceil(chunk_shape.time);
        let band_chunks = num_bands.div_ceil(chunk_shape.embedding);

        tracing::info!(
            "Output grid: {}x{} pixels (raw {}x{}, chunk-aligned), {} chunks ({}x{}x{}x{})",
            width,
            height,
            raw_width,
            raw_height,
            time_chunks * band_chunks * row_chunks * col_chunks,
            time_chunks,
            band_chunks,
            row_chunks,
            col_chunks
        );

        Ok(Self {
            bounds,
            crs,
            resolution,
            num_years,
            start_year,
            num_bands,
            height,
            width,
            chunk_shape,
            chunk_counts: [time_chunks, band_chunks, row_chunks, col_chunks],
        })
    }

    /// Transform WGS84 bounds to the output CRS.
    ///
    /// Uses edge densification to handle non-linear projections accurately.
    fn transform_bounds_to_crs(bounds_wgs84: &[f64; 4], target_crs: &str) -> Result<[f64; 4]> {
        let cache = crs::ProjCache::new();
        crs::transform_bounds_with_densification(
            bounds_wgs84,
            crs::codes::WGS84,
            target_crs,
            &cache,
            10, // Sample 10 points along each edge
        )
    }

    /// Get the total number of spatial chunks.
    pub fn num_spatial_chunks(&self) -> usize {
        self.chunk_counts[2] * self.chunk_counts[3]
    }

    /// Get the total number of chunks (all dimensions).
    pub fn num_chunks(&self) -> usize {
        self.chunk_counts.iter().product()
    }

    /// Get the Zarr array shape.
    pub fn array_shape(&self) -> [usize; 4] {
        [self.num_years, self.num_bands, self.height, self.width]
    }

    /// Enumerate all output chunks.
    pub fn enumerate_chunks(&self) -> impl Iterator<Item = OutputChunk> + '_ {
        (0..self.chunk_counts[0]).flat_map(move |time_idx| {
            (0..self.chunk_counts[2]).flat_map(move |row_idx| {
                (0..self.chunk_counts[3]).map(move |col_idx| OutputChunk {
                    time_idx,
                    row_idx,
                    col_idx,
                })
            })
        })
    }

    /// Get the bounds of a specific chunk in output CRS.
    ///
    /// Uses top-down convention: chunk row 0 is at the top (max_y).
    pub fn chunk_bounds(&self, chunk: &OutputChunk) -> [f64; 4] {
        let chunk_width_crs = self.chunk_shape.width as f64 * self.resolution;
        let chunk_height_crs = self.chunk_shape.height as f64 * self.resolution;

        let min_x = self.bounds[0] + chunk.col_idx as f64 * chunk_width_crs;
        let max_x = (min_x + chunk_width_crs).min(self.bounds[2]);

        // Top-down: row 0 is at max_y, row N is at min_y
        let max_y = self.bounds[3] - chunk.row_idx as f64 * chunk_height_crs;
        let min_y = (max_y - chunk_height_crs).max(self.bounds[1]);

        [min_x, min_y, max_x, max_y]
    }

    /// Get the bounds of a specific chunk in WGS84 coordinates.
    pub fn chunk_bounds_wgs84(&self, chunk: &OutputChunk) -> Result<[f64; 4]> {
        let bounds_crs = self.chunk_bounds(chunk);
        Self::transform_bounds_to_wgs84(&bounds_crs, &self.crs)
    }

    /// Transform bounds from output CRS to WGS84.
    fn transform_bounds_to_wgs84(bounds_crs: &[f64; 4], source_crs: &str) -> Result<[f64; 4]> {
        let cache = crs::ProjCache::new();
        crs::transform_bounds(bounds_crs, source_crs, crs::codes::WGS84, &cache)
    }

    /// Get the pixel bounds of a chunk [start_row, start_col, end_row, end_col].
    pub fn chunk_pixel_bounds(&self, chunk: &OutputChunk) -> [usize; 4] {
        let start_row = chunk.row_idx * self.chunk_shape.height;
        let start_col = chunk.col_idx * self.chunk_shape.width;
        let end_row = (start_row + self.chunk_shape.height).min(self.height);
        let end_col = (start_col + self.chunk_shape.width).min(self.width);

        [start_row, start_col, end_row, end_col]
    }

    /// Get the year for a time index.
    pub fn year_for_time_idx(&self, time_idx: usize) -> i32 {
        self.start_year + time_idx as i32
    }

    /// Convert output CRS coordinates to pixel coordinates.
    ///
    /// Uses top-down convention: row 0 is at max_y (top), increasing southward.
    /// Takes (x, y) coordinates in the output CRS (e.g., meters for EPSG:6933)
    /// and returns (row, col) pixel indices in the output grid.
    pub fn crs_to_pixel(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x - self.bounds[0]) / self.resolution) as usize;
        // Top-down: row 0 is at max_y (top), row increases going south
        let row = ((self.bounds[3] - y) / self.resolution) as usize;
        (row.min(self.height - 1), col.min(self.width - 1))
    }

    /// Convert pixel coordinates to output CRS coordinates (center of pixel).
    ///
    /// Uses top-down convention: row 0 is at max_y (top), increasing southward.
    /// Takes (row, col) pixel indices and returns (x, y) coordinates
    /// in the output CRS.
    pub fn pixel_to_crs(&self, row: usize, col: usize) -> (f64, f64) {
        let x = self.bounds[0] + (col as f64 + 0.5) * self.resolution;
        // Top-down: row 0 is at max_y (top), row increases going south
        let y = self.bounds[3] - (row as f64 + 0.5) * self.resolution;
        (x, y)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> OutputGrid {
        OutputGrid::new(
            [-124.0, 32.0, -114.0, 42.0], // ~California
            "EPSG:4326".to_string(),
            0.0001, // ~10m resolution
            1,
            2024,
            64,
            ChunkShape::default(),
        ).unwrap()
    }

    #[test]
    fn test_grid_dimensions() {
        let grid = create_test_grid();

        // 10 degrees at 0.0001 resolution = 100,000 raw pixels
        // Rounded up to chunk boundary: ceil(100,000/1024) = 98 chunks
        // 98 * 1024 = 100,352 pixels (chunk-aligned)
        assert_eq!(grid.chunk_counts[2], 98);
        assert_eq!(grid.chunk_counts[3], 98);
        assert_eq!(grid.width, 98 * 1024);  // 100,352
        assert_eq!(grid.height, 98 * 1024); // 100,352
    }

    #[test]
    fn test_chunk_enumeration() {
        let grid = OutputGrid::new(
            [0.0, 0.0, 1.0, 1.0],
            "EPSG:4326".to_string(),
            0.5, // 2x2 pixels
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 1,
                width: 1,
            },
        ).unwrap();

        let chunks: Vec<_> = grid.enumerate_chunks().collect();
        assert_eq!(chunks.len(), 4); // 2x2 chunks
    }

    #[test]
    fn test_chunk_bounds() {
        // Grid: [0, 0, 2, 2] with 1x1 chunks = 2x2 grid of chunks
        let grid = OutputGrid::new(
            [0.0, 0.0, 2.0, 2.0],
            "EPSG:4326".to_string(),
            1.0,
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 1,
                width: 1,
            },
        ).unwrap();

        // Top-down convention: row 0 is at max_y (top), row 1 is at min_y (bottom)
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 0,  // Top row
            col_idx: 0,
        };
        let bounds = grid.chunk_bounds(&chunk);
        // x: [0, 1], y: [1, 2] (top row, max_y=2 down to min_y=1)
        assert_eq!(bounds, [0.0, 1.0, 1.0, 2.0]);

        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 1,  // Bottom row
            col_idx: 1,
        };
        let bounds = grid.chunk_bounds(&chunk);
        // x: [1, 2], y: [0, 1] (bottom row, max_y=1 down to min_y=0)
        assert_eq!(bounds, [1.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn test_crs_to_pixel_and_back() {
        let grid = OutputGrid::new(
            [0.0, 0.0, 10.0, 10.0],
            crs::codes::WGS84.to_string(),
            0.01, // 1000x1000 grid
            1,
            2024,
            64,
            ChunkShape::default(),
        ).unwrap();

        // Test a point in the middle
        let (x, y) = (5.0, 5.0);
        let (row, col) = grid.crs_to_pixel(x, y);

        // Should be roughly in the middle
        assert!(row > 400 && row < 600, "row {} not in expected range", row);
        assert!(col > 400 && col < 600, "col {} not in expected range", col);

        // Convert back (pixel center)
        let (x2, y2) = grid.pixel_to_crs(row, col);

        // Should be close to original (within one pixel)
        assert!((x2 - x).abs() < grid.resolution, "x round-trip error too large");
        assert!((y2 - y).abs() < grid.resolution, "y round-trip error too large");
    }

    #[test]
    fn test_crs_to_pixel_bounds_clamping() {
        let grid = OutputGrid::new(
            [0.0, 0.0, 10.0, 10.0],
            crs::codes::WGS84.to_string(),
            0.01,
            1,
            2024,
            64,
            ChunkShape::default(),
        ).unwrap();

        // Test coordinates outside bounds are clamped
        // Top-down: row 0 is at max_y (top), row height-1 is at min_y (bottom)

        // Point below and left of grid: y=-1 < min_y, so row should be at bottom (height-1)
        let (row, col) = grid.crs_to_pixel(-1.0, -1.0);
        assert_eq!(row, grid.height - 1, "y below grid should clamp to bottom row");
        assert_eq!(col, 0);

        // Point above and right of grid: y=100 > max_y, so row should be at top (0)
        let (row, col) = grid.crs_to_pixel(100.0, 100.0);
        assert_eq!(row, 0, "y above grid should clamp to top row");
        assert_eq!(col, grid.width - 1);
    }

    /// Regression test for chunk_bounds computing y from wrong direction.
    ///
    /// The bug: chunk_bounds was computing y from the bottom (min_y) instead of
    /// the top (max_y), causing chunks to be mapped to wrong geographic locations.
    /// This led to R-tree queries finding wrong tiles and reading from incorrect
    /// locations in the source COGs.
    #[test]
    fn test_chunk_bounds_top_down_convention() {
        // Use WGS84 bounds with WGS84 CRS to avoid transformation issues
        // Grid covers 1.0 x 1.0 degrees at 0.001 degree resolution = 1000x1000 pixels
        let grid = OutputGrid::new(
            [0.0, 0.0, 1.0, 1.0],  // [min_lon, min_lat, max_lon, max_lat]
            crs::codes::WGS84.to_string(),
            0.001,  // ~111m resolution at equator
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 1000,
                width: 1000,
            },
        ).unwrap();

        // Should be exactly 1 chunk since grid is 1000x1000 pixels
        assert_eq!(grid.chunk_counts[2], 1, "Expected 1 row chunk");
        assert_eq!(grid.chunk_counts[3], 1, "Expected 1 col chunk");

        // Chunk (0, 0) should cover the ENTIRE grid
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 0,
            col_idx: 0,
        };
        let bounds = grid.chunk_bounds(&chunk);

        // Key assertion: chunk should span the full grid
        assert!((bounds[0] - 0.0).abs() < 0.01, "min_x mismatch: {}", bounds[0]);
        assert!((bounds[2] - 1.0).abs() < 0.01, "max_x mismatch: {}", bounds[2]);
        assert!((bounds[3] - 1.0).abs() < 0.01, "max_y mismatch: {}", bounds[3]);
        assert!((bounds[1] - 0.0).abs() < 0.01, "min_y mismatch: {}", bounds[1]);
    }

    /// Test that chunk row 0 is at the TOP of the grid, not the bottom.
    /// This is the core invariant that was violated by the original bug.
    #[test]
    fn test_chunk_row_zero_at_top() {
        // Grid with 2 row chunks using WGS84
        let grid = OutputGrid::new(
            [0.0, 0.0, 1.0, 2.0],  // 2 degrees tall, 1 degree wide
            crs::codes::WGS84.to_string(),
            0.001,  // 1000 pixels per degree
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 1000,  // 1 degree per chunk
                width: 1000,
            },
        ).unwrap();

        assert_eq!(grid.chunk_counts[2], 2, "Expected 2 row chunks");

        // Row 0 should be at the TOP (max_y)
        let top_chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
        let top_bounds = grid.chunk_bounds(&top_chunk);

        // Row 1 should be at the BOTTOM (min_y)
        let bottom_chunk = OutputChunk { time_idx: 0, row_idx: 1, col_idx: 0 };
        let bottom_bounds = grid.chunk_bounds(&bottom_chunk);

        // Top chunk should have higher y values than bottom chunk
        assert!(
            top_bounds[3] > bottom_bounds[3],
            "Top chunk max_y ({}) should be > bottom chunk max_y ({})",
            top_bounds[3], bottom_bounds[3]
        );
        assert!(
            top_bounds[1] > bottom_bounds[1],
            "Top chunk min_y ({}) should be > bottom chunk min_y ({})",
            top_bounds[1], bottom_bounds[1]
        );

        // Verify specific values (with tolerance for floating point)
        // Top chunk (row 0): y from 1.0 to 2.0
        assert!((top_bounds[1] - 1.0).abs() < 0.01, "Top chunk min_y should be ~1.0, got {}", top_bounds[1]);
        assert!((top_bounds[3] - 2.0).abs() < 0.01, "Top chunk max_y should be ~2.0, got {}", top_bounds[3]);

        // Bottom chunk (row 1): y from 0.0 to 1.0
        assert!((bottom_bounds[1] - 0.0).abs() < 0.01, "Bottom chunk min_y should be ~0.0, got {}", bottom_bounds[1]);
        assert!((bottom_bounds[3] - 1.0).abs() < 0.01, "Bottom chunk max_y should be ~1.0, got {}", bottom_bounds[3]);
    }

    /// Test that pixel coordinates are consistent with chunk bounds.
    /// A pixel in chunk row 0 should have world coordinates within the top chunk's bounds.
    #[test]
    fn test_pixel_world_coords_match_chunk_bounds() {
        let grid = OutputGrid::new(
            [0.0, 0.0, 1.0, 1.0],  // 1x1 degree grid
            crs::codes::WGS84.to_string(),
            0.001,  // 1000x1000 pixels
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 500,  // 2 row chunks
                width: 500,   // 2 col chunks
            },
        ).unwrap();

        // Pixel (0, 0) is in chunk (0, 0) - should be at top-left
        let chunk_0_0 = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
        let bounds_0_0 = grid.chunk_bounds(&chunk_0_0);

        // Get world coordinate for pixel (0, 0)
        let (world_x, world_y) = grid.pixel_to_crs(0, 0);

        // Pixel (0, 0) should be in the top-left, so:
        // - x should be near min_x of chunk (0, 0)
        // - y should be near max_y of chunk (0, 0) (top-down convention)
        assert!(
            world_x >= bounds_0_0[0] - 0.01 && world_x <= bounds_0_0[2] + 0.01,
            "Pixel (0,0) x={} not in chunk bounds [{}, {}]",
            world_x, bounds_0_0[0], bounds_0_0[2]
        );
        assert!(
            world_y >= bounds_0_0[1] - 0.01 && world_y <= bounds_0_0[3] + 0.01,
            "Pixel (0,0) y={} not in chunk bounds [{}, {}]",
            world_y, bounds_0_0[1], bounds_0_0[3]
        );

        // Specifically, pixel (0,0) should be near the TOP of the grid (max_y = 1.0)
        assert!(
            (world_y - 1.0).abs() < 0.01,
            "Pixel (0,0) y={} should be near grid top 1.0",
            world_y
        );
    }

    /// Test the exact scenario that caused the original bug:
    /// Zarr pixel (838, 886) should map to world (552853, 4181018),
    /// which should be in chunk (0, 0) with correct bounds.
    #[test]
    fn test_original_bug_scenario() {
        // Exact grid configuration from the bug
        let grid = OutputGrid::new(
            [-122.5, 37.7, -122.3, 37.85],  // WGS84 filter bounds
            "EPSG:32610".to_string(),       // UTM Zone 10N
            10.0,
            1,
            2024,
            64,
            ChunkShape {
                time: 1,
                embedding: 64,
                height: 1024,
                width: 1024,
            },
        ).unwrap();

        // The problematic pixel
        let test_row = 838;
        let test_col = 886;

        // Get world coordinates for this pixel
        let (world_x, world_y) = grid.pixel_to_crs(test_row, test_col);

        // This pixel is in chunk (0, 0)
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: test_row / 1024,  // 0
            col_idx: test_col / 1024,  // 0
        };
        assert_eq!(chunk.row_idx, 0);
        assert_eq!(chunk.col_idx, 0);

        let bounds = grid.chunk_bounds(&chunk);

        // The world coordinate should be within chunk bounds
        assert!(
            world_x >= bounds[0] && world_x <= bounds[2],
            "Pixel ({},{}) world_x={} not in chunk x bounds [{}, {}]",
            test_row, test_col, world_x, bounds[0], bounds[2]
        );
        assert!(
            world_y >= bounds[1] && world_y <= bounds[3],
            "Pixel ({},{}) world_y={} not in chunk y bounds [{}, {}]",
            test_row, test_col, world_y, bounds[1], bounds[3]
        );

        // Critical: chunk (0, 0) bounds should include world_y ~ 4181018
        // The bug had chunk (0, 0) at y ~ [4172647, 4182887] instead of [4179163, 4189403]
        // The correct chunk should have max_y near 4189403 (grid top), not 4182887
        assert!(
            bounds[3] > 4185000.0,
            "Chunk (0,0) max_y={} is too low - suggests y computed from bottom not top",
            bounds[3]
        );
    }
}
