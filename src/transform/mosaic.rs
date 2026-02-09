//! Mosaicing overlapping tiles using mean aggregation.

use crate::io::WindowData;
use crate::transform::{ReprojectConfig, Reprojector};
use anyhow::Result;
use ndarray::{Array3, Array4};

/// Accumulator for mosaicing multiple tiles using mean aggregation.
///
/// Uses i16 accumulators which can handle up to ~250 overlapping tiles without overflow.
/// (i8 range is -128 to 127; worst case 127 * 250 = 31,750 < i16 max of 32,767)
/// This halves memory usage compared to i32 accumulators.
#[derive(Debug)]
pub struct MosaicAccumulator {
    /// Sum of values for each pixel: (1, bands, height, width)
    sum: Array4<i16>,

    /// Count of contributions for each pixel: (1, height, width)
    count: Array3<u16>,

    /// Number of bands
    bands: usize,

    /// Height in pixels
    height: usize,

    /// Width in pixels
    width: usize,
}

impl MosaicAccumulator {
    /// Create a new accumulator for the given dimensions.
    ///
    /// Shape: (1, bands, height, width) for the output
    pub fn new(bands: usize, height: usize, width: usize) -> Self {
        Self {
            sum: Array4::zeros((1, bands, height, width)),
            count: Array3::zeros((1, height, width)),
            bands,
            height,
            width,
        }
    }

    /// Add a tile's reprojected data to the accumulator.
    ///
    /// The input data should be in shape (bands, height, width) and already
    /// reprojected to the output grid. Values of 0 are treated as nodata.
    ///
    /// Optimized: Uses early-exit check then single-pass accumulation.
    /// For pixels with data: scans bands once to find first non-zero, then accumulates all.
    /// For pixels without data: scans bands once and exits early.
    pub fn add(&mut self, data: &Array3<i8>) {
        let (bands, height, width) = data.dim();
        assert_eq!(bands, self.bands, "Band count mismatch");
        assert_eq!(height, self.height, "Height mismatch");
        assert_eq!(width, self.width, "Width mismatch");

        for row in 0..height {
            for col in 0..width {
                // Quick check: sample first band to skip obviously empty pixels
                // Most empty pixels have all zeros, so checking band 0 is a good heuristic
                let first_val = data[[0, row, col]];
                if first_val == 0 {
                    // Check remaining bands
                    let has_data = (1..bands).any(|b| data[[b, row, col]] != 0);
                    if !has_data {
                        continue; // Skip this pixel entirely
                    }
                }

                // This pixel has data - accumulate all bands
                for band in 0..bands {
                    self.sum[[0, band, row, col]] += data[[band, row, col]] as i16;
                }
                self.count[[0, row, col]] += 1;
            }
        }
    }

    /// Finalize the mosaic by computing the mean.
    ///
    /// Returns an int8 array: (1, bands, height, width)
    ///
    /// Optimized: Uses integer division with rounding instead of f64.
    pub fn finalize(self) -> Array4<i8> {
        let mut result = Array4::zeros((1, self.bands, self.height, self.width));

        for row in 0..self.height {
            for col in 0..self.width {
                let count = self.count[[0, row, col]] as i16;
                if count > 0 {
                    // Pre-compute half count for rounding
                    let half_count = count / 2;
                    for band in 0..self.bands {
                        let sum = self.sum[[0, band, row, col]];
                        // Rounded integer division: (sum + count/2) / count
                        // Handles negative sums correctly
                        let mean = if sum >= 0 {
                            ((sum + half_count) / count) as i8
                        } else {
                            ((sum - half_count) / count) as i8
                        };
                        result[[0, band, row, col]] = mean;
                    }
                }
            }
        }

        result
    }

    /// Get the number of tiles that contributed to each pixel.
    pub fn coverage(&self) -> &Array3<u16> {
        &self.count
    }

    /// Get the maximum overlap count.
    pub fn max_overlap(&self) -> u16 {
        *self.count.iter().max().unwrap_or(&0)
    }

    /// Check if any pixels have data.
    pub fn has_data(&self) -> bool {
        self.count.iter().any(|&c| c > 0)
    }
}

/// Mosaic multiple tile windows into a single output chunk.
///
/// This function:
/// 1. Computes the geographic bounds of each window
/// 2. Reprojects each window to the output CRS/grid
/// 3. Accumulates overlapping pixels using mean
/// 4. Returns the final int8 mosaic
pub fn mosaic_tiles(
    windows: &[WindowData],
    reprojector: &Reprojector,
    reproject_config: &ReprojectConfig,
) -> Result<Array4<i8>> {
    if windows.is_empty() {
        // Return zeros for empty chunks
        return Ok(Array4::zeros((
            1,
            reproject_config.num_bands,
            reproject_config.target_shape.0,
            reproject_config.target_shape.1,
        )));
    }

    let bands = windows[0].data.dim().0;
    let (height, width) = reproject_config.target_shape;

    let mut accumulator = MosaicAccumulator::new(bands, height, width);

    for (i, window_data) in windows.iter().enumerate() {
        // Use the pre-computed intersection bounds from the tile selection phase
        let window_bounds = window_data.bounds_native;

        tracing::debug!(
            "Window {} bounds (native CRS {}): [{:.2}, {:.2}, {:.2}, {:.2}], pixel window: ({}, {}, {}x{})",
            i,
            window_data.tile.crs,
            window_bounds[0], window_bounds[1], window_bounds[2], window_bounds[3],
            window_data.window.x, window_data.window.y, window_data.window.width, window_data.window.height
        );

        // Reproject window to output grid
        let reprojected = reprojector.reproject_tile(
            &window_data.data,
            &window_data.tile.crs,
            &window_bounds,
            reproject_config,
        )?;

        let non_zero = reprojected.iter().filter(|&&v| v != 0).count();
        tracing::debug!(
            "Window {} reprojected: shape={:?}, non_zero={}",
            i, reprojected.dim(), non_zero
        );

        // Add to accumulator
        accumulator.add(&reprojected);
    }

    Ok(accumulator.finalize())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_accumulator_single_tile() {
        let mut acc = MosaicAccumulator::new(2, 2, 2);

        let data = Array3::from_shape_vec(
            (2, 2, 2),
            vec![10i8, 20, 30, 40, 50, 60, 70, 80],
        )
        .unwrap();

        acc.add(&data);

        let result = acc.finalize();

        assert_eq!(result[[0, 0, 0, 0]], 10);
        assert_eq!(result[[0, 0, 0, 1]], 20);
        assert_eq!(result[[0, 1, 0, 0]], 50);
    }

    #[test]
    fn test_accumulator_mean() {
        let mut acc = MosaicAccumulator::new(1, 2, 2);

        // First tile: all 10s
        let data1 = Array3::from_elem((1, 2, 2), 10i8);
        acc.add(&data1);

        // Second tile: all 20s
        let data2 = Array3::from_elem((1, 2, 2), 20i8);
        acc.add(&data2);

        let result = acc.finalize();

        // Mean of 10 and 20 is 15
        assert_eq!(result[[0, 0, 0, 0]], 15);
        assert_eq!(result[[0, 0, 0, 1]], 15);
        assert_eq!(result[[0, 0, 1, 0]], 15);
        assert_eq!(result[[0, 0, 1, 1]], 15);
    }

    #[test]
    fn test_accumulator_nodata() {
        let mut acc = MosaicAccumulator::new(2, 2, 2);

        // First tile: some values
        let data1 = Array3::from_shape_vec(
            (2, 2, 2),
            vec![10i8, 0, 0, 40, 50, 0, 0, 80],
        )
        .unwrap();
        acc.add(&data1);

        // Check coverage
        assert_eq!(acc.count[[0, 0, 0]], 1); // (0,0) has data
        assert_eq!(acc.count[[0, 0, 1]], 0); // (0,1) is nodata (all zeros)
        assert_eq!(acc.count[[0, 1, 0]], 0); // (1,0) is nodata
        assert_eq!(acc.count[[0, 1, 1]], 1); // (1,1) has data
    }

    #[test]
    fn test_accumulator_rounding() {
        let mut acc = MosaicAccumulator::new(1, 1, 1);

        // Add three values: 10, 11, 12 → mean = 11
        let data1 = Array3::from_elem((1, 1, 1), 10i8);
        acc.add(&data1);

        let data2 = Array3::from_elem((1, 1, 1), 11i8);
        acc.add(&data2);

        let data3 = Array3::from_elem((1, 1, 1), 12i8);
        acc.add(&data3);

        let result = acc.finalize();
        assert_eq!(result[[0, 0, 0, 0]], 11);
    }

    #[test]
    fn test_max_overlap() {
        let mut acc = MosaicAccumulator::new(1, 2, 2);

        let data = Array3::from_elem((1, 2, 2), 10i8);

        acc.add(&data);
        acc.add(&data);
        acc.add(&data);

        assert_eq!(acc.max_overlap(), 3);
    }
}
