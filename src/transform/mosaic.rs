//! Mosaicing overlapping tiles using mean aggregation.

use crate::io::WindowData;
use crate::transform::{ReprojectConfig, Reprojector};
use anyhow::Result;
use ndarray::{s, Array2, Array3, Array4, Axis, Zip};

/// Nodata value for AEF embeddings (int8).
const NODATA: i8 = -128;

/// Accumulator for mosaicing multiple tiles using mean aggregation.
///
/// Uses i16 accumulators which can handle up to ~250 overlapping tiles without overflow.
/// (i8 range is -128 to 127; worst case 127 * 250 = 31,750 < i16 max of 32,767)
/// This halves memory usage compared to i32 accumulators.
#[derive(Debug)]
pub struct MosaicAccumulator {
    /// Sum of values for each pixel: (1, bands, height, width)
    sum: Array4<i16>,

    /// Count of contributions for each pixel: (height, width)
    count: Array2<u16>,

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
            count: Array2::zeros((height, width)),
            bands,
            height,
            width,
        }
    }

    /// Add a tile's reprojected data to the accumulator.
    ///
    /// The input data should be in shape (bands, height, width) and already
    /// reprojected to the output grid. Values of -128 are treated as nodata.
    ///
    /// Uses ndarray operations for better performance and readability.
    pub fn add(&mut self, data: &Array3<i8>) {
        let (bands, height, width) = data.dim();
        assert_eq!(bands, self.bands, "Band count mismatch");
        assert_eq!(height, self.height, "Height mismatch");
        assert_eq!(width, self.width, "Width mismatch");

        // Create a mask: true where ANY band has non-nodata value
        // Use fold_axis to check across all bands for each pixel
        let has_data: Array2<bool> = data
            .map(|&v| v != NODATA)
            .fold_axis(Axis(0), false, |acc, &val| *acc || val);

        // Update count where we have data using Zip
        Zip::from(&mut self.count)
            .and(&has_data)
            .for_each(|count, &has| {
                if has {
                    *count += 1;
                }
            });

        // Add data to sum where has_data is true
        // We need to iterate over bands and use the mask
        for b in 0..bands {
            let data_band = data.slice(s![b, .., ..]);
            let mut sum_band = self.sum.slice_mut(s![0, b, .., ..]);

            Zip::from(&mut sum_band)
                .and(&data_band)
                .and(&has_data)
                .for_each(|sum, &val, &has| {
                    if has {
                        *sum += val as i16;
                    }
                });
        }
    }

    /// Finalize the mosaic by computing the mean.
    ///
    /// Returns an int8 array: (1, bands, height, width)
    /// Pixels with no data are set to NODATA (-128).
    ///
    /// Uses ndarray operations for element-wise division with rounding.
    pub fn finalize(self) -> Array4<i8> {
        // Initialize with NODATA
        let mut result = Array4::<i8>::from_elem((1, self.bands, self.height, self.width), NODATA);

        // Process each band using Zip
        for b in 0..self.bands {
            let sum_band = self.sum.slice(s![0, b, .., ..]);
            let mut result_band = result.slice_mut(s![0, b, .., ..]);

            Zip::from(&mut result_band)
                .and(&sum_band)
                .and(&self.count)
                .for_each(|result, &sum, &count| {
                    if count > 0 {
                        let c = count as i16;
                        let half_c = c / 2;
                        // Rounded integer division
                        *result = if sum >= 0 {
                            ((sum + half_c) / c) as i8
                        } else {
                            ((sum - half_c) / c) as i8
                        };
                    }
                });
        }

        result
    }

    /// Get the number of tiles that contributed to each pixel.
    /// Returns a view as 3D array for backwards compatibility.
    pub fn coverage(&self) -> Array3<u16> {
        self.count.clone().insert_axis(Axis(0))
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

        let valid_pixels = reprojected.iter().filter(|&&v| v != -128).count();
        tracing::debug!(
            "Window {} reprojected: shape={:?}, valid_pixels={}",
            i, reprojected.dim(), valid_pixels
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

        // First tile: some values, -128 is nodata
        // Pixel (0,0): both bands have data (10, 50)
        // Pixel (0,1): both bands are nodata (-128, -128)
        // Pixel (1,0): both bands are nodata (-128, -128)
        // Pixel (1,1): both bands have data (40, 80)
        let data1 = Array3::from_shape_vec(
            (2, 2, 2),
            vec![10i8, -128, -128, 40, 50, -128, -128, 80],
        )
        .unwrap();
        acc.add(&data1);

        // Check coverage (using 2D count array)
        assert_eq!(acc.count[[0, 0]], 1); // (0,0) has data
        assert_eq!(acc.count[[0, 1]], 0); // (0,1) is nodata
        assert_eq!(acc.count[[1, 0]], 0); // (1,0) is nodata
        assert_eq!(acc.count[[1, 1]], 1); // (1,1) has data
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
