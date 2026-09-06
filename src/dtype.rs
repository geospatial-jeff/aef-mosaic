//! Pixel data-type abstraction shared across the fetch → mosaic → write pipeline.
//!
//! Different geo-embedding datasets store embeddings in different element types
//! (AEF: quantized `int8`, Spheer: `float32`). To reuse a single COG-read / mosaic /
//! Zarr-write core, pixel buffers are carried as an enum ([`PixelData`] for a decoded
//! COG window, [`ChunkData`] for a mosaiced output chunk), and the mosaic inner loop is
//! generic over the [`Sample`] trait so there is exactly one mosaic code path for all
//! element types.

use ndarray::{Array3, Array4};

/// Element data type of a geo-embedding dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Signed 8-bit integer (e.g. AEF quantized embeddings).
    Int8,
    /// 32-bit float (e.g. Spheer embeddings).
    Float32,
}

impl DataType {
    /// The Zarr v3 data type string for this element type.
    pub fn zarr_str(self) -> &'static str {
        match self {
            DataType::Int8 => "int8",
            DataType::Float32 => "float32",
        }
    }
}

/// A decoded COG window: `(bands, height, width)`, tagged with its element type.
#[derive(Debug, Clone)]
pub enum PixelData {
    /// Signed 8-bit pixels.
    Int8(Array3<i8>),
    /// 32-bit float pixels.
    Float32(Array3<f32>),
}

impl PixelData {
    /// Shape as `(bands, height, width)`.
    pub fn dim(&self) -> (usize, usize, usize) {
        match self {
            PixelData::Int8(a) => a.dim(),
            PixelData::Float32(a) => a.dim(),
        }
    }

    /// The element data type of this window.
    pub fn data_type(&self) -> DataType {
        match self {
            PixelData::Int8(_) => DataType::Int8,
            PixelData::Float32(_) => DataType::Float32,
        }
    }
}

/// A mosaiced output chunk: `(time = 1, bands, height, width)`, tagged with its element type.
#[derive(Debug, Clone)]
pub enum ChunkData {
    /// Signed 8-bit chunk.
    Int8(Array4<i8>),
    /// 32-bit float chunk.
    Float32(Array4<f32>),
}

impl ChunkData {
    /// Shape as `[time, bands, height, width]`.
    pub fn shape(&self) -> &[usize] {
        match self {
            ChunkData::Int8(a) => a.shape(),
            ChunkData::Float32(a) => a.shape(),
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        match self {
            ChunkData::Int8(a) => a.len(),
            ChunkData::Float32(a) => a.len(),
        }
    }

    /// Whether the chunk has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element data type of this chunk.
    pub fn data_type(&self) -> DataType {
        match self {
            ChunkData::Int8(_) => DataType::Int8,
            ChunkData::Float32(_) => DataType::Float32,
        }
    }
}

/// A pixel element type that can be mean-mosaiced.
///
/// The mosaic accumulates a running sum + count per output pixel across overlapping
/// input tiles, then finalizes to the per-pixel mean. `i8` (AEF) rounds a running
/// `i32` sum (sign-correct, half-away-from-zero, matching the original AEF behavior);
/// `f32` (Spheer) averages a running `f64` sum.
pub trait Sample: Copy + Send + Sync + 'static {
    /// Accumulator element type (wider than `Self` to avoid overflow / precision loss).
    type Acc: Copy + Send + Sync;

    /// Output fill / nodata value for pixels with no contributing samples.
    const FILL: Self;
    /// Zero value for the accumulator.
    const ACC_ZERO: Self::Acc;

    /// Is this a valid (non-nodata) input sample?
    fn is_valid(self) -> bool;
    /// Add a valid sample into the accumulator.
    fn accumulate(acc: Self::Acc, sample: Self) -> Self::Acc;
    /// Finalize an accumulator with `count` (> 0) contributing samples into the mean.
    fn finalize(acc: Self::Acc, count: u32) -> Self;

    /// Borrow the matching typed array out of a [`PixelData`], if the dtype matches.
    fn extract(pixels: &PixelData) -> Option<&Array3<Self>>;
    /// Wrap a mosaiced `Array4<Self>` into the [`ChunkData`] enum.
    fn into_chunk(data: Array4<Self>) -> ChunkData;
}

impl Sample for i8 {
    type Acc = i32;

    /// AEF nodata sentinel.
    const FILL: i8 = -128;
    const ACC_ZERO: i32 = 0;

    #[inline]
    fn is_valid(self) -> bool {
        self != -128
    }

    #[inline]
    fn accumulate(acc: i32, sample: i8) -> i32 {
        acc + sample as i32
    }

    #[inline]
    fn finalize(acc: i32, count: u32) -> i8 {
        // Sign-correct rounding to nearest (half away from zero), matching the
        // original AEF BlockAccumulator so int8 output stays byte-identical.
        let c = count as i32;
        let half = c / 2;
        if acc >= 0 {
            ((acc + half) / c) as i8
        } else {
            ((acc - half) / c) as i8
        }
    }

    fn extract(pixels: &PixelData) -> Option<&Array3<i8>> {
        match pixels {
            PixelData::Int8(a) => Some(a),
            _ => None,
        }
    }

    fn into_chunk(data: Array4<i8>) -> ChunkData {
        ChunkData::Int8(data)
    }
}

impl Sample for f32 {
    type Acc = f64;

    /// Float nodata: NaN marks output pixels with no contributing samples.
    const FILL: f32 = f32::NAN;
    const ACC_ZERO: f64 = 0.0;

    #[inline]
    fn is_valid(self) -> bool {
        // Treat NaN and ±inf as nodata. Datasets that use a numeric nodata sentinel
        // (rather than NaN) should have it masked before this point.
        self.is_finite()
    }

    #[inline]
    fn accumulate(acc: f64, sample: f32) -> f64 {
        acc + sample as f64
    }

    #[inline]
    fn finalize(acc: f64, count: u32) -> f32 {
        (acc / count as f64) as f32
    }

    fn extract(pixels: &PixelData) -> Option<&Array3<f32>> {
        match pixels {
            PixelData::Float32(a) => Some(a),
            _ => None,
        }
    }

    fn into_chunk(data: Array4<f32>) -> ChunkData {
        ChunkData::Float32(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i8_sample_finalize_matches_aef_rounding() {
        // Mean of 10 and 20 -> 15 (as in the original AEF accumulator test).
        assert_eq!(i8::finalize(30, 2), 15);
        // Sign-correct rounding: -3 / 2 rounds to -2 (half away from zero).
        assert_eq!(i8::finalize(-3, 2), -2);
        assert_eq!(i8::finalize(3, 2), 2);
    }

    #[test]
    fn test_i8_sample_nodata() {
        assert!(!(-128i8).is_valid());
        assert!(0i8.is_valid());
        assert!(127i8.is_valid());
    }

    #[test]
    fn test_f32_sample_finalize_mean() {
        assert_eq!(f32::finalize(30.0, 2), 15.0);
        assert_eq!(f32::finalize(1.0, 4), 0.25);
    }

    #[test]
    fn test_f32_sample_nodata() {
        assert!(!f32::NAN.is_valid());
        assert!(!f32::INFINITY.is_valid());
        assert!(0.0f32.is_valid());
        assert!((-9.5f32).is_valid());
    }

    #[test]
    fn test_data_type_zarr_str() {
        assert_eq!(DataType::Int8.zarr_str(), "int8");
        assert_eq!(DataType::Float32.zarr_str(), "float32");
    }

    #[test]
    fn test_extract_dtype_mismatch_returns_none() {
        let i = PixelData::Int8(Array3::<i8>::zeros((1, 1, 1)));
        let f = PixelData::Float32(Array3::<f32>::zeros((1, 1, 1)));
        assert!(i8::extract(&i).is_some());
        assert!(i8::extract(&f).is_none());
        assert!(f32::extract(&f).is_some());
        assert!(f32::extract(&i).is_none());
    }
}
