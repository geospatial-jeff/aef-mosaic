//! Per-dataset behavior behind a shared COG-read / mosaic / GeoZarr-write core.
//!
//! Each geo-embedding dataset (AEF, Spheer, …) differs only in a handful of ways:
//! its element type, how its tiles are discovered, its output metadata, and its band
//! naming. Those differences live behind the [`EmbeddingDataset`] trait so the shared
//! pipeline (COG read → mosaic → GeoZarr write) stays dataset-agnostic and a new
//! dataset needs only a new, lightweight implementor.

use crate::dtype::DataType;
use anyhow::Result;
use serde_json::{json, Map, Value};

/// How a dataset's input tiles are discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryMethod {
    /// Read a precomputed parquet tile index (path given by `input.index_path`).
    ParquetIndex,
    /// Scan a prefix for `*.tif` COGs and read each header (path given by `input.index_path`).
    CogFolderScan,
}

/// A geo-embedding dataset: element type, tile discovery, and output metadata.
///
/// Implementors are stateless descriptors selected by the `dataset` config field.
pub trait EmbeddingDataset: Send + Sync {
    /// Machine name, e.g. `"aef"` or `"spheer"`.
    fn name(&self) -> &'static str;

    /// Element data type of the stored embeddings.
    fn data_type(&self) -> DataType;

    /// Default number of embedding bands (informational; the config value takes precedence).
    fn default_num_bands(&self) -> usize;

    /// How this dataset's tiles are discovered.
    fn discovery(&self) -> DiscoveryMethod;

    /// Coordinate names for the `band` dimension (length `num_bands`).
    fn band_names(&self, num_bands: usize) -> Vec<String>;

    /// `geoemb:` group attributes for the output Zarr (model, source, dtype, quantization …).
    ///
    /// `gsd` is the ground sample distance in the output CRS units.
    fn geoemb_attributes(&self, num_bands: usize, gsd: f64) -> Map<String, Value>;
}

/// Resolve a dataset implementor by its config name (case-insensitive).
pub fn dataset_from_name(name: &str) -> Result<Box<dyn EmbeddingDataset>> {
    match name.to_ascii_lowercase().as_str() {
        "aef" => Ok(Box::new(Aef)),
        "spheer" => Ok(Box::new(Spheer)),
        other => anyhow::bail!("Unknown dataset '{}'. Supported: aef, spheer", other),
    }
}

/// AlphaEarth Foundations: int8 quantized embeddings, bottom-up COGs, parquet index.
pub struct Aef;

impl EmbeddingDataset for Aef {
    fn name(&self) -> &'static str {
        "aef"
    }

    fn data_type(&self) -> DataType {
        DataType::Int8
    }

    fn default_num_bands(&self) -> usize {
        64
    }

    fn discovery(&self) -> DiscoveryMethod {
        DiscoveryMethod::ParquetIndex
    }

    fn band_names(&self, num_bands: usize) -> Vec<String> {
        // AEF convention: A00, A01, ..., A63
        (0..num_bands).map(|i| format!("A{:02}", i)).collect()
    }

    fn geoemb_attributes(&self, num_bands: usize, gsd: f64) -> Map<String, Value> {
        let mut attrs = Map::new();
        attrs.insert("geoemb:type".to_string(), json!("pixel"));
        attrs.insert("geoemb:dimensions".to_string(), json!(num_bands));
        attrs.insert(
            "geoemb:model".to_string(),
            json!("https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL"),
        );
        attrs.insert(
            "geoemb:source_data".to_string(),
            json!("https://source.coop/tge-labs/aef/v1/annual/"),
        );
        attrs.insert("geoemb:data_type".to_string(), json!("int8"));
        // GSD in native CRS units (same as resolution in spatial:transform)
        attrs.insert("geoemb:gsd".to_string(), json!(gsd));
        attrs.insert(
            "geoemb:quantization".to_string(),
            json!({
                "method": "signed_square",
                "original_dtype": "float32",
                "quantized_dtype": "int8",
                "formula": "(x / 127.5) ** 2 * sign(x)",
                "valid_range": [-127, 127],
                "nodata": -128
            }),
        );
        attrs
    }
}

/// Spheer FM (Albatross): raw float32 embeddings, standard top-down COGs laid out per
/// MGRS tile and year, discovered by scanning the folder tree.
pub struct Spheer;

impl EmbeddingDataset for Spheer {
    fn name(&self) -> &'static str {
        "spheer"
    }

    fn data_type(&self) -> DataType {
        DataType::Float32
    }

    fn default_num_bands(&self) -> usize {
        100
    }

    fn discovery(&self) -> DiscoveryMethod {
        DiscoveryMethod::CogFolderScan
    }

    fn band_names(&self, num_bands: usize) -> Vec<String> {
        // Generic embedding-dimension names: E000, E001, ...
        (0..num_bands).map(|i| format!("E{:03}", i)).collect()
    }

    fn geoemb_attributes(&self, num_bands: usize, gsd: f64) -> Map<String, Value> {
        // Spheer stores raw float32 embeddings, so there is no quantization block.
        let mut attrs = Map::new();
        attrs.insert("geoemb:type".to_string(), json!("pixel"));
        attrs.insert("geoemb:dimensions".to_string(), json!(num_bands));
        attrs.insert(
            "geoemb:model".to_string(),
            json!("https://huggingface.co/datasets/spheer/spheer-fm-embeddings"),
        );
        attrs.insert(
            "geoemb:source_data".to_string(),
            json!("https://huggingface.co/datasets/spheer/spheer-fm-embeddings"),
        );
        attrs.insert("geoemb:data_type".to_string(), json!("float32"));
        attrs.insert("geoemb:gsd".to_string(), json!(gsd));
        attrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_from_name_aef() {
        let ds = dataset_from_name("aef").unwrap();
        assert_eq!(ds.name(), "aef");
        assert_eq!(ds.data_type(), DataType::Int8);
        assert_eq!(ds.default_num_bands(), 64);
        assert_eq!(ds.discovery(), DiscoveryMethod::ParquetIndex);
    }

    #[test]
    fn test_dataset_from_name_case_insensitive() {
        assert!(dataset_from_name("AEF").is_ok());
        assert!(dataset_from_name("Aef").is_ok());
    }

    #[test]
    fn test_dataset_from_name_unknown() {
        assert!(dataset_from_name("nope").is_err());
    }

    #[test]
    fn test_dataset_from_name_spheer() {
        let ds = dataset_from_name("spheer").unwrap();
        assert_eq!(ds.name(), "spheer");
        assert_eq!(ds.data_type(), DataType::Float32);
        assert_eq!(ds.default_num_bands(), 100);
        assert_eq!(ds.discovery(), DiscoveryMethod::CogFolderScan);
    }

    #[test]
    fn test_spheer_band_names() {
        let names = Spheer.band_names(100);
        assert_eq!(names.len(), 100);
        assert_eq!(names[0], "E000");
        assert_eq!(names[99], "E099");
    }

    #[test]
    fn test_spheer_metadata_no_quantization() {
        let attrs = Spheer.geoemb_attributes(100, 10.0);
        assert_eq!(attrs.get("geoemb:data_type").and_then(|v| v.as_str()), Some("float32"));
        assert_eq!(attrs.get("geoemb:dimensions").and_then(|v| v.as_i64()), Some(100));
        // Spheer is raw float32 — no quantization metadata.
        assert!(attrs.get("geoemb:quantization").is_none());
    }

    #[test]
    fn test_aef_band_names() {
        let names = Aef.band_names(64);
        assert_eq!(names.len(), 64);
        assert_eq!(names[0], "A00");
        assert_eq!(names[9], "A09");
        assert_eq!(names[63], "A63");
    }

    #[test]
    fn test_aef_geoemb_attributes() {
        let attrs = Aef.geoemb_attributes(64, 10.0);
        assert_eq!(attrs.get("geoemb:type").and_then(|v| v.as_str()), Some("pixel"));
        assert_eq!(attrs.get("geoemb:dimensions").and_then(|v| v.as_i64()), Some(64));
        assert_eq!(attrs.get("geoemb:data_type").and_then(|v| v.as_str()), Some("int8"));
        assert_eq!(attrs.get("geoemb:gsd").and_then(|v| v.as_f64()), Some(10.0));
        let q = attrs.get("geoemb:quantization").expect("quantization present");
        assert_eq!(q.get("method").and_then(|v| v.as_str()), Some("signed_square"));
    }
}
