//! Configuration for the AEF mosaic pipeline.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the mosaic pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Input configuration
    pub input: InputConfig,

    /// Output configuration
    pub output: OutputConfig,

    /// Processing configuration
    pub processing: ProcessingConfig,

    /// Optional filter to limit processing area and years
    #[serde(default)]
    pub filter: Option<FilterConfig>,
}

/// Input data configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Path to the parquet index file (local or S3)
    /// Default: AEF v1 index on source.coop
    #[serde(default = "default_index_path")]
    pub index_path: String,

    /// S3 bucket containing COG files
    /// Default: source.coop public bucket
    #[serde(default = "default_cog_bucket")]
    pub cog_bucket: String,
}

fn default_index_path() -> String {
    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet".to_string()
}

fn default_cog_bucket() -> String {
    "us-west-2.opendata.source.coop".to_string()
}

/// Output Zarr configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Local filesystem path for output Zarr array.
    /// If set, output is written to local disk instead of S3.
    /// Mutually exclusive with bucket/prefix.
    #[serde(default)]
    pub local_path: Option<String>,

    /// Output S3 bucket for Zarr array (required if local_path is not set)
    #[serde(default)]
    pub bucket: Option<String>,

    /// Output S3 path prefix (required if local_path is not set)
    #[serde(default)]
    pub prefix: Option<String>,

    /// Output CRS (default: EPSG:6933 - EASE-Grid 2.0 Global, equal-area)
    #[serde(default = "default_output_crs")]
    pub crs: String,

    /// Output resolution in meters
    #[serde(default = "default_resolution")]
    pub resolution: f64,

    /// Number of years/time steps in output
    #[serde(default = "default_num_years")]
    pub num_years: usize,

    /// Starting year for the time dimension
    #[serde(default = "default_start_year")]
    pub start_year: i32,

    /// Number of bands/embedding dimensions
    #[serde(default = "default_num_bands")]
    pub num_bands: usize,

    /// Zarr chunk dimensions
    #[serde(default)]
    pub chunk_shape: ChunkShape,

    /// Use Zarr V3 sharding
    #[serde(default)]
    pub use_sharding: bool,

    /// Shards per dimension (if sharding enabled)
    #[serde(default = "default_shard_shape")]
    pub shard_shape: [usize; 2],

    /// Compression level (0-22 for zstd)
    #[serde(default = "default_compression_level")]
    pub compression_level: i32,
}

impl OutputConfig {
    /// Check if output is to local filesystem.
    pub fn is_local(&self) -> bool {
        self.local_path.is_some()
    }

    /// Get the output path as a display string (local path or s3:// URI).
    /// Panics if config is invalid (call validate() first).
    pub fn path_display(&self) -> String {
        if let Some(path) = &self.local_path {
            path.clone()
        } else {
            format!("s3://{}/{}",
                self.bucket.as_deref().unwrap_or(""),
                self.prefix.as_deref().unwrap_or(""))
        }
    }

    /// Get the local path if this is a local output.
    pub fn local_path(&self) -> Option<&str> {
        self.local_path.as_deref()
    }

    /// Get the S3 bucket if this is an S3 output.
    pub fn bucket(&self) -> Option<&str> {
        self.bucket.as_deref()
    }

    /// Get the S3 prefix if this is an S3 output.
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }
}

/// Zarr chunk shape configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkShape {
    /// Time dimension (typically 1 for single year)
    #[serde(default = "default_time_chunks")]
    pub time: usize,

    /// Embedding dimension (typically 64, full band)
    #[serde(default = "default_embedding_chunks")]
    pub embedding: usize,

    /// Height in pixels
    #[serde(default = "default_spatial_chunks")]
    pub height: usize,

    /// Width in pixels
    #[serde(default = "default_spatial_chunks")]
    pub width: usize,
}

impl Default for ChunkShape {
    fn default() -> Self {
        Self {
            time: 1,
            embedding: 64,
            height: 1024,
            width: 1024,
        }
    }
}

/// Processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Number of concurrent chunk processors
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,

    /// Maximum concurrent COG fetches per chunk
    #[serde(default = "default_cog_fetch_concurrency")]
    pub cog_fetch_concurrency: usize,

    /// Number of Tokio worker threads
    #[serde(default)]
    pub worker_threads: Option<usize>,

    /// Rayon thread pool size for CPU-bound work
    #[serde(default)]
    pub rayon_threads: Option<usize>,

    /// Enable metrics reporting
    #[serde(default = "default_true")]
    pub enable_metrics: bool,

    /// Metrics reporting interval in seconds
    #[serde(default = "default_metrics_interval")]
    pub metrics_interval_secs: u64,

    /// Retry configuration for failed operations
    #[serde(default)]
    pub retry: RetryConfig,

    /// Maximum memory for tile cache in GB
    #[serde(default = "default_tile_cache_gb")]
    pub tile_cache_gb: f64,

    /// Maximum entries in TIFF metadata cache
    #[serde(default = "default_metadata_cache_entries")]
    pub metadata_cache_entries: usize,

    /// Size of meta-tiles for spatial windowing (NxN output chunks per meta-tile)
    #[serde(default = "default_metatile_size")]
    pub metatile_size: usize,

    /// Enable prefetching all tiles for a metatile before processing
    /// When true: fewer, larger S3 requests but processing waits for prefetch
    /// When false: more, smaller requests but I/O overlaps with compute
    #[serde(default)]
    pub enable_prefetch: bool,

    /// Optional path to save metrics JSON after run completes
    #[serde(default)]
    pub metrics_output_path: Option<String>,
}

/// Retry configuration for transient failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// Initial backoff in milliseconds
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,

    /// Maximum backoff in milliseconds
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
        }
    }
}

/// Filter configuration to limit processing area and time range.
/// If not specified, all tiles are processed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat]
    /// Only tiles intersecting this bbox will be processed.
    #[serde(default)]
    pub bounds: Option<[f64; 4]>,

    /// List of years to process.
    /// Only tiles matching these years will be processed.
    /// If empty or not specified, all years are processed.
    #[serde(default)]
    pub years: Option<Vec<i32>>,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            concurrency: 256,
            cog_fetch_concurrency: 8,
            worker_threads: None,
            rayon_threads: None,
            enable_metrics: true,
            metrics_interval_secs: 10,
            retry: RetryConfig::default(),
            tile_cache_gb: 32.0,
            metadata_cache_entries: 10_000,
            metatile_size: 32,
            enable_prefetch: false,
            metrics_output_path: None,
        }
    }
}


impl Config {
    /// Load configuration from a YAML or JSON file.
    /// Format is auto-detected from file extension (.yaml, .yml, or .json).
    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let config: Config = match ext {
            "yaml" | "yml" => serde_yaml::from_str(&contents)?,
            "json" => serde_json::from_str(&contents)?,
            _ => {
                // Try YAML first (it's a superset of JSON)
                serde_yaml::from_str(&contents)?
            }
        };
        Ok(config)
    }

    /// Load configuration from a YAML string.
    pub fn from_yaml(yaml: &str) -> anyhow::Result<Self> {
        let config: Config = serde_yaml::from_str(yaml)?;
        Ok(config)
    }

    /// Load configuration from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let config: Config = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Serialize configuration to YAML.
    pub fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate output destination
        match (&self.output.local_path, &self.output.bucket, &self.output.prefix) {
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                anyhow::bail!("Cannot specify both local_path and bucket/prefix");
            }
            (None, None, _) | (None, _, None) => {
                anyhow::bail!("Must specify either local_path or both bucket and prefix");
            }
            _ => {}
        }

        if self.output.chunk_shape.embedding == 0 {
            anyhow::bail!("Embedding chunk size must be > 0");
        }
        if self.output.chunk_shape.height == 0 || self.output.chunk_shape.width == 0 {
            anyhow::bail!("Spatial chunk sizes must be > 0");
        }
        if self.processing.concurrency == 0 {
            anyhow::bail!("Concurrency must be > 0");
        }
        if self.output.compression_level < 0 || self.output.compression_level > 22 {
            anyhow::bail!("Compression level must be 0-22 for zstd");
        }
        Ok(())
    }
}

// Default value functions for serde
fn default_output_crs() -> String { "EPSG:6933".to_string() }
fn default_resolution() -> f64 { 10.0 }
fn default_time_chunks() -> usize { 1 }
fn default_embedding_chunks() -> usize { 64 }
fn default_spatial_chunks() -> usize { 1024 }
fn default_concurrency() -> usize { 256 }
fn default_cog_fetch_concurrency() -> usize { 8 }
fn default_true() -> bool { true }
fn default_metrics_interval() -> u64 { 10 }
fn default_max_retries() -> usize { 3 }
fn default_initial_backoff_ms() -> u64 { 100 }
fn default_max_backoff_ms() -> u64 { 10000 }
fn default_shard_shape() -> [usize; 2] { [8, 8] }
fn default_num_years() -> usize { 1 }
fn default_start_year() -> i32 { 2024 }
fn default_num_bands() -> usize { 64 }
fn default_compression_level() -> i32 { 3 }
fn default_tile_cache_gb() -> f64 { 32.0 }
fn default_metadata_cache_entries() -> usize { 10_000 }
fn default_metatile_size() -> usize { 32 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_chunk_shape() {
        let shape = ChunkShape::default();
        assert_eq!(shape.time, 1);
        assert_eq!(shape.embedding, 64);
        assert_eq!(shape.height, 1024);
        assert_eq!(shape.width, 1024);
    }

    #[test]
    fn test_config_validation_s3() {
        let config = Config {
            input: InputConfig {
                index_path: "s3://bucket/index.parquet".to_string(),
                cog_bucket: "cog-bucket".to_string(),
            },
            output: OutputConfig {
                local_path: None,
                bucket: Some("output-bucket".to_string()),
                prefix: Some("zarr/".to_string()),
                crs: "EPSG:6933".to_string(),
                resolution: 10.0,
                num_years: 1,
                start_year: 2024_i32,
                num_bands: 64,
                chunk_shape: ChunkShape::default(),
                use_sharding: false,
                shard_shape: [8, 8],
                compression_level: 3,
            },
            processing: ProcessingConfig::default(),
            filter: None,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_local() {
        let config = Config {
            input: InputConfig {
                index_path: "s3://bucket/index.parquet".to_string(),
                cog_bucket: "cog-bucket".to_string(),
            },
            output: OutputConfig {
                local_path: Some("/tmp/output.zarr".to_string()),
                bucket: None,
                prefix: None,
                crs: "EPSG:6933".to_string(),
                resolution: 10.0,
                num_years: 1,
                start_year: 2024_i32,
                num_bands: 64,
                chunk_shape: ChunkShape::default(),
                use_sharding: false,
                shard_shape: [8, 8],
                compression_level: 3,
            },
            processing: ProcessingConfig::default(),
            filter: None,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid() {
        // Both local_path and bucket set - should fail
        let config = Config {
            input: InputConfig {
                index_path: "s3://bucket/index.parquet".to_string(),
                cog_bucket: "cog-bucket".to_string(),
            },
            output: OutputConfig {
                local_path: Some("/tmp/output.zarr".to_string()),
                bucket: Some("bucket".to_string()),
                prefix: None,
                crs: "EPSG:6933".to_string(),
                resolution: 10.0,
                num_years: 1,
                start_year: 2024_i32,
                num_bands: 64,
                chunk_shape: ChunkShape::default(),
                use_sharding: false,
                shard_shape: [8, 8],
                compression_level: 3,
            },
            processing: ProcessingConfig::default(),
            filter: None,
        };

        assert!(config.validate().is_err());
    }
}
