//! Object store configuration for S3 and local filesystem access.
//!
//! This module provides optimized S3 client configuration for high-throughput
//! data transfer, including connection pool tuning and timeout settings.

use anyhow::{Context, Result};

/// Parse an S3 URI into bucket and key components.
///
/// Accepts URIs in the format `s3://bucket/key/path`.
///
/// # Returns
/// A tuple of (bucket, key) on success.
///
/// # Errors
/// Returns an error if the URI is malformed (missing scheme, bucket, or key).
pub fn parse_s3_uri(uri: &str) -> Result<(&str, &str)> {
    let without_scheme = uri
        .strip_prefix("s3://")
        .with_context(|| format!("Invalid S3 URI: expected 's3://' prefix in '{}'", uri))?;

    without_scheme
        .split_once('/')
        .with_context(|| format!("Invalid S3 URI: expected 's3://bucket/key' format in '{}'", uri))
}
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::{ClientOptions, ObjectStore, RetryConfig};
use std::sync::Arc;
use std::time::Duration;

/// Create optimized client options for high-throughput S3 access.
///
/// These settings are tuned for:
/// - EC2 instances with 100 Gbps network
/// - High-concurrency workloads (512+ concurrent requests)
/// - Large file transfers
fn create_client_options() -> ClientOptions {
    ClientOptions::new()
        // Connection timeout: how long to wait for a connection to be established
        .with_connect_timeout(Duration::from_secs(5))
        // Request timeout: total time allowed for a request including retries
        .with_timeout(Duration::from_secs(30))
        // Pool idle timeout: how long to keep idle connections in the pool
        .with_pool_idle_timeout(Duration::from_secs(90))
        // Maximum idle connections per host - increased for 512+ concurrency
        .with_pool_max_idle_per_host(512)
        // HTTP/2 keep-alive settings for long-running connections
        .with_http2_keep_alive_interval(Duration::from_secs(15))
        .with_http2_keep_alive_while_idle()
}

/// Create retry configuration for transient failures.
fn create_retry_config() -> RetryConfig {
    RetryConfig {
        // Maximum number of retries per request
        max_retries: 5,
        // Initial backoff (doubles each retry)
        backoff: object_store::BackoffConfig {
            init_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            base: 2.0,
        },
        // Retry on 429 (rate limiting) and 5xx (server errors)
        retry_timeout: Duration::from_secs(120),
    }
}

/// Create an anonymous S3 client for reading from source.coop (public bucket).
///
/// Used for reading COG tiles and index. No credentials needed.
/// Hardcoded to us-west-2 where source.coop is hosted.
pub fn create_anonymous_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    tracing::info!("Creating anonymous S3 client for bucket: {}", bucket);

    let builder = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-west-2") // source.coop is in us-west-2
        .with_client_options(create_client_options())
        .with_retry(create_retry_config())
        .with_skip_signature(true)
        .with_virtual_hosted_style_request(false);

    Ok(Arc::new(builder.build()?))
}

/// Create an authenticated S3 client for writing.
///
/// Credentials and region are loaded from (in order):
/// - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
/// - AWS config files (~/.aws/credentials, ~/.aws/config)
/// - EC2 instance profile (IMDS)
fn create_authenticated_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    tracing::info!("Creating authenticated S3 client for bucket: {}", bucket);

    let builder = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .with_client_options(create_client_options())
        .with_retry(create_retry_config())
        .with_virtual_hosted_style_request(true);

    Ok(Arc::new(builder.build()?))
}

/// Create a store for reading input COG tiles (anonymous, no credentials).
pub fn create_cog_store(config: &crate::config::Config) -> Result<Arc<dyn ObjectStore>> {
    create_anonymous_store(&config.input.cog_bucket)
}

/// Create a store for writing output Zarr (authenticated).
/// Uses LocalFileSystem if local_path is set, otherwise S3 with credentials.
pub fn create_output_store(config: &crate::config::Config) -> Result<Arc<dyn ObjectStore>> {
    match (&config.output.local_path, &config.output.bucket) {
        (Some(local_path), _) => {
            let path = std::path::Path::new(local_path);
            if !path.exists() {
                std::fs::create_dir_all(path)?;
            }
            tracing::info!("Creating LocalFileSystem store at: {}", path.display());
            Ok(Arc::new(LocalFileSystem::new_with_prefix(path)?))
        }
        (_, Some(bucket)) => create_authenticated_store(bucket),
        _ => anyhow::bail!("Invalid config: no output destination"),
    }
}

/// Get the output path prefix for Zarr.
/// Returns empty string for local (since prefix is baked into store),
/// or the S3 prefix for remote.
pub fn get_output_prefix(config: &crate::config::Config) -> &str {
    if config.output.is_local() {
        "" // LocalFileSystem already has the path as prefix
    } else {
        config.output.prefix().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_authenticated_store() {
        let result = create_authenticated_store("test-bucket");
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_anonymous_store() {
        let result = create_anonymous_store("us-west-2.opendata.source.coop");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_s3_uri() {
        // Valid URIs
        let (bucket, key) = parse_s3_uri("s3://my-bucket/path/to/file.tif").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.tif");

        let (bucket, key) = parse_s3_uri("s3://us-west-2.opendata.source.coop/tge-labs/aef/index.parquet").unwrap();
        assert_eq!(bucket, "us-west-2.opendata.source.coop");
        assert_eq!(key, "tge-labs/aef/index.parquet");

        // Single level key
        let (bucket, key) = parse_s3_uri("s3://bucket/file.txt").unwrap();
        assert_eq!(bucket, "bucket");
        assert_eq!(key, "file.txt");
    }

    #[test]
    fn test_parse_s3_uri_invalid() {
        // Missing scheme
        assert!(parse_s3_uri("bucket/key").is_err());

        // Wrong scheme
        assert!(parse_s3_uri("http://bucket/key").is_err());

        // Missing key (bucket only)
        assert!(parse_s3_uri("s3://bucket").is_err());
    }
}
