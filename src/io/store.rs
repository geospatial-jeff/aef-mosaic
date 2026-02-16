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
use object_store::{ClientOptions, ObjectStore};
use std::sync::Arc;
use std::time::Duration;

/// Create client options for S3 access optimized for high-throughput.
///
/// Key settings:
/// - Large connection pool (256 connections per host) for high concurrency
/// - Keep-alive to reuse TCP connections
/// - Note: HTTP/2 is NOT enabled as S3 path-style URLs don't support it reliably
fn create_client_options() -> ClientOptions {
    ClientOptions::new()
        // Request timeout (per request, not connection)
        .with_timeout(Duration::from_secs(60))
        // Allow many concurrent connections to S3
        .with_pool_max_idle_per_host(256)
        // Keep connections alive longer for reuse
        .with_pool_idle_timeout(Duration::from_secs(90))
        // TCP connection timeout
        .with_connect_timeout(Duration::from_secs(10))
}

/// Create an anonymous S3 client for reading from source.coop (public bucket).
///
/// Used for reading COG tiles and index. No credentials needed.
/// Hardcoded to us-west-2 where source.coop is hosted.
///
/// Connection pool is configured for high throughput (256 connections, HTTP/2).
pub fn create_anonymous_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    tracing::info!(
        "Creating anonymous S3 client for bucket: {} (pool_size=256, idle_timeout=90s)",
        bucket
    );

    let builder = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-west-2")
        .with_client_options(create_client_options())
        .with_skip_signature(true)
        .with_virtual_hosted_style_request(false);

    Ok(Arc::new(builder.build()?))
}

/// Create an authenticated S3 client for writing.
///
/// Credentials are loaded from environment variables:
/// - AWS_ACCESS_KEY_ID (required)
/// - AWS_SECRET_ACCESS_KEY (required)
/// - AWS_REGION or AWS_DEFAULT_REGION (required)
/// - AWS_SESSION_TOKEN (optional, for temporary credentials)
fn create_authenticated_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    let mut missing: Vec<&str> = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        .into_iter()
        .filter(|var| std::env::var(var).is_err())
        .collect();

    if std::env::var("AWS_REGION").is_err() && std::env::var("AWS_DEFAULT_REGION").is_err() {
        missing.push("AWS_REGION");
    }

    if !missing.is_empty() {
        anyhow::bail!("Missing AWS credentials: {}", missing.join(", "));
    }

    tracing::info!("Creating authenticated S3 client for bucket: {}", bucket);

    // Buckets with dots in the name require path-style URLs
    // Virtual-hosted style creates invalid hostnames like "bucket.name.s3.region.amazonaws.com"
    let use_virtual_hosted = !bucket.contains('.');

    let builder = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .with_client_options(create_client_options())
        .with_virtual_hosted_style_request(use_virtual_hosted);

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
    fn test_create_authenticated_store_missing_credentials() {
        // Clear any existing credentials to test error handling
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");

        let result = create_authenticated_store("test-bucket");
        assert!(result.is_err());

        let err = result.unwrap_err().to_string();
        assert!(err.contains("Missing AWS credentials"));
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
