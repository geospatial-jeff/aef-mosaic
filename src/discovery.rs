//! Input tile discovery.
//!
//! Datasets differ in how their tiles are enumerated:
//! - AEF ships a precomputed parquet index ([`DiscoveryMethod::ParquetIndex`]).
//! - Spheer is a folder tree of COGs ([`DiscoveryMethod::CogFolderScan`]); each COG's
//!   header is read to derive its geometry, CRS, and year.
//!
//! Both paths produce an [`InputIndex`], so the rest of the pipeline is unaffected.

use crate::config::Config;
use crate::crs::ProjCache;
use crate::dataset::{DiscoveryMethod, EmbeddingDataset};
use crate::index::{CogTile, InputIndex};
use crate::io::{parse_s3_uri, CogReader};
use anyhow::{Context, Result};
use futures::StreamExt;
use geo::{BoundingRect, Polygon};
use object_store::path::Path;
use object_store::ObjectStore;
use std::sync::Arc;

/// Build the input tile index for the given dataset, dispatching on its discovery method.
pub async fn build_input_index(
    dataset: &dyn EmbeddingDataset,
    config: &Config,
    cog_store: Arc<dyn ObjectStore>,
) -> Result<InputIndex> {
    match dataset.discovery() {
        DiscoveryMethod::ParquetIndex => load_parquet_index(config, cog_store).await,
        DiscoveryMethod::CogFolderScan => scan_cog_folder(config, cog_store).await,
    }
}

/// Load a precomputed parquet tile index (local path or `s3://` URI).
async fn load_parquet_index(config: &Config, cog_store: Arc<dyn ObjectStore>) -> Result<InputIndex> {
    if config.input.index_path.starts_with("s3://") {
        let (_bucket, key) = parse_s3_uri(&config.input.index_path)?;
        let path = Path::from(key);
        InputIndex::from_s3(cog_store, &path).await
    } else {
        InputIndex::from_local_parquet(&config.input.index_path)
    }
}

/// Scan a folder of COGs, reading each header to build the tile index.
///
/// COG keys are enumerated via the HuggingFace tree API for `hf://` repos, or via
/// `ObjectStore::list` otherwise. `input.index_path` is the key prefix to scan. Each
/// `*.tif`/`*.tiff` header supplies dimensions + geotransform (→ native bounds) and the
/// GeoKey CRS; the year is parsed from the file path.
async fn scan_cog_folder(config: &Config, cog_store: Arc<dyn ObjectStore>) -> Result<InputIndex> {
    let mut keys: Vec<String> = if let Some(repo) = config.input.cog_bucket.strip_prefix("hf://") {
        let token = std::env::var("HF_TOKEN")
            .context("HF_TOKEN must be set to list a HuggingFace (hf://) dataset")?;
        list_hf_cogs(repo, &config.input.index_path, &token).await?
    } else {
        list_store_cogs(&cog_store, config).await?
    };

    if keys.is_empty() {
        anyhow::bail!(
            "No .tif/.tiff COGs found under prefix '{}'",
            config.input.index_path
        );
    }

    // Deterministic order (helps reproducibility and debugging).
    keys.sort();
    tracing::info!("Discovered {} COG(s) to scan", keys.len());

    let reader = CogReader::new(cog_store.clone());
    let proj_cache = ProjCache::new();
    let mut tiles: Vec<CogTile> = Vec::with_capacity(keys.len());

    for key in keys {
        let header = reader
            .read_header(&key)
            .await
            .with_context(|| format!("Failed to read COG header: {}", key))?;

        let gt = header
            .geo_transform
            .ok_or_else(|| anyhow::anyhow!("COG {} has no geotransform", key))?;
        let epsg = header
            .epsg
            .ok_or_else(|| anyhow::anyhow!("COG {} has no CRS in its GeoKeyDirectory", key))?;
        let crs = format!("EPSG:{}", epsg);

        // Native bounds from the geotransform corners (works for any orientation).
        let (w, h) = (header.width as f64, header.height as f64);
        let corners = [
            gt.pixel_to_world(0.0, 0.0),
            gt.pixel_to_world(w, 0.0),
            gt.pixel_to_world(0.0, h),
            gt.pixel_to_world(w, h),
        ];
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for (x, y) in corners {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let bounds_native = [min_x, min_y, max_x, max_y];

        // WGS84 footprint + bounds by transforming the native corners.
        let footprint_wgs84 = CogTile::compute_footprint(&bounds_native, &crs, &proj_cache)
            .with_context(|| format!("Failed to compute WGS84 footprint for {}", key))?;
        let bounds_wgs84 = polygon_bounds(&footprint_wgs84);

        let year = parse_year_from_key(&key)
            .with_context(|| format!("Could not determine year from COG path: {}", key))?;

        tiles.push(CogTile {
            tile_id: key.clone(),
            s3_path: key,
            crs,
            bounds_native,
            bounds_wgs84,
            footprint_wgs84,
            resolution: gt.a.abs(),
            year,
        });
    }

    tracing::info!("Folder scan built index from {} COG tiles", tiles.len());
    Ok(InputIndex::from_tiles(tiles))
}

/// Enumerate `.tif`/`.tiff` keys under a prefix via `ObjectStore::list` (S3/local).
async fn list_store_cogs(cog_store: &Arc<dyn ObjectStore>, config: &Config) -> Result<Vec<String>> {
    let list_prefix = if config.input.index_path.starts_with("s3://") {
        let (_bucket, key) = parse_s3_uri(&config.input.index_path)?;
        key.to_string()
    } else {
        config.input.index_path.clone()
    };
    let prefix_path = if list_prefix.is_empty() {
        None
    } else {
        Some(Path::from(list_prefix.as_str()))
    };

    let mut listing = cog_store.list(prefix_path.as_ref());
    let mut keys = Vec::new();
    while let Some(meta) = listing.next().await {
        let meta = meta.context("Failed to list COG store")?;
        let key = meta.location.as_ref();
        if key.ends_with(".tif") || key.ends_with(".tiff") {
            keys.push(key.to_string());
        }
    }
    Ok(keys)
}

/// Enumerate `.tif`/`.tiff` keys under a prefix in a gated HuggingFace dataset repo via
/// the tree API (`/api/datasets/<repo>/tree/main/<prefix>?recursive=true`), following
/// pagination if present.
async fn list_hf_cogs(repo: &str, prefix: &str, token: &str) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let mut keys = Vec::new();
    let mut url = format!(
        "https://huggingface.co/api/datasets/{}/tree/main/{}?recursive=true&expand=true",
        repo,
        prefix.trim_matches('/')
    );

    loop {
        let resp = client
            .get(&url)
            .bearer_auth(token)
            .send()
            .await
            .context("HuggingFace tree API request failed")?
            .error_for_status()
            .context("HuggingFace tree API returned an error status")?;

        // Capture the pagination link before consuming the body.
        let next = resp
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_next_link);

        let items: Vec<serde_json::Value> = resp
            .json()
            .await
            .context("Failed to parse HuggingFace tree API response")?;
        for item in items {
            if item.get("type").and_then(|v| v.as_str()) == Some("file") {
                if let Some(path) = item.get("path").and_then(|v| v.as_str()) {
                    if path.ends_with(".tif") || path.ends_with(".tiff") {
                        keys.push(path.to_string());
                    }
                }
            }
        }

        match next {
            Some(next_url) => url = next_url,
            None => break,
        }
    }

    Ok(keys)
}

/// Parse the `rel="next"` URL from an RFC 5988 `Link` header, if present.
fn parse_next_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        if part.contains("rel=\"next\"") {
            let start = part.find('<')?;
            let end = part.find('>')?;
            return Some(part[start + 1..end].to_string());
        }
    }
    None
}

/// Bounding box `[min_x, min_y, max_x, max_y]` of a polygon.
fn polygon_bounds(poly: &Polygon<f64>) -> [f64; 4] {
    match poly.bounding_rect() {
        Some(rect) => [rect.min().x, rect.min().y, rect.max().x, rect.max().y],
        None => [0.0, 0.0, 0.0, 0.0],
    }
}

/// Parse the acquisition year from a COG path.
///
/// Prefers the file stem (Spheer names files `<year>.tif`), falling back to any 4-digit
/// year found among the path components.
fn parse_year_from_key(key: &str) -> Result<i32> {
    let stem = key
        .rsplit('/')
        .next()
        .unwrap_or(key)
        .trim_end_matches(".tiff")
        .trim_end_matches(".tif");
    if let Ok(year) = stem.parse::<i32>() {
        return Ok(year);
    }

    for component in key.split(['/', '-', '_', '.']) {
        if component.len() == 4 {
            if let Ok(year) = component.parse::<i32>() {
                if (1900..=2200).contains(&year) {
                    return Ok(year);
                }
            }
        }
    }

    anyhow::bail!("no 4-digit year found in path '{}'", key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_year_from_stem() {
        assert_eq!(parse_year_from_key("albatross-EU-v2025/nl-tiles/31UGV/2020.tif").unwrap(), 2020);
        assert_eq!(parse_year_from_key("2017.tiff").unwrap(), 2017);
    }

    #[test]
    fn test_parse_year_from_component() {
        // Stem is not a year, but a path component is.
        assert_eq!(parse_year_from_key("data/2019/tile_abc.tif").unwrap(), 2019);
    }

    #[test]
    fn test_parse_year_none() {
        assert!(parse_year_from_key("data/tile_abc.tif").is_err());
    }

    #[test]
    fn test_parse_next_link() {
        let link = "<https://huggingface.co/api/datasets/x/tree/main?cursor=abc>; rel=\"next\"";
        assert_eq!(
            parse_next_link(link).as_deref(),
            Some("https://huggingface.co/api/datasets/x/tree/main?cursor=abc")
        );
        assert_eq!(parse_next_link("<https://x>; rel=\"prev\"").as_deref(), None);
        assert_eq!(parse_next_link("").as_deref(), None);
    }
}
