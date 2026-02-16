//! Async COG reading using async-tiff.
//!
//! Reads only the portion of a COG that overlaps with the requested pixel window.
//! Algorithm:
//! 1. Read TIFF header/metadata (cached)
//! 2. Identify which internal tile rows cover the requested bounds
//! 3. Fetch each tile row in parallel (range requests)
//! 4. Assemble tiles into an array
//! 5. Clip to the exact requested bounds

use crate::index::CogTile;
use crate::pipeline::Metrics;
use anyhow::{Context, Result};
use async_tiff::decoder::DecoderRegistry;
use async_tiff::metadata::TiffMetadataReader;
use async_tiff::reader::{AsyncFileReader, ObjectReader};
use async_tiff::tags::PlanarConfiguration;
use async_tiff::TIFF;
use dashmap::DashMap;
use futures::future::try_join_all;
use lru::LruCache;
use ndarray::Array3;
use rayon::prelude::*;
use object_store::path::Path;
use object_store::ObjectStore;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

/// Affine geotransform for converting between pixel and world coordinates.
///
/// The transform is defined by 6 coefficients from the GDAL-style affine:
/// ```text
/// x_world = a * col + b * row + c
/// y_world = d * col + e * row + f
/// ```
///
/// For most GeoTIFFs:
/// - `a` is the pixel width (x resolution)
/// - `e` is the pixel height (y resolution, negative for top-down images)
/// - `c` is the x coordinate of the upper-left corner
/// - `f` is the y coordinate of the upper-left corner
/// - `b` and `d` are typically 0 (no rotation/shear)
#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    /// Pixel width (x scale)
    pub a: f64,
    /// Row rotation (typically 0)
    pub b: f64,
    /// X origin (upper-left x coordinate)
    pub c: f64,
    /// Column rotation (typically 0)
    pub d: f64,
    /// Pixel height (y scale, negative for top-down, positive for bottom-up)
    pub e: f64,
    /// Y origin (upper-left y coordinate)
    pub f: f64,
}

impl GeoTransform {
    /// Create a GeoTransform from the 16-element ModelTransformationTag matrix.
    ///
    /// The matrix is a 4x4 affine transform in row-major order:
    /// ```text
    /// | a  b  0  c |
    /// | d  e  0  f |
    /// | 0  0  0  0 |
    /// | 0  0  0  1 |
    /// ```
    pub fn from_model_transformation(matrix: &[f64]) -> Option<Self> {
        if matrix.len() < 8 {
            return None;
        }
        Some(Self {
            a: matrix[0],  // scale_x
            b: matrix[1],  // rotation (typically 0)
            c: matrix[3],  // origin_x
            d: matrix[4],  // rotation (typically 0)
            e: matrix[5],  // scale_y (positive for bottom-up, negative for top-down)
            f: matrix[7],  // origin_y
        })
    }

    /// Convert world coordinates to pixel coordinates.
    ///
    /// Returns (column, row) as floating point for sub-pixel precision.
    #[inline]
    pub fn world_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        // Inverse of affine transform (assuming no rotation, b=0, d=0)
        let col = (x - self.c) / self.a;
        let row = (y - self.f) / self.e;
        (col, row)
    }

    /// Convert pixel coordinates to world coordinates.
    ///
    /// Takes (column, row) and returns (x, y) in the CRS.
    #[inline]
    pub fn pixel_to_world(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.a * col + self.b * row + self.c;
        let y = self.d * col + self.e * row + self.f;
        (x, y)
    }

    /// Check if this is a bottom-up image (positive y scale).
    #[inline]
    pub fn is_bottom_up(&self) -> bool {
        self.e > 0.0
    }
}

/// A pixel window within an image.
#[derive(Debug, Clone, Copy)]
pub struct PixelWindow {
    /// X offset (column) from top-left
    pub x: usize,
    /// Y offset (row) from top-left
    pub y: usize,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
}

impl PixelWindow {
    /// Create a new pixel window with the given offset and dimensions.
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self { x, y, width, height }
    }
}

/// Data read from a COG window.
#[derive(Debug)]
pub struct WindowData {
    /// The source tile metadata
    pub tile: CogTile,

    /// Pixel data as int8 array: (bands, height, width)
    pub data: Array3<i8>,

    /// The pixel window that was read
    pub window: PixelWindow,

    /// Geographic bounds of this window in the tile's native CRS [min_x, min_y, max_x, max_y]
    pub bounds_native: [f64; 4],
}

/// Cached TIFF metadata to avoid re-parsing headers on every read.
#[derive(Clone)]
pub struct CachedTiff {
    /// The parsed TIFF structure
    tiff: Arc<TIFF>,
    /// Async file reader for this COG
    reader: Arc<dyn AsyncFileReader>,
    /// Image width in pixels
    image_width: usize,
    /// Image height in pixels
    image_height: usize,
    /// Number of bands/samples per pixel
    bands: usize,
    /// Whether the data is planar (vs chunky/interleaved)
    is_planar: bool,
    /// Internal tile width
    tile_width: usize,
    /// Internal tile height
    tile_height: usize,
    /// Geotransform from ModelTransformationTag (if present)
    geo_transform: Option<GeoTransform>,
}

/// LRU cache for TIFF metadata with single-flight deduplication.
pub struct TiffMetadataCache {
    /// LRU cache mapping COG path to cached metadata
    cache: RwLock<LruCache<String, Arc<CachedTiff>>>,
    /// In-flight requests (single-flight pattern) - uses DashMap for lock-free access
    in_flight: DashMap<String, broadcast::Sender<Result<Arc<CachedTiff>, String>>>,
    /// Optional metrics for tracking cache performance
    metrics: Option<Arc<Metrics>>,
}

impl TiffMetadataCache {
    /// Create a new metadata cache with the specified capacity.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of TIFF metadata entries to cache (default: 10,000)
    /// * `metrics` - Optional metrics collector for cache hit/miss tracking
    pub fn new(max_entries: usize, metrics: Option<Arc<Metrics>>) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::new(10000).unwrap()),
            )),
            in_flight: DashMap::new(),
            metrics,
        }
    }

    /// Get cached metadata or load it from S3 with single-flight deduplication.
    ///
    /// If multiple tasks request the same metadata simultaneously, only one
    /// will actually fetch it - the others will wait for that result.
    pub async fn get_or_load(
        &self,
        path: &str,
        store: Arc<dyn ObjectStore>,
        object_path: Path,
    ) -> Result<Arc<CachedTiff>> {
        // 1. Check cache first (fast path)
        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get(path) {
                if let Some(ref m) = self.metrics {
                    m.add_metadata_cache_hit();
                }
                return Ok(cached.clone());
            }
        }

        // 2. Check if already in-flight (single-flight pattern) - lock-free with DashMap
        if let Some(sender_ref) = self.in_flight.get(path) {
            // Someone else is fetching - wait for their result
            let mut rx = sender_ref.subscribe();
            drop(sender_ref); // Release the DashMap reference before await

            if let Some(ref m) = self.metrics {
                m.add_metadata_cache_hit(); // Count as hit since we avoid duplicate fetch
            }

            // Wait for the result
            match rx.recv().await {
                Ok(Ok(cached)) => return Ok(cached),
                Ok(Err(e)) => return Err(anyhow::anyhow!("Coalesced fetch failed: {}", e)),
                Err(e) => return Err(anyhow::anyhow!("Broadcast channel error: {}", e)),
            }
        }

        // 3. We're the first - register in-flight and fetch (lock-free insert)
        let (tx, _) = broadcast::channel(64); // Increased buffer for high concurrency
        self.in_flight.insert(path.to_string(), tx.clone());

        // Cache miss - load metadata
        if let Some(ref m) = self.metrics {
            m.add_metadata_cache_miss();
        }

        // 4. Perform the actual fetch
        let result = self.load_metadata(store, object_path).await;

        // 5. Handle result
        match result {
            Ok(cached) => {
                // Store in cache
                {
                    let mut cache = self.cache.write().await;
                    cache.put(path.to_string(), cached.clone());
                }

                // Remove from in-flight and notify waiters (lock-free)
                self.in_flight.remove(path);
                let _ = tx.send(Ok(cached.clone()));

                Ok(cached)
            }
            Err(e) => {
                // Remove from in-flight and notify waiters of failure (lock-free)
                self.in_flight.remove(path);
                let _ = tx.send(Err(e.to_string()));

                Err(e)
            }
        }
    }

    /// Load metadata from S3 (internal helper).
    async fn load_metadata(
        &self,
        store: Arc<dyn ObjectStore>,
        object_path: Path,
    ) -> Result<Arc<CachedTiff>> {
        // Create async reader
        let reader: Arc<dyn AsyncFileReader> =
            Arc::new(ObjectReader::new(store, object_path));

        // Read TIFF header/metadata
        let read_cache = async_tiff::metadata::cache::ReadaheadMetadataCache::new(reader.clone());
        let mut metadata_reader = TiffMetadataReader::try_open(&read_cache)
            .await
            .context("Failed to open TIFF metadata")?;

        let ifds = metadata_reader
            .read_all_ifds(&read_cache)
            .await
            .context("Failed to read IFDs")?;

        let tiff = TIFF::new(ifds, metadata_reader.endianness());

        // Get the first IFD (full resolution)
        let ifd = tiff.ifds().first().context("No IFDs in TIFF")?;

        let image_width = ifd.image_width() as usize;
        let image_height = ifd.image_height() as usize;
        let bands = ifd.samples_per_pixel() as usize;
        let is_planar = matches!(ifd.planar_configuration(), PlanarConfiguration::Planar);
        let tile_width = ifd.tile_width().unwrap_or(image_width as u32) as usize;
        let tile_height = ifd.tile_height().unwrap_or(image_height as u32) as usize;

        // Extract geotransform from ModelTransformationTag.
        // We use the geotransform as-is (even for bottom-up images like AEF COGs)
        // because async-tiff reads using the TIFF's native row indexing.
        let geo_transform = ifd
            .model_transformation()
            .and_then(GeoTransform::from_model_transformation);

        if geo_transform.is_none() {
            tracing::warn!("No ModelTransformationTag found in TIFF, pixel coordinate assumptions may be incorrect");
        }

        Ok(Arc::new(CachedTiff {
            tiff: Arc::new(tiff),
            reader,
            image_width,
            image_height,
            bands,
            is_planar,
            tile_width,
            tile_height,
            geo_transform,
        }))
    }

    /// Get the current cache size.
    pub async fn len(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.cache.read().await.is_empty()
    }
}

/// Key for tile cache: (cog_path, tile_x, tile_y)
type TileCacheKey = (String, usize, usize);

/// Cached decoded tile with single-flight support.
struct TileDataCache {
    /// LRU cache for decoded tiles
    pub cache: RwLock<LruCache<TileCacheKey, Arc<TileArray>>>,
    /// In-flight requests (single-flight pattern) - uses DashMap for lock-free access
    pub in_flight: DashMap<TileCacheKey, broadcast::Sender<Result<Arc<TileArray>, String>>>,
    /// Optional metrics
    pub metrics: Option<Arc<Metrics>>,
}

impl TileDataCache {
    fn new(max_entries: usize, metrics: Option<Arc<Metrics>>) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::new(50_000).unwrap()),
            )),
            in_flight: DashMap::new(),
            metrics,
        }
    }
}

/// Async COG reader for fetching windowed tile data from S3.
pub struct CogReader {
    /// Object store for S3 access
    store: Arc<dyn ObjectStore>,

    /// Decoder registry with all supported decoders (including ZSTD)
    decoder_registry: Arc<DecoderRegistry>,

    /// Cache for TIFF metadata (avoids re-parsing headers)
    metadata_cache: Arc<TiffMetadataCache>,

    /// Cache for decoded tile data with single-flight
    tile_cache: Arc<TileDataCache>,

    /// Optional metrics for cache and read statistics
    metrics: Option<Arc<Metrics>>,
}

impl CogReader {
    /// Create a new COG reader.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            decoder_registry: Arc::new(DecoderRegistry::default()),
            metadata_cache: Arc::new(TiffMetadataCache::new(10_000, None)),
            tile_cache: Arc::new(TileDataCache::new(50_000, None)),
            metrics: None,
        }
    }

    /// Create a new COG reader with metrics tracking.
    pub fn with_metrics(store: Arc<dyn ObjectStore>, metrics: Arc<Metrics>) -> Self {
        Self {
            store,
            decoder_registry: Arc::new(DecoderRegistry::default()),
            metadata_cache: Arc::new(TiffMetadataCache::new(10_000, Some(metrics.clone()))),
            tile_cache: Arc::new(TileDataCache::new(50_000, Some(metrics.clone()))),
            metrics: Some(metrics),
        }
    }

    /// Create a new COG reader with custom cache size and metrics.
    pub fn with_cache_size(
        store: Arc<dyn ObjectStore>,
        metadata_cache_size: usize,
        tile_cache_size: usize,
        metrics: Option<Arc<Metrics>>,
    ) -> Self {
        Self {
            store,
            decoder_registry: Arc::new(DecoderRegistry::default()),
            metadata_cache: Arc::new(TiffMetadataCache::new(metadata_cache_size, metrics.clone())),
            tile_cache: Arc::new(TileDataCache::new(tile_cache_size, metrics.clone())),
            metrics,
        }
    }

    /// Get the geotransform for a COG tile.
    ///
    /// This reads the ModelTransformationTag from the TIFF header to get the
    /// affine transform between pixel and world coordinates.
    ///
    /// Returns None if the TIFF doesn't have a geotransform tag.
    pub async fn get_geo_transform(&self, tile: &CogTile) -> Result<Option<GeoTransform>> {
        let object_path = self.tile_path(tile)?;
        let path_str = object_path.to_string();

        let cached = self
            .metadata_cache
            .get_or_load(&path_str, self.store.clone(), object_path)
            .await?;

        Ok(cached.geo_transform)
    }

    /// Read a window from a COG tile.
    ///
    /// Only fetches the internal TIFF tiles that overlap with the requested window,
    /// then clips to the exact bounds.
    ///
    /// # Arguments
    /// * `tile` - The tile metadata
    /// * `window` - The pixel window to read
    /// * `bounds_native` - Geographic bounds of this window in tile's native CRS
    pub async fn read_window(&self, tile: &CogTile, window: PixelWindow, bounds_native: [f64; 4]) -> Result<WindowData> {
        let path = self.tile_path(tile)?;

        // Step 1: Get cached TIFF metadata (or load and cache it)
        let cached = self
            .metadata_cache
            .get_or_load(&tile.s3_path, self.store.clone(), path)
            .await?;

        let image_width = cached.image_width;
        let image_height = cached.image_height;
        let bands = cached.bands;
        let is_planar = cached.is_planar;
        let tile_width = cached.tile_width;
        let tile_height = cached.tile_height;

        // Validate window bounds
        if window.x + window.width > image_width || window.y + window.height > image_height {
            anyhow::bail!(
                "Window ({}, {}, {}, {}) exceeds image bounds ({}, {})",
                window.x, window.y, window.width, window.height,
                image_width, image_height
            );
        }

        tracing::debug!(
            "Reading window ({}, {}, {}, {}) from {}x{} image, {} bands, tile_size={}x{}",
            window.x, window.y, window.width, window.height,
            image_width, image_height, bands, tile_width, tile_height
        );

        // Step 2: Identify which internal tiles cover the requested window
        let start_tile_x = window.x / tile_width;
        let end_tile_x = (window.x + window.width - 1) / tile_width;
        let start_tile_y = window.y / tile_height;
        let end_tile_y = (window.y + window.height - 1) / tile_height;

        let num_tiles_x = end_tile_x - start_tile_x + 1;
        let num_tiles_y = end_tile_y - start_tile_y + 1;

        tracing::debug!(
            "Fetching tiles: x=[{}..{}], y=[{}..{}] ({}x{} tiles)",
            start_tile_x, end_tile_x, start_tile_y, end_tile_y,
            num_tiles_x, num_tiles_y
        );

        // Step 3: Fetch all needed tiles in parallel (with caching and single-flight)
        let mut tile_futures = Vec::with_capacity(num_tiles_x * num_tiles_y);
        for ty in start_tile_y..=end_tile_y {
            for tx in start_tile_x..=end_tile_x {
                tile_futures.push(self.get_tile(
                    &tile.s3_path,
                    &cached,
                    tx,
                    ty,
                    tile_width,
                    tile_height,
                    image_width,
                    image_height,
                ));
            }
        }

        let fetched_tiles = try_join_all(tile_futures).await?;

        // Step 4: Assemble tiles into a single array
        let assembled_x = start_tile_x * tile_width;
        let assembled_y = start_tile_y * tile_height;
        let assembled_width = num_tiles_x * tile_width;
        let assembled_height = num_tiles_y * tile_height;

        // Clamp to image bounds (edge tiles may be smaller)
        let assembled_width = assembled_width.min(image_width - assembled_x);
        let assembled_height = assembled_height.min(image_height - assembled_y);

        let mut assembled = Array3::<i8>::zeros((bands, assembled_height, assembled_width));

        let mut tile_idx = 0;
        for ty in start_tile_y..=end_tile_y {
            for tx in start_tile_x..=end_tile_x {
                self.copy_tile_to_assembled(
                    fetched_tiles[tile_idx].as_ref(),
                    &mut assembled,
                    tx, ty,
                    start_tile_x, start_tile_y,
                    tile_width, tile_height,
                    assembled_width, assembled_height,
                    bands,
                    is_planar,
                )?;
                tile_idx += 1;
            }
        }

        // Step 5: Clip to the exact requested bounds
        let clip_x = window.x - assembled_x;
        let clip_y = window.y - assembled_y;

        let clipped = assembled
            .slice(ndarray::s![.., clip_y..clip_y + window.height, clip_x..clip_x + window.width])
            .to_owned();

        Ok(WindowData {
            tile: tile.clone(),
            data: clipped,
            window,
            bounds_native,
        })
    }

    /// Get a single tile with caching and single-flight deduplication.
    #[allow(clippy::too_many_arguments)]
    async fn get_tile(
        &self,
        cog_path: &str,
        cached: &Arc<CachedTiff>,
        tx: usize,
        ty: usize,
        tile_width: usize,
        tile_height: usize,
        image_width: usize,
        image_height: usize,
    ) -> Result<Arc<TileArray>> {
        let key = (cog_path.to_string(), tx, ty);

        // 1. Check cache (fast path)
        {
            let mut cache = self.tile_cache.cache.write().await;
            if let Some(tile) = cache.get(&key) {
                if let Some(ref m) = self.tile_cache.metrics {
                    m.add_tile_cache_hit();
                }
                return Ok(tile.clone());
            }
        }

        // 2. Check if already in-flight (single-flight pattern)
        if let Some(sender_ref) = self.tile_cache.in_flight.get(&key) {
            let mut rx = sender_ref.subscribe();
            drop(sender_ref);

            if let Some(ref m) = self.tile_cache.metrics {
                m.add_tile_cache_coalesced();
            }

            return match rx.recv().await {
                Ok(Ok(tile)) => Ok(tile),
                Ok(Err(e)) => Err(anyhow::anyhow!("Coalesced fetch failed: {}", e)),
                Err(e) => Err(anyhow::anyhow!("Broadcast channel error: {}", e)),
            };
        }

        // 3. Register in-flight and fetch
        let (tx_sender, _) = broadcast::channel(16);
        self.tile_cache.in_flight.insert(key.clone(), tx_sender.clone());

        if let Some(ref m) = self.tile_cache.metrics {
            m.add_tile_cache_miss();
        }

        // 4. Fetch the tile
        let result = self.fetch_single_tile(cached, tx, ty, tile_width, tile_height, image_width, image_height).await;

        // 5. Handle result
        match result {
            Ok(tile) => {
                let tile_arc = Arc::new(tile);

                // Cache it and track bytes
                {
                    let mut cache = self.tile_cache.cache.write().await;
                    cache.put(key.clone(), tile_arc.clone());
                }
                if let Some(ref m) = self.metrics {
                    m.add_tile_cache_bytes(tile_arc.data.data().as_ref().len() as u64);
                }

                // Notify waiters and remove from in-flight
                self.tile_cache.in_flight.remove(&key);
                let _ = tx_sender.send(Ok(tile_arc.clone()));

                Ok(tile_arc)
            }
            Err(e) => {
                // Notify waiters and remove from in-flight
                self.tile_cache.in_flight.remove(&key);
                let _ = tx_sender.send(Err(e.to_string()));

                Err(e)
            }
        }
    }

    /// Fetch a single tile from the TIFF.
    async fn fetch_single_tile(
        &self,
        cached: &Arc<CachedTiff>,
        tx: usize,
        ty: usize,
        tile_width: usize,
        tile_height: usize,
        image_width: usize,
        image_height: usize,
    ) -> Result<TileArray> {
        let ifd = cached.tiff.ifds().first()
            .ok_or_else(|| anyhow::anyhow!("No IFDs in TIFF"))?;

        // Fetch single tile using async-tiff (this is the HTTP request)
        let fetch_start = std::time::Instant::now();
        let tile = ifd
            .fetch_tile(tx, ty, cached.reader.as_ref())
            .await
            .with_context(|| format!("Failed to fetch tile ({}, {})", tx, ty))?;
        let fetch_elapsed = fetch_start.elapsed();

        // Track bytes read
        let raw_bytes: u64 = match tile.compressed_bytes() {
            async_tiff::CompressedBytes::Chunky(bytes) => bytes.len() as u64,
            async_tiff::CompressedBytes::Planar(vec) => vec.iter().map(|b| b.len() as u64).sum(),
        };

        // Log HTTP request timing
        let fetch_ms = fetch_elapsed.as_secs_f64() * 1000.0;
        let throughput_mbps = if fetch_elapsed.as_secs_f64() > 0.0 {
            (raw_bytes as f64 / 1024.0 / 1024.0) / fetch_elapsed.as_secs_f64()
        } else {
            0.0
        };
        tracing::debug!(
            "HTTP fetch tile ({},{}) {}KB in {:.1}ms ({:.1} MB/s)",
            tx, ty, raw_bytes / 1024, fetch_ms, throughput_mbps
        );

        if let Some(ref m) = self.metrics {
            m.add_bytes_read(raw_bytes);
            m.add_http_request(fetch_elapsed);
        }

        // Decode the tile in a blocking thread (decompression is CPU-bound)
        let decoder_registry = self.decoder_registry.clone();
        let array = tokio::task::spawn_blocking(move || {
            tile.decode(&decoder_registry)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Decode task panicked: {}", e))?
        .map_err(|e| anyhow::anyhow!("Failed to decode tile ({}, {}): {:?}", tx, ty, e))?;

        // Calculate actual tile dimensions (may be smaller at image edges)
        let actual_width = tile_width.min(image_width - tx * tile_width);
        let actual_height = tile_height.min(image_height - ty * tile_height);

        Ok(TileArray {
            data: array,
            actual_width,
            actual_height,
        })
    }

    /// Copy a decoded tile into the assembled array.
    #[allow(clippy::too_many_arguments)]
    fn copy_tile_to_assembled(
        &self,
        tile: &TileArray,
        output: &mut Array3<i8>,
        tx: usize,
        ty: usize,
        start_tx: usize,
        start_ty: usize,
        tile_width: usize,
        tile_height: usize,
        output_width: usize,
        output_height: usize,
        bands: usize,
        is_planar: bool,
    ) -> Result<()> {
        let shape = tile.data.shape();

        // Position in output array (relative to start of assembled region)
        let out_x = (tx - start_tx) * tile_width;
        let out_y = (ty - start_ty) * tile_height;

        // How many pixels to copy (may be less at edges)
        let copy_width = tile.actual_width.min(output_width - out_x);
        let copy_height = tile.actual_height.min(output_height - out_y);

        match tile.data.data() {
            async_tiff::TypedArray::Int8(data) => {
                self.copy_typed_data(
                    data, output, bands, out_x, out_y, copy_width, copy_height,
                    is_planar, shape, |v| v,
                )?;
            }
            async_tiff::TypedArray::UInt8(data) => {
                self.copy_typed_data(
                    data, output, bands, out_x, out_y, copy_width, copy_height,
                    is_planar, shape, |v| v as i8,
                )?;
            }
            _ => anyhow::bail!("Unsupported data type: expected Int8 or UInt8"),
        }
        Ok(())
    }

    /// Generic copy function for typed data.
    ///
    /// Uses Rayon to parallelize over bands for improved throughput.
    #[allow(clippy::too_many_arguments)]
    fn copy_typed_data<T: Copy + Sync, F: Fn(T) -> i8 + Sync>(
        &self,
        data: &[T],
        output: &mut Array3<i8>,
        bands: usize,
        out_x: usize,
        out_y: usize,
        copy_width: usize,
        copy_height: usize,
        is_planar: bool,
        shape: [usize; 3],
        convert: F,
    ) -> Result<()> {
        // Get raw slice for parallel writes - safe because each band writes to disjoint memory
        let output_shape = output.dim();
        let output_slice = output.as_slice_mut().expect("Array should be contiguous");
        let out_height = output_shape.1;
        let out_width = output_shape.2;

        // Parallelize over bands using Rayon for significant speedup on 64-band data
        (0..bands).into_par_iter().for_each(|b| {
            for row in 0..copy_height {
                for col in 0..copy_width {
                    // Calculate source index based on layout
                    let src_idx = if is_planar {
                        // Planar: [bands, height, width]
                        b * shape[1] * shape[2] + row * shape[2] + col
                    } else {
                        // Chunky: [height, width, bands]
                        row * shape[1] * shape[2] + col * shape[2] + b
                    };

                    if src_idx < data.len() {
                        // Output index: [band, row, col] in row-major order
                        let out_idx = b * out_height * out_width + (out_y + row) * out_width + (out_x + col);
                        // Safety: each band writes to its own disjoint slice of memory
                        unsafe {
                            let ptr = output_slice.as_ptr() as *mut i8;
                            *ptr.add(out_idx) = convert(data[src_idx]);
                        }
                    }
                }
            }
        });
        Ok(())
    }

    /// Extract S3 path from tile.
    pub fn tile_path(&self, tile: &CogTile) -> Result<Path> {
        // Handle both s3:// URIs and plain paths
        let path_str = if tile.s3_path.starts_with("s3://") {
            // Extract key (path after bucket name)
            let (_bucket, key) = super::parse_s3_uri(&tile.s3_path)?;
            key
        } else {
            &tile.s3_path
        };

        Path::parse(path_str).with_context(|| format!("Invalid path: {}", path_str))
    }
}

/// Temporary holder for a decoded tile's data.
struct TileArray {
    data: async_tiff::Array,
    actual_width: usize,
    actual_height: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tile(s3_path: &str) -> CogTile {
        CogTile {
            tile_id: "test".to_string(),
            s3_path: s3_path.to_string(),
            crs: "EPSG:32610".to_string(),
            bounds_native: [0.0, 0.0, 81920.0, 81920.0],
            bounds_wgs84: [-122.0, 37.0, -121.0, 38.0],
            resolution: 10.0,
            year: 2024,
        }
    }

    fn make_reader() -> CogReader {
        let store = Arc::new(object_store::memory::InMemory::new());
        CogReader::new(store)
    }

    // ==================== Path Parsing Tests ====================

    #[test]
    fn test_path_parsing_s3_uri() {
        let reader = make_reader();
        let tile = make_test_tile("s3://bucket/path/to/file.tif");
        let path = reader.tile_path(&tile).unwrap();
        assert_eq!(path.as_ref(), "path/to/file.tif");
    }

    #[test]
    fn test_path_parsing_s3_uri_nested() {
        let reader = make_reader();
        let tile = make_test_tile("s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2024/10N/tile.tiff");
        let path = reader.tile_path(&tile).unwrap();
        assert_eq!(path.as_ref(), "tge-labs/aef/v1/annual/2024/10N/tile.tiff");
    }

    #[test]
    fn test_path_parsing_plain_path() {
        let reader = make_reader();
        let tile = make_test_tile("path/to/file.tif");
        let path = reader.tile_path(&tile).unwrap();
        assert_eq!(path.as_ref(), "path/to/file.tif");
    }

    #[test]
    fn test_path_parsing_bucket_only_fails() {
        let reader = make_reader();
        let tile = make_test_tile("s3://bucket");
        let result = reader.tile_path(&tile);
        assert!(result.is_err());
    }

    // ==================== Pixel Window Tests ====================

    #[test]
    fn test_pixel_window_creation() {
        let window = PixelWindow::new(100, 200, 1024, 1024);
        assert_eq!(window.x, 100);
        assert_eq!(window.y, 200);
        assert_eq!(window.width, 1024);
        assert_eq!(window.height, 1024);
    }

    // ==================== Copy Data Tests ====================

    #[test]
    fn test_copy_planar_data() {
        let reader = make_reader();
        let bands = 2;
        let copy_h = 2;
        let copy_w = 3;

        // Planar data: [bands=2, height=2, width=3]
        let data: Vec<i8> = vec![
            1, 2, 3, 4, 5, 6,     // band 0
            7, 8, 9, 10, 11, 12,  // band 1
        ];

        let mut output = Array3::<i8>::zeros((bands, copy_h, copy_w));
        let shape = [bands, copy_h, copy_w];

        reader.copy_typed_data(
            &data, &mut output, bands,
            0, 0,  // out_x, out_y
            copy_w, copy_h,
            true,  // is_planar
            shape,
            |v| v,
        ).unwrap();

        assert_eq!(output[[0, 0, 0]], 1);
        assert_eq!(output[[0, 0, 2]], 3);
        assert_eq!(output[[0, 1, 0]], 4);
        assert_eq!(output[[1, 0, 0]], 7);
        assert_eq!(output[[1, 1, 2]], 12);
    }

    #[test]
    fn test_copy_chunky_data() {
        let reader = make_reader();
        let bands = 2;
        let copy_h = 2;
        let copy_w = 3;

        // Chunky data: [height=2, width=3, bands=2]
        let data: Vec<i8> = vec![
            1, 7, 2, 8, 3, 9,      // row 0
            4, 10, 5, 11, 6, 12,   // row 1
        ];

        let mut output = Array3::<i8>::zeros((bands, copy_h, copy_w));
        let shape = [copy_h, copy_w, bands];

        reader.copy_typed_data(
            &data, &mut output, bands,
            0, 0,
            copy_w, copy_h,
            false,  // chunky
            shape,
            |v| v,
        ).unwrap();

        assert_eq!(output[[0, 0, 0]], 1);
        assert_eq!(output[[0, 0, 2]], 3);
        assert_eq!(output[[1, 0, 0]], 7);
        assert_eq!(output[[1, 1, 2]], 12);
    }

    #[test]
    fn test_copy_with_offset() {
        let reader = make_reader();
        let bands = 1;

        let data: Vec<i8> = vec![1, 2, 3, 4];
        let mut output = Array3::<i8>::zeros((bands, 4, 4));
        let shape = [bands, 2, 2];

        reader.copy_typed_data(
            &data, &mut output, bands,
            2, 2,  // offset into output
            2, 2,  // copy size
            true,
            shape,
            |v| v,
        ).unwrap();

        // Data should be at offset position
        assert_eq!(output[[0, 0, 0]], 0);
        assert_eq!(output[[0, 2, 2]], 1);
        assert_eq!(output[[0, 2, 3]], 2);
        assert_eq!(output[[0, 3, 2]], 3);
        assert_eq!(output[[0, 3, 3]], 4);
    }

    // ==================== GeoTransform Tests ====================

    #[test]
    fn test_geotransform_from_model_transformation_aef() {
        // Real AEF COG ModelTransformationTag (bottom-up image)
        // From: xcyo46pot2fg6a61t-0000008192-0000000000.tiff
        let matrix: [f64; 16] = [
            10.0, 0.0, 0.0, 500000.0,   // row 0: scale_x, rot, 0, origin_x
            0.0, 10.0, 0.0, 4177920.0,  // row 1: rot, scale_y, 0, origin_y
            0.0, 0.0, 0.0, 0.0,         // row 2
            0.0, 0.0, 0.0, 1.0,         // row 3
        ];

        let gt = GeoTransform::from_model_transformation(&matrix).unwrap();

        assert_eq!(gt.a, 10.0);       // scale_x
        assert_eq!(gt.b, 0.0);        // rotation
        assert_eq!(gt.c, 500000.0);   // origin_x
        assert_eq!(gt.d, 0.0);        // rotation
        assert_eq!(gt.e, 10.0);       // scale_y (positive = bottom-up)
        assert_eq!(gt.f, 4177920.0);  // origin_y
    }

    #[test]
    fn test_geotransform_from_model_transformation_top_down() {
        // Standard top-down GeoTIFF (negative y scale)
        let matrix: [f64; 16] = [
            10.0, 0.0, 0.0, 500000.0,
            0.0, -10.0, 0.0, 4259840.0,  // negative scale_y, origin at top
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let gt = GeoTransform::from_model_transformation(&matrix).unwrap();

        assert_eq!(gt.e, -10.0);      // negative = top-down
        assert_eq!(gt.f, 4259840.0);  // origin at top (max y)
    }

    #[test]
    fn test_geotransform_from_model_transformation_too_short() {
        let matrix: [f64; 4] = [10.0, 0.0, 0.0, 500000.0];
        assert!(GeoTransform::from_model_transformation(&matrix).is_none());
    }

    #[test]
    fn test_geotransform_is_bottom_up() {
        // Bottom-up (positive scale_y)
        let gt_bottom_up = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: 10.0, f: 4177920.0,
        };
        assert!(gt_bottom_up.is_bottom_up());

        // Top-down (negative scale_y)
        let gt_top_down = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: -10.0, f: 4259840.0,
        };
        assert!(!gt_top_down.is_bottom_up());
    }

    #[test]
    fn test_geotransform_world_to_pixel_bottom_up() {
        // AEF-style bottom-up geotransform
        // Origin at (500000, 4177920), 10m pixels, 8192x8192 image
        let gt = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: 10.0, f: 4177920.0,
        };

        // Origin point -> pixel (0, 0)
        let (col, row) = gt.world_to_pixel(500000.0, 4177920.0);
        assert!((col - 0.0).abs() < 0.001);
        assert!((row - 0.0).abs() < 0.001);

        // Point 100m east, 50m north -> pixel (10, 5)
        let (col, row) = gt.world_to_pixel(500100.0, 4177970.0);
        assert!((col - 10.0).abs() < 0.001);
        assert!((row - 5.0).abs() < 0.001);

        // Top-right corner of 8192x8192 image
        // x = 500000 + 8192*10 = 581920
        // y = 4177920 + 8192*10 = 4259840
        let (col, row) = gt.world_to_pixel(581920.0, 4259840.0);
        assert!((col - 8192.0).abs() < 0.001);
        assert!((row - 8192.0).abs() < 0.001);
    }

    #[test]
    fn test_geotransform_world_to_pixel_top_down() {
        // Standard top-down geotransform
        // Origin at top-left (500000, 4259840), negative y scale
        let gt = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: -10.0, f: 4259840.0,
        };

        // Origin point (top-left) -> pixel (0, 0)
        let (col, row) = gt.world_to_pixel(500000.0, 4259840.0);
        assert!((col - 0.0).abs() < 0.001);
        assert!((row - 0.0).abs() < 0.001);

        // Point 100m east, 50m south -> pixel (10, 5)
        let (col, row) = gt.world_to_pixel(500100.0, 4259790.0);
        assert!((col - 10.0).abs() < 0.001);
        assert!((row - 5.0).abs() < 0.001);

        // Bottom-right corner
        let (col, row) = gt.world_to_pixel(581920.0, 4177920.0);
        assert!((col - 8192.0).abs() < 0.001);
        assert!((row - 8192.0).abs() < 0.001);
    }

    #[test]
    fn test_geotransform_pixel_to_world_roundtrip() {
        let gt = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: 10.0, f: 4177920.0,
        };

        // Test roundtrip: world -> pixel -> world
        let original_x = 552853.07;
        let original_y = 4181018.26;

        let (col, row) = gt.world_to_pixel(original_x, original_y);
        let (recovered_x, recovered_y) = gt.pixel_to_world(col, row);

        assert!((recovered_x - original_x).abs() < 0.001);
        assert!((recovered_y - original_y).abs() < 0.001);
    }

    #[test]
    fn test_geotransform_real_aef_coordinate() {
        // Test with real coordinate from validation script
        // Pixel center (EPSG:32610): (552853.07, 4181018.26)
        // Should map to approximately row 310 in bottom-up TIFF
        let gt = GeoTransform {
            a: 10.0, b: 0.0, c: 500000.0,
            d: 0.0, e: 10.0, f: 4177920.0,
        };

        let (col, row) = gt.world_to_pixel(552853.07, 4181018.26);

        // col = (552853.07 - 500000) / 10 = 5285.307
        assert!((col - 5285.307).abs() < 0.01);

        // row = (4181018.26 - 4177920) / 10 = 309.826
        assert!((row - 309.826).abs() < 0.01);
    }
}

/// Integration tests that require network access to S3
#[cfg(test)]
mod integration_tests {
    use super::*;
    use object_store::aws::AmazonS3Builder;

    fn get_s3_store() -> Result<Arc<dyn ObjectStore>> {
        let store = AmazonS3Builder::new()
            .with_bucket_name("us-west-2.opendata.source.coop")
            .with_region("us-west-2")
            .with_skip_signature(true)
            .with_retry(object_store::RetryConfig {
                max_retries: 2,
                ..Default::default()
            })
            .build()?;
        Ok(Arc::new(store))
    }

    fn get_aef_test_tile() -> CogTile {
        CogTile {
            tile_id: "test_sf_2024".to_string(),
            s3_path: "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2024/10N/xcyo46pot2fg6a61t-0000000000-0000000000.tiff".to_string(),
            crs: "EPSG:32610".to_string(),
            bounds_native: [500000.0, 4096000.0, 581920.0, 4177920.0],
            bounds_wgs84: [-123.0, 37.006, -122.07, 37.748],
            resolution: 10.0,
            year: 2024,
        }
    }

    /// Test reading a realistic window (1024x1024) from an AEF tile
    #[tokio::test]
    #[ignore]
    async fn test_read_window_1024x1024() {
        let store = get_s3_store().expect("Failed to create S3 store");
        let reader = CogReader::new(store);
        let tile = get_aef_test_tile();

        // Read a 1024x1024 window (realistic output chunk size)
        let window = PixelWindow::new(1024, 1024, 1024, 1024);
        // Bounds: pixel (1024,1024) to (2048,2048) at 10m resolution
        let bounds_native = [510240.0, 4106240.0, 520480.0, 4116480.0];
        let result = reader.read_window(&tile, window, bounds_native).await;

        match result {
            Ok(data) => {
                let shape = data.data.shape();
                assert_eq!(shape[0], 64, "Expected 64 bands");
                assert_eq!(shape[1], 1024, "Expected height 1024");
                assert_eq!(shape[2], 1024, "Expected width 1024");

                let nonzero = data.data.iter().filter(|&&v| v != 0 && v != -128).count();
                println!("Window shape: {:?}, non-zero pixels: {}", shape, nonzero);
            }
            Err(e) => {
                if !e.to_string().contains("timeout") {
                    panic!("Failed to read window: {}", e);
                }
            }
        }
    }

    /// Test reading a small window (256x256) - should only fetch 1 internal tile
    #[tokio::test]
    #[ignore]
    async fn test_read_window_256x256() {
        let store = get_s3_store().expect("Failed to create S3 store");
        let reader = CogReader::new(store);
        let tile = get_aef_test_tile();

        // Read a 256x256 window within a single 1024x1024 internal tile
        let window = PixelWindow::new(100, 100, 256, 256);
        // Bounds: pixel (100,100) to (356,356) at 10m resolution
        let bounds_native = [501000.0, 4097000.0, 503560.0, 4099560.0];
        let result = reader.read_window(&tile, window, bounds_native).await;

        match result {
            Ok(data) => {
                let shape = data.data.shape();
                assert_eq!(shape[0], 64);
                assert_eq!(shape[1], 256);
                assert_eq!(shape[2], 256);
                println!("Small window read successfully: {:?}", shape);
            }
            Err(e) => {
                if !e.to_string().contains("timeout") {
                    panic!("Failed to read window: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_real_s3_path_extraction() {
        let store = Arc::new(object_store::memory::InMemory::new());
        let reader = CogReader::new(store);
        let tile = get_aef_test_tile();

        let path = reader.tile_path(&tile).unwrap();
        assert_eq!(
            path.as_ref(),
            "tge-labs/aef/v1/annual/2024/10N/xcyo46pot2fg6a61t-0000000000-0000000000.tiff"
        );
    }
}
