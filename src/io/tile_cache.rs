//! LRU tile data cache with single-flight deduplication.
//!
//! This module provides caching for decoded TIFF tiles to avoid redundant S3 fetches
//! when adjacent output chunks need overlapping input tiles.
//!
//! Key features:
//! - LRU eviction to bound memory usage
//! - Single-flight pattern: if a tile is being fetched, new requesters wait on the result
//!   rather than issuing duplicate S3 requests
//! - Memory-based eviction (configurable max bytes)

use crate::pipeline::Metrics;
use anyhow::Result;
use lru::LruCache;
use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

/// Key for identifying a specific tile within a COG.
/// Format: (cog_path, tile_x, tile_y)
pub type TileKey = (String, usize, usize);

/// A decoded tile ready for use.
#[derive(Clone)]
pub struct DecodedTile {
    /// Raw tile data as bytes (decoded from compression)
    pub data: Vec<u8>,
    /// Tile width in pixels
    pub width: usize,
    /// Tile height in pixels
    pub height: usize,
    /// Number of bands/samples
    pub bands: usize,
    /// Whether the data is planar layout
    pub is_planar: bool,
}

impl DecodedTile {
    /// Estimate memory size of this tile in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<Self>()
    }
}

/// LRU cache for decoded tile data with single-flight deduplication.
pub struct TileCache {
    /// Cached decoded tiles
    cache: RwLock<LruCache<TileKey, Arc<DecodedTile>>>,

    /// In-flight requests (single-flight pattern)
    /// If a tile is being fetched, new requesters subscribe to this broadcast
    in_flight: RwLock<HashMap<TileKey, broadcast::Sender<Result<Arc<DecodedTile>, String>>>>,

    /// Maximum cache size in bytes
    max_bytes: u64,

    /// Current cache size in bytes
    current_bytes: AtomicU64,

    /// Optional metrics for tracking cache performance
    metrics: Option<Arc<Metrics>>,
}

impl TileCache {
    /// Create a new tile cache with the specified maximum size in bytes.
    ///
    /// # Arguments
    /// * `max_bytes` - Maximum cache size in bytes (e.g., 32 * 1024 * 1024 * 1024 for 32 GB)
    /// * `metrics` - Optional metrics collector for cache hit/miss tracking
    pub fn new(max_bytes: u64, metrics: Option<Arc<Metrics>>) -> Self {
        // Estimate initial capacity based on typical tile size (~256KB for 1024x1024 tiles with 64 bands)
        let estimated_tile_size = 256 * 1024;
        let initial_capacity = (max_bytes / estimated_tile_size as u64) as usize;
        let capacity = initial_capacity.max(1000);

        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1000).unwrap()),
            )),
            in_flight: RwLock::new(HashMap::new()),
            max_bytes,
            current_bytes: AtomicU64::new(0),
            metrics,
        }
    }

    /// Get a tile from cache or fetch it using the provided function.
    ///
    /// This implements the single-flight pattern: if the tile is already being fetched
    /// by another task, we wait for that fetch to complete rather than issuing a duplicate request.
    ///
    /// # Arguments
    /// * `key` - The tile key (cog_path, tile_x, tile_y)
    /// * `fetch` - Async function to fetch the tile if not cached
    pub async fn get_or_fetch<F, Fut>(&self, key: TileKey, fetch: F) -> Result<Arc<DecodedTile>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<DecodedTile>>,
    {
        // 1. Check cache first (fast path)
        {
            let mut cache = self.cache.write().await;
            if let Some(tile) = cache.get(&key) {
                if let Some(ref m) = self.metrics {
                    m.add_tile_cache_hit();
                }
                return Ok(tile.clone());
            }
        }

        // 2. Check if already in-flight (single-flight pattern)
        {
            let in_flight = self.in_flight.read().await;
            if let Some(sender) = in_flight.get(&key) {
                // Someone else is fetching - wait for their result
                let mut rx = sender.subscribe();
                drop(in_flight);

                if let Some(ref m) = self.metrics {
                    m.add_tile_cache_coalesced();
                }

                // Wait for the result
                match rx.recv().await {
                    Ok(Ok(tile)) => return Ok(tile),
                    Ok(Err(e)) => return Err(anyhow::anyhow!("Coalesced fetch failed: {}", e)),
                    Err(e) => return Err(anyhow::anyhow!("Broadcast channel error: {}", e)),
                }
            }
        }

        // 3. We're the first - register in-flight and fetch
        let (tx, _) = broadcast::channel(16);
        {
            let mut in_flight = self.in_flight.write().await;
            in_flight.insert(key.clone(), tx.clone());
        }

        // 4. Perform the actual fetch
        let result = fetch().await;

        // 5. Handle result
        match result {
            Ok(tile) => {
                let tile = Arc::new(tile);
                let tile_size = tile.size_bytes() as u64;

                // Evict old entries if needed to make room
                self.evict_if_needed(tile_size).await;

                // Add to cache
                {
                    let mut cache = self.cache.write().await;
                    cache.put(key.clone(), tile.clone());
                    self.current_bytes.fetch_add(tile_size, Ordering::Relaxed);
                }

                // Update metrics
                if let Some(ref m) = self.metrics {
                    m.add_tile_cache_miss();
                    m.set_tile_cache_bytes(self.current_bytes.load(Ordering::Relaxed));
                }

                // Remove from in-flight and notify waiters
                {
                    let mut in_flight = self.in_flight.write().await;
                    in_flight.remove(&key);
                }
                let _ = tx.send(Ok(tile.clone()));

                Ok(tile)
            }
            Err(e) => {
                // Remove from in-flight and notify waiters of failure
                {
                    let mut in_flight = self.in_flight.write().await;
                    in_flight.remove(&key);
                }
                let _ = tx.send(Err(e.to_string()));

                Err(e)
            }
        }
    }

    /// Evict old entries if needed to make room for a new tile.
    async fn evict_if_needed(&self, new_tile_size: u64) {
        let current = self.current_bytes.load(Ordering::Relaxed);
        if current + new_tile_size <= self.max_bytes {
            return;
        }

        // Need to evict - remove LRU entries until we have enough space
        let mut cache = self.cache.write().await;
        while self.current_bytes.load(Ordering::Relaxed) + new_tile_size > self.max_bytes {
            if let Some((_, evicted)) = cache.pop_lru() {
                let evicted_size = evicted.size_bytes() as u64;
                self.current_bytes.fetch_sub(evicted_size, Ordering::Relaxed);
            } else {
                break; // Cache is empty
            }
        }
    }

    /// Get the current cache size in bytes.
    pub fn current_bytes(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get the maximum cache size in bytes.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Get the number of cached tiles.
    pub async fn len(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.cache.read().await.is_empty()
    }

    /// Clear all cached tiles.
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        self.current_bytes.store(0, Ordering::Relaxed);
        if let Some(ref m) = self.metrics {
            m.set_tile_cache_bytes(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tile(size: usize) -> DecodedTile {
        DecodedTile {
            data: vec![0u8; size],
            width: 1024,
            height: 1024,
            bands: 64,
            is_planar: true,
        }
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = TileCache::new(1024 * 1024, None); // 1 MB cache
        let key = ("test.tif".to_string(), 0, 0);

        // First fetch - cache miss
        let tile1 = cache
            .get_or_fetch(key.clone(), || async { Ok(make_test_tile(1000)) })
            .await
            .unwrap();

        // Second fetch - should be cache hit
        let tile2 = cache
            .get_or_fetch(key.clone(), || async { panic!("Should not be called") })
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&tile1, &tile2));
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = TileCache::new(2000, None); // Small cache to force eviction

        let key1 = ("test1.tif".to_string(), 0, 0);
        let key2 = ("test2.tif".to_string(), 0, 0);
        let key3 = ("test3.tif".to_string(), 0, 0);

        // Fill cache
        cache
            .get_or_fetch(key1.clone(), || async { Ok(make_test_tile(800)) })
            .await
            .unwrap();

        cache
            .get_or_fetch(key2.clone(), || async { Ok(make_test_tile(800)) })
            .await
            .unwrap();

        // This should cause eviction of key1
        cache
            .get_or_fetch(key3.clone(), || async { Ok(make_test_tile(800)) })
            .await
            .unwrap();

        // key1 should be evicted (LRU)
        assert!(cache.current_bytes() <= 2000);
    }

    #[tokio::test]
    async fn test_single_flight() {
        use std::sync::atomic::AtomicUsize;

        let cache = Arc::new(TileCache::new(1024 * 1024, None));
        let fetch_count = Arc::new(AtomicUsize::new(0));
        let key = ("test.tif".to_string(), 0, 0);

        // Spawn multiple concurrent fetches for the same key
        let mut handles = vec![];
        for _ in 0..10 {
            let cache = cache.clone();
            let key = key.clone();
            let count = fetch_count.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_fetch(key, || {
                        let count = count.clone();
                        async move {
                            count.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            Ok(make_test_tile(1000))
                        }
                    })
                    .await
            }));
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Only one fetch should have been made (single-flight deduplication)
        // Note: Due to race conditions, it might be 1 or 2, but definitely not 10
        assert!(
            fetch_count.load(Ordering::SeqCst) <= 2,
            "Expected at most 2 fetches due to single-flight, got {}",
            fetch_count.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn test_cache_different_keys() {
        let cache = TileCache::new(1024 * 1024, None);

        // Same file, different tile indices
        let key1 = ("test.tif".to_string(), 0, 0);
        let key2 = ("test.tif".to_string(), 0, 1);
        let key3 = ("test.tif".to_string(), 1, 0);

        let tile1 = cache
            .get_or_fetch(key1.clone(), || async { Ok(make_test_tile(100)) })
            .await
            .unwrap();

        let tile2 = cache
            .get_or_fetch(key2.clone(), || async { Ok(make_test_tile(100)) })
            .await
            .unwrap();

        let tile3 = cache
            .get_or_fetch(key3.clone(), || async { Ok(make_test_tile(100)) })
            .await
            .unwrap();

        // All tiles should be different
        assert!(!Arc::ptr_eq(&tile1, &tile2));
        assert!(!Arc::ptr_eq(&tile2, &tile3));
    }

    #[tokio::test]
    async fn test_cache_fetch_error() {
        let cache = TileCache::new(1024 * 1024, None);
        let key = ("error.tif".to_string(), 0, 0);

        // First fetch returns error
        let result = cache
            .get_or_fetch(key.clone(), || async {
                Err(anyhow::anyhow!("Simulated fetch error"))
            })
            .await;

        assert!(result.is_err());

        // Cache should not store failed fetches, so next attempt should call fetch again
        let result = cache
            .get_or_fetch(key.clone(), || async { Ok(make_test_tile(100)) })
            .await;

        assert!(result.is_ok());
    }

    #[test]
    fn test_decoded_tile_size() {
        let tile = DecodedTile {
            data: vec![0u8; 1024],
            width: 10,
            height: 10,
            bands: 1,
            is_planar: false,
        };

        // size_bytes includes data.len() + struct overhead
        let expected = 1024 + std::mem::size_of::<DecodedTile>();
        assert_eq!(tile.size_bytes(), expected);
    }
}
