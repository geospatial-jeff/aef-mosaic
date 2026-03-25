//! Checkpoint/resume system for multi-day processing.
//!
//! Tracks completed chunks in a checkpoint file adjacent to output.
//! On restart, already-completed chunks are skipped.

use crate::config::Config;
use crate::index::OutputChunk;
use anyhow::Result;
use dashmap::DashSet;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;

/// Default checkpoint file name (stored in .checkpoint/ directory).
const CHECKPOINT_FILENAME: &str = "checkpoint.json";

/// Generate checkpoint filename with optional prefix for multi-VM support.
fn checkpoint_filename(prefix: Option<&str>) -> String {
    match prefix {
        Some(p) => format!("checkpoint.{}.json", p),
        None => CHECKPOINT_FILENAME.to_string(),
    }
}

/// Command sent to the background checkpoint writer.
enum CheckpointCommand {
    /// Mark a chunk as completed (completion is already stored in DashSet).
    MarkCompleted,
    /// Flush checkpoint to storage immediately.
    Flush,
    /// Shutdown the background writer.
    Shutdown,
}

/// Checkpoint file format stored as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    /// Schema version for forward compatibility.
    version: u32,
    /// Hash of configuration to detect incompatible restarts.
    config_hash: String,
    /// List of completed chunk keys ("time:row:col").
    completed_chunks: Vec<String>,
    /// Unix timestamp (seconds since epoch) of last checkpoint write.
    last_checkpoint: u64,
}


/// Manages checkpoint state for resumable processing.
///
/// Uses a lock-free concurrent set for fast lookups and a background
/// writer for periodic persistence.
pub struct CheckpointManager {
    #[allow(dead_code)]
    store: Arc<dyn ObjectStore>,
    #[allow(dead_code)]
    path: Path,
    completed: Arc<DashSet<String>>,
    #[allow(dead_code)]
    config_hash: String,
    checkpoint_tx: mpsc::Sender<CheckpointCommand>,
    _writer_handle: tokio::task::JoinHandle<()>,
}

impl CheckpointManager {
    /// Load existing checkpoint or create new one.
    ///
    /// Returns an error if the checkpoint exists but has a different config hash,
    /// indicating an incompatible configuration change.
    pub async fn load_or_create(
        store: Arc<dyn ObjectStore>,
        output_prefix: &str,
        config: &Config,
        interval_secs: u64,
    ) -> Result<Self> {
        let filename = checkpoint_filename(config.processing.checkpoint.prefix.as_deref());
        let checkpoint_path = format!("{}/.checkpoint/{}", output_prefix.trim_end_matches('/'), filename);
        let path = Path::from(checkpoint_path);
        let config_hash = compute_config_hash(config);
        let completed = Arc::new(DashSet::new());

        // Try to load existing checkpoint
        match store.get(&path).await {
            Ok(get_result) => {
                let bytes = get_result.bytes().await?;
                let data: CheckpointData = serde_json::from_slice(&bytes)?;

                // Validate config hash
                if data.config_hash != config_hash {
                    anyhow::bail!(
                        "Checkpoint config mismatch. Expected hash '{}', found '{}'. \
                         Delete {} to restart with new configuration.",
                        config_hash,
                        data.config_hash,
                        path
                    );
                }

                // Load completed chunks
                for key in data.completed_chunks {
                    completed.insert(key);
                }

                tracing::info!(
                    "Loaded checkpoint: {} chunks completed (last saved {})",
                    completed.len(),
                    data.last_checkpoint
                );
            }
            Err(object_store::Error::NotFound { .. }) => {
                tracing::info!("No existing checkpoint found, starting fresh");
            }
            Err(e) => {
                return Err(e.into());
            }
        }

        // Start background writer
        let (checkpoint_tx, checkpoint_rx) = mpsc::channel::<CheckpointCommand>(1024);
        let writer_handle = Self::spawn_background_writer(
            store.clone(),
            path.clone(),
            completed.clone(),
            config_hash.clone(),
            checkpoint_rx,
            interval_secs,
        );

        Ok(Self {
            store,
            path,
            completed,
            config_hash,
            checkpoint_tx,
            _writer_handle: writer_handle,
        })
    }

    /// Check if a chunk is already completed (fast, lock-free).
    pub fn is_completed(&self, chunk: &OutputChunk) -> bool {
        let key = chunk_key(chunk);
        self.completed.contains(&key)
    }

    /// Mark a chunk as completed (async, non-blocking).
    ///
    /// The chunk is immediately visible to `is_completed()` but may not
    /// be persisted until the next periodic flush.
    pub fn mark_completed(&self, chunk: &OutputChunk) {
        let key = chunk_key(chunk);
        self.completed.insert(key);
        // Best-effort send - if channel is full, the periodic flush will still pick it up
        let _ = self.checkpoint_tx.try_send(CheckpointCommand::MarkCompleted);
    }

    /// Get the number of completed chunks.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Flush checkpoint to storage immediately.
    pub async fn flush(&self) -> Result<()> {
        // Send flush command and wait a bit for the writer to process it
        let _ = self.checkpoint_tx.send(CheckpointCommand::Flush).await;

        // Give the writer a moment to flush
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }

    /// Shutdown the checkpoint manager, flushing final state.
    pub async fn shutdown(self) -> Result<()> {
        let _ = self.checkpoint_tx.send(CheckpointCommand::Shutdown).await;
        // Wait for the writer to finish
        let _ = self._writer_handle.await;
        Ok(())
    }

    /// Spawn the background checkpoint writer task.
    fn spawn_background_writer(
        store: Arc<dyn ObjectStore>,
        path: Path,
        completed: Arc<DashSet<String>>,
        config_hash: String,
        mut rx: mpsc::Receiver<CheckpointCommand>,
        interval_secs: u64,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
            let mut last_count = 0usize;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Periodic flush if there are new completions
                        let current_count = completed.len();
                        if current_count > last_count {
                            if let Err(e) = write_checkpoint(&store, &path, &completed, &config_hash).await {
                                tracing::warn!("Failed to write checkpoint: {}", e);
                            } else {
                                tracing::debug!("Checkpoint saved: {} chunks completed", current_count);
                                last_count = current_count;
                            }
                        }
                    }
                    cmd = rx.recv() => {
                        match cmd {
                            Some(CheckpointCommand::MarkCompleted) => {
                                // The completion is already recorded in the DashSet,
                                // so we don't need to do anything here - just let the
                                // periodic flush pick it up.
                            }
                            Some(CheckpointCommand::Flush) => {
                                if let Err(e) = write_checkpoint(&store, &path, &completed, &config_hash).await {
                                    tracing::warn!("Failed to write checkpoint on flush: {}", e);
                                } else {
                                    last_count = completed.len();
                                }
                            }
                            Some(CheckpointCommand::Shutdown) | None => {
                                // Final flush before shutdown
                                if let Err(e) = write_checkpoint(&store, &path, &completed, &config_hash).await {
                                    tracing::warn!("Failed to write final checkpoint: {}", e);
                                } else {
                                    tracing::info!("Final checkpoint saved: {} chunks completed", completed.len());
                                }
                                return;
                            }
                        }
                    }
                }
            }
        })
    }
}

/// Write checkpoint to storage.
async fn write_checkpoint(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    completed: &DashSet<String>,
    config_hash: &str,
) -> Result<()> {
    // Collect completed chunks (fast, just cloning references)
    let chunks_snapshot: Vec<String> = completed.iter().map(|r| r.clone()).collect();
    let config_hash = config_hash.to_string();

    // Move CPU-intensive work (sorting, serialization) to blocking thread pool
    let json = tokio::task::spawn_blocking(move || {
        let mut chunks = chunks_snapshot;
        chunks.sort();

        let data = CheckpointData {
            version: 1,
            config_hash,
            completed_chunks: chunks,
            last_checkpoint: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        serde_json::to_vec_pretty(&data)
    })
    .await
    .map_err(|e| anyhow::anyhow!("Checkpoint serialization task failed: {}", e))??;

    store.put(path, json.into()).await?;

    Ok(())
}

/// Generate a unique key for a chunk.
fn chunk_key(chunk: &OutputChunk) -> String {
    format!("{}:{}:{}", chunk.time_idx, chunk.row_idx, chunk.col_idx)
}

/// Compute a hash of configuration fields that affect output compatibility.
fn compute_config_hash(config: &Config) -> String {
    let mut hasher = Sha256::new();

    // Hash fields that affect output format/content
    hasher.update(config.output.crs.as_bytes());
    hasher.update(config.output.resolution.to_le_bytes());
    hasher.update(config.output.chunk_shape.height.to_le_bytes());
    hasher.update(config.output.chunk_shape.width.to_le_bytes());
    hasher.update(config.output.chunk_shape.embedding.to_le_bytes());
    hasher.update(config.output.num_bands.to_le_bytes());

    // Include bounds if specified
    if let Some(filter) = &config.filter {
        if let Some(bounds) = &filter.bounds {
            for b in bounds {
                hasher.update(b.to_le_bytes());
            }
        }
    }

    let result = hasher.finalize();
    format!("sha256:{:x}", result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ChunkShape, InputConfig, OutputConfig, ProcessingConfig, ShardingConfig,
    };

    fn test_config() -> Config {
        Config {
            input: InputConfig {
                index_path: "test".to_string(),
                cog_bucket: "test".to_string(),
            },
            output: OutputConfig {
                local_path: Some("/tmp/test".to_string()),
                bucket: None,
                prefix: None,
                crs: "EPSG:4326".to_string(),
                resolution: 10.0,
                num_bands: 64,
                chunk_shape: ChunkShape::default(),
                sharding: ShardingConfig::default(),
                compression_level: 3,
                years: None,
            },
            processing: ProcessingConfig::default(),
            filter: None,
        }
    }

    #[test]
    fn test_chunk_key() {
        let chunk = OutputChunk {
            time_idx: 1,
            row_idx: 2,
            col_idx: 3,
        };
        assert_eq!(chunk_key(&chunk), "1:2:3");
    }

    #[test]
    fn test_config_hash_deterministic() {
        let config = test_config();
        let hash1 = compute_config_hash(&config);
        let hash2 = compute_config_hash(&config);
        assert_eq!(hash1, hash2);
        assert!(hash1.starts_with("sha256:"));
    }

    #[test]
    fn test_config_hash_changes_with_resolution() {
        let config1 = test_config();
        let mut config2 = test_config();
        config2.output.resolution = 20.0;

        let hash1 = compute_config_hash(&config1);
        let hash2 = compute_config_hash(&config2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_checkpoint_data_serialization() {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let data = CheckpointData {
            version: 1,
            config_hash: "sha256:abc123".to_string(),
            completed_chunks: vec!["0:0:0".to_string(), "0:0:1".to_string()],
            last_checkpoint: timestamp,
        };

        let json = serde_json::to_string(&data).unwrap();
        let parsed: CheckpointData = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.config_hash, "sha256:abc123");
        assert_eq!(parsed.completed_chunks.len(), 2);
    }
}
