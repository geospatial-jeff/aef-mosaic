//! Async Zarr writing using zarrs.

use crate::config::Config;
use crate::index::{OutputChunk, OutputGrid};
use anyhow::Result;
use futures::StreamExt;
use ndarray::Array4;
use object_store::{ObjectStore, ObjectStoreExt};
use std::sync::Arc;
use zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder};
use zarrs::group::GroupBuilder;
use zarrs_object_store::AsyncObjectStore;


/// Async Zarr writer for output chunks.
///
/// Note: The zarrs::Array type is designed for concurrent writes to different chunks.
/// Each chunk write is independent, so we don't need a mutex for synchronization.
/// The underlying AsyncObjectStore handles concurrent access to the storage backend.
pub struct ZarrWriter {
    /// The Zarr array (thread-safe for concurrent writes to different chunks)
    array: Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,

    /// Output grid configuration
    output_grid: Arc<OutputGrid>,
}

impl ZarrWriter {
    /// Create a new Zarr array for writing.
    pub async fn create(
        store: Arc<dyn ObjectStore>,
        path: &str,
        output_grid: Arc<OutputGrid>,
        config: &Config,
    ) -> Result<Self> {
        // Delete any existing embeddings array to ensure fresh write
        let embeddings_prefix = if path.is_empty() {
            object_store::path::Path::from("embeddings")
        } else {
            object_store::path::Path::from(format!("{}/embeddings", path))
        };

        // List and delete all existing objects under the embeddings path (parallelized)
        let existing: Vec<_> = store
            .list(Some(&embeddings_prefix))
            .collect::<Vec<_>>()
            .await;

        if !existing.is_empty() {
            tracing::info!("Deleting {} existing objects under {:?} (parallel)", existing.len(), embeddings_prefix);

            // Parallel deletion using buffer_unordered for high concurrency
            use futures::stream::{self, StreamExt as _};
            let store_ref = &store;
            let delete_results: Vec<_> = stream::iter(existing)
                .filter_map(|result| async move { result.ok() })
                .map(|meta| async move {
                    let location = meta.location.clone();
                    match store_ref.delete(&meta.location).await {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            tracing::warn!("Failed to delete {:?}: {}", location, e);
                            Err(e)
                        }
                    }
                })
                .buffer_unordered(64) // Process up to 64 deletes concurrently
                .collect()
                .await;

            let deleted_count = delete_results.iter().filter(|r| r.is_ok()).count();
            tracing::debug!("Deleted {} objects", deleted_count);
        }

        let zarr_store = Arc::new(AsyncObjectStore::new(store.clone()));

        // Create root group - handle empty path for local filesystem
        let group_path = if path.is_empty() { "/".to_string() } else { format!("{}/", path) };
        let group = GroupBuilder::new().build(zarr_store.clone(), &group_path)?;
        group.async_store_metadata().await?;

        // Define array shape: (time, bands, height, width)
        let shape = output_grid.array_shape();
        let chunk_shape = &config.output.chunk_shape;

        // Build the array - handle empty path for local filesystem
        let array_path = if path.is_empty() { "/embeddings".to_string() } else { format!("{}/embeddings", path) };

        let chunk_grid = vec![
            chunk_shape.time as u64,
            chunk_shape.embedding as u64,
            chunk_shape.height as u64,
            chunk_shape.width as u64,
        ];

        let mut builder = ArrayBuilder::new(
            vec![shape[0] as u64, shape[1] as u64, shape[2] as u64, shape[3] as u64],
            chunk_grid,
            "int8",  // Data type as string
            0i8,     // Fill value
        );

        // Add dimension names
        builder.dimension_names(Some(vec![
            Some("time".to_string()),
            Some("band".to_string()),
            Some("y".to_string()),
            Some("x".to_string()),
        ]));

        // Configure zstd compression
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(
            config.output.compression_level,
            false,
        ))]);

        // Add attributes for geospatial metadata
        let mut attributes = serde_json::Map::new();
        attributes.insert("crs".to_string(), serde_json::json!(output_grid.crs));
        attributes.insert(
            "transform".to_string(),
            serde_json::json!([
                output_grid.resolution,
                0.0,
                output_grid.bounds[0],
                0.0,
                -output_grid.resolution,
                output_grid.bounds[3]
            ]),
        );
        attributes.insert(
            "bounds".to_string(),
            serde_json::json!(output_grid.bounds),
        );
        attributes.insert(
            "resolution".to_string(),
            serde_json::json!(output_grid.resolution),
        );
        builder.attributes(attributes);

        let array = builder.build(zarr_store.clone(), &array_path)?;
        array.async_store_metadata().await?;

        tracing::info!(
            "Created Zarr array at {} with shape {:?}",
            array_path,
            shape
        );

        Ok(Self {
            array,
            output_grid,
        })
    }

    /// Write a chunk to the Zarr array asynchronously.
    ///
    /// Handles partial edge chunks by padding to full chunk size.
    pub async fn write_chunk_async(&self, chunk: &OutputChunk, data: Array4<i8>) -> Result<()> {
        let indices = chunk.chunk_indices();
        let chunk_indices: Vec<u64> = indices.iter().map(|&i| i as u64).collect();

        let shape = data.shape();
        let chunk_shape = &self.output_grid.chunk_shape;
        let expected_shape = [1, chunk_shape.embedding, chunk_shape.height, chunk_shape.width];

        tracing::debug!(
            "Writing chunk {:?}: shape={:?}, expected={:?}",
            chunk_indices, shape, expected_shape
        );

        // Check if this is a partial edge chunk that needs padding
        let needs_padding = shape[0] != expected_shape[0]
            || shape[1] != expected_shape[1]
            || shape[2] != expected_shape[2]
            || shape[3] != expected_shape[3];

        let chunk_elements: Vec<i8> = if needs_padding {
            // Pad to full chunk size with fill value (0)
            let mut padded = Array4::<i8>::zeros((
                expected_shape[0],
                expected_shape[1],
                expected_shape[2],
                expected_shape[3],
            ));
            // Copy data into the padded array
            padded
                .slice_mut(ndarray::s![..shape[0], ..shape[1], ..shape[2], ..shape[3]])
                .assign(&data);
            tracing::debug!(
                "Padded chunk {:?} from {:?} to {:?}",
                chunk_indices, shape, expected_shape
            );
            padded.iter().copied().collect()
        } else {
            data.iter().copied().collect()
        };

        tracing::debug!("Chunk {:?}: {} elements", chunk_indices, chunk_elements.len());

        // Use async_store_chunk with chunk indices directly
        // zarrs::Array is safe for concurrent writes to different chunks
        self.array
            .async_store_chunk(&chunk_indices, chunk_elements.as_slice())
            .await
            .map_err(|e| {
                tracing::error!(
                    "Chunk {:?} write failed: {:?}. Data shape: {:?}, elements: {}",
                    chunk_indices, e, shape, chunk_elements.len()
                );
                anyhow::anyhow!("Failed to write chunk {:?}: {:?}", chunk_indices, e)
            })?;

        tracing::debug!("Chunk {:?} write completed", chunk_indices);
        Ok(())
    }

    /// Get the output grid.
    pub fn output_grid(&self) -> &OutputGrid {
        &self.output_grid
    }

    /// Finalize the Zarr array.
    ///
    /// Note: This is currently a no-op since zarrs handles writes synchronously
    /// and metadata is stored during `create()`. This method exists for API
    /// consistency and future extensibility (e.g., writing consolidated metadata).
    pub fn finalize(&self) -> Result<()> {
        tracing::info!("Zarr array finalized");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_indices() {
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 5,
            col_idx: 10,
        };

        let indices = chunk.chunk_indices();
        assert_eq!(indices, [0, 0, 5, 10]);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::config::{ChunkShape, OutputConfig};
    use object_store::local::LocalFileSystem;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_config(chunk_shape: ChunkShape) -> Config {
        Config {
            input: crate::config::InputConfig {
                index_path: "test".to_string(),
                cog_bucket: "test".to_string(),
            },
            output: OutputConfig {
                bucket: None,
                prefix: None,
                local_path: Some("test".to_string()),
                crs: "EPSG:4326".to_string(),
                resolution: 10.0,
                chunk_shape,
                num_years: 1,
                start_year: 2024,
                num_bands: 4,
                compression_level: 3,
                use_sharding: false,
                shard_shape: [1, 1],
            },
            processing: crate::config::ProcessingConfig::default(),
            aws: crate::config::AwsConfig {
                region: "us-west-2".to_string(),
            },
            filter: None,
        }
    }

    fn create_test_grid(height: usize, width: usize, chunk_shape: &ChunkShape) -> OutputGrid {
        OutputGrid {
            bounds: [0.0, 0.0, width as f64 * 10.0, height as f64 * 10.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            num_years: 1,
            start_year: 2024,
            num_bands: 4,
            height,
            width,
            chunk_shape: chunk_shape.clone(),
            chunk_counts: [
                1,
                1,
                (height + chunk_shape.height - 1) / chunk_shape.height,
                (width + chunk_shape.width - 1) / chunk_shape.width,
            ],
        }
    }

    #[tokio::test]
    async fn test_zarr_writer_creates_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let zarr_path = temp_dir.path().join("test.zarr");
        std::fs::create_dir_all(&zarr_path).unwrap();

        let store = Arc::new(LocalFileSystem::new_with_prefix(&zarr_path).unwrap());
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = create_test_config(chunk_shape.clone());
        let grid = Arc::new(create_test_grid(8, 8, &chunk_shape));

        let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Check metadata files exist
        let zarr_json = zarr_path.join("zarr.json");
        let array_json = zarr_path.join("embeddings").join("zarr.json");

        assert!(zarr_json.exists(), "Root zarr.json should exist");
        assert!(array_json.exists(), "Array zarr.json should exist");

        writer.finalize().unwrap();
    }

    #[tokio::test]
    async fn test_zarr_writer_writes_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let zarr_path = temp_dir.path().join("test.zarr");
        std::fs::create_dir_all(&zarr_path).unwrap();

        let store = Arc::new(LocalFileSystem::new_with_prefix(&zarr_path).unwrap());
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = create_test_config(chunk_shape.clone());
        let grid = Arc::new(create_test_grid(8, 8, &chunk_shape));

        let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Create test data - full chunk size
        let data = Array4::from_elem((1, 4, 4, 4), 42i8);
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 0,
            col_idx: 0,
        };

        // Write chunk
        let result = writer.write_chunk_async(&chunk, data).await;
        println!("Write result: {:?}", result);
        result.unwrap();

        writer.finalize().unwrap();

        // Check chunk file exists - Zarr v3 stores chunks at c/<indices...>
        let chunk_path = zarr_path.join("embeddings").join("c").join("0").join("0").join("0").join("0");

        // List all files in embeddings directory
        println!("Files in {:?}:", zarr_path.join("embeddings"));
        list_dir_recursive(&zarr_path.join("embeddings"), 0);

        assert!(chunk_path.exists(), "Chunk file should exist at {:?}", chunk_path);
    }

    #[tokio::test]
    async fn test_zarr_writer_writes_partial_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let zarr_path = temp_dir.path().join("test.zarr");
        std::fs::create_dir_all(&zarr_path).unwrap();

        let store = Arc::new(LocalFileSystem::new_with_prefix(&zarr_path).unwrap());
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = create_test_config(chunk_shape.clone());
        // Grid is 6x6, so edge chunks will be partial (2 pixels instead of 4)
        let grid = Arc::new(create_test_grid(6, 6, &chunk_shape));

        let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Create test data - partial chunk (edge case: 2x2 instead of 4x4)
        let data = Array4::from_elem((1, 4, 2, 2), 42i8);
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 1, // Second row - partial
            col_idx: 1, // Second col - partial
        };

        // Write partial chunk - should be padded internally
        let result = writer.write_chunk_async(&chunk, data).await;
        println!("Partial chunk write result: {:?}", result);
        result.unwrap();

        writer.finalize().unwrap();

        // Check chunk file exists
        let chunk_path = zarr_path.join("embeddings").join("c").join("0").join("0").join("1").join("1");
        println!("Files in {:?}:", zarr_path.join("embeddings"));
        list_dir_recursive(&zarr_path.join("embeddings"), 0);

        assert!(chunk_path.exists(), "Partial chunk file should exist at {:?}", chunk_path);
    }

    fn list_dir_recursive(path: &PathBuf, depth: usize) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                let indent = "  ".repeat(depth);
                println!("{}{:?}", indent, path.file_name().unwrap());
                if path.is_dir() {
                    list_dir_recursive(&path, depth + 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod production_test {
    use super::*;
    use crate::config::{ChunkShape, OutputConfig};
    use object_store::local::LocalFileSystem;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_zarr_writer_relative_path() {
        // This test mimics production setup with relative path
        let test_dir = PathBuf::from("target/test-zarr");
        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir_all(&test_dir).unwrap();

        let store = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = crate::config::Config {
            input: crate::config::InputConfig {
                index_path: "test".to_string(),
                cog_bucket: "test".to_string(),
            },
            output: OutputConfig {
                bucket: None,
                prefix: None,
                local_path: Some("test".to_string()),
                crs: "EPSG:4326".to_string(),
                resolution: 10.0,
                chunk_shape: chunk_shape.clone(),
                num_years: 1,
                start_year: 2024,
                num_bands: 4,
                compression_level: 3,
                use_sharding: false,
                shard_shape: [1, 1],
            },
            processing: crate::config::ProcessingConfig::default(),
            aws: crate::config::AwsConfig {
                region: "us-west-2".to_string(),
            },
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            num_years: 1,
            start_year: 2024,
            num_bands: 4,
            height: 8,
            width: 8,
            chunk_shape: chunk_shape.clone(),
            chunk_counts: [1, 1, 2, 2],
        });

        let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Create and write a chunk
        let data = Array4::from_elem((1, 4, 4, 4), 42i8);
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 0,
            col_idx: 0,
        };

        let result = writer.write_chunk_async(&chunk, data).await;
        println!("Write result: {:?}", result);
        result.unwrap();

        writer.finalize().unwrap();

        // Verify files exist
        let chunk_path = test_dir.join("embeddings").join("c").join("0").join("0").join("0").join("0");
        println!("Looking for chunk at: {:?}", chunk_path);
        println!("Directory contents:");
        
        fn list_recursive(path: &std::path::PathBuf, depth: usize) {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let indent = "  ".repeat(depth);
                    println!("{}{:?}", indent, p.file_name().unwrap());
                    if p.is_dir() {
                        list_recursive(&p, depth + 1);
                    }
                }
            }
        }
        list_recursive(&test_dir, 0);

        assert!(chunk_path.exists(), "Chunk should exist at {:?}", chunk_path);
        
        // Cleanup
        std::fs::remove_dir_all(&test_dir).ok();
    }

    #[tokio::test]
    async fn test_zarr_writer_dyn_store() {
        use crate::config::ChunkShape;

        // This test uses Arc<dyn ObjectStore> like production
        let test_dir = std::path::PathBuf::from("target/test-zarr-dyn");
        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir_all(&test_dir).unwrap();

        // Use Arc<dyn ObjectStore> like production
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&test_dir).unwrap());
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = crate::config::Config {
            input: crate::config::InputConfig {
                index_path: "test".to_string(),
                cog_bucket: "test".to_string(),
            },
            output: crate::config::OutputConfig {
                bucket: None,
                prefix: None,
                local_path: Some("test".to_string()),
                crs: "EPSG:4326".to_string(),
                resolution: 10.0,
                chunk_shape: chunk_shape.clone(),
                num_years: 1,
                start_year: 2024,
                num_bands: 4,
                compression_level: 3,
                use_sharding: false,
                shard_shape: [1, 1],
            },
            processing: crate::config::ProcessingConfig::default(),
            aws: crate::config::AwsConfig {
                region: "us-west-2".to_string(),
            },
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            num_years: 1,
            start_year: 2024,
            num_bands: 4,
            height: 8,
            width: 8,
            chunk_shape: chunk_shape.clone(),
            chunk_counts: [1, 1, 2, 2],
        });

        let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Create and write a chunk
        let data = Array4::from_elem((1, 4, 4, 4), 42i8);
        let chunk = OutputChunk {
            time_idx: 0,
            row_idx: 0,
            col_idx: 0,
        };

        let result = writer.write_chunk_async(&chunk, data).await;
        println!("Write result (dyn store): {:?}", result);
        result.unwrap();

        writer.finalize().unwrap();

        // Verify files exist
        let chunk_path = test_dir.join("embeddings").join("c").join("0").join("0").join("0").join("0");
        println!("Looking for chunk at: {:?}", chunk_path);
        
        fn list_recursive(path: &std::path::PathBuf, depth: usize) {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let indent = "  ".repeat(depth);
                    println!("{}{:?}", indent, p.file_name().unwrap());
                    if p.is_dir() {
                        list_recursive(&p, depth + 1);
                    }
                }
            }
        }
        println!("Directory contents:");
        list_recursive(&test_dir, 0);

        assert!(chunk_path.exists(), "Chunk should exist at {:?}", chunk_path);
        
        // Cleanup
        std::fs::remove_dir_all(&test_dir).ok();
    }
}

#[cfg(test)]
mod concurrent_test {
    use super::*;
    use crate::config::ChunkShape;
    use futures::stream::{self, StreamExt};
    use object_store::ObjectStore;

    #[tokio::test]
    async fn test_zarr_writer_concurrent_writes() {
        let test_dir = std::path::PathBuf::from("target/test-zarr-concurrent");
        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir_all(&test_dir).unwrap();

        let store: Arc<dyn ObjectStore> = Arc::new(
            object_store::local::LocalFileSystem::new_with_prefix(&test_dir).unwrap()
        );
        let chunk_shape = ChunkShape {
            time: 1,
            embedding: 4,
            height: 4,
            width: 4,
        };
        let config = crate::config::Config {
            input: crate::config::InputConfig {
                index_path: "test".to_string(),
                cog_bucket: "test".to_string(),
            },
            output: crate::config::OutputConfig {
                bucket: None,
                prefix: None,
                local_path: Some("test".to_string()),
                crs: "EPSG:4326".to_string(),
                resolution: 10.0,
                chunk_shape: chunk_shape.clone(),
                num_years: 1,
                start_year: 2024,
                num_bands: 4,
                compression_level: 3,
                use_sharding: false,
                shard_shape: [1, 1],
            },
            processing: crate::config::ProcessingConfig::default(),
            aws: crate::config::AwsConfig {
                region: "us-west-2".to_string(),
            },
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            num_years: 1,
            start_year: 2024,
            num_bands: 4,
            height: 8,
            width: 8,
            chunk_shape: chunk_shape.clone(),
            chunk_counts: [1, 1, 2, 2],
        });

        let writer = Arc::new(ZarrWriter::create(store, "", grid, &config).await.unwrap());

        // Write 4 chunks concurrently (like production)
        let chunks = vec![
            OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 },
            OutputChunk { time_idx: 0, row_idx: 0, col_idx: 1 },
            OutputChunk { time_idx: 0, row_idx: 1, col_idx: 0 },
            OutputChunk { time_idx: 0, row_idx: 1, col_idx: 1 },
        ];

        let results: Vec<_> = stream::iter(chunks.clone())
            .map(|chunk| {
                let w = writer.clone();
                async move {
                    let data = Array4::from_elem((1, 4, 4, 4), 42i8);
                    w.write_chunk_async(&chunk, data).await
                }
            })
            .buffer_unordered(4)
            .collect()
            .await;

        for (i, r) in results.iter().enumerate() {
            println!("Chunk {} result: {:?}", i, r);
            assert!(r.is_ok(), "Chunk {} should succeed", i);
        }

        writer.finalize().unwrap();

        // Check all chunks exist
        for chunk in &chunks {
            let chunk_path = test_dir
                .join("embeddings")
                .join("c")
                .join("0")
                .join("0")
                .join(chunk.row_idx.to_string())
                .join(chunk.col_idx.to_string());
            println!("Checking: {:?}", chunk_path);
            assert!(chunk_path.exists(), "Chunk {:?} should exist at {:?}", chunk, chunk_path);
        }

        std::fs::remove_dir_all(&test_dir).ok();
    }
}
