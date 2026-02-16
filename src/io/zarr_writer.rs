//! Async Zarr writing using zarrs.

use crate::config::Config;
use crate::crs::epsg_to_proj_definition;
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

        // Create root group - zarrs requires paths to start with /
        let group_path = if path.is_empty() { "/".to_string() } else { format!("/{}", path) };
        let group = GroupBuilder::new().build(zarr_store.clone(), &group_path)?;
        group.async_store_metadata().await?;

        // Define array shape: (time, bands, height, width)
        let shape = output_grid.array_shape();
        let chunk_shape = &config.output.chunk_shape;

        // Build the array - zarrs requires paths to start with /
        let array_path = if path.is_empty() { "/embeddings".to_string() } else { format!("/{}/embeddings", path) };

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
        // Add GeoZarr proj: namespace attribute for emerging GeoZarr ecosystem compatibility
        attributes.insert("proj:code".to_string(), serde_json::json!(output_grid.crs));

        // Add CRS as PROJ definition for additional CRS identification
        if let Some(proj_def) = epsg_to_proj_definition(&output_grid.crs) {
            attributes.insert("crs_proj4".to_string(), serde_json::json!(proj_def));
        }

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

        // Add time dimension metadata
        attributes.insert("start_year".to_string(), serde_json::json!(output_grid.start_year()));
        attributes.insert("num_years".to_string(), serde_json::json!(output_grid.num_years()));
        attributes.insert("years".to_string(), serde_json::json!(output_grid.years));

        // Add band names (AEF convention: A00, A01, ..., A63)
        let band_names: Vec<String> = (0..output_grid.num_bands)
            .map(|i| format!("A{:02}", i))
            .collect();
        attributes.insert("band_names".to_string(), serde_json::json!(band_names));

        // Add CF Conventions attribute for compatibility
        attributes.insert("Conventions".to_string(), serde_json::json!("CF-1.8"));

        builder.attributes(attributes);

        let array = builder.build(zarr_store.clone(), &array_path)?;
        array.async_store_metadata().await?;

        // Create coordinate arrays for xarray compatibility
        Self::create_coordinate_arrays(
            &zarr_store,
            &path,
            &output_grid,
        ).await?;

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
    /// Chunks are always full-sized because OutputGrid rounds dimensions up to chunk boundaries.
    pub async fn write_chunk_async(&self, chunk: &OutputChunk, data: Array4<i8>) -> Result<()> {
        let indices = chunk.chunk_indices();
        let chunk_indices: [u64; 4] = [
            indices[0] as u64,
            indices[1] as u64,
            indices[2] as u64,
            indices[3] as u64,
        ];

        let shape = data.shape();
        let chunk_shape = &self.output_grid.chunk_shape;
        let expected_shape = [1, chunk_shape.embedding, chunk_shape.height, chunk_shape.width];

        // Verify chunk is full-sized (OutputGrid guarantees this by rounding up dimensions)
        debug_assert_eq!(
            shape,
            expected_shape.as_slice(),
            "Chunk shape mismatch: got {:?}, expected {:?}. OutputGrid should ensure all chunks are full-sized.",
            shape, expected_shape
        );

        tracing::debug!(
            "Writing chunk {:?}: shape={:?}",
            chunk_indices, shape
        );

        // Pass slice directly to avoid copy (data should be contiguous)
        let chunk_data = data.as_slice().expect("chunk data should be contiguous");

        self.array
            .async_store_chunk(&chunk_indices, chunk_data)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Chunk {:?} write failed: {:?}. Data shape: {:?}",
                    chunk_indices, e, shape
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

    /// Create coordinate arrays for xarray compatibility.
    ///
    /// Creates three 1D arrays:
    /// - `/x` - Float64 array of x-coordinates (column centers)
    /// - `/y` - Float64 array of y-coordinates (row centers)
    /// - `/time` - Int32 array of years
    async fn create_coordinate_arrays(
        zarr_store: &Arc<AsyncObjectStore<Arc<dyn ObjectStore>>>,
        path: &str,
        output_grid: &OutputGrid,
    ) -> Result<()> {
        // Compute coordinate values
        let x_coords: Vec<f64> = (0..output_grid.width)
            .map(|i| output_grid.bounds[0] + (i as f64 + 0.5) * output_grid.resolution)
            .collect();

        let y_coords: Vec<f64> = (0..output_grid.height)
            .map(|i| output_grid.bounds[3] - (i as f64 + 0.5) * output_grid.resolution)
            .collect();

        let time_coords: Vec<i32> = output_grid.years.clone();

        // Create x coordinate array
        let x_path = if path.is_empty() { "/x".to_string() } else { format!("/{}/x", path) };
        let x_array = ArrayBuilder::new(
            vec![output_grid.width as u64],
            vec![output_grid.width as u64], // Single chunk
            "float64",
            0.0f64,
        )
        .dimension_names(Some(vec![Some("x".to_string())]))
        .build(zarr_store.clone(), &x_path)?;
        x_array.async_store_metadata().await?;
        x_array.async_store_chunk(&[0], &x_coords).await?;

        // Create y coordinate array
        let y_path = if path.is_empty() { "/y".to_string() } else { format!("/{}/y", path) };
        let y_array = ArrayBuilder::new(
            vec![output_grid.height as u64],
            vec![output_grid.height as u64], // Single chunk
            "float64",
            0.0f64,
        )
        .dimension_names(Some(vec![Some("y".to_string())]))
        .build(zarr_store.clone(), &y_path)?;
        y_array.async_store_metadata().await?;
        y_array.async_store_chunk(&[0], &y_coords).await?;

        // Create time coordinate array
        let time_path = if path.is_empty() { "/time".to_string() } else { format!("/{}/time", path) };
        let num_years = output_grid.num_years();
        let time_array = ArrayBuilder::new(
            vec![num_years as u64],
            vec![num_years as u64], // Single chunk
            "int32",
            0i32,
        )
        .dimension_names(Some(vec![Some("time".to_string())]))
        .build(zarr_store.clone(), &time_path)?;
        time_array.async_store_metadata().await?;
        time_array.async_store_chunk(&[0], &time_coords).await?;

        tracing::info!(
            "Created coordinate arrays: x[{}], y[{}], time[{}]",
            x_coords.len(),
            y_coords.len(),
            time_coords.len()
        );

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
            filter: None,
        }
    }

    fn create_test_grid(height: usize, width: usize, chunk_shape: &ChunkShape) -> OutputGrid {
        OutputGrid {
            bounds: [0.0, 0.0, width as f64 * 10.0, height as f64 * 10.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            years: vec![2024],
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
    async fn test_zarr_writer_attributes() {
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

        let _writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

        // Read and parse the array metadata
        let array_json = zarr_path.join("embeddings").join("zarr.json");
        let metadata_str = std::fs::read_to_string(&array_json).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();

        // Verify attributes exist and have correct values
        let attrs = metadata.get("attributes").expect("attributes should exist");

        // CF-style CRS attribute
        assert_eq!(attrs.get("crs").and_then(|v| v.as_str()), Some("EPSG:4326"));

        // GeoZarr proj: namespace attribute
        assert_eq!(attrs.get("proj:code").and_then(|v| v.as_str()), Some("EPSG:4326"));

        // CF Conventions marker
        assert_eq!(attrs.get("Conventions").and_then(|v| v.as_str()), Some("CF-1.8"));

        // Geospatial metadata
        assert!(attrs.get("transform").is_some(), "transform should exist");
        assert!(attrs.get("bounds").is_some(), "bounds should exist");
        assert_eq!(attrs.get("resolution").and_then(|v| v.as_f64()), Some(10.0));

        // Time metadata
        assert_eq!(attrs.get("start_year").and_then(|v| v.as_i64()), Some(2024));
        assert_eq!(attrs.get("num_years").and_then(|v| v.as_i64()), Some(1));
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

    // Note: test_zarr_writer_writes_partial_chunk was removed because OutputGrid now
    // guarantees all chunks are full-sized by rounding dimensions up to chunk boundaries.

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
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            years: vec![2024],
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
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            years: vec![2024],
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
            filter: None,
        };

        let grid = Arc::new(OutputGrid {
            bounds: [0.0, 0.0, 80.0, 80.0],
            crs: "EPSG:4326".to_string(),
            resolution: 10.0,
            years: vec![2024],
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
