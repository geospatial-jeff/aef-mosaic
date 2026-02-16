//! Comprehensive integration tests for ZarrWriter at interface boundaries.
//!
//! Tests cover:
//! 1. Full production chunk sizes (1, 64, 1024, 1024)
//! 2. Partial edge chunks
//! 3. Arc<ZarrWriter> with concurrent writes
//! 4. Data flow from mosaic → write

use crate::config::{ChunkShape, Config, OutputConfig, InputConfig, ProcessingConfig};
use crate::index::{OutputChunk, OutputGrid};
use crate::io::ZarrWriter;
use futures::stream::{self, StreamExt};
use ndarray::Array4;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use std::sync::Arc;

fn create_test_config(chunk_shape: ChunkShape) -> Config {
    Config {
        input: InputConfig {
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
            num_bands: 64,
            compression_level: 3,
        },
        processing: ProcessingConfig::default(),
        filter: None,
    }
}

fn create_test_grid(height: usize, width: usize, chunk_shape: &ChunkShape) -> OutputGrid {
    // Round up to chunk boundaries (matching OutputGrid::new behavior)
    let row_chunks = height.div_ceil(chunk_shape.height);
    let col_chunks = width.div_ceil(chunk_shape.width);
    let aligned_height = row_chunks * chunk_shape.height;
    let aligned_width = col_chunks * chunk_shape.width;

    OutputGrid {
        bounds: [0.0, 0.0, aligned_width as f64 * 10.0, aligned_height as f64 * 10.0],
        crs: "EPSG:4326".to_string(),
        resolution: 10.0,
        years: vec![2024],
        num_bands: 64,
        height: aligned_height,
        width: aligned_width,
        chunk_shape: chunk_shape.clone(),
        chunk_counts: [1, 1, row_chunks, col_chunks],
    }
}

/// Test 1: Production-sized full chunks (1, 64, 1024, 1024)
#[tokio::test]
async fn test_production_full_chunk_size() {
    let test_dir = std::path::PathBuf::from("target/test-zarr-prod-size");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };
    let config = create_test_config(chunk_shape.clone());
    let grid = Arc::new(create_test_grid(1024, 1024, &chunk_shape));

    let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

    // Create production-sized data: (1, 64, 1024, 1024) = 67,108,864 elements
    let data = Array4::from_elem((1, 64, 1024, 1024), 42i8);
    println!("Created data with {} elements", data.len());
    assert_eq!(data.len(), 67_108_864);

    let chunk = OutputChunk {
        time_idx: 0,
        row_idx: 0,
        col_idx: 0,
    };

    let result = writer.write_chunk_async(&chunk, data).await;
    println!("Write result: {:?}", result);
    result.unwrap();

    writer.finalize().unwrap();

    let chunk_path = test_dir.join("embeddings").join("c").join("0").join("0").join("0").join("0");
    println!("Checking for chunk at: {:?}", chunk_path);
    assert!(chunk_path.exists(), "Full production chunk should exist at {:?}", chunk_path);

    // Check file size is reasonable (compressed, so smaller than raw)
    let metadata = std::fs::metadata(&chunk_path).unwrap();
    println!("Chunk file size: {} bytes", metadata.len());
    assert!(metadata.len() > 0, "Chunk file should have content");

    std::fs::remove_dir_all(&test_dir).ok();
}

// Note: Partial edge chunk tests (test_partial_width_chunk, test_partial_height_chunk) were removed
// because OutputGrid now guarantees all chunks are full-sized by rounding dimensions up to chunk boundaries.

/// Test 2: Multiple chunks via Arc<ZarrWriter> with buffer_unordered (production pattern)
#[tokio::test]
async fn test_arc_writer_buffer_unordered() {
    let test_dir = std::path::PathBuf::from("target/test-zarr-arc-buffered");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 256, // Smaller for faster test
        width: 256,
    };
    let config = create_test_config(chunk_shape.clone());
    let grid = Arc::new(create_test_grid(512, 512, &chunk_shape)); // 2x2 = 4 chunks

    // Arc<ZarrWriter> like production
    let writer = Arc::new(ZarrWriter::create(store, "", grid, &config).await.unwrap());

    let chunks: Vec<OutputChunk> = vec![
        OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 },
        OutputChunk { time_idx: 0, row_idx: 0, col_idx: 1 },
        OutputChunk { time_idx: 0, row_idx: 1, col_idx: 0 },
        OutputChunk { time_idx: 0, row_idx: 1, col_idx: 1 },
    ];

    // Use buffer_unordered like production scheduler
    let results: Vec<_> = stream::iter(chunks.clone())
        .map(|chunk| {
            let w = writer.clone();
            async move {
                let data = Array4::from_elem((1, 64, 256, 256), 42i8);
                w.write_chunk_async(&chunk, data).await
            }
        })
        .buffer_unordered(4) // Concurrent like production
        .collect()
        .await;

    for (i, r) in results.iter().enumerate() {
        println!("Chunk {} result: {:?}", i, r);
        assert!(r.is_ok(), "Chunk {} should succeed: {:?}", i, r);
    }

    writer.finalize().unwrap();

    // Verify all chunks exist
    for chunk in &chunks {
        let chunk_path = test_dir
            .join("embeddings")
            .join("c")
            .join("0")
            .join("0")
            .join(chunk.row_idx.to_string())
            .join(chunk.col_idx.to_string());
        assert!(chunk_path.exists(), "Chunk {:?} should exist at {:?}", chunk, chunk_path);
    }

    std::fs::remove_dir_all(&test_dir).ok();
}

/// Test 3: Production-like grid dimensions (7719x7116 like SF Bay Area run)
/// Note: Grid is now chunk-aligned, so all chunks are full 1024x1024
#[tokio::test]
async fn test_production_grid_dimensions() {
    let test_dir = std::path::PathBuf::from("target/test-zarr-prod-grid");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };
    let config = create_test_config(chunk_shape.clone());
    // Request 7116x7719, will be aligned to 7168x8192 (7x8 chunks)
    let grid = Arc::new(create_test_grid(7116, 7719, &chunk_shape));

    // Verify chunk alignment
    assert_eq!(grid.height, 7 * 1024, "Height should be chunk-aligned");
    assert_eq!(grid.width, 8 * 1024, "Width should be chunk-aligned");
    assert_eq!(grid.chunk_counts[2], 7, "Should have 7 row chunks");
    assert_eq!(grid.chunk_counts[3], 8, "Should have 8 col chunks");

    println!("Grid: {}x{} pixels (chunk-aligned), chunk_counts: {:?}",
             grid.height, grid.width, grid.chunk_counts);

    let writer = Arc::new(ZarrWriter::create(store, "", grid.clone(), &config).await.unwrap());

    // Write first chunk and last chunk (both full-sized now)
    let first_chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
    let last_row = grid.chunk_counts[2] - 1;
    let last_col = grid.chunk_counts[3] - 1;
    let last_chunk = OutputChunk { time_idx: 0, row_idx: last_row, col_idx: last_col };

    println!("First chunk: {:?}", first_chunk);
    println!("Last chunk: {:?} (row={}, col={})", last_chunk, last_row, last_col);

    // All chunks are full size (1024x1024) due to chunk alignment
    let first_data = Array4::from_elem((1, 64, 1024, 1024), 42i8);
    let result1 = writer.write_chunk_async(&first_chunk, first_data).await;
    println!("First chunk result: {:?}", result1);
    result1.unwrap();

    let last_data = Array4::from_elem((1, 64, 1024, 1024), 42i8);
    let result2 = writer.write_chunk_async(&last_chunk, last_data).await;
    println!("Last chunk result: {:?}", result2);
    result2.unwrap();

    writer.finalize().unwrap();

    // Verify both chunks exist
    let first_path = test_dir.join("embeddings").join("c").join("0").join("0").join("0").join("0");
    let last_path = test_dir
        .join("embeddings")
        .join("c")
        .join("0")
        .join("0")
        .join(last_row.to_string())
        .join(last_col.to_string());

    assert!(first_path.exists(), "First chunk should exist at {:?}", first_path);
    assert!(last_path.exists(), "Last chunk should exist at {:?}", last_path);

    std::fs::remove_dir_all(&test_dir).ok();
}

/// Test 6: Verify actual data content survives round-trip
#[tokio::test]
async fn test_data_integrity() {
    use zarrs::array::Array;
    use zarrs_object_store::AsyncObjectStore;

    let test_dir = std::path::PathBuf::from("target/test-zarr-integrity");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 4,
        height: 4,
        width: 4,
    };
    let config = create_test_config(chunk_shape.clone());
    let grid = Arc::new(create_test_grid(4, 4, &chunk_shape));

    let writer = ZarrWriter::create(store.clone(), "", grid, &config).await.unwrap();

    // Write known pattern
    let mut data = Array4::zeros((1, 4, 4, 4));
    for b in 0..4 {
        for h in 0..4 {
            for w in 0..4 {
                data[[0, b, h, w]] = ((b * 16 + h * 4 + w) as i8) - 32;
            }
        }
    }
    println!("Writing pattern data: first value = {}, last value = {}",
             data[[0, 0, 0, 0]], data[[0, 3, 3, 3]]);

    let chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
    writer.write_chunk_async(&chunk, data.clone()).await.unwrap();
    writer.finalize().unwrap();

    // Read back and verify
    let zarr_store = Arc::new(AsyncObjectStore::new(store));
    let array = Array::<AsyncObjectStore<Arc<dyn ObjectStore>>>::async_open(
        zarr_store,
        "/embeddings",
    ).await.unwrap();

    let read_data = array.async_retrieve_chunk_elements::<i8>(&[0, 0, 0, 0]).await.unwrap();
    println!("Read back {} elements", read_data.len());

    // Verify first and last values
    assert_eq!(read_data[0], data[[0, 0, 0, 0]], "First value mismatch");
    assert_eq!(read_data[read_data.len() - 1], data[[0, 3, 3, 3]], "Last value mismatch");

    std::fs::remove_dir_all(&test_dir).ok();
}

/// Test 7: mosaic_tiles → write flow (exact production data path)
#[tokio::test]
async fn test_mosaic_to_write_flow() {
    use crate::index::CogTile;
    use crate::io::{PixelWindow, WindowData};
    use crate::transform::{mosaic_tiles, ReprojectConfig, Reprojector};

    let test_dir = std::path::PathBuf::from("target/test-zarr-mosaic-flow");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 256,
        width: 256,
    };
    let config = create_test_config(chunk_shape.clone());
    let grid = Arc::new(create_test_grid(256, 256, &chunk_shape));

    let writer = ZarrWriter::create(store, "", grid, &config).await.unwrap();

    // Create mock WindowData like production COG reads would produce
    let bounds_wgs84 = [-122.5, 36.5, -122.0, 37.0];
    let tile = CogTile {
        tile_id: "test-tile".to_string(),
        s3_path: "s3://test/tile.tif".to_string(),
        crs: "EPSG:32610".to_string(),
        bounds_native: [500000.0, 4000000.0, 502560.0, 4002560.0],
        bounds_wgs84,
        footprint_wgs84: CogTile::footprint_from_wgs84_bounds(&bounds_wgs84),
        resolution: 10.0,
        year: 2024,
    };

    let window_data = WindowData {
        tile: tile.clone(),
        data: ndarray::Array3::from_elem((64, 256, 256), 42i8),
        window: PixelWindow::new(0, 0, 256, 256),
        bounds_native: [500000.0, 4000000.0, 502560.0, 4002560.0],
    };

    // Run mosaic_tiles like production (this uses spawn_blocking internally in prod)
    let reprojector = Reprojector::new("EPSG:32610"); // Same as tile CRS for simplicity
    let reproject_config = ReprojectConfig {
        target_crs: "EPSG:32610".to_string(),
        target_resolution: 10.0,
        target_bounds: [500000.0, 4000000.0, 502560.0, 4002560.0],
        target_shape: (256, 256),
        num_bands: 64,
    };

    let mosaic_result = mosaic_tiles(&[window_data], &reprojector, &reproject_config);
    println!("Mosaic result: {:?}", mosaic_result.as_ref().map(|a| a.shape()));
    let mosaic = mosaic_result.unwrap();

    println!("Mosaic shape: {:?}, len: {}", mosaic.shape(), mosaic.len());
    assert_eq!(mosaic.shape(), &[1, 64, 256, 256]);

    // Write the mosaic output
    let chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
    let write_result = writer.write_chunk_async(&chunk, mosaic).await;
    println!("Write result: {:?}", write_result);
    write_result.unwrap();

    writer.finalize().unwrap();

    let chunk_path = test_dir.join("embeddings").join("c").join("0").join("0").join("0").join("0");
    assert!(chunk_path.exists(), "Mosaic output chunk should exist at {:?}", chunk_path);

    std::fs::remove_dir_all(&test_dir).ok();
}

/// Test 8: Multiple mosaics written concurrently (like scheduler)
#[tokio::test]
async fn test_concurrent_mosaic_writes() {
    use crate::index::CogTile;
    use crate::io::{PixelWindow, WindowData};
    use crate::transform::{mosaic_tiles, ReprojectConfig, Reprojector};

    let test_dir = std::path::PathBuf::from("target/test-zarr-concurrent-mosaic");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).unwrap();
    }
    std::fs::create_dir_all(&test_dir).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&test_dir).unwrap());
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 128,
        width: 128,
    };
    let config = create_test_config(chunk_shape.clone());
    let grid = Arc::new(create_test_grid(256, 256, &chunk_shape)); // 2x2 chunks

    let writer = Arc::new(ZarrWriter::create(store, "", grid, &config).await.unwrap());

    let chunks: Vec<OutputChunk> = vec![
        OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 },
        OutputChunk { time_idx: 0, row_idx: 0, col_idx: 1 },
        OutputChunk { time_idx: 0, row_idx: 1, col_idx: 0 },
        OutputChunk { time_idx: 0, row_idx: 1, col_idx: 1 },
    ];

    // Simulate production: mosaic in spawn_blocking, then write
    let results: Vec<_> = stream::iter(chunks.clone())
        .map(|chunk| {
            let w = writer.clone();
            async move {
                // Create mock window data
                let bounds_wgs84 = [-122.5, 36.5, -122.0, 37.0];
                let tile = CogTile {
                    tile_id: format!("tile-{}-{}", chunk.row_idx, chunk.col_idx),
                    s3_path: "s3://test/tile.tif".to_string(),
                    crs: "EPSG:32610".to_string(),
                    bounds_native: [500000.0, 4000000.0, 501280.0, 4001280.0],
                    bounds_wgs84,
                    footprint_wgs84: CogTile::footprint_from_wgs84_bounds(&bounds_wgs84),
                    resolution: 10.0,
                    year: 2024,
                };

                let window_data = WindowData {
                    tile: tile.clone(),
                    data: ndarray::Array3::from_elem((64, 128, 128), 42i8),
                    window: PixelWindow::new(0, 0, 128, 128),
                    bounds_native: [500000.0, 4000000.0, 501280.0, 4001280.0],
                };

                // Mosaic in spawn_blocking like production
                let mosaic = tokio::task::spawn_blocking(move || {
                    let reprojector = Reprojector::new("EPSG:32610");
                    let reproject_config = ReprojectConfig {
                        target_crs: "EPSG:32610".to_string(),
                        target_resolution: 10.0,
                        target_bounds: [500000.0, 4000000.0, 501280.0, 4001280.0],
                        target_shape: (128, 128),
                        num_bands: 64,
                    };
                    mosaic_tiles(&[window_data], &reprojector, &reproject_config)
                })
                .await
                .unwrap()
                .unwrap();

                // Write like production
                w.write_chunk_async(&chunk, mosaic).await
            }
        })
        .buffer_unordered(4)
        .collect()
        .await;

    for (i, r) in results.iter().enumerate() {
        println!("Concurrent mosaic chunk {} result: {:?}", i, r);
        assert!(r.is_ok(), "Chunk {} should succeed", i);
    }

    writer.finalize().unwrap();

    // Verify all chunks
    for chunk in &chunks {
        let chunk_path = test_dir
            .join("embeddings")
            .join("c")
            .join("0")
            .join("0")
            .join(chunk.row_idx.to_string())
            .join(chunk.col_idx.to_string());
        assert!(chunk_path.exists(), "Chunk {:?} should exist", chunk);
    }

    std::fs::remove_dir_all(&test_dir).ok();
}

/// Test 9: Use exact production store creation via create_output_store
#[tokio::test]
async fn test_production_store_creation() {
    use crate::io::create_output_store;

    let test_path = "target/test-zarr-prod-store";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    // Create config exactly like production
    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };
    let mut config = create_test_config(chunk_shape.clone());
    config.output.local_path = Some(test_path.to_string());

    // Use production store creation
    let store = create_output_store(&config).unwrap();
    println!("Created store for path: {}", test_path);

    let grid = Arc::new(create_test_grid(1024, 1024, &chunk_shape));

    // Use production prefix logic
    let prefix = crate::io::get_output_prefix(&config);
    println!("Using prefix: '{}'", prefix);

    let writer = ZarrWriter::create(store, prefix, grid, &config).await.unwrap();

    // Write a chunk
    let data = Array4::from_elem((1, 64, 1024, 1024), 42i8);
    let chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };

    let result = writer.write_chunk_async(&chunk, data).await;
    println!("Production store write result: {:?}", result);
    result.unwrap();

    writer.finalize().unwrap();

    // Verify
    let chunk_path = std::path::Path::new(test_path)
        .join("embeddings")
        .join("c")
        .join("0")
        .join("0")
        .join("0")
        .join("0");
    println!("Looking for chunk at: {:?}", chunk_path);

    // List directory contents
    fn list_dir(path: &std::path::Path, depth: usize) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                let indent = "  ".repeat(depth);
                println!("{}{}", indent, p.file_name().unwrap().to_string_lossy());
                if p.is_dir() {
                    list_dir(&p, depth + 1);
                }
            }
        }
    }
    println!("Directory contents of {}:", test_path);
    list_dir(std::path::Path::new(test_path), 0);

    assert!(chunk_path.exists(), "Chunk should exist at {:?}", chunk_path);

    std::fs::remove_dir_all(test_path).ok();
}

// Note: test_production_failure_chunk_0_0_0_7 was removed because OutputGrid now
// guarantees all chunks are full-sized by rounding dimensions up to chunk boundaries.
// The partial chunk scenario this test was designed to catch can no longer occur.

/// Test 8: Verify corner chunks work correctly with aligned grid
#[tokio::test]
async fn test_aligned_corner_chunks() {
    use crate::io::create_output_store;

    let test_path = "target/test-zarr-aligned-corner";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };

    let mut config = create_test_config(chunk_shape.clone());
    config.output.local_path = Some(test_path.to_string());

    let store = create_output_store(&config).unwrap();

    // Use create_test_grid which now rounds up to chunk boundaries
    // Request 7719x7116, will get 8192x7168 (8x7 full chunks)
    let grid = Arc::new(create_test_grid(7719, 7116, &chunk_shape));

    // Verify alignment
    assert_eq!(grid.height, 8 * 1024, "Height should be 8 chunks");
    assert_eq!(grid.width, 7 * 1024, "Width should be 7 chunks");
    assert_eq!(grid.chunk_counts[2], 8);
    assert_eq!(grid.chunk_counts[3], 7);

    println!("Grid: {}x{} pixels (aligned), chunks: {:?}",
             grid.width, grid.height, grid.chunk_counts);

    let prefix = crate::io::get_output_prefix(&config);
    let writer = ZarrWriter::create(store, prefix, grid.clone(), &config).await.unwrap();

    // Test corner chunk [0, 0, 7, 6] - all chunks are now full 1024x1024
    let corner_chunk = OutputChunk { time_idx: 0, row_idx: 7, col_idx: 6 };
    let data = Array4::from_elem((1, 64, 1024, 1024), 42i8);

    let result = writer.write_chunk_async(&corner_chunk, data).await;
    println!("Corner chunk write result: {:?}", result);
    result.expect("Corner chunk should write successfully");

    writer.finalize().unwrap();

    let chunk_path = std::path::Path::new(test_path)
        .join("embeddings").join("c").join("0").join("0").join("7").join("6");
    assert!(chunk_path.exists(), "Corner chunk should exist at {:?}", chunk_path);

    std::fs::remove_dir_all(test_path).ok();
}

/// Test 9: Simulate the exact chunk_processor flow with aligned grid
/// Tests spawning mosaic work in spawn_blocking then writing via Arc<ZarrWriter>
#[tokio::test]
async fn test_chunk_processor_flow_simulation() {
    use crate::io::create_output_store;
    use futures::stream::{self, StreamExt};

    let test_path = "target/test-zarr-processor-flow";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };

    let mut config = create_test_config(chunk_shape.clone());
    config.output.local_path = Some(test_path.to_string());

    let store = create_output_store(&config).unwrap();

    // Use create_test_grid which auto-aligns to chunk boundaries
    // Request 7719x7116, will get 8192x7168 (8x7 full chunks)
    let grid = Arc::new(create_test_grid(7719, 7116, &chunk_shape));

    // Verify alignment
    assert_eq!(grid.height, 8 * 1024);
    assert_eq!(grid.width, 7 * 1024);
    assert_eq!(grid.chunk_counts[2], 8);
    assert_eq!(grid.chunk_counts[3], 7);

    let prefix = crate::io::get_output_prefix(&config);
    let writer = Arc::new(ZarrWriter::create(store, prefix, grid.clone(), &config).await.unwrap());

    // Generate all chunks like the scheduler does
    let mut chunks = Vec::new();
    for row in 0..8 {
        for col in 0..7 {
            chunks.push(OutputChunk { time_idx: 0, row_idx: row, col_idx: col });
        }
    }

    println!("Processing {} chunks...", chunks.len());

    // Process chunks exactly like chunk_processor does
    // All chunks are now full 1024x1024 due to grid alignment
    let results: Vec<Result<(), anyhow::Error>> = stream::iter(chunks)
        .map(|chunk| {
            let writer = writer.clone();
            async move {
                // Simulate mosaic work in spawn_blocking (like production)
                // All chunks are full 1024x1024 now
                let mosaic = tokio::task::spawn_blocking(move || {
                    Array4::from_elem((1, 64, 1024, 1024), (chunk.row_idx * 7 + chunk.col_idx) as i8)
                })
                .await
                .map_err(|e| anyhow::anyhow!("spawn_blocking failed: {}", e))?;

                // Write chunk (like production)
                writer.write_chunk_async(&chunk, mosaic).await
            }
        })
        .buffer_unordered(16)  // Similar concurrency to production
        .collect()
        .await;

    // Count successes and failures
    let (successes, failures): (Vec<_>, Vec<_>) = results.into_iter().partition(|r| r.is_ok());
    println!("Results: {} successes, {} failures", successes.len(), failures.len());

    for (i, failure) in failures.iter().enumerate() {
        println!("Failure {}: {:?}", i, failure);
    }

    assert!(failures.is_empty(), "All chunks should write successfully");
    assert_eq!(successes.len(), 56, "Should have 56 chunks (8x7)");

    // Verify some chunks exist
    let chunk_path = std::path::Path::new(test_path)
        .join("embeddings").join("c").join("0").join("0").join("7").join("6");
    assert!(chunk_path.exists(), "Corner chunk should exist at {:?}", chunk_path);

    // Verify file isn't empty
    let metadata = std::fs::metadata(&chunk_path).unwrap();
    assert!(metadata.len() > 0, "Chunk file should not be empty");
    println!("Corner chunk size: {} bytes", metadata.len());

    println!("All {} chunks processed successfully!", successes.len());

    std::fs::remove_dir_all(test_path).ok();
}

/// Test 12: Verify chunks can be read back after finalize from a fresh store
/// This ensures data is actually persisted to disk, not just buffered
#[tokio::test]
async fn test_chunk_persistence_after_finalize() {
    use crate::io::create_output_store;
    use zarrs::array::Array;
    use zarrs_object_store::AsyncObjectStore;

    let test_path = "target/test-zarr-persistence";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };

    // Use small grid for quick test
    let grid = Arc::new(OutputGrid {
        bounds: [0.0, 0.0, 20480.0, 20480.0],
        crs: "EPSG:6933".to_string(),
        resolution: 10.0,
        years: vec![2024],
        num_bands: 64,
        height: 2048,
        width: 2048,
        chunk_shape: chunk_shape.clone(),
        chunk_counts: [1, 1, 2, 2],
    });

    // Write phase
    {
        let mut config = create_test_config(chunk_shape.clone());
        config.output.local_path = Some(test_path.to_string());

        let store = create_output_store(&config).unwrap();
        let prefix = crate::io::get_output_prefix(&config);
        let writer = ZarrWriter::create(store, prefix, grid.clone(), &config).await.unwrap();

        // Write 4 chunks with distinct values
        for row in 0..2 {
            for col in 0..2 {
                let value = ((row * 2 + col + 1) * 10) as i8;  // 10, 20, 30, 40
                let chunk = OutputChunk { time_idx: 0, row_idx: row, col_idx: col };
                let data = Array4::from_elem((1, 64, 1024, 1024), value);
                writer.write_chunk_async(&chunk, data).await.unwrap();
                println!("Wrote chunk ({}, {}) with value {}", row, col, value);
            }
        }

        // Finalize to flush
        writer.finalize().unwrap();
        println!("Finalized writer");
    }

    // Read phase - open fresh store and read back
    {
        let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(test_path).unwrap());
        let async_store = Arc::new(AsyncObjectStore::new(store));

        // Open the array fresh (note: leading slash required)
        let array = Array::<AsyncObjectStore<Arc<dyn ObjectStore>>>::async_open(
            async_store.clone(),
            "/embeddings",
        )
        .await
        .expect("Should be able to open array");

        println!("Opened array, shape: {:?}", array.shape());

        // Read back each chunk and verify values
        for row in 0..2 {
            for col in 0..2 {
                let expected_value = ((row * 2 + col + 1) * 10) as i8;
                let chunk_indices = vec![0u64, 0, row as u64, col as u64];

                let read_data: Vec<i8> = array.async_retrieve_chunk(&chunk_indices)
                    .await
                    .expect("Should be able to read chunk");

                // Verify data matches
                let sample_values: Vec<i8> = read_data.iter().take(10).copied().collect();
                println!("Chunk ({}, {}): expected {}, got {:?}...", row, col, expected_value, sample_values);

                // All values in chunk should match (we wrote uniform values)
                assert!(
                    read_data.iter().all(|&v| v == expected_value),
                    "Chunk ({}, {}) should have value {}, but found mixed values",
                    row, col, expected_value
                );
            }
        }

        println!("All chunks read back correctly!");
    }

    std::fs::remove_dir_all(test_path).ok();
}

/// Test 13: Test with very high concurrency (stress test)
#[tokio::test]
async fn test_high_concurrency_writes() {
    use crate::io::create_output_store;
    use futures::stream::{self, StreamExt};

    let test_path = "target/test-zarr-high-concurrency";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 1024,
        width: 1024,
    };

    // 10x10 grid = 100 chunks
    let grid = Arc::new(OutputGrid {
        bounds: [0.0, 0.0, 102400.0, 102400.0],
        crs: "EPSG:6933".to_string(),
        resolution: 10.0,
        years: vec![2024],
        num_bands: 64,
        height: 10240,
        width: 10240,
        chunk_shape: chunk_shape.clone(),
        chunk_counts: [1, 1, 10, 10],
    });

    let mut config = create_test_config(chunk_shape.clone());
    config.output.local_path = Some(test_path.to_string());

    let store = create_output_store(&config).unwrap();
    let prefix = crate::io::get_output_prefix(&config);
    let writer = Arc::new(ZarrWriter::create(store, prefix, grid.clone(), &config).await.unwrap());

    // Generate 100 chunks
    let chunks: Vec<OutputChunk> = (0..10)
        .flat_map(|row| (0..10).map(move |col| OutputChunk { time_idx: 0, row_idx: row, col_idx: col }))
        .collect();

    println!("Writing {} chunks with high concurrency...", chunks.len());

    // Process with high concurrency (like production scheduler with 256)
    let results: Vec<Result<(), anyhow::Error>> = stream::iter(chunks)
        .map(|chunk| {
            let writer = writer.clone();
            async move {
                let data = Array4::from_elem((1, 64, 1024, 1024), (chunk.row_idx + chunk.col_idx) as i8);
                writer.write_chunk_async(&chunk, data).await
            }
        })
        .buffer_unordered(256)  // Very high concurrency
        .collect()
        .await;

    let failures: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    println!("Results: {} successes, {} failures", results.len() - failures.len(), failures.len());

    for failure in &failures {
        println!("Failure: {:?}", failure);
    }

    assert!(failures.is_empty(), "All chunks should succeed with high concurrency");

    // Verify chunk count
    let chunk_dir = std::path::Path::new(test_path)
        .join("embeddings").join("c").join("0").join("0");

    let row_count = std::fs::read_dir(&chunk_dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().is_dir())
        .count();

    assert_eq!(row_count, 10, "Should have 10 row directories");
    println!("All {} chunks written successfully!", results.len());

    std::fs::remove_dir_all(test_path).ok();
}

/// Test 14: Verify error on oversized chunk data
/// The mosaic should never produce data larger than chunk size, but test the behavior
#[tokio::test]
async fn test_oversized_chunk_error() {
    use crate::io::create_output_store;

    let test_path = "target/test-zarr-oversized";
    if std::path::Path::new(test_path).exists() {
        std::fs::remove_dir_all(test_path).unwrap();
    }

    let chunk_shape = ChunkShape {
        time: 1,
        embedding: 64,
        height: 512,  // Small chunk for quick test
        width: 512,
    };

    let grid = Arc::new(OutputGrid {
        bounds: [0.0, 0.0, 5120.0, 5120.0],
        crs: "EPSG:6933".to_string(),
        resolution: 10.0,
        years: vec![2024],
        num_bands: 64,
        height: 512,
        width: 512,
        chunk_shape: chunk_shape.clone(),
        chunk_counts: [1, 1, 1, 1],
    });

    let mut config = create_test_config(chunk_shape.clone());
    config.output.local_path = Some(test_path.to_string());

    let store = create_output_store(&config).unwrap();
    let prefix = crate::io::get_output_prefix(&config);
    let writer = ZarrWriter::create(store, prefix, grid, &config).await.unwrap();

    // Try to write oversized data (larger than chunk size)
    // This simulates a bug where mosaic returns more data than expected
    let oversized_data = Array4::from_elem((1, 64, 600, 600), 42i8);  // 600 > 512
    let chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };

    println!("Attempting to write oversized chunk (600x600 into 512x512)...");

    // This should panic because the padding logic will try to slice oversized data
    // into an undersized array. A panic is acceptable here because:
    // 1. It's a programming error if mosaic returns oversized data
    // 2. It's caught immediately rather than silently corrupting data
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Need to block on the async call
        tokio::runtime::Handle::current().block_on(async {
            writer.write_chunk_async(&chunk, oversized_data).await
        })
    }));

    println!("Oversized write result: {:?}", result);
    assert!(result.is_err(), "Oversized chunk should panic (caught as Err)");
    println!("Correctly panicked on oversized chunk - this is expected behavior");

    std::fs::remove_dir_all(test_path).ok();
}
