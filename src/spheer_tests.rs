//! End-to-end tests for the Spheer path (float32, top-down COGs, folder-scan discovery),
//! exercised against synthetic fixtures under `tests/fixtures/spheer`.
//!
//! Regenerate the fixtures with:
//!   `pixi run python tests/fixtures/generate_spheer_fixtures.py`
//!
//! Fixture layout (EPSG:32631, 10 m, 4 bands, 64x64, top-down):
//! - `31UGV/2020.tif` — tile A, native bounds [500000, 4259360, 500640, 4260000],
//!   values A[b, r, c] = 1000*b + 10*r + c
//! - `31UGW/2020.tif` — tile B (east-adjacent), native bounds [500640, 4259360, 501280, 4260000],
//!   values B[b, r, c] = 5000 + 1000*b + 10*r + c

use crate::config::{
    ChunkShape, Config, InputConfig, OutputConfig, ProcessingConfig, ShardingConfig,
};
use crate::dtype::{ChunkData, PixelData};
use crate::index::{CogTile, OutputChunk, OutputGrid};
use crate::io::{CogReader, PixelWindow, ZarrWriter};
use crate::transform::{mosaic_tiles, ReprojectConfig};
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use std::sync::Arc;

const FIXTURE_ROOT: &str = "tests/fixtures/spheer";
// Union of the two tiles' native bounds (EPSG:32631).
const UNION_BOUNDS: [f64; 4] = [500000.0, 4259360.0, 501280.0, 4260000.0];

fn fixtures_present() -> bool {
    std::path::Path::new(FIXTURE_ROOT)
        .join("albatross-EU-v2025/nl-tiles/31UGV/2020.tif")
        .exists()
}

fn spheer_config() -> Config {
    Config {
        dataset: "spheer".to_string(),
        input: InputConfig {
            // Scan the whole store (fixtures rooted at FIXTURE_ROOT).
            index_path: "".to_string(),
            cog_bucket: "unused".to_string(),
        },
        output: OutputConfig {
            local_path: Some("unused".to_string()),
            bucket: None,
            prefix: None,
            crs: "EPSG:32631".to_string(),
            resolution: 10.0,
            num_bands: 4,
            chunk_shape: ChunkShape {
                time: 1,
                embedding: 4,
                height: 64,
                width: 128,
            },
            sharding: ShardingConfig {
                enabled: false,
                shard_shape: [16, 16],
            },
            compression_level: 3,
            years: Some(vec![2020]),
        },
        processing: ProcessingConfig::default(),
        filter: None,
    }
}

fn fixture_store() -> Arc<dyn ObjectStore> {
    Arc::new(LocalFileSystem::new_with_prefix(FIXTURE_ROOT).unwrap())
}

async fn build_index() -> crate::index::InputIndex {
    let config = spheer_config();
    let dataset = config.dataset().unwrap();
    crate::discovery::build_input_index(dataset.as_ref(), &config, fixture_store())
        .await
        .unwrap()
}

/// Read both fixture tiles in full and mosaic them into the union grid (identity CRS).
async fn read_and_mosaic() -> ChunkData {
    let store = fixture_store();
    let reader = CogReader::new(store.clone());
    let index = build_index().await;
    let tiles = index.all_tiles();
    let tile_a = tiles
        .iter()
        .find(|t| t.s3_path.contains("31UGV"))
        .expect("tile A present");
    let tile_b = tiles
        .iter()
        .find(|t| t.s3_path.contains("31UGW"))
        .expect("tile B present");

    let win_a = reader
        .read_window(tile_a, PixelWindow::new(0, 0, 64, 64), tile_a.bounds_native)
        .await
        .unwrap();
    let win_b = reader
        .read_window(tile_b, PixelWindow::new(0, 0, 64, 64), tile_b.bounds_native)
        .await
        .unwrap();

    let rc = ReprojectConfig {
        target_crs: "EPSG:32631".to_string(),
        target_resolution: 10.0,
        target_bounds: UNION_BOUNDS,
        target_shape: (64, 128),
        num_bands: 4,
    };
    mosaic_tiles(&[win_a, win_b], &rc).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_from_cog_scan_builds_index() {
    if !fixtures_present() {
        eprintln!("skipping: run tests/fixtures/generate_spheer_fixtures.py to create fixtures");
        return;
    }
    let index = build_index().await;
    assert_eq!(index.len(), 2, "should discover both fixture tiles");

    for tile in index.all_tiles() {
        assert_eq!(tile.crs, "EPSG:32631", "CRS from GeoKeys");
        assert_eq!(tile.year, 2020, "year parsed from filename");
        assert!((tile.resolution - 10.0).abs() < 1e-6);
    }

    let a = index
        .all_tiles()
        .iter()
        .find(|t| t.s3_path.contains("31UGV"))
        .unwrap();
    // Native bounds derived from the geotransform + dims.
    assert!((a.bounds_native[0] - 500000.0).abs() < 1e-3, "min_x");
    assert!((a.bounds_native[1] - 4259360.0).abs() < 1e-3, "min_y");
    assert!((a.bounds_native[2] - 500640.0).abs() < 1e-3, "max_x");
    assert!((a.bounds_native[3] - 4260000.0).abs() < 1e-3, "max_y");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_spheer_read_window_is_top_down_float32() {
    if !fixtures_present() {
        return;
    }
    let store = fixture_store();
    let reader = CogReader::new(store.clone());
    let index = build_index().await;
    let tile = index
        .all_tiles()
        .iter()
        .find(|t| t.s3_path.contains("31UGV"))
        .unwrap();

    let wd = reader
        .read_window(tile, PixelWindow::new(0, 0, 64, 64), tile.bounds_native)
        .await
        .unwrap();

    assert!(!wd.is_bottom_up, "standard COG should be read as top-down");
    match &wd.data {
        PixelData::Float32(a) => {
            assert_eq!(a.dim(), (4, 64, 64));
            // Orientation preserved: row 0 is north (value grows southward with r).
            assert_eq!(a[[0, 0, 0]], 0.0, "band0 north-west");
            assert_eq!(a[[0, 63, 0]], 630.0, "band0 south-west");
            assert_eq!(a[[0, 0, 63]], 63.0, "band0 north-east");
            assert_eq!(a[[1, 0, 0]], 1000.0, "band1 north-west");
        }
        _ => panic!("expected Float32 pixel data"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_spheer_mosaic_float32() {
    if !fixtures_present() {
        return;
    }
    match read_and_mosaic().await {
        ChunkData::Float32(a) => {
            assert_eq!(a.shape(), &[1, 4, 64, 128]);
            // Tile A on the left (cols 0..64), tile B on the right (cols 64..128),
            // vertical orientation preserved (no flip).
            assert_eq!(a[[0, 0, 0, 0]], 0.0, "A band0 nw");
            assert_eq!(a[[0, 0, 63, 0]], 630.0, "A band0 sw");
            assert_eq!(a[[0, 1, 0, 0]], 1000.0, "A band1 nw");
            assert_eq!(a[[0, 0, 0, 64]], 5000.0, "B band0 nw");
            assert_eq!(a[[0, 0, 0, 127]], 5063.0, "B band0 ne");
            assert_eq!(a[[0, 0, 63, 127]], 5693.0, "B band0 se");
        }
        _ => panic!("expected Float32 chunk"),
    }
}

/// Read a small window from a REAL Spheer COG on HuggingFace, mosaic it (identity
/// reprojection), write a GeoZarr, and assert the mosaic matches the source read.
///
/// Network + gated-access test: skipped unless `HF_TOKEN` is set. Writes output under
/// `SPHEER_TEST_DIR` (default /tmp/spheer_hf_test) so the Python cross-check can compare
/// the GeoZarr against the source COG via rasterio.
///
/// Run: `HF_TOKEN=... cargo test -- --ignored test_spheer_hf_small_mosaic --nocapture`
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_spheer_hf_small_mosaic() {
    let token = match std::env::var("HF_TOKEN") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skip: HF_TOKEN not set");
            return;
        }
    };
    let out_dir = std::env::var("SPHEER_TEST_DIR").unwrap_or_else(|_| "/tmp/spheer_hf_test".into());
    std::fs::create_dir_all(&out_dir).unwrap();

    let store = crate::io::create_hf_store("spheer/spheer-fm-embeddings", &token, 32).unwrap();
    let reader = CogReader::new(store);

    let key = "albatross-EU-v2025/nl-tiles/31UGV/2020.tif";
    let hdr = reader.read_header(key).await.unwrap();
    let gt = hdr.geo_transform.expect("geotransform");
    let epsg = hdr.epsg.expect("epsg");
    let crs = format!("EPSG:{}", epsg);
    println!(
        "HF COG {}: {}x{} epsg={} gt=(a{},e{},c{},f{})",
        key, hdr.width, hdr.height, epsg, gt.a, gt.e, gt.c, gt.f
    );

    // A 256x256 window aligned to the 512-px internal-tile grid (fully inside one tile).
    let (px, py, w, h) = (5120usize, 5120usize, 256usize, 256usize);
    let (x0, y0) = gt.pixel_to_world(px as f64, py as f64);
    let (x1, y1) = gt.pixel_to_world((px + w) as f64, (py + h) as f64);
    let bounds_native = [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)];

    let (fx0, fy0) = gt.pixel_to_world(0.0, 0.0);
    let (fx1, fy1) = gt.pixel_to_world(hdr.width as f64, hdr.height as f64);
    let tile = CogTile {
        tile_id: key.to_string(),
        s3_path: key.to_string(),
        crs: crs.clone(),
        bounds_native: [fx0.min(fx1), fy0.min(fy1), fx0.max(fx1), fy0.max(fy1)],
        bounds_wgs84: [0.0; 4],
        footprint_wgs84: CogTile::footprint_from_wgs84_bounds(&[0.0, 0.0, 1.0, 1.0]),
        resolution: gt.a.abs(),
        year: 2020,
    };

    let win = reader
        .read_window(&tile, PixelWindow::new(px, py, w, h), bounds_native)
        .await
        .unwrap();
    assert!(!win.is_bottom_up, "Spheer COG must read as top-down");
    let src = match &win.data {
        PixelData::Float32(a) => a.clone(),
        other => panic!("expected Float32, got {:?} dtype", other.data_type()),
    };
    let nbands = src.dim().0;
    println!(
        "read window {:?}; samples [0,0,0]={} [50,100,150]={}",
        src.dim(),
        src[[0, 0, 0]],
        src[[50.min(nbands - 1), 100, 150]]
    );

    // Identity mosaic (output CRS/res/grid == source).
    let rc = ReprojectConfig {
        target_crs: crs.clone(),
        target_resolution: gt.a.abs(),
        target_bounds: bounds_native,
        target_shape: (h, w),
        num_bands: nbands,
    };
    let mosaic = mosaic_tiles(&[win], &rc).unwrap();
    let marr = match &mosaic {
        ChunkData::Float32(a) => a.clone(),
        _ => panic!("expected Float32 chunk"),
    };
    assert_eq!(marr.shape(), &[1, nbands, h, w]);
    for b in [0usize, nbands / 2, nbands - 1] {
        for (r, c) in [(0usize, 0usize), (10, 20), (h - 1, w - 1)] {
            assert_eq!(marr[[0, b, r, c]], src[[b, r, c]], "mosaic != source at b{b} r{r} c{c}");
        }
    }
    println!("in-process check OK: identity mosaic matches source read");

    // Write a small GeoZarr for the Python cross-check against rasterio.
    let grid = Arc::new(OutputGrid {
        bounds: bounds_native,
        crs: crs.clone(),
        resolution: gt.a.abs(),
        years: vec![2020],
        num_bands: nbands,
        height: h,
        width: w,
        chunk_shape: ChunkShape { time: 1, embedding: nbands, height: h, width: w },
        chunk_counts: [1, 1, 1, 1],
    });
    let mut cfg = spheer_config();
    cfg.output.crs = crs.clone();
    cfg.output.num_bands = nbands;
    cfg.output.chunk_shape = ChunkShape { time: 1, embedding: nbands, height: h, width: w };

    let zpath = format!("{}/mosaic.zarr", out_dir);
    let _ = std::fs::remove_dir_all(&zpath);
    std::fs::create_dir_all(&zpath).unwrap();
    let ostore: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&zpath).unwrap());
    let writer = Arc::new(ZarrWriter::create(ostore, "", grid, &cfg).await.unwrap());
    let chunk = OutputChunk { time_idx: 0, row_idx: 0, col_idx: 0 };
    let wc = writer.clone();
    tokio::task::spawn_blocking(move || wc.write_chunk_dyn(&chunk, mosaic))
        .await
        .unwrap()
        .unwrap();
    writer.finalize().unwrap();

    // Record window params for the Python comparison.
    std::fs::write(
        format!("{}/window.json", out_dir),
        format!(
            r#"{{"key":"{key}","px":{px},"py":{py},"w":{w},"h":{h},"bands":{nbands},"zarr":"{zpath}","epsg":{epsg}}}"#
        ),
    )
    .unwrap();
    println!("wrote GeoZarr {} + window.json", zpath);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_spheer_zarr_write_float32_metadata() {
    use zarrs::array::Array;
    use zarrs_object_store::AsyncObjectStore;

    if !fixtures_present() {
        return;
    }

    let mosaic = read_and_mosaic().await;

    let tmp = tempfile::TempDir::new().unwrap();
    let out_store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(tmp.path()).unwrap());

    let grid = Arc::new(OutputGrid {
        bounds: UNION_BOUNDS,
        crs: "EPSG:32631".to_string(),
        resolution: 10.0,
        years: vec![2020],
        num_bands: 4,
        height: 64,
        width: 128,
        chunk_shape: ChunkShape {
            time: 1,
            embedding: 4,
            height: 64,
            width: 128,
        },
        chunk_counts: [1, 1, 1, 1],
    });

    let config = spheer_config();
    let writer = Arc::new(
        ZarrWriter::create(out_store.clone(), "", grid, &config)
            .await
            .unwrap(),
    );

    let chunk = OutputChunk {
        time_idx: 0,
        row_idx: 0,
        col_idx: 0,
    };
    let writer_clone = writer.clone();
    tokio::task::spawn_blocking(move || writer_clone.write_chunk_dyn(&chunk, mosaic))
        .await
        .unwrap()
        .unwrap();
    writer.finalize().unwrap();

    // Reopen from a fresh store and verify float32 values survive the round-trip.
    let read_store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(tmp.path()).unwrap());
    let async_store = Arc::new(AsyncObjectStore::new(read_store));
    let array = Array::<AsyncObjectStore<Arc<dyn ObjectStore>>>::async_open(async_store, "/embeddings")
        .await
        .unwrap();

    let data = array
        .async_retrieve_chunk::<Vec<f32>>(&[0, 0, 0, 0])
        .await
        .unwrap();
    // C-order [time=1, band=4, y=64, x=128]; index of (band, row, col):
    let idx = |b: usize, r: usize, c: usize| (b * 64 + r) * 128 + c;
    assert_eq!(data[idx(0, 0, 0)], 0.0);
    assert_eq!(data[idx(0, 0, 64)], 5000.0);
    assert_eq!(data[idx(0, 63, 127)], 5693.0);
    assert_eq!(data[idx(1, 0, 0)], 1000.0);

    // Group metadata: float32, and NO quantization block (Spheer is raw float32).
    let group_json = tmp.path().join("zarr.json");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(group_json).unwrap()).unwrap();
    let attrs = meta.get("attributes").expect("group attributes");
    assert_eq!(
        attrs.get("geoemb:data_type").and_then(|v| v.as_str()),
        Some("float32")
    );
    assert!(
        attrs.get("geoemb:quantization").is_none(),
        "float32 dataset must not declare quantization"
    );

    // Array metadata: data_type is float32.
    let array_json = tmp.path().join("embeddings").join("zarr.json");
    let ameta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(array_json).unwrap()).unwrap();
    assert_eq!(
        ameta.get("data_type").and_then(|v| v.as_str()),
        Some("float32")
    );
}
