#!/usr/bin/env python3
"""
E2E validation: Compare Zarr output to source COG using cosine similarity.

Outputs a GeoTIFF showing per-pixel cosine similarity.
"""

import argparse
import os
import sys
import numpy as np
import zarr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
import yaml


NODATA = -128
COG_MOSAIC_CACHE = "cog_mosaic_cache.tif"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="similarity_map.tif")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 1. Read the Zarr array
    print("Reading Zarr...")
    root = zarr.open_group(args.zarr_path, mode='r')
    zarr_arr = root['embeddings']
    attrs = dict(zarr_arr.attrs)

    zarr_data = np.array(zarr_arr[0, :, :, :])  # Shape: (64, H, W)
    print(f"  Zarr shape: {zarr_data.shape}")

    bounds = attrs['bounds']  # [min_x, min_y, max_x, max_y]
    resolution = attrs['resolution']
    output_crs = attrs['crs']
    _, height, width = zarr_data.shape

    # 2. Read the COG (via VRT) - reproject to match Zarr grid
    print("Reading COG...")

    # Get the VRT path from config filter bounds
    # For now, find the first matching tile
    import pandas as pd
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    import io

    index_path = config['input']['index_path']
    s3 = boto3.client('s3', config=BotoConfig(signature_version=UNSIGNED))
    bucket = index_path.split('/')[2]
    key = '/'.join(index_path.split('/')[3:])
    response = s3.get_object(Bucket=bucket, Key=key)
    tiles_df = pd.read_parquet(io.BytesIO(response['Body'].read()))

    # Convert Zarr bounds (EPSG:6933) to WGS84 for tile filtering
    transformer = Transformer.from_crs(output_crs, "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds[0], bounds[1])
    max_lon, max_lat = transformer.transform(bounds[2], bounds[3])
    print(f"  Zarr bounds (WGS84): [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]")

    # Filter by year from Zarr metadata
    start_year = attrs.get('start_year')
    if start_year:
        tiles_df = tiles_df[tiles_df['year'] == start_year]

    # Filter tiles by Zarr bounds
    tiles_df = tiles_df[
        (tiles_df['wgs84_west'] < max_lon) &
        (tiles_df['wgs84_east'] > min_lon) &
        (tiles_df['wgs84_south'] < max_lat) &
        (tiles_df['wgs84_north'] > min_lat)
    ]
    print(f"  Tiles overlapping Zarr bounds: {len(tiles_df)}")

    if len(tiles_df) == 0:
        print("ERROR: No tiles found overlapping Zarr bounds")
        return 1

    dst_transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Check for cached COG mosaic
    if os.path.exists(COG_MOSAIC_CACHE):
        print(f"  Loading cached mosaic from {COG_MOSAIC_CACHE}...")
        with rasterio.open(COG_MOSAIC_CACHE) as src:
            cog_data = src.read().astype(np.int8)
        print(f"  Mosaic shape: {cog_data.shape}")
    else:
        # Read and reproject ALL overlapping COGs, then compute mean mosaic
        print("  Building mosaic from COGs (will be cached)...")

        # Accumulate sum and count for mean calculation (use float32 for accumulation)
        mosaic_sum = np.zeros((64, height, width), dtype=np.float32)
        mosaic_count = np.zeros((height, width), dtype=np.int16)

        with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
            for i, (_, tile) in enumerate(tiles_df.iterrows()):
                vrt_path = tile['path'].replace('s3://', '/vsis3/').replace('.tiff', '.vrt')
                print(f"  [{i+1}/{len(tiles_df)}] Reading {vrt_path.split('/')[-1]}...")

                # Temporary buffer for this tile
                tile_data = np.full((64, height, width), NODATA, dtype=np.int8)

                with rasterio.open(vrt_path) as src:
                    reproject(
                        source=rasterio.band(src, list(range(1, 65))),
                        destination=tile_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=NODATA,
                        dst_transform=dst_transform,
                        dst_crs=output_crs,
                        dst_nodata=NODATA,
                        resampling=Resampling.nearest,
                    )

                # Accumulate where valid (check band 0 for nodata)
                valid = tile_data[0] != NODATA
                mosaic_sum[:, valid] += tile_data[:, valid].astype(np.float32)
                mosaic_count[valid] += 1

        # Compute mean and convert back to int8
        print("  Computing mean mosaic...")
        with np.errstate(divide='ignore', invalid='ignore'):
            mosaic_mean = mosaic_sum / mosaic_count[np.newaxis, :, :]

        # Round to nearest integer and clip to int8 range
        cog_data = np.clip(np.round(mosaic_mean), -127, 127).astype(np.int8)
        # Set nodata where count is 0
        cog_data[:, mosaic_count == 0] = NODATA

        print(f"  Mosaic shape: {cog_data.shape}")

        # Save cache
        print(f"  Saving mosaic cache to {COG_MOSAIC_CACHE}...")
        with rasterio.open(
            COG_MOSAIC_CACHE,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=64,
            dtype=np.int8,
            crs=output_crs,
            transform=dst_transform,
            nodata=NODATA,
        ) as dst:
            dst.write(cog_data)

    # 3. Compare them - compute cosine similarity per pixel
    print("Computing cosine similarity...")

    # Mask where either is nodata
    zarr_valid = zarr_data[0] != NODATA
    cog_valid = cog_data[0] != NODATA
    valid_mask = zarr_valid & cog_valid

    # Dot product and norms
    dot_product = np.sum(zarr_data * cog_data, axis=0)
    zarr_norm = np.sqrt(np.sum(zarr_data ** 2, axis=0))
    cog_norm = np.sqrt(np.sum(cog_data ** 2, axis=0))

    # Cosine similarity (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = dot_product / (zarr_norm * cog_norm)

    # Set invalid pixels to NaN
    similarity[~valid_mask] = np.nan

    # Save as GeoTIFF
    print(f"Saving {args.output}...")
    with rasterio.open(
        args.output,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=output_crs,
        transform=dst_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(similarity.astype(np.float32), 1)

    # Print statistics
    valid_sims = similarity[~np.isnan(similarity)]
    print(f"\nResults:")
    print(f"  Valid pixels: {len(valid_sims):,} / {height * width:,}")
    print(f"  Mean similarity: {np.mean(valid_sims):.4f}")
    print(f"  Min similarity:  {np.min(valid_sims):.4f}")
    print(f"  Max similarity:  {np.max(valid_sims):.4f}")
    print(f"  Std similarity:  {np.std(valid_sims):.4f}")
    print(f"\n  % below 0.9: {np.sum(valid_sims < 0.9) / len(valid_sims) * 100:.1f}%")
    print(f"  % below 0.5: {np.sum(valid_sims < 0.5) / len(valid_sims) * 100:.1f}%")

    if np.mean(valid_sims) >= 0.95:
        print("\nPASS")
        return 0
    else:
        print(f"\nFAIL (mean={np.mean(valid_sims):.4f})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
