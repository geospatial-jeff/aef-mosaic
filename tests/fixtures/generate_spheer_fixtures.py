"""Generate synthetic Spheer-like COG fixtures for the Rust integration tests.

Creates two east-adjacent, standard **top-down** float32 COGs in EPSG:32631 (UTM 31N),
georeferenced via ModelPixelScale + ModelTiepoint (as GDAL/rasterio COGs are), laid out
like Spheer's `albatross-EU-v2025/nl-tiles/<MGRS>/<year>.tif`.

Pixel values are deterministic so the Rust test can verify no vertical flip and correct
east-west stitching:

    tile_A[b, r, c] = 1000*b + 10*r + c
    tile_B[b, r, c] = 5000 + 1000*b + 10*r + c

Run: `pixi run python tests/fixtures/generate_spheer_fixtures.py`
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

BANDS = 4
SIZE = 64          # 64 x 64 pixels
RES = 10.0         # 10 m pixels
NORTH = 4260000.0  # top edge (max y)
WEST_A = 500000.0  # tile A left edge
WEST_B = WEST_A + SIZE * RES  # tile B is immediately east of A
CRS = "EPSG:32631"

FIXTURE_ROOT = Path(__file__).parent / "spheer" / "albatross-EU-v2025" / "nl-tiles"


def make_data(base: float) -> np.ndarray:
    data = np.zeros((BANDS, SIZE, SIZE), dtype=np.float32)
    for b in range(BANDS):
        for r in range(SIZE):
            for c in range(SIZE):
                data[b, r, c] = base + 1000 * b + 10 * r + c
    return data


def write_cog(path: Path, west: float, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Top-down transform: origin at the top-left, negative y-scale.
    transform = from_origin(west, NORTH, RES, RES)
    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": BANDS,
        "dtype": "float32",
        "crs": CRS,
        "transform": transform,
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    print(f"wrote {path}  west={west}  shape={data.shape}")


def main() -> None:
    write_cog(FIXTURE_ROOT / "31UGV" / "2020.tif", WEST_A, make_data(0.0))
    write_cog(FIXTURE_ROOT / "31UGW" / "2020.tif", WEST_B, make_data(5000.0))


if __name__ == "__main__":
    main()
