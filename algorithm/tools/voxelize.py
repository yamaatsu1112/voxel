#!/usr/bin/env python3
"""Voxelize an OBJ mesh at multiple resolutions using Open3D.

Output format (one file per resolution):
  - First line: "resolution <N>"
  - Remaining lines: "x y z" (integer voxel coordinates, 0-indexed)

Usage:
  python tools/voxelize.py model.obj -o output_dir/ --resolutions 64,128,256,512
  python tools/voxelize.py model.obj  # writes to dataset/svo/
"""

import argparse
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import open3d as o3d


@contextmanager
def triangulated_mesh_path(mesh_path: str):
    path = Path(mesh_path)
    if path.suffix.lower() != ".obj":
        yield mesh_path
        return

    with tempfile.NamedTemporaryFile(
        "w", suffix=".obj", prefix=f"{path.stem}_triangulated_", delete=False
    ) as output:
        temp_path = output.name
        with open(mesh_path) as input_file:
            for line in input_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    output.write(line)
                    continue

                parts = stripped.split()
                if parts[0] != "f" or len(parts) <= 4:
                    output.write(line)
                    continue

                vertices = parts[1:]
                for i in range(1, len(vertices) - 1):
                    output.write(f"f {vertices[0]} {vertices[i]} {vertices[i + 1]}\n")

    try:
        yield temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)


def voxelize_mesh(mesh_path: str, resolution: int) -> np.ndarray:
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")

    with triangulated_mesh_path(mesh_path) as readable_mesh_path:
        mesh = o3d.io.read_triangle_mesh(readable_mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_extent = max(extent)
    if max_extent <= 0:
        raise RuntimeError("Mesh has zero extent")

    center = bbox.get_center()
    half = max_extent / 2.0
    origin = center - np.array([half, half, half])
    max_bound = origin + np.array([max_extent, max_extent, max_extent])
    voxel_size = max_extent / resolution

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=voxel_size,
        min_bound=origin,
        max_bound=max_bound,
    )

    voxels = []
    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index
        x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            voxels.append((x, y, z))

    return np.array(voxels, dtype=np.uint32)


def write_voxel_file(voxels: np.ndarray, resolution: int, path: str) -> None:
    with open(path, "w") as f:
        f.write(f"resolution {resolution}\n")
        for x, y, z in voxels:
            f.write(f"{x} {y} {z}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxelize OBJ mesh using Open3D")
    parser.add_argument("mesh", help="Path to OBJ file")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="Output directory (default: dataset/svo/)"
    )
    parser.add_argument(
        "--resolutions",
        default="64,128,256,512",
        help="Comma-separated list of resolutions (default: 64,128,256,512)",
    )
    args = parser.parse_args()

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    mesh_name = Path(args.mesh).stem

    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent.parent
        output_dir = script_dir / "dataset" / "svo"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for res in resolutions:
        print(f"Voxelizing {args.mesh} at resolution {res}...", file=sys.stderr)
        voxels = voxelize_mesh(args.mesh, res)

        out_path = output_dir / f"{mesh_name}_{res}.txt"
        write_voxel_file(voxels, res, str(out_path))
        print(
            f"  -> {out_path} ({len(voxels)} voxels)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
