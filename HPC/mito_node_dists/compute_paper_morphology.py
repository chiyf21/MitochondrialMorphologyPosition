#!/usr/bin/env python3
"""Compute paper morphology descriptors from masks only; SWC supplies a radius proxy."""
import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mitochondria_feature_encoder import EncoderConfig, encode_one

PAPER_FEATURES = [
    "mito SA", "mito CA", "skel CA", "skel Circum", "convex hull compactness",
    "PC1 Length", "PC1 inertia moment", "PC1 symmetry", "PC1 CA", "PC1 Circum",
    "PC2 Length", "PC2 inertia moment", "PC2 symmetry", "PC2 CA", "PC2 Circum",
    "PC3 Length", "PC3 inertia moment", "PC3 symmetry", "PC3 CA", "PC3 Circum",
    "mito diameter", "length_along_skeleton_nm",
]


@lru_cache(maxsize=256)
def load_swc(root, body):
    folder = Path(root) / "skeleton_metadata" / body
    manifest = json.loads((folder / "neuron_pointcloud_manifest.json").read_text())
    path = Path(manifest["skeleton_swc"])
    if not path.is_file():
        path = folder / "skeleton" / f"{body}.swc"
    points, radii = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 7:
            points.append([float(fields[2]), float(fields[3]), float(fields[4])])
            radii.append(float(fields[5]))
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if not len(points) or not len(radii):
        raise ValueError(f"No SWC nodes: {path}")
    return cKDTree(points), radii


def compute_one(item):
    root, row, spacing = item
    root = Path(root)
    base = {k: row.get(k, "") for k in ("record_id", "source_run_id", "body_id", "cell_type", "global_instance", "voxel_count")}
    try:
        meta = json.loads((root / row["metadata_path"]).read_text())
        arr = np.load(root / row["mask_path"], mmap_mode="r", allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"mask must be 3D: {arr.shape}")
        x0, y0, z0, x1, y1, z1 = map(int, meta["bbox_xyz"])
        extent_xyz = np.array([x1 - x0, y1 - y0, z1 - z0])
        shape_xyz = np.array(arr.shape[::-1])
        if not (np.array_equal(extent_xyz, shape_xyz) or np.array_equal(extent_xyz + 1, shape_xyz)):
            raise ValueError(f"bbox/mask mismatch: bbox={meta['bbox_xyz']}, mask={arr.shape}")

        # Tight crop after one sparse scan; avoids dense operations over padded crops.
        coords = np.nonzero(arr)
        if not len(coords[0]):
            raise ValueError("empty mask")
        lo = [int(c.min()) for c in coords]
        hi = [int(c.max()) + 1 for c in coords]
        mask = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] > 0)
        bbox_zyx = (slice(z0 + lo[0], z0 + hi[0]), slice(y0 + lo[1], y0 + hi[1]), slice(x0 + lo[2], x0 + hi[2]))
        seed = int(row["global_instance"])
        cfg = EncoderConfig(spacing_zyx_nm=tuple(spacing), min_voxels=1)
        morph = encode_one(mask, bbox_zyx, None, cfg, np.random.default_rng(seed))

        # SWC radius is in nm; use its nearest node as a circular local-section estimate.
        xyz_nm = np.array([morph["centroid_x_voxel"] * spacing[2], morph["centroid_y_voxel"] * spacing[1], morph["centroid_z_voxel"] * spacing[0]])
        tree, radii = load_swc(str(root), str(row["body_id"]))
        _, nearest = tree.query(xyz_nm, k=1)
        radius = float(radii[int(nearest)])
        values = {k: morph.get(k, math.nan) for k in PAPER_FEATURES if k in morph}
        values.update({"skel CA": math.pi * radius ** 2, "skel Circum": 2 * math.pi * radius,
                       "skeleton_radius_nm": radius, "skeleton_section_method": "nearest_SWC_radius_circle_estimate"})
        result = {**base, **values,
                  "centroid_x_voxel": morph["centroid_x_voxel"], "centroid_y_voxel": morph["centroid_y_voxel"],
                  "centroid_z_voxel": morph["centroid_z_voxel"], "voxel_size_zyx_nm": ",".join(map(str, spacing)), "status": "ok"}
        return result
    except Exception as exc:
        return {**base, "status": f"error: {type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--spacing-zyx-nm", type=float, nargs=3, default=(30., 8., 8.))
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with (args.dataset_root / "metadata/instances.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = rows[args.start_index:]
    if args.limit:
        rows = rows[:args.limit]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "source_run_id", "body_id", "cell_type", "global_instance", "voxel_count", *PAPER_FEATURES,
              "skeleton_radius_nm", "skeleton_section_method", "centroid_x_voxel", "centroid_y_voxel", "centroid_z_voxel", "voxel_size_zyx_nm", "status"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.output_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for i, result in enumerate(pool.map(compute_one, [(str(args.dataset_root), row, args.spacing_zyx_nm) for row in rows], chunksize=1), 1):
            writer.writerow(result)
            if i == 1 or i % 100 == 0 or i == len(rows):
                print(f"processed {i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
