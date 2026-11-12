"""Compute the available paper morphometrics from the LC21 instance masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA

from paper_voxel_geometry import cart_2_spherical, find_cross_section, get_foreground_stats

FEATURES = ["size", "mito SA", "mito CA", "convex hull compactness"] + [
    f"PC{i} {name}" for i in (1, 2, 3)
    for name in ("Length", "inertia moment", "symmetry", "CA", "Circum")
] + ["mito diameter", "length along skeleton", "SBI"]
IDENTITY = ["record_id", "body_id", "cell_type", "split", "source_mito_label",
            "center_x", "center_y", "center_z", "mask_voxels", "status"]


def packed_rows(points, minimum, shape):
    # Integer row IDs preserve the authors' integer-coordinate uniqueness test.
    return np.ravel_multi_index(tuple((points - minimum).T), tuple(shape))


def symmetry(points, axis):
    base = points.astype(np.int64)
    flipped = base.copy()
    flipped[:, axis] *= -1
    minimum = np.minimum(base.min(0), flipped.min(0))
    shape = np.maximum(base.max(0), flipped.max(0)) - minimum + 1
    original = np.unique(packed_rows(base, minimum, shape))
    reflected = np.unique(packed_rows(flipped, minimum, shape))
    return 2 - len(np.union1d(original, reflected)) / len(original)


def diameter(sample):
    # A farthest pair is on the convex hull. Restrict the distance matrix
    # to hull vertices, and chunk the rare degenerate or very large case.
    try:
        sample = sample[ConvexHull(sample).vertices]
    except QhullError:
        pass
    if len(sample) <= 4096:
        return float(pdist(sample).max())
    return max(float(cdist(sample[i:i + 256], sample).max())
               for i in range(0, len(sample), 256))


def compute(row):
    result = {name: row.get(name, "") for name in IDENTITY}
    try:
        directory = Path(row["relative_dir"])
        meta = json.loads((directory / "metadata.json").read_text())
        if tuple(meta["voxel_size_nm_xyz"]) != (8, 8, 8):
            raise ValueError("Expected 8 nm isotropic FlyEM source")
        x, y, z = meta["center_xyz"]
        result.update(center_x=x, center_y=y, center_z=z)
        raw = np.load(directory / "mask.npy", mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3:
            raise ValueError("Expected ZYX mask")
        bounds = [np.flatnonzero(np.any(raw, axis=tuple(i for i in range(3) if i != a)))
                  for a in range(3)]
        if any(not len(b) for b in bounds):
            raise ValueError("Empty mask")
        tight = raw[tuple(slice(int(b[0]), int(b[-1]) + 1) for b in bounds)] != 0
        # Original box is [min-2, max+2), which leaves two low-side and one
        # high-side background voxels around the foreground.
        mask = np.pad(tight, ((2, 1),) * 3)
        origin = np.asarray(meta["bbox_xyz"][:3]) + np.array([b[0] for b in bounds[::-1]])
        coords = np.argwhere(tight)[:, ::-1] + origin
        result["mask_voxels"] = len(coords)
        size = float(meta["source_voxel_count"])
        if size <= 0 or len(coords) < 300:
            raise ValueError("Invalid source size or too few voxels for 1% PCA")
        box = np.array([[0, 0, 0], mask.shape])
        _, area = get_foreground_stats(mask, box, return_COM=False)
        boundary = tight & ~binary_erosion(tight)
        hull = ConvexHull(np.argwhere(boundary)[:, ::-1]).area
        count = len(coords) // 100
        seed = int.from_bytes(hashlib.sha256(str(row["record_id"]).encode()).digest()[:8], "big")
        rng = random.Random(seed)
        pca = PCA().fit(coords[rng.sample(range(len(coords)), count)])
        projected = pca.transform(coords)
        result.update({"size": size, "mito SA": float(area), "convex hull compactness": float(hull)})
        for axis in range(3):
            theta, phi = cart_2_spherical(*pca.components_[axis])[1:]
            plane = find_cross_section(theta, phi, mask)
            plane_area, perimeter = get_foreground_stats(plane, box, return_COM=False)
            prefix = f"PC{axis + 1} "
            orthogonal = [i for i in range(3) if i != axis]
            result.update({
                prefix + "Length": float(np.ptp(projected[:, axis])),
                prefix + "inertia moment": float(np.square(projected[:, orthogonal]).sum()),
                prefix + "symmetry": symmetry(projected, axis),
                prefix + "CA": int(plane_area),
                prefix + "Circum": float(perimeter),
            })
        result["mito diameter"] = diameter(coords[rng.sample(range(len(coords)), count)])
        radius = (3 * size / (4 * np.pi)) ** (1 / 3)
        result["SBI"] = float(np.log10(4 * np.pi * radius ** 2 / area))
        result["status"] = "ok_geometry"
    except Exception as exc:
        result["status"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["records"][:args.limit]
    if not rows or args.workers < 1:
        parser.error("Provide instances and at least one worker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.csv")
    if args.output.exists():
        raise FileExistsError(args.output)
    failures = 0
    with partial.open("x", newline="") as stream, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(stream, IDENTITY + FEATURES)
        writer.writeheader()
        for i, row in enumerate(pool.map(compute, rows, chunksize=1), 1):
            writer.writerow(row)
            failures += row["status"] != "ok_geometry"
            if i % 100 == 0 or i == len(rows):
                stream.flush()
                print(f"{i}/{len(rows)} processed; {failures} errors", flush=True)
    partial.replace(args.output)
    if failures:
        raise SystemExit(f"{failures} instances need review")


if __name__ == "__main__":
    main()
