#!/usr/bin/env python3
"""Materialize physical-nm copies of explicitly identified Fish2 segmentation-grid SWCs."""

import argparse
import json
from pathlib import Path

SEGMENTATION_VOXEL_NM = (16, 16, 15)


def normalize(source, output, body):
    folder = source / str(body)
    path = folder / "skeleton" / f"{body}.swc"
    lines = path.read_text().splitlines()
    if any("coordinate_units" in line for line in lines if line.startswith("#")):
        raise ValueError(f"Expected an unmarked segmentation-grid SWC: {path}")
    target = output / str(body) / "skeleton" / path.name
    target.parent.mkdir(parents=True, exist_ok=False)
    with target.open("w") as f:
        f.write('# Coordinate normalization: segmentation XYZ grid x [16,16,15] nm; radii retained in nm.\n')
        for line in lines:
            if line.startswith("#") or not line.strip():
                f.write(line + "\n")
                continue
            fields = line.split()
            for i, scale in enumerate(SEGMENTATION_VOXEL_NM, 2):
                fields[i] = str(float(fields[i]) * scale)
            f.write(" ".join(fields) + "\n")
    manifest = json.loads((folder / "neuron_pointcloud_manifest.json").read_text())
    manifest["skeleton_swc"] = str(target.resolve())
    manifest["source_swc"] = str(path.resolve())
    manifest["swc_xyz_nm_per_source_unit"] = list(SEGMENTATION_VOXEL_NM)
    (target.parent.parent / "neuron_pointcloud_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("body_ids", nargs="+", type=int)
    a = p.parse_args()
    for body_id in a.body_ids:
        normalize(a.source, a.output, body_id)
