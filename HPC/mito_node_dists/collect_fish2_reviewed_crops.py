#!/usr/bin/env python3
"""Copy exactly the instance crops used by an aligned Fish2 embedding panel."""

import argparse
import csv
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FILES = ("mask.npy", "em.npy", "metadata.json")


def copy_instance(source, staging, row):
    run, body, record = (row[k] for k in ("source_run_id", "body_id", "record_id"))
    instance = int(row["global_instance"])
    if not body.isdecimal() or Path(run).name != run or record != f"{run}__{body}__{instance:06d}":
        raise ValueError(f"Invalid instance key: {record}")
    original = source / run / "results" / body / "instances" / f"instance_{instance:06d}"
    metadata = json.loads((original / "metadata.json").read_text())
    if str(metadata["body_id"]) != body or int(metadata["global_instance"]) != instance:
        raise ValueError(f"Crop metadata mismatch: {record}")
    relative = Path(body) / record
    target = staging / relative
    target.mkdir(parents=True)
    for name in FILES:
        src, dst = original / name, target / name
        shutil.copy2(src, dst)
        if src.stat().st_size != dst.stat().st_size:
            raise ValueError(f"Incomplete file copy: {dst}")
    return {
        **row, "mask_path": str(relative / "mask.npy"), "em_path": str(relative / "em.npy"),
        "metadata_path": str(relative / "metadata.json"),
        "bbox_xyz": json.dumps(metadata["bbox_xyz"], separators=(",", ":")),
        "voxel_count": metadata["voxel_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neuron_mito_root", type=Path)
    parser.add_argument("panel_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.panel_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != len({r["record_id"] for r in rows}):
        raise ValueError("Expected a nonempty panel with unique record IDs")
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(
            lambda row: copy_instance(args.neuron_mito_root, staging, row), rows
        ), 1):
            results.append(result)
            if i % 500 == 0 or i == len(rows):
                print(f"Copied {i}/{len(rows)} instances", flush=True)
    with (staging / "instance_paths.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    staging.chmod(0o755)
    staging.rename(args.output)
    print(f"{len(results)} instances from {len({r['body_id'] for r in results})} neurons at {args.output}")


if __name__ == "__main__":
    main()
