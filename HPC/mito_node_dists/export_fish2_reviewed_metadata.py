"""Map FISH2 mask centroids to the final manually reviewed skeleton edges."""
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from export_reviewed_fish2_handcraft import CENTROID, SPACING, digest, map_body

KEY = ["record_id", "body_id"]
LABELS = ("axon", "dendrite", "soma", "cell_body_fiber", "connecting_cable", "ambiguous")


def mask_centroid(row):
    bbox = np.asarray(json.loads(row.bbox_xyz))
    mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False)
    if tuple((bbox[3:] - bbox[:3])[::-1]) != mask.shape:
        raise ValueError(f"Mask shape and bbox disagree: {row.record_id}")
    size = np.count_nonzero(mask)
    if not size or size != row.voxel_count:
        raise ValueError(f"Mask voxel count disagrees: {row.record_id}")
    means = []
    for axis in range(3):
        marginal = np.count_nonzero(mask, axis=tuple(i for i in range(3) if i != axis))
        means.append(np.dot(np.arange(mask.shape[axis]), marginal) / size)
    return bbox[:3] + means[::-1]


def process_body(task):
    body, rows, snapshot, original_root = task
    centers = np.asarray([mask_centroid(row) for row in rows.itertuples(index=False)])
    rows = rows.copy()
    rows[CENTROID] = centers
    node_path = snapshot / "automatic" / str(body) / "skeleton_nodes.parquet"
    nodes = pd.read_parquet(node_path, columns=["x_em", "y_em", "z_em"]).to_numpy() * SPACING
    # The old dashboard could contain SWC coordinates in segmentation voxels.
    # Pick the frame that is geometrically consistent with this body's crops.
    medians = [float(np.median(cKDTree(nodes * scale).query(centers * SPACING)[0]))
               for scale in (1, np.array([16, 16, 15]))]
    scaled = medians[1] < medians[0]
    mapped, hashes = map_body((int(body), rows, snapshot, scaled), review_file_root=original_root)
    mapped = pd.DataFrame(mapped)
    distances = mapped.compartment_distance_nm
    alignment_bad = distances.median() > 5000 or max(medians) < 3 * min(medians)
    mapped["compartment_qc"] = np.select(
        [np.full(len(mapped), alignment_bad), distances.gt(1000), mapped.compartment.eq("ambiguous")],
        ["skeleton_alignment_review", "farther_than_1000_nm", "reviewed_ambiguous"],
        default="ok")
    mapped["body_id"] = body
    mapped[CENTROID] = centers
    counts = mapped.compartment.value_counts()
    summary = dict(body_id=body, compartment_review_status="corrected",
                   compartment_review_revision=int(mapped.compartment_review_revision.iloc[0]),
                   skeleton_xyz_rescaled=scaled, skeleton_alignment_qc="review" if alignment_bad else "ok",
                   median_compartment_distance_nm=float(distances.median()),
                   compartment_qc_ok_count=int(mapped.compartment_qc.eq("ok").sum()),
                   **{f"compartment_{label}_count": int(counts.get(label, 0)) for label in LABELS})
    return mapped, summary, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument("original_review_root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    files = {name: args.dataset / name for name in
             ("instances.csv", "neuron_index.csv", "handcrafted_features22.csv")}
    instances, neurons, hand = [pd.read_csv(path, dtype={"body_id": str, "record_id": str},
                                           float_precision="round_trip") for path in files.values()]
    if args.output.exists():
        raise FileExistsError(args.output)
    if (instances.duplicated("record_id").any() or hand.duplicated("record_id").any()
            or neurons.duplicated("body_id").any()
            or set(map(tuple, instances[KEY].to_numpy())) != set(map(tuple, hand[KEY].to_numpy()))):
        raise ValueError("The instance tables do not have a unique common identity mapping")
    panel = pd.DataFrame(json.loads((args.snapshot / "panel.json").read_text()))
    expected = neurons.set_index("body_id").instance_count.sort_index()
    observed = instances.groupby("body_id").size()
    panel_counts = panel.assign(body_id=panel.body_id.astype(str)).set_index("body_id").instances.sort_index()
    if not expected.equals(observed) or not expected.equals(panel_counts):
        raise ValueError("Snapshot and dataset disagree on body IDs or instance counts")
    if set(zip(panel.body_id.astype(str), panel.cell_type)) != set(zip(neurons.body_id, neurons.cell_type)):
        raise ValueError("Snapshot and dataset disagree on cell types")
    if not instances.array_axis_order.eq("ZYX").all() or not instances.voxel_size_xyz_nm.eq("8,8,30").all():
        raise ValueError("Expected FISH2 ZYX arrays at 8 × 8 × 30 nm (XYZ)")
    tasks = [(body, rows, args.snapshot.resolve(), args.original_review_root)
             for body, rows in instances.groupby("body_id")]
    results, summaries, hashes = [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (mapped, summary, sources) in enumerate(pool.map(process_body, tasks), start=1):
            results.append(mapped)
            summaries.append(summary)
            hashes.update(sources)
            if i % 15 == 0 or i == len(tasks):
                print(f"Mapped {i}/{len(tasks)} neurons", flush=True)
    labels = pd.concat(results, ignore_index=True)
    if len(labels) != len(instances) or labels.duplicated(KEY).any():
        raise ValueError("Incomplete or duplicate instance labels")
    merged = instances.merge(labels, on=KEY, validate="one_to_one")
    if len(merged) != len(instances):
        raise ValueError("Mapped record IDs did not join back to the original manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "instances_with_compartment.csv": merged,
        "handcrafted_features22_with_compartment.csv": hand.merge(labels, on=KEY, validate="one_to_one"),
        "neuron_index_with_compartment.csv": neurons.merge(pd.DataFrame(summaries), on="body_id", validate="one_to_one"),
        "instance_compartments.csv": instances[KEY + ["cell_type"]].merge(labels, on=KEY, validate="one_to_one"),
    }
    for name, df in outputs.items():
        df.to_csv(args.output / name, index=False)
    report = dict(instances=len(labels), neurons=len(neurons),
                  compartment_counts=labels.compartment.value_counts().to_dict(),
                  qc_counts=labels.compartment_qc.value_counts().to_dict(),
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                      cwd=Path(__file__).resolve().parent).strip(),
                  input_sha256={name: digest(path) for name, path in files.items()},
                  output_sha256={name: digest(args.output / name) for name in outputs},
                  review_snapshot=str(args.snapshot.resolve()), review_file_sha256=hashes)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("instances", "neurons", "compartment_counts", "qc_counts")}, indent=2))


if __name__ == "__main__":
    main()
