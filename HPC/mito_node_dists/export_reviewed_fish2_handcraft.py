#!/usr/bin/env python3
"""Export reviewed FISH2 handcraft embeddings with the nearest reviewed skeleton-edge compartment."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SPACING = np.array([8., 8., 30.])
LABEL = "compartment_boundary_fill"
FEATURES = [
    "size", "mito SA", "mito CA", "convex hull compactness",
    *(f"PC{i} {metric}" for i in range(1, 4)
      for metric in ("Length", "inertia moment", "symmetry", "CA", "Circum")),
    "mito diameter", "length_along_skeleton_nm", "SBI",
]
KEY = ["record_id", "body_id", "cell_type", "source_run_id", "global_instance"]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


CENTROID = ["centroid_x_voxel", "centroid_y_voxel", "centroid_z_voxel"]


def map_body(task):
    body, records, reviews, scaled = task
    marker = reviews / "corrected" / str(body) / "review.json"
    review = json.loads(marker.read_text())
    if review["state"] != "corrected" or int(review["body_id"]) != body:
        raise ValueError(f"Invalid review: {marker}")
    original = Path(review["files"]["compartment_edges.parquet"])
    bridge = Path(review["files"]["compartment_bridges.parquet"])
    nodes_file = reviews / "automatic" / str(body) / "skeleton_nodes.parquet"
    nodes = pd.read_parquet(nodes_file, columns=["rowId", "x_em", "y_em", "z_em"])
    edges = pd.read_parquet(original)
    bridges = pd.read_parquet(bridge)
    pairs = np.concatenate([edges[["u", "v"]].to_numpy(int),
                            bridges[["u_node_index", "v_node_index"]].to_numpy(int)])
    if not len(pairs) or pairs.min() < 0 or pairs.max() >= len(nodes):
        raise ValueError(f"Invalid reviewed skeleton indices: {body}")
    swc_ids = nodes.rowId.to_numpy()
    if not np.array_equal(swc_ids[pairs[:len(edges)]],
                          edges[["parent_id_original", "child_id_original"]].to_numpy()):
        raise ValueError(f"Reviewed edges do not match skeleton node IDs: {body}")
    if not np.array_equal(swc_ids[pairs[len(edges):]], bridges[["u_rowId", "v_rowId"]].to_numpy()):
        raise ValueError(f"Reviewed bridges do not match skeleton node IDs: {body}")
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy(float) * SPACING
    if scaled:
        # The review stored segmentation-grid XYZ as if they were nanometers.
        xyz *= [16, 16, 15]
    start, end = xyz[pairs[:, 0]], xyz[pairs[:, 1]]
    delta = end - start
    length_sq = np.einsum("ij,ij->i", delta, delta)
    denom = np.where(length_sq > 0, length_sq, 1.)
    labels = pd.concat([edges[LABEL], bridges[LABEL]], ignore_index=True).to_numpy()
    edge_ids = np.concatenate([edges.edge_id.to_numpy(int), bridges.bridge_id.to_numpy(int)])
    if set(labels) - {"axon", "dendrite", "soma", "cell_body_fiber", "connecting_cable", "ambiguous"}:
        raise ValueError(f"Invalid compartment labels: {body}")
    output = []
    for row in records.itertuples(index=False):
        center = np.array([getattr(row, field) for field in CENTROID]) * SPACING
        offset = center - start
        t = np.clip(np.einsum("ij,ij->i", offset, delta) / denom, 0., 1.)
        residual = offset - t[:, None] * delta
        distances = np.einsum("ij,ij->i", residual, residual)
        best = int(distances.argmin())
        # Record proximity to another compartment for auditing boundary assignments.
        different = labels != labels[best]
        next_distance = float(np.sqrt(distances[different].min())) if different.any() else None
        output.append(dict(record_id=row.record_id, compartment=labels[best],
                           compartment_review_revision=review["revision"],
                           skeleton_xyz_rescaled=scaled,
                           skeleton_edge_type="original" if best < len(edges) else "bridge",
                           skeleton_edge_id=int(edge_ids[best]),
                           skeleton_edge_u_swc=int(swc_ids[pairs[best, 0]]),
                           skeleton_edge_v_swc=int(swc_ids[pairs[best, 1]]),
                           compartment_distance_nm=float(np.sqrt(distances[best])),
                           nearest_other_compartment_nm=next_distance))
    sources = {str(f): digest(f) for f in (marker, nodes_file, original, bridge)}
    return output, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("existing", "new", "reviews", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--normalized-lengths", required=True, type=Path)
    args = parser.parse_args()
    old, new = [pd.read_csv(p, float_precision="round_trip") for p in (args.existing, args.new)]
    reviewed = {int(p.parent.name) for p in (args.reviews / "corrected").glob("*/review.json")}
    old = old[old.body_id.isin(reviewed)].copy()
    if not set(new.body_id) <= reviewed or set(old.body_id) & set(new.body_id):
        raise ValueError("New bodies must have a review and be absent from the existing panel")
    cols = KEY + CENTROID + ["voxel_count", "status", "qc"] + [c for c in FEATURES if c not in {"size", "SBI"}]
    hand = pd.concat([old[cols], new[cols]], ignore_index=True)
    if not hand.record_id.is_unique or not hand.status.eq("ok").all() or not hand.qc.eq("ok").all():
        raise ValueError("Duplicate records or unsuccessful feature computation")
    if not np.isfinite(hand[CENTROID].to_numpy()).all():
        raise ValueError("At least one mask centroid is unavailable")
    normalized = pd.read_csv(args.normalized_lengths)
    if not normalized.qc.eq("ok").all() or not normalized.record_id.is_unique:
        raise ValueError("Normalized skeleton length calculation was incomplete")
    scaled = set(normalized.body_id)
    if set(normalized.record_id) != set(hand.loc[hand.body_id.isin(scaled), "record_id"]):
        raise ValueError("Normalized lengths must cover every instance of the rescaled bodies")
    corrected_lengths = hand.record_id.map(normalized.set_index("record_id").length_along_skeleton_nm)
    hand.loc[hand.body_id.isin(scaled), "length_along_skeleton_nm"] = corrected_lengths
    hand["size"] = hand.voxel_count * np.prod(SPACING)
    radius = np.cbrt(3 * hand["size"] / (4 * np.pi))
    hand["SBI"] = np.log10((4 * np.pi * radius**2) / hand["mito SA"].where(hand["mito SA"] > 0))
    tasks = [(int(body), group, args.reviews, body in scaled) for body, group in hand.groupby("body_id")]
    mapped, hashes = [], {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        for rows, sources in pool.map(map_body, tasks):
            mapped.extend(rows)
            hashes.update(sources)
    out = hand[KEY + FEATURES].merge(pd.DataFrame(mapped), on="record_id", how="left", validate="one_to_one")
    if len(out) != len(hand) or out.compartment.isna().any():
        raise ValueError("Incomplete compartment mapping")
    if out.groupby("body_id").compartment_distance_nm.median().gt(5000).any():
        raise ValueError("Large median assignment distance indicates another coordinate mismatch")
    out["compartment_qc"] = np.select(
        [out.compartment_distance_nm.gt(1000), out.compartment.eq("ambiguous")],
        ["farther_than_1000_nm", "reviewed_ambiguous"], default="ok")
    out = out[[c for c in out if c not in FEATURES] + FEATURES]
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "fish2_handcraft_embedding_22_with_compartment.csv"
    out.to_csv(csv_path, index=False)
    summary = {
        "neurons": int(out.body_id.nunique()), "instances": len(out), "feature_dimensions": len(FEATURES),
        "new_neurons": int(new.body_id.nunique()), "new_instances": len(new),
        "rescaled_skeleton_bodies": sorted(scaled),
        "compartment_qc": out.compartment_qc.value_counts().to_dict(),
        "by_type": out.groupby("cell_type").agg(neurons=("body_id", "nunique"), instances=("record_id", "size")).to_dict("index"),
        "compartment_counts": out.compartment.value_counts().to_dict(),
        "missing_features": {c: int(n) for c, n in out[FEATURES].isna().sum().items() if n},
        "distance_nm_p95": float(out.compartment_distance_nm.quantile(.95)),
        "distance_over_1000nm": int(out.compartment_distance_nm.gt(1000).sum()),
        "bridge_assignments": int(out.skeleton_edge_type.eq("bridge").sum()),
        "input_sha256": {str(p): digest(p) for p in (args.existing, args.new, args.normalized_lengths)},
        "review_input_sha256": hashes, "output_sha256": digest(csv_path),
    }
    (args.output / "export_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "review_input_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
