#!/usr/bin/env python3
"""Compute per-instance shape and EM texture features for the canonical dataset."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mitochondria_feature_encoder import EncoderConfig, encode_one, extract_pixel_ultrastructure_features

MORPH_FEATURES = [
    "size", "mito SA", "mito CA", "convex hull compactness",
    "PC1 Length", "PC1 inertia moment", "PC1 symmetry", "PC1 CA", "PC1 Circum",
    "PC2 Length", "PC2 inertia moment", "PC2 symmetry", "PC2 CA", "PC2 Circum",
    "PC3 Length", "PC3 inertia moment", "PC3 symmetry", "PC3 CA", "PC3 Circum",
    "mito diameter", "SBI", "sphericity",
]
PIXEL_FEATURES = [
    "pixel_internal_contrast", "pixel_internal_edge_density",
    "pixel_orientation_coherence", "pixel_normal_pc1_angle_deg",
    "pixel_internal_depth_index", "pixel_dominant_spatial_scale_nm",
    "pixel_spectral_entropy",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--spacing-zyx-nm", type=float, nargs=3, default=(30., 8., 8.))
    ap.add_argument("--run-id", action="append")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    root = args.dataset_root
    with (root / "metadata/instances.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.run_id:
        selected = set(args.run_id)
        rows = [r for r in rows if r["source_run_id"] in selected]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        ap.error("No instances selected")

    cfg = EncoderConfig(spacing_zyx_nm=tuple(args.spacing_zyx_nm), min_voxels=1)
    morph_rows, pixel_rows, combined_rows = [], [], []
    errors = []
    for i, row in enumerate(rows, 1):
        base = {k: row.get(k, "") for k in ("record_id", "source_run_id", "body_id", "cell_type", "global_instance", "voxel_count")}
        try:
            meta = json.loads((root / row["metadata_path"]).read_text(encoding="utf-8"))
            mask = np.load(root / row["mask_path"], allow_pickle=False) > 0
            raw = np.load(root / row["em_path"], allow_pickle=False)
            if mask.ndim != 3 or raw.shape != mask.shape or not mask.any():
                raise ValueError(f"invalid or mismatched arrays: mask={mask.shape}, em={raw.shape}")
            x0, y0, z0, x1, y1, z1 = map(int, meta["bbox_xyz"])
            extent_xyz = np.array([x1-x0, y1-y0, z1-z0])
            shape_xyz = np.array(mask.shape[::-1])
            if not (np.array_equal(extent_xyz, shape_xyz) or np.array_equal(extent_xyz + 1, shape_xyz)):
                raise ValueError(f"bbox/mask mismatch: bbox={meta['bbox_xyz']}, mask={mask.shape}")
            sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            seed = int.from_bytes(hashlib.sha256(row["record_id"].encode()).digest()[:8], "little")
            morph = encode_one(mask, sl, raw, cfg, np.random.default_rng(seed))
            pixel = extract_pixel_ultrastructure_features(raw, mask, voxel_size=tuple(args.spacing_zyx_nm))
            mrow = {**base, **{k: morph.get(k, math.nan) for k in MORPH_FEATURES},
                    "mean_em_intensity": morph.get("mean_em_intensity", math.nan),
                    "centroid_x_voxel": morph["centroid_x_voxel"], "centroid_y_voxel": morph["centroid_y_voxel"],
                    "centroid_z_voxel": morph["centroid_z_voxel"], "touches_bbox_border": int(np.any(mask[[0,-1],:,:]) or np.any(mask[:,[0,-1],:]) or np.any(mask[:,:,[0,-1]])),
                    "voxel_size_zyx_nm": ",".join(map(str, args.spacing_zyx_nm)), "status": "ok"}
            prow = {**base, **{k: pixel.get(k, math.nan) for k in PIXEL_FEATURES}, "status": "ok"}
            morph_rows.append(mrow); pixel_rows.append(prow); combined_rows.append({**mrow, **{k: pixel.get(k, math.nan) for k in PIXEL_FEATURES}})
        except Exception as e:
            err = {**base, "status": f"error: {type(e).__name__}: {e}"}
            errors.append(err); morph_rows.append(err); pixel_rows.append(err); combined_rows.append(err)
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"processed {i}/{len(rows)}; errors={len(errors)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("morphology_features.csv", morph_rows), ("pixel_features.csv", pixel_rows), ("handcrafted_features.csv", combined_rows), ("errors.csv", errors)]:
        fields = list(dict.fromkeys(k for r in data for k in r))
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)
    summary = {"instances_selected": len(rows), "morphology_features": MORPH_FEATURES, "pixel_features": PIXEL_FEATURES,
               "spacing_zyx_nm": list(args.spacing_zyx_nm), "errors": len(errors),
               "source_run_ids": sorted(set(r["source_run_id"] for r in rows))}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"WROTE {args.output_dir}; rows={len(rows)}; errors={len(errors)}")

if __name__ == "__main__":
    main()
