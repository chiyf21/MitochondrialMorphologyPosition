#!/usr/bin/env python3
"""Generate V3 Step 05 review candidates for the frozen FISH2 experiment panel."""

import argparse
import csv
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from compare_pc_compartments import STAGES, prepare, replay

LABEL = "compartment_boundary_fill"
LABELS = ("axon", "dendrite", "soma", "cell_body_fiber", "connecting_cable", "ambiguous")
TYPES = ("GC", "CON", "VMN", "PC", "MON2", "MON1")
EXPERIMENT = Path("experiments/distinguish_compartment_of_neuron")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def candidate(task):
    row, neuron_root, root = task
    body = row["body_id"]
    target = root / "automatic" / str(body)
    status = root / "batch_status" / f"{body}.json"
    if target.exists():
        return
    work = Path(tempfile.mkdtemp(prefix=f"{body}-", dir=root / "work"))
    log = work / "pipeline.log"
    try:
        with log.open("w") as f, redirect_stdout(f), redirect_stderr(f):
            source = neuron_root / str(body)
            skel = prepare(source, work, body)
            if (source / "panel_annotation.json").exists():
                shutil.copy2(source / "panel_annotation.json", skel.parent)
            # The old bridge builder reads V11 geometry. Step 04 and 05 read V3 labels.
            hashes = replay(root / "original_scripts", work, body, bridge_from_v3_if_v11_unavailable=True)
            nodes = pd.read_parquet(skel / f"{body}_annotated_skeleton_v3.parquet")
            final = work / EXPERIMENT / "v3_boundary_component_fill" / str(body)
            edges = pd.read_parquet(final / f"{body}_annotated_edges_v3_boundary_fill.parquet")
            bridges = pd.read_parquet(final / f"{body}_bridge_edges_v3_boundary_fill.parquet")
            xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]
            if not np.isfinite(xyz).all() or len(edges) + len(bridges) != len(nodes) - 1:
                raise ValueError("Candidate coordinates or tree edge count are invalid")
            stage = work / "candidate"
            stage.mkdir()
            nodes.to_parquet(stage / "skeleton_nodes.parquet", index=False)
            edges.to_parquet(stage / "compartment_edges.parquet", index=False)
            bridges.to_parquet(stage / "compartment_bridges.parquet", index=False)
            connections = pd.read_parquet(skel / f"{body}_synapse_skeleton_attachments_v3.parquet")
            connections.to_parquet(stage / "synapse_attachments.parquet", index=False)
            diagnostics = json.loads((skel / f"{body}_compartment_summary_v3.json").read_text())
            write_json(stage / "candidate.json", {
                **row, "method": "V3 Step 05", "step01_arbors": diagnostics["main_arbors"],
                "soma": diagnostics["soma"],
                "bridge_geometry": "V11" if (skel / f"{body}_annotated_edges_v11_science_like.parquet").exists() else "V3",
                "edge_counts": edges[LABEL].value_counts().to_dict(),
                "bridge_counts": bridges[LABEL].value_counts().to_dict(), "source_script_sha256": hashes,
                "work_directory": str(work),
            })
            os.rename(stage, target)
        write_json(status, {"state": "ready"})
    except Exception as exc:
        write_json(status, {"state": "error", "error": str(exc), "log": str(log)})
        print(body, "error", str(exc), flush=True)
    else:
        print(body, "ready", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("features", type=Path)
    p.add_argument("neurons", type=Path)
    p.add_argument("legacy", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    root = args.output.resolve()
    for folder in ("automatic", "corrected", "drafts", "batch_status", "work", "original_scripts"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name in STAGES:
        source, dest = args.legacy / name, root / "original_scripts" / name
        if not dest.exists():
            shutil.copy2(source, dest)
    rows = {}
    with args.features.open(newline="") as f:
        for record in csv.DictReader(f):
            body, typ = int(record["body_id"]), record["cell_type"]
            if typ not in TYPES:
                continue
            row = rows.setdefault(body, dict(body_id=body, cell_type=typ, instances=0))
            if typ != row["cell_type"]:
                raise ValueError(f"Conflicting cell types for {body}")
            row["instances"] += 1
    panel = sorted(rows.values(), key=lambda r: (TYPES.index(r["cell_type"]), r["body_id"]))
    write_json(root / "panel.json", panel)
    print(f"{len(panel)} bodies, {sum(row['instances'] for row in panel)} instances", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(candidate, ((row, args.neurons, root) for row in panel)))


if __name__ == "__main__":
    main()
