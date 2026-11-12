#!/usr/bin/env python3
"""Replay the original FISH2 compartment stages for one body and compare with NAVis."""

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STAGES = [
    "01_seed_compartments.py", "02_refine_compartments.py", "03_connect_components.py",
    "04_bridge_aware_fill.py", "05_boundary_fill.py",
]
XYZ = ["x", "y", "z"]
STYLE = {
    "axon": ("Axon", "#e57224"), "dendrite": ("Dendrite", "#2879bc"),
    "connecting_cable": ("Linker", "#973db2"), "cell_body_fiber": ("Cell body fiber", "#348457"),
    "soma": ("Soma", "#a31e5b"), "ambiguous": ("Ambiguous", "#8c7a49"),
    "unassigned": ("Unassigned", "#aeb4bf"), "boundary": ("NAVis boundary", "#30334f"),
}


def prepare(source, root, body):
    neuron = root / "data" / "neuron_downloads" / str(body)
    skeleton = neuron / "skeleton"
    skeleton.mkdir(parents=True)
    swc = source / "skeleton" / f"{body}.swc"
    nodes = pd.read_csv(swc, sep=r"\s+", comment="#", header=None,
                        names=["rowId", "swc_type", *XYZ, "radius", "link"])
    xyz = nodes[XYZ].to_numpy()
    nodes[["x_em", "y_em", "z_em"]] = xyz / [8, 8, 30]
    nodes[XYZ] = xyz / [16, 16, 15]
    nodes.to_parquet(skeleton / f"{body}_skeleton_em.parquet", index=False)
    shutil.copy2(source / "neuron_synapse_connections.parquet", neuron)
    return skeleton


def replay(original, root, body):
    snapshot = root / "original_scripts"
    snapshot.mkdir()
    for stage in STAGES:
        path = snapshot / stage
        shutil.copy2(original / stage, path)
        if stage.startswith(("01", "02")):
            previous = sys.argv
            try:
                sys.argv = [str(path), str(body)]
                runpy.run_path(str(path), init_globals={"ROOT": root})
            finally:
                sys.argv = previous
            continue
        source = path.read_text()
        old_root = "ROOT = Path('/home/cyf/JaneliaEM')"
        if source.count(old_root) != 1:
            raise ValueError(f"Unexpected root declaration in {stage}")
        namespace = {"__name__": "legacy_stage", "__file__": str(path), "WORK_ROOT": root}
        previous = sys.argv
        try:
            sys.argv = [str(path), str(body)]
            exec(compile(source.replace(old_root, "ROOT = WORK_ROOT"), str(path), "exec"), namespace)
        finally:
            sys.argv = previous
        if "process" in namespace:
            result = namespace["process"](body)
            print(stage, result, flush=True)
            if result["status"] != "success":
                raise RuntimeError(result)
    return {stage: hashlib.sha256((snapshot / stage).read_bytes()).hexdigest() for stage in STAGES}


def add_edges(fig, xyz, pairs, labels, col, bridge=False):
    for key, (name, color) in STYLE.items():
        keep = labels == key
        if not keep.any():
            continue
        points = np.full((int(keep.sum()), 3, 3), np.nan, dtype=np.float32)
        points[:, :2] = xyz[pairs[keep]]
        x, y, z = points.reshape(-1, 3).T
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=f"{name} (bridge)" if bridge else name,
            legendgroup=f"{key}-{bridge}", showlegend=(col == 1 or key in {"unassigned", "boundary"}),
            line=dict(color=color, width=3 if bridge or key == "unassigned" else 5,
                      dash="dash" if bridge else "solid"),
            opacity=.72 if bridge else 1, hovertemplate=f"{name}<extra></extra>",
        ), row=1, col=col)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body", type=int)
    parser.add_argument("source", type=Path)
    parser.add_argument("original_scripts", type=Path)
    parser.add_argument("navis_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    body = args.body
    root = args.output.resolve()
    skeleton = prepare(args.source, root, body)
    hashes = replay(args.original_scripts, root, body)
    nodes = pd.read_parquet(skeleton / f"{body}_annotated_skeleton_v3.parquet")
    final = root / "experiments" / "distinguish_compartment_of_neuron" / "v3_boundary_component_fill" / str(body)
    edges = pd.read_parquet(final / f"{body}_annotated_edges_v3_boundary_fill.parquet")
    bridges = pd.read_parquet(final / f"{body}_bridge_edges_v3_boundary_fill.parquet")
    label = "compartment_boundary_fill"
    pairs = edges[["u", "v"]].to_numpy(int)
    bridge_pairs = bridges[["u_node_index", "v_node_index"]].to_numpy(int)
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]
    navis = pd.read_csv(args.navis_csv).set_index("node_id").loc[nodes.rowId]
    if not np.allclose(xyz * 1000, navis[XYZ].to_numpy(), atol=.001, rtol=0):
        raise ValueError("The original pipeline and NAVis are using different skeleton coordinates.")
    refined = pd.read_parquet(skeleton / f"{body}_annotated_edges_v11_science_like.parquet")
    refined_pairs = refined[["u", "v"]].to_numpy(int)
    navis_labels = navis.compartment.replace({"linker": "connecting_cable"}).to_numpy()
    left, right = navis_labels[pairs].T
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=["Original FISH2 pipeline | step 02", "NAVis SFC"], horizontal_spacing=.03)
    add_edges(fig, xyz, refined_pairs, refined.compartment.to_numpy(), 1)
    refined_end = len(fig.data)
    add_edges(fig, xyz, pairs, edges[label].to_numpy(), 1)
    add_edges(fig, xyz, bridge_pairs, bridges[label].to_numpy(), 1, bridge=True)
    final_end = len(fig.data)
    add_edges(fig, xyz, pairs, np.where(left == right, left, "boundary"), 2)
    for trace in fig.data[refined_end:final_end]:
        trace.visible = False
    buttons = [
        dict(label=name, method="update", args=[
            {"visible": [start <= i < end or i >= final_end for i in range(len(fig.data))]},
            {"annotations[0].text": f"Original FISH2 pipeline | {name}"},
        ])
        for name, start, end in [("Step 02", 0, refined_end), ("Step 05", refined_end, final_end)]
    ]
    for scene in ("scene", "scene2"):
        fig.layout[scene].update(aspectmode="data", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)")
    fig.update_layout(
        title=f"PC / {body}  |  Compartment comparison",
        template="plotly_white", margin=dict(t=100, b=35, l=10, r=10),
        legend=dict(orientation="h", x=0, y=-.02),
        updatemenus=[dict(buttons=buttons, x=0, y=1.14, xanchor="left", yanchor="top")],
    )
    sync = """
const plot = document.getElementById('{plot_id}');
let syncing = false;
plot.on('plotly_relayout', async (event) => {
  const source = ['scene', 'scene2'].find(scene => event[scene + '.camera']);
  if (syncing || !source) return;
  const target = source === 'scene' ? 'scene2' : 'scene';
  syncing = true;
  try { await Plotly.relayout(plot, {[target + '.camera']: event[source + '.camera']}); }
  finally { syncing = false; }
});
"""
    fig.write_html(root / "fish2-pc-compartment-comparison.html", post_script=sync,
                   default_height="94vh", config={"displaylogo": False, "scrollZoom": True})
    edges[[label, "length_nm"]].groupby(label).agg(["count", "sum"]).to_csv(root / "original_edge_summary.csv")
    summary = {
        "body_id": body, "nodes": len(nodes), "original_edges": edges[label].value_counts().to_dict(),
        "step_02_edges": refined.compartment.value_counts().to_dict(),
        "bridge_edges": bridges[label].value_counts().to_dict(), "legacy_script_sha256": hashes,
    }
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
