#!/usr/bin/env python3
"""Display PC synapses on sampled segmented voxels and overlay V11."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_pc_compartments import add_edges

BODY = 100076106
XYZ = ["x_nm", "y_nm", "z_nm"]
COLORS = {"pre": "#df5319", "post": "#147abf"}


def sample_cloud(path, size=85000):
    rle = np.load(path, mmap_mode="r", allow_pickle=False)
    if rle.ndim != 2 or rle.shape[1] != 4 or (rle[:, 3] <= 0).any():
        raise ValueError("Expected X, Y, Z, positive run length")
    ends = np.cumsum(rle[:, 3], dtype=np.int64)
    rng = np.random.default_rng(100076106)
    offsets = rng.choice(int(ends[-1]), min(size, int(ends[-1])), replace=False)
    rows = np.searchsorted(ends, offsets, side="right")
    points = rle[rows, :3].astype(np.float32)
    points[:, 0] += offsets - (ends[rows] - rle[rows, 3])
    return points * np.array([.016, .016, .015], np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rle", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    skel = args.run / "data" / "neuron_downloads" / str(BODY) / "skeleton"
    def read(suffix):
        return pd.read_parquet(skel / f"{BODY}_{suffix}.parquet")

    cloud = sample_cloud(args.rle)
    nodes = read("annotated_skeleton_v11_science_like")
    edges = read("annotated_edges_v11_science_like")
    attached = read("synapse_node_attachments_v11_science_like")
    info = json.loads((skel / f"{BODY}_compartment_summary_v11_science_like.json").read_text())
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]
    pairs = edges[["u", "v"]].to_numpy(int)
    if not np.array_equal(nodes.rowId.to_numpy()[pairs],
                          edges[["parent_id_original", "child_id_original"]].to_numpy()):
        raise ValueError("Skeleton node IDs do not match V11 edges")
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    fig.add_trace(go.Scatter3d(
        x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], name="Neuron mask",
        mode="markers", marker=dict(size=1.7, opacity=.24, color="#9299a5"), hoverinfo="skip",
    ), row=1, col=1)
    stats = {}
    for role, color in COLORS.items():
        subset = attached[attached.role.eq(role)]
        points = subset.groupby(XYZ, as_index=False).size()
        coords = points[XYZ].to_numpy() / 1000
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            name=f"{role.upper()} | {len(points)} unique sites",
            marker=dict(size=5, opacity=.93, color=color, line=dict(color="white", width=.65)),
            customdata=points["size"], hovertemplate=f"{role.upper()} | %{{customdata}} endpoint records<extra></extra>",
        ), row=1, col=1)
        used = subset[subset.attachment_status.eq("attached")]
        representative = info["synapses"][f"{role}_representative_rowId"]
        point = xyz[nodes.rowId.eq(representative)][0]
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
            name=f"V11 {role.upper()} representative", marker=dict(size=9, color=color, symbol="diamond",
                                                                  line=dict(color="#242b35", width=1.5)),
            hovertemplate=f"SWC node {representative}<extra></extra>",
        ), row=1, col=1)
        stats[role] = {
            "endpoint_records": len(subset), "unique_sites": len(points), "used_records": len(used),
            "representative_node": representative,
            "used_records_by_v11_node_label": nodes.set_index("rowId").loc[used.nearest_rowId, "compartment"].value_counts().to_dict(),
        }

    overlay_start = len(fig.data)
    add_edges(fig, xyz, pairs, edges.compartment.to_numpy(), 1)
    for trace in fig.data[overlay_start:]:
        trace.visible = False
        trace.name = "V11 " + trace.name
    buttons = [dict(label=label, method="update",
                    args=[{"visible": [show or i < overlay_start for i in range(len(fig.data))]}])
               for label, show in [("Synapses + point cloud", False), ("Show V11 compartments", True)]]
    fig.update_layout(
        title=dict(text=f"PC {BODY} | PRE and POST synapses<br><sup>{len(cloud):,} sampled neuron voxels</sup>", x=.02),
        template="plotly_white", margin=dict(t=125, b=25, l=10, r=10),
        legend=dict(orientation="h", x=0, y=1.01),
        scene=dict(aspectmode="data", uirevision="pc", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)"),
        updatemenus=[dict(buttons=buttons, x=1, xanchor="right", y=1.14)],
    )
    fig.write_html(args.output / "fish2-pc-100076106-synapses.html", default_height="94vh",
                   config={"displaylogo": False, "scrollZoom": True})
    result = dict(body_id=BODY, sampled_voxels=len(cloud), source_rle=str(args.rle), synapses=stats)
    (args.output / "synapse_cloud_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
