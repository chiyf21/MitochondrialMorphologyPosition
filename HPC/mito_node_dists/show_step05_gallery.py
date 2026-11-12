#!/usr/bin/env python3
"""Build a 3D gallery of archived and rerun FISH2 V3 Step 05 outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from compare_pc_compartments import add_edges

SAMPLES = [(101487522, "PC"), (103005807, ""), (104131472, "GC"), (101082794, "")]
RERUN_BODY = 100076106
LABEL = "compartment_boundary_fill"


def load_sample(node_path, result, body):
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(result / f"{body}_annotated_edges_v3_boundary_fill.parquet")
    bridges = pd.read_parquet(result / f"{body}_bridge_edges_v3_boundary_fill.parquet")
    pairs = edges[["u", "v"]].to_numpy(int)
    expected = edges[["parent_id_original", "child_id_original"]].to_numpy(int)
    if not np.array_equal(nodes.rowId.to_numpy()[pairs], expected):
        raise ValueError(f"Node order does not match the Step 05 edges for {body}")
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]
    plotted_lengths = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    if not np.allclose(plotted_lengths * 1000, edges.length_nm, atol=.01, rtol=1e-5):
        raise ValueError(f"Physical edge lengths do not match for {body}")
    return xyz, edges, bridges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neuron_root", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("pc_rerun", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    groups, summaries = [], []
    for body, cell_type in [*SAMPLES, (RERUN_BODY, "PC")]:
        is_rerun = body == RERUN_BODY
        node_root = args.pc_rerun / "data" / "neuron_downloads" if is_rerun else args.neuron_root
        result_root = (
            args.pc_rerun / "experiments" / "distinguish_compartment_of_neuron" / "v3_boundary_component_fill"
            if is_rerun else args.archive
        )
        node_path = node_root / str(body) / "skeleton" / f"{body}_annotated_skeleton_v3.parquet"
        xyz, edges, bridges = load_sample(node_path, result_root / str(body), body)
        counts = edges[LABEL].value_counts().to_dict()
        start = len(fig.data)
        add_edges(fig, xyz, edges[["u", "v"]].to_numpy(int), edges[LABEL].to_numpy(), 1)
        add_edges(fig, xyz, bridges[["u_node_index", "v_node_index"]].to_numpy(int),
                  bridges[LABEL].to_numpy(), 1, bridge=True)
        groups.append(range(start, len(fig.data)))
        summary = {
            "body_id": body, "cell_type": cell_type, "source": "Recent rerun" if is_rerun else "Historical Step 05 output",
            "edges": len(edges), "edge_labels": counts, "bridges": len(bridges),
            "node_path": str(node_path), "step05_directory": str(result_root / str(body)),
        }
        summaries.append(summary)

    def title(row):
        counts = row["edge_labels"]
        coverage = counts.get("ambiguous", 0) / row["edges"]
        return (
            f"{row['cell_type']} {row['body_id']} | V3 Step 05<br><sup>{row['source']} | "
            f"{counts.get('soma', 0)} soma edges | {coverage:.1%} ambiguous | "
            f"{row['bridges']} inferred bridges</sup>"
        )

    buttons = [
        dict(label=f"{row['cell_type']} {row['body_id']}".strip(), method="update",
             args=[{"visible": [i in group for i in range(len(fig.data))]}, {"title.text": title(row)}])
        for row, group in zip(summaries, groups)
    ]
    for i, trace in enumerate(fig.data):
        trace.visible = i in groups[0]
    fig.update_layout(
        title=dict(text=title(summaries[0]), x=.02), template="plotly_white",
        margin=dict(t=125, b=25, l=10, r=10), legend=dict(orientation="h", x=0, y=1.01),
        scene=dict(aspectmode="data", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)"),
        updatemenus=[dict(buttons=buttons, x=1, xanchor="right", y=1.14)],
    )
    fig.write_html(args.output / "fish2-step05-gallery.html", default_height="94vh",
                   config={"displaylogo": False, "scrollZoom": True})
    (args.output / "step05_gallery_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for row in summaries:
        print(row["body_id"], row["edges"], row["edge_labels"], flush=True)


if __name__ == "__main__":
    main()
