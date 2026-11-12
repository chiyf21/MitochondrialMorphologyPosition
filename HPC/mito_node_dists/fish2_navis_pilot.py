#!/usr/bin/env python3
"""Run NAVis SFC on a FISH2 pilot panel and export node labels and a 3D dashboard."""

import argparse
import hashlib
import json
from pathlib import Path

import navis
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree

XYZ = ["x", "y", "z"]
SPACING_NM = np.array([8, 8, 30])
MAX_SNAP_NM = 1000
PANEL = {
    "GC": 103696549, "CON": 100001316, "VMN": 116512011,
    "PC": 100076106, "MON2": 100000128, "MON1": 100041389,
}
STYLES = {
    "axon": ("Axon", "#e57224"), "dendrite": ("Dendrite", "#2879bc"),
    "linker": ("Linker", "#973db2"), "cellbodyfiber": ("Cell body fiber", "#45945a"),
    "boundary": ("Boundary", "#30334f"), "unassigned": ("Unassigned", "#aeb4bf"),
    "pre": ("Presynaptic", "#9c3524"), "post": ("Postsynaptic", "#16806d"),
}


def split_neuron(root, body):
    folder = root / str(body)
    swc = folder / "skeleton" / f"{body}.swc"
    connections_path = folder / "neuron_synapse_connections.parquet"
    endpoints_path = folder / "synapse_endpoints.parquet"
    header = swc.read_text().splitlines()[:10]
    if not any('"coordinate_units": "nanometers"' in line for line in header):
        raise ValueError(f"Expected an SWC in nanometers: {swc}")
    neuron = navis.read_swc(swc, units="nm")
    nodes = neuron.nodes.copy()
    graph = neuron.graph.to_undirected()
    main_ids = max(nx.connected_components(graph), key=len)
    main = navis.subset_neuron(neuron, list(main_ids))
    connections = pd.read_parquet(connections_path)
    keys = set(connections.loc[connections.bodyId_pre.eq(body), "pre_endpoint_key"])
    keys.update(connections.loc[connections.bodyId_post.eq(body), "post_endpoint_key"])
    endpoints = pd.read_parquet(endpoints_path)
    endpoints = endpoints[endpoints.endpoint_key.isin(keys)].drop_duplicates("endpoint_key").copy()
    endpoints[XYZ] = endpoints[["x_em", "y_em", "z_em"]].to_numpy() * SPACING_NM
    distances, indices = cKDTree(nodes[XYZ].to_numpy()).query(endpoints[XYZ])
    endpoints["node_id"] = nodes.node_id.to_numpy()[indices]
    endpoints["distance_nm"] = distances
    endpoints["used"] = endpoints.node_id.isin(main_ids) & (distances <= MAX_SNAP_NM)
    connectors = endpoints.loc[endpoints.used, ["node_id", *XYZ, "endpoint_role"]].copy()
    connectors["type"] = connectors.pop("endpoint_role").map({"pre": 0, "post": 1})
    connectors["connector_id"] = np.arange(len(connectors))
    nodes["compartment"] = "unassigned"
    status = "Split"
    if set(connectors.type) == {0, 1}:
        main.connectors = connectors
        labelled = navis.split_axon_dendrite(
            main, metric="synapse_flow_centrality", flow_thresh=0.9,
            cellbodyfiber=False, reroot_soma=False, label_only=True,
        )
        labels = labelled.nodes.set_index("node_id").compartment
        nodes["compartment"] = nodes.node_id.map(labels.astype("object")).fillna("unassigned")
        if not {"axon", "dendrite"} <= set(nodes.compartment):
            status = "No axon-dendrite separation"
        unexpected = set(nodes.compartment) - set(STYLES)
        if unexpected:
            raise ValueError(f"Unexpected NAVis labels: {unexpected}")
    else:
        status = "Insufficient pre/post synapses"
    summary = {
        "body_id": body, "status": status, "nodes": len(nodes), "main_nodes": len(main_ids),
        "components": nx.number_connected_components(graph),
        "endpoint_counts": endpoints.endpoint_role.value_counts().to_dict(),
        "used_counts": endpoints.loc[endpoints.used, "endpoint_role"].value_counts().to_dict(),
        "snap_p95_nm": round(float(np.quantile(distances, .95)), 1),
        "labels": nodes.compartment.value_counts().to_dict(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [swc, connections_path, endpoints_path]},
    }
    return nodes, endpoints, summary


def add_traces(fig, nodes, endpoints):
    indexed = nodes.set_index("node_id")
    edges = nodes[nodes.parent_id.isin(indexed.index)]
    parent = indexed.loc[edges.parent_id]
    child_pos = edges[XYZ].to_numpy() / 1000
    parent_pos = parent[XYZ].to_numpy() / 1000
    labels = edges.compartment.to_numpy()
    edge_labels = np.where(labels == parent.compartment.to_numpy(), labels, "boundary")
    for key, (name, color) in STYLES.items():
        if key in {"pre", "post"}:
            points = endpoints.loc[endpoints.used & endpoints.endpoint_role.eq(key), XYZ] / 1000
            fig.add_trace(go.Scatter3d(
                x=points.x, y=points.y, z=points.z, name=name, mode="markers",
                marker=dict(size=3, color=color, opacity=.68), visible="legendonly",
            ))
        elif key == "boundary":
            cut = edge_labels == key
            points = (child_pos[cut] + parent_pos[cut]) / 2
            if len(points):
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2], name=name,
                    mode="markers", marker=dict(size=4, color=color, symbol="diamond"),
                ))
        else:
            keep = edge_labels == key
            if not keep.any():
                continue
            points = np.full((int(keep.sum()), 3, 3), np.nan, dtype=np.float32)
            points[:, 0] = child_pos[keep]
            points[:, 1] = parent_pos[keep]
            x, y, z = points.reshape(-1, 3).T
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, name=name, mode="lines", hoverinfo="skip",
                line=dict(color=color, width=2 if key == "unassigned" else 5),
            ))


def title(cell_type, summary):
    used = summary["used_counts"]
    coverage = 1 - summary["labels"].get("unassigned", 0) / summary["nodes"]
    return (
        f"{cell_type} / {summary['body_id']}<br><sup>"
        f"{summary['status']} | {coverage:.1%} of nodes labelled | "
        f"{used.get('pre', 0)} pre / {used.get('post', 0)} post used | "
        f"{summary['components']} SWC components</sup>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neuron_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    summaries, groups = [], []
    for cell_type, body in PANEL.items():
        nodes, endpoints, summary = split_neuron(args.neuron_root, body)
        nodes.to_csv(args.output / f"{body}_compartments.csv", index=False)
        endpoints.to_csv(args.output / f"{body}_synapse_mapping.csv", index=False)
        start = len(fig.data)
        add_traces(fig, nodes, endpoints)
        groups.append(range(start, len(fig.data)))
        summaries.append({"cell_type": cell_type, **summary})
        print(cell_type, summary, flush=True)
    visibility = [trace.visible if trace.visible is not None else True for trace in fig.data]
    buttons = []
    for group, summary in zip(groups, summaries):
        buttons.append(dict(
            label=f"{summary['cell_type']} / {summary['body_id']}", method="update",
            args=[{"visible": [v if i in group else False for i, v in enumerate(visibility)]},
                  {"title.text": title(summary["cell_type"], summary),
                   "scene.camera": {"eye": dict(x=1.4, y=1.4, z=1.2)}}],
        ))
    default = list(PANEL).index("PC")
    for i, trace in enumerate(fig.data):
        trace.visible = visibility[i] if i in groups[default] else False
    fig.update_layout(
        title=dict(text=title(summaries[default]["cell_type"], summaries[default]), x=.02),
        template="plotly_white", margin=dict(l=10, r=10, b=10, t=104),
        scene=dict(aspectmode="data", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)"),
        legend=dict(orientation="h", y=1.01, x=0),
        updatemenus=[dict(buttons=buttons, active=default, x=.99, xanchor="right", y=1.11, yanchor="top")],
    )
    fig.write_html(
        args.output / "fish2-navis-dashboard.html", include_plotlyjs=True,
        default_height="95vh", config={"displaylogo": False, "scrollZoom": True},
    )
    (args.output / "run_summary.json").write_text(json.dumps({
        "navis": navis.__version__, "method": "synapse_flow_centrality",
        "flow_thresh": .9, "max_snap_nm": MAX_SNAP_NM, "neurons": summaries,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
