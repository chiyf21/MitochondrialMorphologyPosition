#!/usr/bin/env python3
"""Inspect stored FISH2 compartment steps and the V3 axon candidates for one PC."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_pc_compartments import add_edges

XYZ_EM = ["x_em", "y_em", "z_em"]
BODY = 100076106


def candidate_traces(fig, xyz, edges, nodes, minimum, total_pre):
    parents = edges.rooted_parent_node_index.to_numpy(int)
    children = edges.rooted_child_node_index.to_numpy(int)
    soma = nodes.is_soma_node.to_numpy(bool)
    seed = nodes.distance_to_soma_geodesic_nm.eq(0).to_numpy()
    seed_index = int(np.flatnonzero(seed)[0])
    descendants = [[] for _ in range(len(nodes))]
    for i, (p, c) in enumerate(zip(parents, children)):
        if p >= 0 and c >= 0:
            descendants[p].append((i, c))
    main = {seed_index}
    queue = [seed_index]
    for p in queue:
        for _, c in descendants[p]:
            if c not in main:
                main.add(c)
                queue.append(c)
    pre = edges.subtree_pre_weight.to_numpy(float)
    post = edges.subtree_post_weight.to_numpy(float)
    support = pre + post
    coverage = pre / total_pre
    purity = np.divide(pre, support, out=np.zeros_like(pre), where=support > 0)
    eligible = np.isin(children, list(main)) & ~soma[children] & (support >= minimum)
    selected = [
        ("Best purity at ≥25% coverage", eligible & (coverage >= .25), purity, "#8053c9"),
        ("Best coverage at ≥65% purity", eligible & (purity >= .65), coverage, "#168c75"),
    ]
    pairs = edges[["u", "v"]].to_numpy(int)
    result = []
    for label, mask, metric, color in selected:
        best = int(np.argmax(np.where(mask, metric, -1)))
        if not mask[best]:
            raise ValueError(f"No candidate for {label}")
        branch = [best]
        queue = [children[best]]
        for p in queue:
            for e, c in descendants[p]:
                branch.append(e)
                queue.append(c)
        points = np.full((len(branch), 3, 3), np.nan, dtype=np.float32)
        points[:, :2] = xyz[pairs[branch]]
        x, y, z = points.reshape(-1, 3).T
        name = f"{label}: {coverage[best]:.1%} / {purity[best]:.1%}"
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=name,
            line=dict(width=6, color=color), legendgroup=label, hoverinfo="skip",
        ), row=1, col=1)
        cut = xyz[pairs[best]].mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[cut[0]], y=[cut[1]], z=[cut[2]], mode="markers", name=name,
            marker=dict(size=7, color=color, symbol="diamond"), legendgroup=label,
            showlegend=False, hovertemplate=f"{name}<extra></extra>",
        ), row=1, col=1)
        result.append(dict(edge_id=int(edges.edge_id.iloc[best]), capture=float(coverage[best]),
                           purity=float(purity[best]), pre_weight=float(pre[best]), post_weight=float(post[best])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    skel = args.run / "data" / "neuron_downloads" / str(BODY) / "skeleton"
    exp = args.run / "experiments" / "distinguish_compartment_of_neuron"
    def read(folder, suffix):
        return pd.read_parquet(folder / f"{BODY}_{suffix}.parquet")

    nodes = read(skel, "annotated_skeleton_v3")
    xyz = nodes[XYZ_EM].to_numpy() * [.008, .008, .030]
    v3 = read(skel, "annotated_edges_v3")
    v11 = read(skel, "annotated_edges_v11_science_like")
    nodes_v11 = read(skel, "annotated_skeleton_v11_science_like")
    bridges = read(exp / "derived_component_bridging" / str(BODY), "derived_bridge_edges")
    s4 = exp / "v3_bridge_aware_fill" / str(BODY)
    s5 = exp / "v3_boundary_component_fill" / str(BODY)
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    stages = [
        ("01 | V3 seed labels", v3, None, "compartment"),
        ("02 | V11 independent split", v11, None, "compartment"),
        ("03 | V11 with geometric bridges", v11, bridges, "compartment"),
        ("04 | V3 chain fill", read(s4, "annotated_edges_v3_bridge_aware"),
         read(s4, "bridge_edges_v3_bridge_aware"), "compartment_bridge_aware"),
        ("05 | V3 boundary fill", read(s5, "annotated_edges_v3_boundary_fill"),
         read(s5, "bridge_edges_v3_boundary_fill"), "compartment_boundary_fill"),
    ]
    views = []
    stage_summaries = []
    for title, edges, links, col in stages:
        start = len(fig.data)
        pairs = edges[["u", "v"]].to_numpy(int)
        if not np.array_equal(nodes.rowId.to_numpy()[pairs],
                              edges[["parent_id_original", "child_id_original"]].to_numpy()):
            raise ValueError(f"Skeleton node IDs do not match {title}")
        add_edges(fig, xyz, pairs, edges[col].to_numpy(), 1)
        if links is not None:
            add_edges(fig, xyz, links[["u_node_index", "v_node_index"]].to_numpy(int),
                      links[col].to_numpy(), 1, bridge=True)
        if title.startswith(("02", "03")):
            ids = nodes_v11.loc[nodes_v11.compartment.eq("soma"), "rowId"]
            points = xyz[nodes.rowId.isin(ids)]
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name="Soma seed",
                marker=dict(size=7, color="#a31e5b"),
            ), row=1, col=1)
        stats = edges[col].value_counts().to_dict()
        subtitle = (f"{stats.get('ambiguous', 0):,} ambiguous original edges | "
                    f"{stats.get('soma', 0):,} soma edges | {0 if links is None else len(links)} bridges")
        views.append((title, subtitle, range(start, len(fig.data))))
        stage_summaries.append(dict(stage=title, original_edges=stats, bridges=0 if links is None else len(links)))
    start = len(fig.data)
    add_edges(fig, xyz, v3[["u", "v"]].to_numpy(int), np.full(len(v3), "unassigned"), 1)
    soma_root = int(np.flatnonzero(nodes.distance_to_soma_geodesic_nm.eq(0))[0])
    total = float(v3.loc[v3.rooted_parent_node_index.eq(soma_root), "subtree_pre_weight"].sum())
    candidates = candidate_traces(fig, xyz, v3, nodes, 3, total)
    views.append(("Diagnostic | Axon candidate subtrees",
                  "Legend values: PRE coverage / PRE purity. Diamonds mark subtree bases.",
                  range(start, len(fig.data))))
    overlays = range(len(fig.data), len(fig.data) + 2)
    attachments = read(skel, "synapse_skeleton_attachments_v3")
    for role, color in [("pre", "#aa3b20"), ("post", "#1673b8")]:
        q = attachments[attachments.role.eq(role) & attachments.attachment_status.eq("attached")]
        points = q[["x_em_nm", "y_em_nm", "z_em_nm"]].to_numpy() / 1000
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            name=f"{role.upper()} synapses ({len(q)})", marker=dict(size=4, color=color, opacity=.72),
            customdata=q[["confidence", "attachment_distance_nm"]].to_numpy(),
            hovertemplate="Confidence %{customdata[0]:.2f}<br>Snap distance %{customdata[1]:.0f} nm<extra></extra>",
        ), row=1, col=1)

    def title(view):
        return f"PC {BODY} | {view[0]}<br><sup>{view[1]}</sup>"

    def visibility(view):
        return [True if i in view[2] else (view[0].startswith("Diagnostic") or "legendonly")
                if i in overlays else False for i in range(len(fig.data))]
    for trace, visible in zip(fig.data, visibility(views[0])):
        trace.visible = visible
    fig.update_layout(
        title=dict(text=title(views[0]), x=.02), template="plotly_white",
        margin=dict(t=125, b=25, l=10, r=10),
        legend=dict(orientation="h", x=0, y=1.01),
        scene=dict(aspectmode="data", uirevision="pc", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)"),
        updatemenus=[dict(
            buttons=[dict(label=v[0], method="update", args=[{"visible": visibility(v)}, {"title.text": title(v)}])
                     for v in views], x=1, xanchor="right", y=1.14,
        )],
    )
    fig.write_html(args.output / "fish2-pc-100076106-steps.html", default_height="94vh",
                   config={"displaylogo": False, "scrollZoom": True})
    result = dict(body_id=BODY, stages=stage_summaries, weighted_main_pre=total, axon_candidates=candidates)
    (args.output / "pc_step_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
