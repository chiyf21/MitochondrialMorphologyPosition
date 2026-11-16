#!/usr/bin/env python3
"""Compare the original Step 02 and NAVis on the same FISH2 panel."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_pc_compartments import XYZ, add_edges, prepare

SYNC_CAMERAS = """
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


def run_body(source, scratch, body, script):
    skeleton = prepare(source, scratch, body)
    subprocess.run([
        sys.executable, "-c",
        "import runpy, sys; from pathlib import Path; "
        "script, root, body = sys.argv[1:]; sys.argv = [script, body]; "
        "runpy.run_path(script, init_globals={'ROOT': Path(root)})",
        str(script), str(scratch), str(body),
    ], check=True)
    nodes = pd.read_parquet(skeleton / f"{body}_annotated_skeleton_v11_science_like.parquet")
    edges = pd.read_parquet(skeleton / f"{body}_annotated_edges_v11_science_like.parquet")
    summary = json.loads((skeleton / f"{body}_compartment_summary_v11_science_like.json").read_text())
    return nodes, edges, summary


def compare(fig, nodes, edges, navis_csv):
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]
    navis = pd.read_csv(navis_csv).set_index("node_id").loc[nodes.rowId]
    if not np.allclose(xyz * 1000, navis[XYZ].to_numpy(), atol=.001, rtol=0):
        raise ValueError("The compared skeleton coordinates differ")
    pairs = edges[["u", "v"]].to_numpy(int)
    left, right = navis.compartment.replace({"linker": "connecting_cable"}).to_numpy()[pairs].T
    labels = edges.compartment.to_numpy()
    add_edges(fig, xyz, pairs, labels, 1)
    soma = nodes.compartment.eq("soma").to_numpy()
    fig.add_trace(go.Scatter3d(
        x=xyz[soma, 0], y=xyz[soma, 1], z=xyz[soma, 2], mode="markers",
        name="Soma nodes", marker=dict(size=6, color="#a31e5b"),
        text=nodes.loc[soma, "rowId"].astype(str), hovertemplate="Soma SWC node %{text}<extra></extra>",
    ), row=1, col=1)
    add_edges(fig, xyz, pairs, np.where(left == right, left, "boundary"), 2)
    jointly_labeled = ((left == right) & np.isin(left, ["axon", "dendrite"])
                       & np.isin(labels, ["axon", "dendrite"]))
    return {
        "soma_nodes": int(soma.sum()),
        "step02_edges": edges.compartment.value_counts().to_dict(),
        "shared_axon_dendrite_edges": int(jointly_labeled.sum()),
        "matching_axon_dendrite_edges": int((jointly_labeled & (left == labels)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neuron_root", type=Path)
    parser.add_argument("legacy_step02", type=Path)
    parser.add_argument("navis_output", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True)
    snapshot = root / args.legacy_step02.name
    shutil.copy2(args.legacy_step02, snapshot)
    panel = json.loads((args.navis_output / "run_summary.json").read_text())["neurons"]
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Original Step 02", "NAVis SFC"], horizontal_spacing=.03,
    )
    groups, results = [], []
    for sample in panel:
        body, cell_type = sample["body_id"], sample["cell_type"]
        start = len(figure.data)
        result = {"body_id": body, "cell_type": cell_type, "navis_status": sample["status"]}
        try:
            nodes, edges, summary = run_body(args.neuron_root / str(body), root, body, snapshot)
            result.update(mode=summary["mode"], fallback=summary["fallback"],
                          selected_capture_fraction=summary["selected_capture_fraction"],
                          soma=summary["soma"], synapses=summary["synapses"])
            result.update(compare(figure, nodes, edges, args.navis_output / f"{body}_compartments.csv"))
        except subprocess.CalledProcessError as error:
            result["error"] = str(error)
        else:
            groups.append((result, range(start, len(figure.data))))
        results.append(result)
        print(json.dumps(result), flush=True)
    (root / "step02_panel_summary.json").write_text(json.dumps({
        "legacy_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "neurons": results,
    }, indent=2) + "\n")
    if not groups:
        raise RuntimeError("Step 02 returned no viewable results")
    def title(result):
        synapses = result["synapses"]
        mode = result["fallback"].get("mode", "subtree arbors").replace("_", " ")
        return (
            f"{result['cell_type']} / {result['body_id']}<br><sup>"
            f"Step 02: {mode} | {synapses['attached_pre']} pre / {synapses['attached_post']} post"
            f" | {result['soma_nodes']} soma node(s) | NAVis: {result['navis_status']}</sup>"
        )
    buttons = [dict(label=f"{row['cell_type']} / {row['body_id']}", method="update",
                    args=[{"visible": [i in group for i in range(len(figure.data))]},
                          {"title.text": title(row)}]) for row, group in groups]
    default = next((i for i, (row, _) in enumerate(groups) if row["cell_type"] == "MON2"), 0)
    row, group = groups[default]
    for i, trace in enumerate(figure.data):
        trace.visible = i in group
    for scene in ("scene", "scene2"):
        figure.layout[scene].update(aspectmode="data", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)")
    figure.update_layout(
        title=title(row), template="plotly_white", margin=dict(t=115, b=35, l=10, r=10),
        legend=dict(orientation="h", x=0, y=-.02),
        updatemenus=[dict(buttons=buttons, active=default, x=1, y=1.14, xanchor="right")],
    )
    figure.write_html(root / "fish2-step02-panel.html", post_script=SYNC_CAMERAS,
                      default_height="94vh", config={"displaylogo": False, "scrollZoom": True})


if __name__ == "__main__":
    main()
