"""Overlay a saved PC30 partition on its existing UMAP coordinates."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("setting")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    keys = ["embedding_row", "body_id", "record_id"]
    dtypes = {"body_id": str, "record_id": str}
    coordinates = pd.read_csv(args.coordinates, dtype=dtypes, usecols=keys + ["umap_1", "umap_2"])
    assignments = pd.read_csv(args.assignments, dtype=dtypes, usecols=keys + [args.setting])
    rows = coordinates.merge(assignments, on=keys, validate="one_to_one")
    xy = rows[["umap_1", "umap_2"]].to_numpy()
    if len(rows) != len(coordinates) or len(rows) != len(assignments) or not np.isfinite(xy).all():
        raise ValueError("UMAP coordinates and cluster labels do not align")
    labels = rows[args.setting].to_numpy()
    clusters, counts = np.unique(labels, return_counts=True)
    palette = plt.get_cmap("tab10").colors
    colors = np.asarray(palette)[np.searchsorted(clusters, labels) % len(palette)]
    order = np.random.default_rng(50000).permutation(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 6.1), layout="constrained")
    ax.scatter(*xy[order].T, c=colors[order], s=5, lw=0, alpha=0.7, rasterized=True)
    legend = []
    for i, (cluster, count) in enumerate(zip(clusters, counts)):
        color = palette[i % len(palette)]
        points = xy[labels == cluster]
        center = np.median(points, axis=0)
        point = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
        ax.text(*point, f"C{i + 1}", ha="center", va="center", fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1.6))
        legend.append(Line2D([], [], marker="o", ls="", color=color, markersize=8,
                             label=f"C{i + 1}   {count:,}"))
        print(f"C{i + 1} (raw label {cluster}): {count} ({count / len(rows):.1%})")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", aspect="equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=legend, title="Cluster / mitochondria", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, labelspacing=1)
    fig.suptitle(f"FISH2 PC30  |  {args.setting}  |  {len(rows):,} mitochondria", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(ext), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
