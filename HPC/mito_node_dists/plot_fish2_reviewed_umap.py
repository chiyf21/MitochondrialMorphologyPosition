"""Plot paired FISH2 UMAPs from the reviewed cohort and epoch-50 exports."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

CELL_TYPES = ("GC", "CON", "VMN", "PC", "MON2", "MON1")
COLORS = {"soma": "#CC79A7", "axon": "#0072B2", "dendrite": "#D55E00",
          "ambiguous": "#8A929C", "unknown": "#414852", "mixed": "#009E73",
          "primary_neurite": "#E69F00", "neurite": "#009E73"}
KEYS = ["source_run_id", "body_id", "global_instance"]
PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="euclidean",
              random_state=0, n_components=2, n_jobs=1)
IMAGE_RUN = "mito_newmae_image_cls_only_refine_20261110_01"


def image_matrix(frame, runs, fish_mapping):
    run = runs / IMAGE_RUN
    protocol = runs / "fish2_reviewed_81_protocol"
    legacy_rows = json.loads((run / "cohort.json").read_text())
    legacy = pd.DataFrame([row["key"] for row in legacy_rows], columns=KEYS)
    legacy["embedding_row"] = np.arange(len(legacy))
    legacy = legacy.astype({"body_id": str})
    fish = pd.read_csv(fish_mapping, dtype={"record_id": str}).set_index(
        "record_id", verify_integrity=True
    )
    matrix = np.full((len(frame), 768), np.nan, dtype=np.float32)
    for is_legacy, directory in (
        (True, protocol / "cls_only_legacy/image_12"),
        (False, protocol / "fish2_14_image_cls_only"),
    ):
        meta = json.loads((directory / "complete.json").read_text())
        expected = run / "checkpoints/refine_epoch050.pt"
        if Path(meta["checkpoint"]).resolve() != expected.resolve():
            raise ValueError(f"Unexpected checkpoint in {directory}")
        array = np.load(directory / "image_12.npy", mmap_mode="r")
        count = len(legacy) if is_legacy else len(fish)
        if array.shape != (count, 768) or meta["instances"] != count:
            raise ValueError(f"Invalid embedding dimensions in {directory}")
        selected = frame["source_run_id"].eq("e16_legacy_repair_reinfer_20261105")
        if is_legacy:
            indices = frame.loc[selected, KEYS].merge(
                legacy, on=KEYS, how="left", validate="one_to_one", sort=False
            )["embedding_row"]
        else:
            selected = ~selected
            if not frame.loc[selected, "source_run_id"].eq(
                "e16_legacy_repair_reinfer_20261112_fish2_14"
            ).all():
                raise ValueError("Unexpected segmentation run")
            indices = fish.loc[frame.loc[selected, "record_id"], "embedding_row"]
        if indices.isna().any() or not indices.between(0, count - 1).all():
            raise ValueError(f"Unmatched embedding rows in {directory}")
        matrix[selected] = array[indices.to_numpy(dtype=int)]
    return matrix


def draw(ax, points, labels, palette, limits, size):
    for label, color in palette.items():
        x, y = points[labels == label].T
        ax.scatter(x, y, s=size, color=color, alpha=0.68, linewidths=0,
                   rasterized=True)
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color("#D0D5DA")


def plot(frame, coords, output):
    from matplotlib.lines import Line2D

    labels = frame["compartment"].fillna("unknown").str.strip().str.lower()
    groups = sorted(labels.unique(), key=lambda name: name not in {"ambiguous", "unknown"})
    palette = {name: COLORS.get(name, plt.get_cmap("tab20")(i))
               for i, name in enumerate(groups)}
    handles = [Line2D([], [], marker="o", linestyle="", color=palette[name],
                      label=f"{name.replace('_', ' ').title()} ({(labels == name).sum():,})")
               for name in groups]
    titles = ("Handcrafted features (22D)", "Image CLS only (epoch 50, L12, 768D)")
    limits = [(np.array([p.min(axis=0), p.max(axis=0)]) + np.array([[-.5], [.5]])).T
              for p in coords]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, points, title, lim in zip(axes, coords, titles, limits):
        draw(ax, points, labels, palette, lim, 7)
        ax.set_title(title)
    fig.suptitle(f"FISH2 · {len(frame):,} mitochondria from {frame.body_id.nunique()} neurons")
    fig.legend(handles=handles, loc="lower center", ncol=len(groups), frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "fish2_umap_compartment.png", dpi=230)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for k, cell in enumerate(CELL_TYPES):
        selected = frame["cell_type"].eq(cell)
        for j, points in enumerate(coords):
            ax = axes[k // 2, (k % 2) * 2 + j]
            draw(ax, points[selected], labels[selected], palette, limits[j], 10)
            ax.set_title(f"{cell} · {'Handcraft' if j == 0 else 'Image CLS'} · n={selected.sum():,}")
    fig.suptitle("FISH2 · Compartment labels by cell type")
    fig.legend(handles=handles, loc="lower center", ncol=len(groups), frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(output / "fish2_umap_by_cell_type.png", dpi=230)
    plt.close(fig)
    for j, prefix in enumerate(("handcraft", "image_cls")):
        frame[[f"{prefix}_umap1", f"{prefix}_umap2"]] = coords[j]
    frame[["record_id", "body_id", "cell_type", "compartment", "compartment_qc",
           "handcraft_umap1", "handcraft_umap2", "image_cls_umap1", "image_cls_umap2"
           ]].to_csv(output / "fish2_umap_coordinates.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handcraft", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("fish_mapping", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.handcraft, dtype={"body_id": str, "record_id": str})
    if (len(frame) != 5568 or frame.body_id.nunique() != 81
            or frame.record_id.isna().any() or frame.record_id.duplicated().any()
            or set(frame.cell_type) != set(CELL_TYPES)):
        raise ValueError("Expected the complete reviewed-81 cohort")
    features = frame.loc[:, "size":"SBI"].to_numpy(dtype=float)
    if features.shape != (len(frame), 22):
        raise ValueError("Expected 22 contiguous paper features")
    image = image_matrix(frame, args.runs, args.fish_mapping)
    coords = []
    for name, matrix in (("handcraft", features), ("image CLS", image)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Missing or non-finite {name} input")
        print(f"Fitting {name}: {matrix.shape}", flush=True)
        xy = UMAP(**PARAMS).fit_transform(StandardScaler().fit_transform(matrix))
        if xy.shape != (len(frame), 2) or not np.isfinite(xy).all():
            raise ValueError(f"Invalid {name} coordinates")
        coords.append(xy)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(frame, coords, args.output)
    print(pd.crosstab(frame.cell_type, frame.compartment).to_string())


if __name__ == "__main__":
    main()
