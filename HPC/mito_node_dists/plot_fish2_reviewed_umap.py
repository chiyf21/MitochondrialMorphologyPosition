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
POINT_RUN = "mito_newmae_point_cls_only_refine_20261110_01"
FUSED_JOB = "56174094a696eb461195"


def infer_fish_fusion(runs, fish_mapping, checkpoint, output):
    import sys
    import torch

    protocol = runs / "fish2_reviewed_81_protocol"
    sys.path.insert(0, str(protocol))
    import run_compartment_modalities as existing

    if fish_mapping.resolve() != (existing.FISH / "instance_map.csv").resolve():
        raise ValueError("Fish inputs must match the existing protocol")
    rows = existing.fish_rows()
    points = np.load(existing.FISH / "pointclouds.npy", mmap_mode="r")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261111)
    device = torch.device("cuda:0")
    existing.bootstrap_run(checkpoint.parent.parent)
    train, model, grouping = existing.load_fusion_model(checkpoint, device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            inp = train.prepare(batch, points, model.point, grouping,
                                model.image, device, 0, False, point_only=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, fused, _ = model(inp, views=1, loss=False)
            chunks.append(fused[12][:, 0].float().cpu().numpy())
            if start % 128 == 0:
                print(f"New batch: {start + len(batch)}/{len(rows)}", flush=True)
    matrix = np.concatenate(chunks)
    if matrix.shape != (len(rows), 384) or not np.isfinite(matrix).all():
        raise ValueError("Invalid new-batch fused_12 output")
    np.save(output / "fish2_14_point_only_fused_12.npy", matrix)
    return matrix


def fusion_matrix(frame, runs, fish_mapping, output):
    run = runs / POINT_RUN
    checkpoint = run / "checkpoints/refine_epoch050.pt"
    cache = runs / "mito_embedding_dashboard_20261030_01/cache" / FUSED_JOB
    request = json.loads((cache / "manifest.json").read_text())["request"]
    if (Path(request["experiment"]).resolve() != run.resolve()
            or request["representation"] != "fused_12" or request["epoch"] != 50
            or not Path(request["checkpoint"]).samefile(checkpoint)):
        raise ValueError("Unexpected dashboard checkpoint or representation")
    legacy_rows = json.loads((run / "cohort.json").read_text())
    legacy = pd.DataFrame([row["key"] for row in legacy_rows], columns=KEYS)
    ids = np.load(cache / "ids.npy", allow_pickle=False)
    cached = np.load(cache / "embeddings.npy", mmap_mode="r")
    if not np.array_equal(ids, np.arange(len(legacy))) or cached.shape != (len(legacy), 384):
        raise ValueError("Expected full canonical cache in cohort order")
    legacy = legacy.astype({"body_id": str}).set_index(KEYS)
    if not legacy.index.is_unique:
        raise ValueError("Duplicate legacy cohort keys")
    legacy["embedding_row"] = np.arange(len(legacy))
    fish = pd.read_csv(fish_mapping, dtype={"record_id": str, "body_id": str})
    if (not fish.record_id.is_unique or not np.array_equal(fish.embedding_row, np.arange(len(fish)))):
        raise ValueError("Expected unique fish IDs in inference order")
    fish = fish.set_index("record_id")
    selected = frame["source_run_id"].eq("e16_legacy_repair_reinfer_20261105")
    legacy_keys = pd.MultiIndex.from_frame(frame.loc[selected, KEYS])
    fish_rows = fish.loc[frame.loc[~selected, "record_id"]]
    if not np.array_equal(fish_rows[KEYS].to_numpy(), frame.loc[~selected, KEYS].to_numpy()):
        raise ValueError("New batch IDs do not match the reviewed cohort")
    matrix = np.empty((len(frame), 384), dtype=np.float32)
    matrix[selected] = cached[legacy.loc[legacy_keys, "embedding_row"].to_numpy()]
    inferred = infer_fish_fusion(runs, fish_mapping, checkpoint, output)
    matrix[~selected] = inferred[fish_rows.embedding_row.to_numpy()]
    print(f"Fused_12 aligned: {selected.sum()} cached + {(~selected).sum()} inferred", flush=True)
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
    titles = ("Handcrafted features (22D)", "Point-only refine fusion (epoch 50, L12, 384D)")
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
            ax.set_title(f"{cell} · {'Handcraft' if j == 0 else 'Point-only fusion'} · n={selected.sum():,}")
    fig.suptitle("FISH2 · Compartment labels by cell type")
    fig.legend(handles=handles, loc="lower center", ncol=len(groups), frameon=False)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(output / "fish2_umap_by_cell_type.png", dpi=230)
    plt.close(fig)
    for j, prefix in enumerate(("handcraft", "fused_12")):
        frame[[f"{prefix}_umap1", f"{prefix}_umap2"]] = coords[j]
    frame[["record_id", "body_id", "cell_type", "compartment", "compartment_qc",
           "handcraft_umap1", "handcraft_umap2", "fused_12_umap1", "fused_12_umap2"
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
    args.output.mkdir(parents=True, exist_ok=True)
    fusion = fusion_matrix(frame, args.runs, args.fish_mapping, args.output)
    coords = []
    for name, matrix in (("handcraft", features), ("point-only fused_12", fusion)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Missing or non-finite {name} input")
        print(f"Fitting {name}: {matrix.shape}", flush=True)
        xy = UMAP(**PARAMS).fit_transform(StandardScaler().fit_transform(matrix))
        if xy.shape != (len(frame), 2) or not np.isfinite(xy).all():
            raise ValueError(f"Invalid {name} coordinates")
        coords.append(xy)
    plot(frame, coords, args.output)
    print(pd.crosstab(frame.cell_type, frame.compartment).to_string())


if __name__ == "__main__":
    main()
