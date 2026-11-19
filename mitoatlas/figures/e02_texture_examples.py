"""Find FISH2 shape-matched pairs that visibly differ in raw EM."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
from scipy import ndimage

from mitoatlas.states import matched_pairs
from mitoatlas.synapse_comparisons import arrays, texture


SPACING = np.array([30, 8, 8])
METRICS = ["crop_scaled_contrast", "in_plane_roughness"]
STEM = "E02_morphology_matched_EM_examples"


def centered_mask(row):
    mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False) != 0
    # Center by mass and sample onto a common 24 nm grid in native orientation.
    offsets = np.rint((np.stack(np.nonzero(mask), axis=1) - ndimage.center_of_mass(mask))
                      * SPACING / 24).astype(np.int32)
    if np.any((offsets < -32768) | (offsets > 32767)):
        raise ValueError(f"Mask too large to index at 24 nm: {row.record_id}")
    z, y, x = offsets.astype(np.uint16).astype(np.int64).T
    return frozenset((z << 32 | y << 16 | x).tolist())


def draw_pair(pair, records, position):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.9), layout="constrained")
    ids = [pair.record_a, pair.record_b]
    images = [arrays(records.loc[record]) for record in ids]
    sampled = [em.ravel()[::max(1, em.size // 20000)] for em, _ in images]
    low, high = np.quantile(np.concatenate(sampled), [0.01, 0.99])
    fig.suptitle(f"Pair {position:02d}   |   {pair.cell_type} · {pair.compartment}   |   "
                 f"3D mask IoU {pair.mask_iou:.2f}", fontsize=14)
    for row, (record, (em, mask)) in enumerate(zip(ids, images)):
        z = mask.sum(axis=(1, 2)).argmax()
        y = mask.sum(axis=(0, 2)).argmax()
        for col, (image, outline, dy, title) in enumerate([
            (mask.any(axis=0), None, 8, "Mask · XY projection"),
            (em[z], mask[z], 8, "Raw EM · XY"),
            (em[:, y], mask[:, y], 30, "Raw EM · XZ"),
        ]):
            ax = axes[row, col]
            height, width = np.array(image.shape) * [dy, 8]
            extent = [-width / 2, width / 2, height / 2, -height / 2]
            ax.imshow(image, cmap="gray_r" if col == 0 else "gray", interpolation="nearest",
                      extent=extent, vmin=0 if col == 0 else low, vmax=1 if col == 0 else high)
            if outline is not None and outline.any() and not outline.all():
                ax.contour(outline, levels=[0.5], colors=["#5ccdc0"], linewidths=0.7,
                           extent=extent, origin="upper")
            ax.set_title(title if row == 0 else "", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            scale = AnchoredSizeBar(ax.transData, 200, "200 nm", "lower right",
                                    color="black" if col == 0 else "white",
                                    frameon=True, size_vertical=5, pad=0.25)
            scale.patch.set(facecolor="white" if col == 0 else "black", edgecolor="none", alpha=0.85)
            ax.add_artist(scale)
        axes[row, 0].set_ylabel(f"{'AB'[row]} · {records.loc[record, 'body_id']}", fontsize=12)
    for col in range(3):
        x_radius = max(ax.get_xlim()[1] for ax in axes[:, col])
        y_radius = max(ax.get_ylim()[0] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_xlim(-x_radius, x_radius)
            ax.set_ylim(y_radius, -y_radius)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("candidate_pairs", type=Path, nargs="?")
    parser.add_argument("output", type=Path)
    parser.add_argument("--large-min-voxels", type=int,
                        help="Search de-duplicated 21D shape neighbors among instances this large")
    parser.add_argument("--large-neighbors", type=int, default=8)
    parser.add_argument("--min-mask-iou", type=float, default=0.60)
    args = parser.parse_args()
    if (args.candidate_pairs is None) == (args.large_min_voxels is None):
        parser.error("Provide either candidate_pairs or --large-min-voxels")
    if args.large_min_voxels is not None and args.large_min_voxels < 1:
        parser.error("--large-min-voxels must be positive")
    records = pd.read_csv(args.handoff / "metadata/instances.csv", dtype={"body_id": str})
    records = records.set_index("record_id", drop=False)
    hand = pd.read_csv(args.handoff / "embeddings/handcrafted_22.csv").set_index("record_id")
    columns = ["cell_type", "compartment", "record_a", "record_b", "geometry_distance"]
    if args.large_min_voxels is None:
        pairs = pd.read_csv(args.candidate_pairs, usecols=columns)
    else:
        large = records.loc[records.compartment_eligible & records.voxel_count.ge(args.large_min_voxels)]
        morphology = hand.loc[large.record_id]
        if not morphology["PC1 CA"].equals(morphology["mito CA"]):
            raise ValueError("Expected duplicate PC1 CA and mito CA")
        feature_names = [
            name for name in json.loads((args.handoff / "metadata/handcrafted_columns.json").read_text())
            if name != "mito CA"
        ]
        with TemporaryDirectory() as temporary:
            temp = Path(temporary)
            matched_pairs(large.reset_index(drop=True), {"handcrafted": morphology[feature_names]}, args.large_neighbors, temp)
            pairs = pd.read_csv(temp / "candidate_pairs.csv", usecols=columns)
    pairs = pairs[pairs.compartment.isin(["axon", "dendrite"])].copy()
    pairs["geometry_percentile"] = pairs.groupby(["cell_type", "compartment"]).geometry_distance.rank(pct=True)
    pairs = pairs[pairs.geometry_percentile <= 0.5].copy()
    pairs["pair_key"] = [tuple(sorted(ids)) for ids in zip(pairs.record_a, pairs.record_b)]
    pairs = pairs.drop_duplicates("pair_key").drop(columns="pair_key")
    for feature, label in [("size", "volume_ratio"), ("PC1 Length", "length_ratio")]:
        a, b = [pairs[col].map(hand[feature]) for col in ("record_a", "record_b")]
        pairs[label] = np.maximum(a, b) / np.minimum(a, b)
    pairs = pairs[(pairs.volume_ratio <= 1.15) & (pairs.length_ratio <= 1.15)].copy()
    print(f"Pairs after handcraft filters: {len(pairs)}", flush=True)
    ids = sorted(set(pairs.record_a) | set(pairs.record_b))
    rows = list(records.loc[ids].itertuples(index=False))
    with ThreadPoolExecutor(max_workers=8) as pool:
        masks = dict(zip(ids, pool.map(centered_mask, rows)))
    def iou(a, b):
        left, right = masks[a], masks[b]
        overlap = len(left & right)
        return overlap / (len(left) + len(right) - overlap)
    pairs["mask_iou"] = [iou(a, b) for a, b in zip(pairs.record_a, pairs.record_b)]
    pairs = pairs[pairs.mask_iou >= args.min_mask_iou].copy()
    print(f"Pairs after 3D overlap filter: {len(pairs)}", flush=True)
    ids = sorted(set(pairs.record_a) | set(pairs.record_b))
    with ThreadPoolExecutor(max_workers=8) as pool:
        textures = pd.DataFrame(pool.map(texture, records.loc[ids].itertuples(index=False))).set_index("record_id")
    for col in METRICS:
        pairs[f"{col}_a"] = pairs.record_a.map(textures[col])
        pairs[f"{col}_b"] = pairs.record_b.map(textures[col])
        pairs[f"{col}_difference"] = (pairs[f"{col}_a"] - pairs[f"{col}_b"]).abs()
    # Rank pairs by the mean of their two within-compartment difference percentiles.
    ranks = pairs.groupby("compartment")[[f"{col}_difference" for col in METRICS]].rank(pct=True)
    pairs["texture_rank_score"] = ranks.mean(axis=1, skipna=False)
    for side in "ab":
        pairs[f"body_{side}"] = pairs[f"record_{side}"].map(records.body_id)
        pairs[f"voxel_count_{side}"] = pairs[f"record_{side}"].map(records.voxel_count)
        pairs[f"volume_um3_{side}"] = pairs[f"voxel_count_{side}"] * 8 * 8 * 30 / 1e9
    selected = []
    for compartment, group in pairs.groupby("compartment", sort=True):
        used_bodies = set()
        group = group.dropna(subset=["texture_rank_score"])
        for pair in group.sort_values(["texture_rank_score", "mask_iou"], ascending=False).itertuples(index=False):
            bodies = {pair.body_a, pair.body_b}
            if bodies & used_bodies:
                continue
            selected.append(pair._asdict())
            used_bodies.update(bodies)
            if len(used_bodies) == 6:
                break
    if not selected:
        raise ValueError("No shape-matched pairs with finite EM texture measurements")
    picked = pd.DataFrame(selected)
    picked.insert(0, "example", range(1, len(picked) + 1))
    print(picked[["example", "cell_type", "compartment", "mask_iou", "texture_rank_score"]].to_string(index=False))
    args.output.mkdir(parents=True, exist_ok=False)
    picked.to_csv(args.output / f"{STEM}.csv", index=False)
    (args.output / "selection.json").write_text(json.dumps({
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }, indent=2) + "\n")
    with PdfPages(args.output / f"{STEM}.pdf") as pdf:
        for pair in picked.itertuples(index=False):
            fig = draw_pair(pair, records, pair.example)
            pdf.savefig(fig, dpi=240)
            fig.savefig(args.output / f"{STEM}_{pair.example:02d}.png", dpi=145)
            plt.close(fig)


if __name__ == "__main__":
    main()
