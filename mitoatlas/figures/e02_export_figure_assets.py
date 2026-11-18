"""Export point clouds, native XY slices and 21D cosine for E02 example pairs."""
import argparse
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cosine

from fish2_downstream.common import dump, sha
from mitoatlas.handoff import inputs
from mitoatlas.morphology import standardize
from mitoatlas.synapse_comparisons import arrays


def surface(mask, voxel_xyz_nm):
    points = np.column_stack(np.nonzero(mask & ~binary_erosion(mask)))[:, ::-1]
    points = (points - (points.min(axis=0) + points.max(axis=0)) / 2) * voxel_xyz_nm
    return points


def draw_cloud(points, radius, folder):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho")
    ax.scatter(*points.T, s=1.15, c="#253548", edgecolors="none", depthshade=False)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.22)
    ax.view_init(elev=23, azim=-58)
    ax.set_axis_off()
    fig.savefig(folder / "point_cloud.pdf", transparent=True)
    fig.savefig(folder / "point_cloud.png", dpi=240, transparent=True)
    plt.close(fig)


def export_slices(em, low, high, folder):
    folder.mkdir()
    for z, plane in enumerate(em):
        scaled = (plane.astype(np.float64) - low) / (high - low)
        pixels = np.rint(np.clip(scaled, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(folder / f"xy_z{z:04}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("examples_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frame, matrices, hashes = inputs(args.handoff)
    hand = matrices["handcrafted"].set_axis(frame.record_id)
    if (hand.shape[1] != 22 or not hand["mito CA"].equals(hand["PC1 CA"])):
        raise ValueError("Expected 22 handcraft columns with identical mito CA and PC1 CA")
    hand = hand.drop(columns="mito CA")
    eligible = frame.loc[frame.compartment_eligible, "record_id"]
    values, kept = standardize(hand.loc[eligible])
    if kept != list(hand.columns):
        raise ValueError("Expected all 21 handcraft dimensions to have nonzero variance")
    features = pd.DataFrame(values, index=eligible, columns=kept)
    metadata = frame.set_index("record_id")

    pairs = pd.read_csv(args.examples_csv, dtype={"body_a":str, "body_b":str})
    pairs = pairs[pairs.compartment.isin(["axon", "dendrite"])].sort_values(
        ["compartment", "mask_iou", "example"], ascending=[True, False, True]
    ).groupby("compartment", sort=False).head(2).copy()
    if pairs.groupby("compartment").size().to_dict() != {"axon": 2, "dendrite": 2}:
        raise ValueError("Expected two example pairs per compartment")
    pairs["pair"] = [f"{row.compartment}_pair_{row.example:02}" for row in pairs.itertuples()]
    args.output.mkdir(parents=True, exist_ok=False)
    instance_rows = []
    for pair in pairs.itertuples(index=False):
        ids = [pair.record_a, pair.record_b]
        records = [metadata.loc[record] for record in ids]
        crops = [arrays(record) for record in records]
        # The same native-intensity window is used for both instances in a pair.
        samples = [em.ravel()[::max(1, em.size // 20_000)] for em, _ in crops]
        low, high = np.quantile(np.concatenate(samples), [0.01, 0.99])
        if not high > low:
            raise ValueError(f"Degenerate pooled intensity window for {pair.pair}")
        voxel_xyz_nm = np.array([8.0, 8.0, 30.0])
        points = [surface(mask, voxel_xyz_nm) for _, mask in crops]
        radius = 0.6 * max(np.max(np.ptp(cloud, axis=0)) for cloud in points)
        pairs.loc[pairs.pair.eq(pair.pair), "cosine_21d_zscore"] = 1 - cosine(*features.loc[ids].to_numpy())
        pairs.loc[pairs.pair.eq(pair.pair), "cosine_21d_raw"] = 1 - cosine(*hand.loc[ids].to_numpy())
        for side, record_id, record, (em, _), cloud in zip("AB", ids, records, crops, points):
            path = Path(pair.pair) / f"{side}_{record_id}"
            folder = args.output / path
            folder.mkdir(parents=True)
            draw_cloud(cloud, radius, folder)
            export_slices(em, low, high, folder / "em_xy")
            instance_rows.append(dict(pair=pair.pair, side=side, record_id=record_id,
                                      body_id=record.body_id, folder=path.as_posix(), xy_slices=em.shape[0],
                                      y_pixels=em.shape[1], x_pixels=em.shape[2], surface_points=len(cloud),
                                      em_window_low=float(low), em_window_high=float(high)))

    leading = ["pair", "example", "compartment", "cell_type", "cosine_21d_zscore", "cosine_21d_raw"]
    pairs[leading + [col for col in pairs if col not in leading]].to_csv(args.output / "pairs.csv", index=False)
    instances = pd.DataFrame(instance_rows)
    instances.to_csv(args.output / "instances.csv", index=False)
    hand.loc[instances.record_id].to_csv(args.output / "handcraft_21d_raw.csv")
    features.loc[instances.record_id].to_csv(args.output / "handcraft_21d_zscore.csv")
    dump(args.output / "run.json", dict(handoff=str(args.handoff), examples_csv=str(args.examples_csv),
         examples_sha256=sha(args.examples_csv), input_sha256=hashes, features=kept,
         commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         zscore_cohort="handoff compartment_eligible instances", zscore_instances=len(eligible),
         voxel_xyz_nm=[8, 8, 30], seed=None))
    (args.output / "README.md").write_text(
        "# E02 figure assets\n\n"
        "Two axon and two dendrite pairs with highest mask IoU in the E02 example shortlist.\n\n"
        "- `pairs.csv` reports 21D cosine after z-scoring on the reviewed axon/dendrite cohort, "
        "as well as cosine in the original feature units. The duplicate mito CA column is omitted.\n"
        "- `instances.csv` maps each folder to its record ID and lists the slice count and grayscale window.\n"
        "- Each instance has a mask-surface point cloud in vector PDF and transparent PNG. "
        "The two clouds in each pair use the same physical scale and viewing angle.\n"
        "- `em_xy/` contains one PNG per original Z index, starting at zero, in original XY pixel dimensions. "
        "The pair shares a single grayscale window from the pooled EM 1st–99th percentiles.\n",
        encoding="utf-8",
    )
    print(pairs[leading + ["mask_iou"]].to_string(index=False))
    print(instances[["pair", "side", "xy_slices", "surface_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
