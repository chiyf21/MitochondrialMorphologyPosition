"""Export wider native FISH2 XY views with exact instance-mask contours."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

from cloudvolume import CloudVolume
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours


def load_crop(volume, row, width):
    x0, y0, z0, x1, y1, z1 = json.loads(row["bbox_xyz"])
    x, y = (x0 + x1 - width) // 2, (y0 + y1 - width) // 2
    if row["array_axis_order"] != "ZYX" or x > x0 or y > y0 or x + width < x1 or y + width < y1:
        raise ValueError(f"Invalid context bounds for {row['record_id']}")
    data = np.asarray(volume[x:x + width, y:y + width, z0:z1])[..., 0].transpose(2, 1, 0)
    em = np.load(row["em_path"], mmap_mode="r", allow_pickle=False)
    mask = np.load(row["mask_path"], mmap_mode="r", allow_pickle=False) != 0
    zyx = np.s_[:, y0 - y:y1 - y, x0 - x:x1 - x]
    raw_center = data[zyx]
    if data.shape != (z1 - z0, width, width) or raw_center.shape != em.shape or mask.shape != em.shape:
        raise ValueError(f"Source or mask shape does not match bbox for {row['record_id']}")
    if not mask.any():
        raise ValueError(f"Empty mask for {row['record_id']}")
    # Check the source against the saved instance, separately from the coordinate math.
    missing = np.all(data == 1, axis=(1, 2))
    verified = mask & ~missing[:, None, None]
    if not verified.any() or not np.array_equal(raw_center[verified], em[verified]):
        raise ValueError(f"Nonblank source EM differs inside {row['record_id']} mask")
    full_mask = np.zeros(data.shape, dtype=bool)
    full_mask[zyx] = mask
    return data, full_mask, [x, y, z0, x + width, y + width, z1], float(np.mean(raw_center == em)), missing


def outline(mask):
    return [list(map(tuple, path[:, ::-1] - 1))
            for path in find_contours(np.pad(mask, 1), level=0.5)]


def export_instance(folder, data, mask, z0, low, high, missing):
    raw_dir = folder / "em_xy_raw"
    overlay_dir = folder / "em_xy_boundary"
    raw_dir.mkdir(parents=True)
    overlay_dir.mkdir()
    for index, (plane, target) in enumerate(zip(data, mask)):
        filename = f"xy_z{index:04d}_global_z{z0 + index:05d}.png"
        Image.fromarray(plane).save(raw_dir / filename)
        scaled = np.rint(np.clip((plane.astype(float) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        overlay = Image.fromarray(scaled).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for path in outline(target):
            draw.line(path, fill="#5CE1D2", width=2, joint="curve")
        if missing[index]:
            draw.text((8, 8), "MISSING RAW EM | MASK ONLY", fill="white")
        overlay.save(overlay_dir / filename)


def preview_index(mask, missing):
    return int(np.where(missing, -1, mask.sum(axis=(1, 2))).argmax())


def page(pair, group, low, high):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.55), layout="constrained")
    fig.suptitle(f"Pair {int(pair[-2:]):02d}   |   {pair.split('_')[0]}", fontsize=14)
    for ax, (entry, data, mask, bounds, _, missing) in zip(axes, group):
        index = preview_index(mask, missing)
        ax.imshow(data[index], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        for path in outline(mask[index]):
            ax.plot(*np.asarray(path).T, color="#5CE1D2", linewidth=1.0)
        ax.set_title(f"{entry['side']} · {entry['body_id']} · global Z {bounds[2] + index}", fontsize=11)
        ax.set_axis_off()
        scale = AnchoredSizeBar(ax.transData, 500 / 8, "500 nm", "lower right",
                                color="white", frameon=True, pad=0.25)
        scale.patch.set(facecolor="black", edgecolor="none", alpha=0.85)
        ax.add_artist(scale)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("examples_instances_csv", type=Path)
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    with args.metadata_csv.open() as file:
        metadata = {row["record_id"]: row for row in csv.DictReader(file)}
    with args.examples_instances_csv.open() as file:
        examples = list(csv.DictReader(file))
    volume = CloudVolume(args.source, mip=0, parallel=1, progress=False, fill_missing=False)
    if not np.array_equal(volume.resolution, [8, 8, 30]):
        raise ValueError(f"Unexpected source resolution {volume.resolution}")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    pair_ids = dict.fromkeys(row["pair"] for row in examples)
    with PdfPages(args.output / "context_gallery.pdf") as pdf:
        for pair in pair_ids:
            group = [(row, *load_crop(volume, metadata[row["record_id"]], args.width))
                     for row in examples if row["pair"] == pair]
            if [row["side"] for row, *_ in group] != ["A", "B"]:
                raise ValueError(f"Expected A and B for {pair}")
            samples = [data.ravel()[::max(1, data.size // 20000)] for _, data, *_ in group]
            low, high = np.quantile(np.concatenate(samples), [0.01, 0.99])
            if not high > low:
                raise ValueError(f"Degenerate display range for {pair}")
            for entry, data, mask, bounds, agreement, missing in group:
                export_instance(args.output / entry["folder"], data, mask, bounds[2], low, high, missing)
                summary.append(dict(pair=pair, side=entry["side"], body_id=entry["body_id"],
                                    record_id=entry["record_id"], folder=entry["folder"],
                                    bbox_xyz=json.dumps(bounds), xy_slices=len(data),
                                    width_px=args.width, field_width_nm=args.width * 8,
                                    representative_global_z=bounds[2] + preview_index(mask, missing),
                                    missing_raw_global_z=json.dumps((np.flatnonzero(missing) + bounds[2]).tolist()),
                                    full_old_crop_pixel_agreement=agreement,
                                    nonblank_in_mask_pixel_agreement=1, display_min=low, display_max=high))
            fig = page(pair, group, low, high)
            pdf.savefig(fig, dpi=240)
            fig.savefig(args.output / pair / "context_preview.png", dpi=160)
            plt.close(fig)
            print(pair, [f"{r['side']} {value:.2%}" for r, _, _, _, value, _ in group], flush=True)
    with (args.output / "context_instances.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "run.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "voxel_xyz_nm": [8, 8, 30],
    }, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# FISH2 wide-field EM context\n\n"
        "Each instance folder has original 8-bit EM in em_xy_raw and a view with its segmentation "
        "contour in em_xy_boundary. Each pair shares a display window for the contour views. "
        "Filenames give the zero-based crop index and the global Z coordinate.\n\n"
        "The gallery uses the largest-area mask slice with raw EM for each instance. "
        "context_instances.csv gives the crop bounds as half-open XYZ and the original-crop pixel agreement. "
        "Source placeholder sections are kept as-is, marked as missing on the overlays, and listed by global Z.\n"
    )


if __name__ == "__main__":
    main()
