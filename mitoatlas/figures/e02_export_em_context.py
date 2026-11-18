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
from PIL import Image, ImageDraw, ImageFont
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


SCALE_NM = 600
XY_NM_PER_PIXEL = 8
SCALE_FONT = ImageFont.truetype(
    str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 22
)


def add_png_scale(image):
    label = f"{SCALE_NM} nm"
    length = SCALE_NM // XY_NM_PER_PIXEL
    left, top, right, bottom = SCALE_FONT.getbbox(label)
    text_width, text_height = right - left, bottom - top
    pad, gap, thickness = 8, 5, 7
    panel_width = max(length, text_width) + 2 * pad
    y1, x1 = image.height - 8, image.width - 8
    x0, y0 = x1 - panel_width, y1 - (2 * pad + text_height + thickness + gap)
    if x0 < 0 or y0 < 0:
        raise ValueError("PNG too small for the physical scale bar")
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill="black")
    bar_x = x0 + (panel_width - length) // 2
    bar_y = y1 - pad - thickness
    draw.rectangle((bar_x, bar_y, bar_x + length - 1, bar_y + thickness - 1), fill="white")
    draw.text((x0 + (panel_width - text_width) // 2 - left, y0 + pad - top),
              label, fill="white", font=SCALE_FONT)
    return image


def save_png_preview(output, pair, rows):
    fig, axes = pair_axes(pair)
    for ax, row in zip(axes, rows):
        z = int(row["representative_global_z"])
        folder = output / row["folder"] / "em_xy_boundary"
        images = list(folder.glob(f"xy_z*_global_z{z:05d}.png"))
        if len(images) != 1:
            raise ValueError(f"Expected one preview image in {folder} for global Z {z}")
        with Image.open(images[0]) as image:
            ax.imshow(image, interpolation="nearest")
        label_axis(ax, row, z)
    fig.savefig(output / pair / "context_preview.png", dpi=160)
    plt.close(fig)


def export_instance(folder, data, mask, z0, low, high, missing):
    raw_dir = folder / "em_xy_raw"
    overlay_dir = folder / "em_xy_boundary"
    raw_dir.mkdir(parents=True)
    overlay_dir.mkdir()
    scale_dir = folder / "em_xy_with_scale"
    scale_dir.mkdir()
    for index, (plane, target) in enumerate(zip(data, mask)):
        filename = f"xy_z{index:04d}_global_z{z0 + index:05d}.png"
        raw = Image.fromarray(plane)
        raw.save(raw_dir / filename)
        add_png_scale(raw).save(scale_dir / filename)
        scaled = np.rint(np.clip((plane.astype(float) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        overlay = Image.fromarray(scaled).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for path in outline(target):
            draw.line(path, fill="#5CE1D2", width=2, joint="curve")
        if missing[index]:
            draw.text((8, 8), "MISSING RAW EM | MASK ONLY", fill="white")
        add_png_scale(overlay).save(overlay_dir / filename)


def preview_index(mask, missing):
    return int(np.where(missing, -1, mask.sum(axis=(1, 2))).argmax())


def pair_axes(pair):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.55), layout="constrained")
    fig.suptitle(f"Pair {int(pair[-2:]):02d}   |   {pair.split('_')[0]}", fontsize=14)
    return fig, axes


def label_axis(ax, entry, z):
    ax.set_title(f"{entry['side']} · {entry['body_id']} · global Z {z}", fontsize=11)
    ax.set_axis_off()


def page(pair, group, low, high):
    fig, axes = pair_axes(pair)
    for ax, (entry, data, mask, bounds, _, missing) in zip(axes, group):
        index = preview_index(mask, missing)
        ax.imshow(data[index], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        for path in outline(mask[index]):
            ax.plot(*np.asarray(path).T, color="#5CE1D2", linewidth=1.0)
        label_axis(ax, entry, bounds[2] + index)
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
            pair_rows = []
            for entry, data, mask, bounds, agreement, missing in group:
                export_instance(args.output / entry["folder"], data, mask, bounds[2], low, high, missing)
                pair_rows.append(dict(pair=pair, side=entry["side"], body_id=entry["body_id"],
                                    record_id=entry["record_id"], folder=entry["folder"],
                                    bbox_xyz=json.dumps(bounds), xy_slices=len(data),
                                    width_px=args.width, field_width_nm=args.width * 8,
                                    representative_global_z=bounds[2] + preview_index(mask, missing),
                                    missing_raw_global_z=json.dumps((np.flatnonzero(missing) + bounds[2]).tolist()),
                                    full_old_crop_pixel_agreement=agreement,
                                    nonblank_in_mask_pixel_agreement=1, display_min=low, display_max=high))
            fig = page(pair, group, low, high)
            pdf.savefig(fig, dpi=240)
            plt.close(fig)
            save_png_preview(args.output, pair, pair_rows)
            summary.extend(pair_rows)
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
        "contour in em_xy_boundary. em_xy_with_scale contains the grayscale view with a scale bar. "
        "The boundary PNGs also have the scale bar. Each pair shares a display window for the contour views. "
        "Filenames give the zero-based crop index and the global Z coordinate.\n\n"
        "The gallery uses the largest-area mask slice with raw EM for each instance. "
        "context_instances.csv gives the crop bounds as half-open XYZ and the original-crop pixel agreement. "
        "Source placeholder sections are kept as-is, marked as missing on the overlays, and listed by global Z.\n"
    )


if __name__ == "__main__":
    main()
