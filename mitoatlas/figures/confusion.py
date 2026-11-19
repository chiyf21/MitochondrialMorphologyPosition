"""Draw FISH2 out-of-fold confusion matrices from E03 result tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

TASKS = (
    ("R1", "per_group"),
    ("R1", "per_mito"),
    ("R2", "per_mito"),
)
REPRESENTATIONS = ("handcrafted", "concat")
COLORS = ("#f7fcf8", "#54b8aa", "#102a63")
COLORMAP = LinearSegmentedColormap.from_list("fish2_confusion", COLORS, N=257)
SCALE = Normalize(vmin=0, vmax=1)
TICK_SIZE = 33
R2_TICK_SIZE = 2 * TICK_SIZE
matplotlib.rcParams["pdf.fonttype"] = 42


def save(fig, output, name, *, fit_labels=False):
    for ext in ("png", "pdf"):
        fig.savefig(output/f"{name}.{ext}", dpi=300,
                    bbox_inches="tight" if fit_labels else None, pad_inches=.15)
    plt.close(fig)


def colorbar(output):
    fig = plt.figure(figsize=(3.5, 7.3))
    ax = fig.add_axes([.12, .10, .15, .80])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=SCALE, cmap=COLORMAP), cax=ax,
                 ticks=np.linspace(0, 1, 6))
    ax.tick_params(labelsize=R2_TICK_SIZE)
    save(fig, output, "confusion_colorbar", fit_labels=True)


def draw(ax, fractions, tick_size=TICK_SIZE):
    ax.imshow(fractions, norm=SCALE, cmap=COLORMAP, interpolation="nearest")
    ax.set_xticks(range(len(fractions.columns)), fractions.columns,
                  rotation=45 if len(fractions) > 2 else 0, ha="right" if len(fractions) > 2 else "center")
    ax.set_yticks(range(len(fractions)), fractions.index)
    ax.tick_params(axis="both", labelsize=tick_size)


def render(source, output, individual=False):
    metrics = pd.read_csv(source/"classification.csv").set_index(["task","representation","scope"])
    output.mkdir(parents=True, exist_ok=True)
    for task, scope in TASKS:
        if task == "R1" and scope == "per_mito":
            present = [(task,rep,scope) in metrics.index for rep in REPRESENTATIONS]
            if not any(present):
                continue  # Older classification summaries may only contain R1 group scores.
            if not all(present):
                raise ValueError("Single-mito R1 needs both representations")
        stem = "R1_mito" if task == "R1" and scope == "per_mito" else task
        frames = {}
        for rep in REPRESENTATIONS:
            counts = pd.read_csv(source/f"{task}_{rep}_{scope}_confusion.csv", index_col=0)
            if (list(counts.index) != list(counts.columns) or
                    (counts.to_numpy() < 0).any() or counts.to_numpy().sum() <= 0):
                raise ValueError(f"Invalid confusion matrix: {task} {rep}")
            frames[rep] = counts
        a, b = frames.values()
        if list(a.index) != list(b.index) or not a.sum(axis=1).equals(b.sum(axis=1)):
            raise ValueError(f"Representations do not share the same scored {task} cohort")
        normalized = {rep:counts.div(counts.sum(axis=1), axis=0).fillna(0)
                      for rep,counts in frames.items()}
        for rep in REPRESENTATIONS:
            frames[rep].to_csv(output/f"{stem}_{rep}_counts.csv", index_label="true_class")
            normalized[rep].to_csv(output/f"{stem}_{rep}_row_fraction.csv", index_label="true_class")
        groups = [REPRESENTATIONS]
        if individual:
            groups.extend((rep,) for rep in REPRESENTATIONS)
        for group in groups:
            fig, axes = plt.subplots(1, len(group), figsize=(8.7*len(group),8.6),
                                     squeeze=False,layout="constrained")
            axes = axes.ravel()
            for ax, rep in zip(axes,group):
                draw(ax,normalized[rep],tick_size=R2_TICK_SIZE if task == "R2" else TICK_SIZE)
            if task == "R2":
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for ax in axes:
                    ticks = ax.get_xticklabels()
                    # Check the label bounding boxes after layout at the requested font size.
                    left, right = (tick.get_window_extent(renderer) for tick in ticks)
                    if left.x1 + 8 > right.x0:
                        plt.setp(ticks, rotation=45, ha="right", rotation_mode="anchor")
            name = f"{stem}_confusion" if len(group)==2 else f"{stem}_{group[0]}_confusion"
            save(fig, output, name, fit_labels=task == "R2")
    colorbar(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--handcraft-label",help=argparse.SUPPRESS)  # Compatibility with old redraw commands.
    p.add_argument("--individual",action="store_true")
    args = p.parse_args()
    render(args.results,args.output,args.individual)


if __name__ == "__main__":
    main()
