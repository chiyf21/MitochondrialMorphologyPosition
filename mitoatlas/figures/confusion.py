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
    ("R1", "per_group", "Cell-type classification", "neuron"),
    ("R1", "per_mito", "Cell-type classification", "mitochondrion"),
    ("R2", "per_mito", "Compartment classification", "mitochondrion"),
)
REPRESENTATIONS = ("handcrafted", "concat")
COLORS = ("#f7fcf8", "#54b8aa", "#102a63")
COLORMAP = LinearSegmentedColormap.from_list("fish2_confusion", COLORS, N=257)
SCALE = Normalize(vmin=0, vmax=1)
AXIS_SIZE = 24
TICK_SIZE = 22
matplotlib.rcParams["pdf.fonttype"] = 42


def save(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output/f"{name}.{ext}", dpi=300)
    plt.close(fig)


def colorbar(output):
    fig, ax = plt.subplots(figsize=(3.05, 7.3), layout="constrained")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=SCALE, cmap=COLORMAP), cax=ax,
                 ticks=np.linspace(0, 1, 6))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_ylabel("Fraction within true class", fontsize=AXIS_SIZE, labelpad=14)
    ax.yaxis.set_label_position("right")
    save(fig, output, "confusion_colorbar")


def draw(ax, fractions, name, metric, sample_count):
    ax.imshow(fractions, norm=SCALE, cmap=COLORMAP, interpolation="nearest")
    ax.set_xticks(range(len(fractions.columns)), fractions.columns,
                  rotation=45 if len(fractions) > 2 else 0, ha="right" if len(fractions) > 2 else "center")
    ax.set_yticks(range(len(fractions)), fractions.index)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlabel("Predicted class", fontsize=AXIS_SIZE, labelpad=12)
    ax.set_ylabel("True class", fontsize=AXIS_SIZE, labelpad=12)
    ax.set_title(f"{name} · Macro F1 {metric.macro_f1:.3f} · n={sample_count:,}",
                 fontsize=16, pad=12)


def render(source, output, handcraft_label="Handcrafted", individual=False):
    metrics = pd.read_csv(source/"classification.csv").set_index(["task","representation","scope"])
    names = {"handcrafted":handcraft_label,"concat":"Concat"}
    output.mkdir(parents=True, exist_ok=True)
    for task, scope, title, unit in TASKS:
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
            fig, axes = plt.subplots(1, len(group), figsize=(17.4,8.6) if len(group)==2 else (8.7,8.6),
                                     squeeze=False,layout="constrained")
            axes = axes.ravel()
            for ax, rep in zip(axes,group):
                draw(ax,normalized[rep],names[rep],metrics.loc[(task,rep,scope)],frames[rep].to_numpy().sum())
            fig.suptitle(f"{title} · one prediction per {unit}",fontsize=19)
            name = f"{stem}_confusion" if len(group)==2 else f"{stem}_{group[0]}_confusion"
            save(fig, output, name)
    colorbar(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--handcraft-label",default="Handcrafted")
    p.add_argument("--individual",action="store_true")
    args = p.parse_args()
    render(args.results,args.output,args.handcraft_label,args.individual)


if __name__ == "__main__":
    main()
