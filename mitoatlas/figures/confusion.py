"""Draw FISH2 out-of-fold confusion matrices from E03 result tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = (
    ("R1", "per_group", "Cell-type classification", "neuron"),
    ("R2", "per_mito", "Compartment classification", "mitochondrion"),
)
REPRESENTATIONS = ("handcrafted", "concat")


def draw(ax, counts, name, metric):
    fractions = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    plot = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for i, j in np.ndindex(counts.shape):
        frac, count = fractions.iloc[i, j], counts.iloc[i, j]
        ax.text(j, i, f"{count:,}", ha="center", va="center",
                color="white" if frac >= .55 else "#242424",
                fontsize=9 if len(counts) > 2 else 13)
    ax.set_xticks(range(len(counts.columns)), counts.columns,
                  rotation=45 if len(counts) > 2 else 0, ha="right" if len(counts) > 2 else "center")
    ax.set_yticks(range(len(counts)), counts.index)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title=f"{name} · Macro F1 {metric.macro_f1:.3f} · n={counts.to_numpy().sum():,}")
    ax.set_title(ax.get_title(), fontsize=11)
    return plot


def render(source, output, handcraft_label="Handcrafted", individual=False):
    metrics = pd.read_csv(source/"classification.csv").set_index(["task","representation","scope"])
    names = {"handcrafted":handcraft_label,"concat":"Concat"}
    output.mkdir(parents=True, exist_ok=True)
    for task, scope, title, unit in TASKS:
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
        groups = [REPRESENTATIONS]
        if individual:
            groups.extend((rep,) for rep in REPRESENTATIONS)
        for group in groups:
            fig, axes = plt.subplots(1, len(group), figsize=(11.7,5.2) if len(group)==2 else (6.1,5.4),
                                     squeeze=False,layout="constrained")
            axes = axes.ravel()
            for ax, rep in zip(axes,group):
                image = draw(ax,frames[rep],names[rep],metrics.loc[(task,rep,scope)])
            fig.colorbar(image,ax=axes,label="Fraction within true class",shrink=.81)
            fig.suptitle(f"{title} · one prediction per {unit}",fontsize=13)
            name = f"{task}_confusion" if len(group)==2 else f"{task}_{group[0]}_confusion"
            for ext in ("png","pdf"):
                fig.savefig(output/f"{name}.{ext}",dpi=300)
            plt.close(fig)


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
