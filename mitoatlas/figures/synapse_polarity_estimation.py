"""Render absolute and paired pre/post AUROC views from the frozen neuron table."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .synapse_polarity_paired import MODELS


PALETTE = ("#607884", "#634199")
NEGATIVE = "#AF6947"
INK = "#2B2433"
GRID = "#DAD5DF"


def strip_offsets(values):
    """Stack within narrow horizontal bins while preserving the AUROC coordinates."""
    bins = np.minimum((values / 0.0275).astype(int), 36)
    height = np.bincount(bins).max()
    offsets = np.zeros(len(values))
    rng = np.random.default_rng(20261117)
    for bin_id in np.unique(bins):
        ids = np.flatnonzero(bins == bin_id)
        ids = rng.permutation(ids)
        offsets[ids] = (np.arange(len(ids)) - (len(ids) - 1) / 2) * .53 / max(height - 1, 1)
    return offsets


def draw_absolute(ax, scores):
    ax.axvline(.5, color="#8b8392", lw=1, ls=(0, (3, 3)), zorder=1)
    for i, (values, color) in enumerate(zip(scores.T, PALETTE)):
        ax.scatter(values, i + strip_offsets(values), color=color, alpha=.55, s=11, linewidths=0)
        mean = values.mean()
        ax.scatter(mean, i + .36, marker="D", s=32, color=color,
                   edgecolors="white", linewidths=.65, zorder=3)
        ax.text(mean, i + .46, f"{mean:.3f}", ha="center", va="bottom", color=color, fontsize=11)
    ax.set(xlim=(-.025, 1.025), ylim=(-.39, 1.65), xlabel="Within-neuron AUROC")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 1], ["Handcrafted", "Model"])
    ax.tick_params(axis="y", length=3, pad=6)
    ax.spines[["top", "right"]].set_visible(False)

def draw_delta(ax, gain, center, lo, hi):
    edges = np.arange(-20, 21) / 20
    tied = np.isclose(gain, 0, atol=1e-12)
    bins = np.histogram(gain[~tied], bins=edges)[0]
    assert bins.sum() + tied.sum() == len(gain)
    colors = [NEGATIVE if edge < 0 else PALETTE[1] for edge in edges[:-1]]
    ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge", color=colors, alpha=.87, linewidth=0)
    ax.axvline(0, ymax=.90, color="#8b8392", lw=1, ls=(0, (3, 3)))
    ax.bar(0, tied.sum(), width=.012, color="#8b8392", linewidth=0)
    ax.grid(axis="y", color=GRID, lw=.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    xmin = min(-.25, np.floor(gain.min() / .25) * .25)
    xmax = max(.5, np.ceil(gain.max() / .25) * .25)
    height = max(bins.max(), tied.sum())
    ax.errorbar(center, height * 1.105, xerr=[[center-lo], [hi-center]], fmt="o", color=PALETTE[1],
                ecolor=INK, elinewidth=1.5, capsize=3, markersize=3.3, zorder=3)
    ax.text(.98, .969, f"{center:+.3f}  [95% CI {lo:.3f}, {hi:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=INK)
    ax.set(ylim=(0, height * 1.24), xlim=(xmin - .025, xmax + .025),
           xlabel="ΔAUROC (Model − Handcrafted)", ylabel="Neurons")
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ticks = np.arange(xmin, xmax + .01, .25)
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ystep = max(1, int(np.ceil(height / 50)) * 10)
    ax.set_yticks(np.arange(0, height + 1, ystep))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--separate", action="store_true", help="Write two standalone vector PDFs.")
    args = parser.parse_args()
    rows = pd.read_csv(args.paired_csv, dtype={"body_id": str})
    scores = rows[list(MODELS)].to_numpy()
    if not rows.body_id.is_unique or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Expected one finite paired AUROC record per neuron")
    gain = scores[:, 1] - scores[:, 0]
    np.testing.assert_allclose(gain, rows.delta_concat_minus_handcrafted)
    bootstrap = np.random.default_rng(20261117).integers(len(gain), size=(1999, len(gain)))
    lo, hi = np.quantile(gain[bootstrap].mean(axis=1), [.025, .975])
    center = gain.mean()
    print(f"{len(gain)} neurons; paired mean gain {center:.6f}; 95% CI {lo:.6f}, {hi:.6f}")

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.separate:
        fig, ax = plt.subplots(figsize=(4.65, 3.2), layout="constrained")
        draw_absolute(ax, scores)
        fig.savefig(args.output.with_name(f"{args.output.name}_AUROC_distribution.pdf"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.65, 3.2), layout="constrained")
        draw_delta(ax, gain, center, lo, hi)
        fig.savefig(args.output.with_name(f"{args.output.name}_delta_AUROC.pdf"))
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(9.3, 3.2), layout="constrained")
        left, right = fig.subfigures(1, 2, wspace=.09)
        draw_absolute(left.subplots(), scores)
        draw_delta(right.subplots(), gain, center, lo, hi)
        for ext in (".pdf", ".png"):
            fig.savefig(args.output.with_suffix(ext), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
