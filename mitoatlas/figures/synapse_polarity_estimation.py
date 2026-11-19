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
    for bin_id in np.unique(bins):
        ids = np.flatnonzero(bins == bin_id)
        ids = ids[np.argsort(values[ids], kind="stable")]
        offsets[ids] = (np.arange(len(ids)) - (len(ids) - 1) / 2) * .53 / max(height - 1, 1)
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_csv", type=Path)
    parser.add_argument("output", type=Path)
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
    fig = plt.figure(figsize=(9.1, 3.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.05, 1.05], wspace=.09)
    ax = left.subplots()
    ax.axvline(.5, color="#8b8392", lw=1, ls=(0, (3, 3)), zorder=1)
    for i, (values, color) in enumerate(zip(scores.T, PALETTE)):
        ax.scatter(values, i + strip_offsets(values), color=color, alpha=.55, s=11, linewidths=0)
        mean = values.mean()
        ax.scatter(mean, i + .36, marker="D", s=32, color=color,
                   edgecolors="white", linewidths=.65, zorder=3)
        ax.text(mean, i + .46, f"{mean:.3f}", ha="center", va="bottom", color=color, fontsize=11)
    ax.set(xlim=(-.025, 1.025), ylim=(-.39, 1.65), xlabel="Within-neuron AUROC")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 1], ["21D", "Concat"])
    ax.tick_params(axis="y", length=0, pad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)

    hist, effect = right.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3.2, 1.1]})
    edges = np.arange(-20, 21) / 20
    bins = np.histogram(gain, bins=edges)[0]
    colors = [NEGATIVE if edge < 0 else PALETTE[1] for edge in edges[:-1]]
    hist.bar(edges[:-1] + .002, bins, width=.046, align="edge", color=colors, alpha=.87, linewidth=0)
    hist.axvline(0, color="#8b8392", lw=1, ls=(0, (3, 3)))
    hist.set_ylabel("Neurons")
    hist.grid(axis="y", color=GRID, lw=.55)
    hist.set_axisbelow(True)
    hist.tick_params(axis="x", bottom=False)
    hist.spines[["top", "right", "bottom"]].set_visible(False)

    xmin = min(-.25, np.floor(gain.min() / .25) * .25)
    xmax = max(.5, np.ceil(gain.max() / .25) * .25)
    effect.axvline(0, color="#8b8392", lw=1, ls=(0, (3, 3)))
    effect.hlines(.35, lo, hi, color=INK, linewidth=2.4)
    effect.scatter(center, .35, color=PALETTE[1], edgecolor="white", linewidth=.55, s=39, zorder=3)
    effect.text(center, .59, f"{center:+.3f}  [{lo:.3f}, {hi:.3f}]",
                ha="center", va="bottom", fontsize=10.5, color=INK)
    effect.set(ylim=(0, 1), xlim=(xmin - .025, xmax + .025),
               xlabel="Paired ΔAUROC  (Concat − 21D)")
    effect.set_xticks(np.arange(xmin, xmax + .01, .25), [f"{x:g}" for x in np.arange(xmin, xmax + .01, .25)])
    effect.set_yticks([])
    effect.spines[["top", "right", "left"]].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(args.output.with_suffix(ext), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
