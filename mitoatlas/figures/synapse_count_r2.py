"""Compare representation-only FISH2 synapse count R² by endpoint role."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    roles = ["pre", "post"]
    models = ["concat_only", "21D_only"]
    scores = pd.read_csv(args.scores).query('scale == "compartment" and scope == "pooled" and regressor == "ridge"')
    selected = scores[scores.model.isin(models) & scores.target.isin(roles)]
    table = selected.pivot(index="target", columns="model", values="r2").reindex(index=roles, columns=models)
    if not np.isfinite(table.to_numpy()).all() or len(selected[["units", "neurons"]].drop_duplicates()) != 1:
        raise ValueError("Expected matched finite representation-only pre/post scores")
    print(table.to_string())

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(4.4, 3.25), layout="constrained")
    locations = np.arange(len(roles))
    for model, label, color, offset in zip(models, ("Concat", "21D"), ("#634199", "#607884"), (-.19, .19)):
        bars = ax.bar(locations + offset, table[model], width=.32, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11, color="#2B2433")
    ax.axhline(0, color="#2B2433", linewidth=.8)
    ax.set_xticks(locations, ["Pre", "Post"])
    ax.set_ylabel("R²", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_xlim(-.52, 1.52)
    low, high = table.to_numpy().min(), table.to_numpy().max()
    span = max(high, 0) - min(low, 0)
    ax.set_ylim(low - .13 * span if low < 0 else 0, max(.1, high + .32 * span))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        fig.savefig(args.output.with_suffix(extension), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
