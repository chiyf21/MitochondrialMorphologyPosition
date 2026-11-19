"""Plot paired within-neuron pre/post AUROCs from saved out-of-fold scores."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


MODELS = ("handcrafted_only", "concat_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oof", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    keys = ["body_id", "record_id"]
    rows = pd.read_csv(args.oof, dtype={key: str for key in keys})
    if (rows.duplicated(keys).any() or not rows[MODELS[0]].between(0, 1).all()
            or not rows[MODELS[1]].between(0, 1).all()
            or not rows.is_pre.isin([0, 1]).all()
            or not rows.groupby("body_id").cell_type_fold.nunique().eq(1).all()):
        raise ValueError("Expected one valid OOF score per instance and one fold per neuron")

    pairs = pd.DataFrame([
        dict(body_id=body_id, n_pre=int(group.is_pre.sum()),
             n_post=int(len(group) - group.is_pre.sum()),
             **{model: roc_auc_score(group.is_pre, group[model]) for model in MODELS})
        for body_id, group in rows.groupby("body_id") if group.is_pre.nunique() == 2
    ])
    if pairs.empty:
        raise ValueError("No neuron contains both labels")
    x, y = (pairs[model].to_numpy() for model in MODELS)
    pairs["delta_concat_minus_handcrafted"] = y - x
    args.output.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(args.output / "pre_post_neuron_AUROC.csv", index=False)

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.1, 5.1), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#70707b", linestyle=(0, (4, 4)), linewidth=1.4, zorder=1)
    ax.scatter(x, y, s=32, color="#522a91", alpha=0.48, linewidths=0, zorder=2)
    ax.set(xlabel="21D handcrafted AUROC", ylabel="Concat AUROC",
           xlim=(-0.025, 1.025), ylim=(-0.025, 1.025), aspect="equal")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    tick_labels = ["0", "0.25", "0.5", "0.75", "1"]
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, labelpad=7)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, labelpad=7)
    ax.tick_params(labelsize=14, length=5, width=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(args.output / f"pre_post_neuron_AUROC.{ext}", dpi=300)
    plt.close(fig)

    ties = np.isclose(x, y, rtol=0, atol=1e-12)
    print(f"Paired neurons: {len(pairs)} of {rows.body_id.nunique()}; instances: {len(rows)}")
    print(f"Mean AUROC, 21D: {x.mean():.6f}; Concat: {y.mean():.6f}")
    print(f"Concat better: {sum((y > x) & ~ties)}; tied: {sum(ties)}; 21D better: {sum((x > y) & ~ties)}")


if __name__ == "__main__":
    main()
