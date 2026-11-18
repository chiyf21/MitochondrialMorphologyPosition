"""Plot the E01 fold-out and in-sample R² comparisons."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = (
    ("ridge_five_fold_r2", "Ridge · held-out neurons", "#102a63", "o"),
    ("ridge_in_sample_r2", "Ridge · in sample", "#178477", "s"),
    ("ols_in_sample_r2", "OLS · in sample", "#a85623", "^"),
)
LABELS = {
    "size": "Volume",
    "mito SA": "Surface area",
    "convex hull compactness": "Convex hull surface area",
    "mito diameter": "Diameter",
    "length_along_skeleton_nm": "Length along skeleton",
    "SBI": "SBI",
}
PC_LABELS = ("length", "inertia moment", "symmetry", "cross-sectional area", "circumference")
PC_FIELDS = ("Length", "inertia moment", "symmetry", "CA", "Circum")
for axis in ("PC1", "PC2", "PC3"):
    LABELS.update({f"{axis} {field}": f"{axis} {label}" for field, label in zip(PC_FIELDS, PC_LABELS)})


def render(source, output):
    data = pd.read_csv(source)
    data = data.loc[~data.duplicate_target].set_index("target", verify_integrity=True)
    if set(data.index) != set(LABELS):
        raise ValueError("Expected the 21 distinct E01 targets")
    shown = data.loc[list(LABELS)]
    x = shown[[col for col, *_ in SERIES]].to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Missing or non-finite R²")
    rows = np.arange(len(shown), dtype=float)
    rows[6:] += .55
    rows[11:] += .55
    rows[16:] += .55
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(9.8, 11.6), layout="constrained")
    ax.axvline(0, color="#656565", linewidth=.85)
    ax.grid(axis="x", color="#dddddd", linewidth=.7)
    ax.set_axisbelow(True)
    for offset, (col, label, color, marker) in zip((-.19, 0, .19), SERIES):
        ax.scatter(shown[col], rows+offset, s=34, color=color, marker=marker, label=label, zorder=3)
    ax.set_yticks(rows, LABELS.values(), fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set(xlabel="R²", xlim=(min(-.035, x.min()-.025), 1.025))
    ax.xaxis.label.set_fontsize(15)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5,1), ncol=3, frameon=False, fontsize=10)
    output.mkdir(parents=True, exist_ok=True)
    shown.to_csv(output/"E01_feature_recovery_R2.csv")
    for extension in ("pdf", "png"):
        fig.savefig(output/f"E01_feature_recovery_R2.{extension}", dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("comparison", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    render(args.comparison, args.output)


if __name__ == "__main__":
    main()
