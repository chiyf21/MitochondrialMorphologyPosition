"""Plot the 21 E01 in-sample OLS R² scores in descending order."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    data = data.loc[~data.duplicate_target].set_index("target")
    if not data.index.is_unique or set(data.index) != set(LABELS):
        raise ValueError("Expected the 21 distinct E01 targets")
    scores = data["ols_in_sample_r2"].sort_values(ascending=False, kind="stable")
    if not np.isfinite(scores).all():
        raise ValueError("Missing or non-finite R²")
    rows = np.arange(len(scores))
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(9.8, 10.8), layout="constrained")
    ax.grid(axis="x", color="#dddddd", linewidth=.7)
    ax.set_axisbelow(True)
    ax.scatter(scores, rows, s=43, color="#102a63", zorder=3)
    ax.set_yticks(rows, [LABELS[target] for target in scores.index], fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set(xlabel="R²", xlim=(min(0, scores.min()-.025), max(1.025, scores.max()+.025)))
    ax.xaxis.label.set_fontsize(15)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output/"E01_feature_recovery_R2.csv")
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
