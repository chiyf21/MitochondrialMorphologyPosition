"""Export morphology-module figures for any pipeline run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures.confusion import render



def save(fig,path):
    for extension in ("png","pdf"):
        fig.savefig(path.with_suffix(f".{extension}"),dpi=300,bbox_inches="tight")
    plt.close(fig)


def recovery(folder):
    data = pd.read_csv(folder/"recovery.csv")
    data = data.loc[~data.duplicate_target].sort_values(["descriptor_family","weighted_r2"])
    fig,ax = plt.subplots(figsize=(8,8))
    for index,row in enumerate(data.itertuples(index=False)):
        color = "#ad3d3e" if row.hull_definition=="convex_hull" else "#246775"
        ax.hlines(index,row.r2_low,row.r2_high,color=color,linewidth=1.3)
        ax.plot(row.weighted_r2,index,"o",color=color,markersize=4)
    ax.set_yticks(np.arange(len(data)),data.target)
    ax.axvline(0,color="#777777",linewidth=.8)
    ax.set_xlabel("Neuron-weighted out-of-fold R²")
    save(fig,folder/"feature_recovery")


def confusion(folder):
    render(folder,folder)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment",choices=["E01","E03"])
    p.add_argument("result",type=Path)
    args = p.parse_args()
    {"E01":recovery,"E03":confusion}[args.experiment](args.result)


if __name__=="__main__":
    main()
