"""Export morphology-module figures for any pipeline run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
    figures = [("R1","per_group","Neuron type","Neurons"),("R2","per_mito","Compartment","Mitochondria")]
    metrics = pd.read_csv(folder/"classification.csv")
    sources = {rep: {} for rep in ("handcrafted","concat")}
    for row in metrics.itertuples(index=False):
        sources[row.representation][row.task,row.scope] = row
    for task,scope,title,unit in figures:
        fig, axes = plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
        for ax,rep in zip(axes,sources):
            data = pd.read_csv(folder/f"{task}_{rep}_{scope}_confusion.csv",index_col=0)
            rel = data.div(data.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
            color = ax.imshow(rel,vmin=0,vmax=1,cmap="Blues")
            for i,j in np.ndindex(data.shape):
                ax.text(j,i,int(data.iloc[i,j]),ha="center",va="center",fontsize=8,
                        color="white" if rel.iloc[i,j]>.55 else "#262626")
            metric = sources[rep][task,scope]
            label = {"handcrafted":"Handcrafted","concat":"Concat"}[rep]
            ax.set_title(f"{label} · Macro F1 {metric.macro_f1:.3f}")
            ax.set_xticks(range(len(data.columns)),data.columns,rotation=45,ha="right")
            ax.set_yticks(range(len(data.index)),data.index)
            ax.set(xlabel="Predicted",ylabel="True")
        fig.colorbar(color,ax=axes,label="Fraction within true class",shrink=.82)
        fig.suptitle(f"{title} classification · {unit.lower()}")
        save(fig,folder/f"{task}_confusion")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment",choices=["E01","E03"])
    p.add_argument("result",type=Path)
    args = p.parse_args()
    {"E01":recovery,"E03":confusion}[args.experiment](args.result)


if __name__=="__main__":
    main()
