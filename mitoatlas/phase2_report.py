"""Render Phase II feasibility, held-out results and figures from frozen outputs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fish2_downstream.common import read
from .synapse_report import table


def save(fig, dest, name):
    dest.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(dest/f"{name}.{ext}",dpi=300)
    plt.close(fig)


def uncertainty(directory, effect):
    fig,ax = plt.subplots(figsize=(7,3.8))
    for i,(ref,group) in enumerate(effect.groupby("reference",sort=False)):
        y = np.arange(len(group))+i*.2
        ax.errorbar(group.gain,y,xerr=[group.gain-group.lo,group.hi-group.gain],
                    fmt="o",capsize=3,label=ref)
    ax.set_yticks(np.arange(effect.task.nunique())+.1,effect.task.unique())
    ax.axvline(0,color="black",linewidth=.8)
    ax.set_xlabel("Held-out loss improvement with Concat")
    ax.legend(frameon=False)
    save(fig,directory/"figures","improvement")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("handoff",type=Path)
    args = p.parse_args()
    root = args.out
    audit = read(root/"inputs/audit.json")
    counts = {name:read(root/name/"sample_counts.json") for name in ("E07","E08","E09","E10")}
    metrics = {name:pd.read_csv(root/name/"metrics.csv") for name in ("E07","E09")}
    effects = {name:pd.read_csv(root/name/"paired_effects.csv") for name in metrics}
    for name,effect in effects.items():
        uncertainty(root/name,effect)

    labels = pd.read_csv(root/"E08/target_type_coverage.csv").dropna(subset="target_type").sort_values("contacts")
    fig,ax = plt.subplots(figsize=(7,5.5))
    ax.barh(labels.target_type,labels.contacts,color="#228097")
    ax.set_xlabel("Archived output contacts with a mapped target type")
    save(fig,root/"E08/figures","target_label_coverage")

    p9 = pd.read_parquet(root/"E09/predictions.parquet")
    p9["observed"] = p9.observed.astype(float)
    p9["predicted"] = p9.predicted.astype(float)
    fig,axes = plt.subplots(1,3,figsize=(10,3.4),sharex=True,sharey=True)
    for ax,(task,data) in zip(axes,p9.groupby("task",sort=False)):
        for model,color in (("coverage","#888888"),("concat","#147a90")):
            d = data[data.model.eq(model)]
            ax.scatter(d.observed,d.predicted,s=9,alpha=.55,label=model,color=color)
        ax.plot([0,10],[0,10],color="black",linewidth=.7)
        ax.set(title=task,xlabel="Observed rarefied partners")
    axes[0].set_ylabel("Predicted rarefied partners")
    axes[-1].legend(frameon=False,loc="lower right")
    save(fig,root/"E09/figures","wiring_observed_vs_predicted")
    typ = pd.read_csv(args.handoff/"metadata/instances.csv",usecols=["body_id","cell_type"]).drop_duplicates()
    p9["absolute_error"] = abs(p9.observed-p9.predicted)
    wide = p9.pivot(index=["task","body_id"],columns="model",values="absolute_error").reset_index()
    wide = wide.merge(typ,on="body_id",validate="many_to_one")
    wide["gain_vs_coverage"] = wide.coverage-wide.concat
    wide["gain_vs_handcraft"] = wide.handcraft-wide.concat
    within = wide.groupby(["task","cell_type"]).agg(bodies=("body_id","size"),
        gain_vs_coverage=("gain_vs_coverage","mean"),gain_vs_handcraft=("gain_vs_handcraft","mean")).reset_index()
    within.to_csv(root/"E09/within_type_oof_descriptive.csv",index=False)

    m10 = pd.read_csv(root/"E10/match_feasibility.csv")
    fig,ax = plt.subplots(figsize=(6,3.6))
    ax.plot(m10.threshold_um,m10.dual_side_pairs,"o-",label="Unique mitochondrial pairs")
    ax.plot(m10.threshold_um,m10.matched_pairs,"o-",label="With a matched alternate partner")
    ax.set(xticks=m10.threshold_um,xlabel="Maximum distance on either side (µm)",ylabel="Pairs")
    ax.legend(frameon=False)
    save(fig,root/"E10/figures","matched_null_coverage")

    details = {
        "E07":[f"{audit['observed_pre_sites']:,} archived coordinate-defined pre sites; "
               f"{counts['E07']['nearest_count']['rows']:,} linked within 5 µm across {counts['E07']['nearest_count']['bodies']} bodies.",
               "An independent presynaptic site ID was unavailable. The original archive’s endpoint keys were validated one-to-one against exact coordinates. Fan-out means distinct recorded postsynaptic contacts per pre endpoint.",
               "Held-out count models use Poisson regression. Binary labels are single versus multiple recorded contacts. Compare mean pooling within 5 µm and the nearest mitochondrion on exactly the same sites.",
               table(metrics["E07"][["task","model","mae","poisson_deviance","auroc","average_precision","positive_prevalence"]].fillna("—")),
               table(effects["E07"]), "![Paired held-out effects](figures/improvement.png)"],
        "E08":[f"{audit['target_type_contacts']} of {audit['observed_contacts_with_partner_ids']:,} contacts have mapped target types. "
               f"{counts['E08']['linked_typed_contacts']} also have an associated mitochondrion.",
               "The contact-level label set requires at least 20 contacts from five different source neurons in every training fold. No type passed. Segment composition uses nonoverlapping 10 µm arclength bins on the reviewed skeleton and needs at least three labeled contacts per segment plus two supported classes. It is blocked by label availability.",
               "![Mapped types](figures/target_label_coverage.png)"],
        "E09":["The wiring endpoint is expected distinct partner bodies among ten sampled archived output contacts. Rarefaction holds sample depth constant.",
               "A uses all or axonal mitochondria. D uses dendritic mitochondria and dendritic geometry against axonal output diversity. Only three neurons have at least ten typed output contacts, which blocks the target-composition task.",
               "The within-type table is descriptive held-out error for the shared model, with no separate per-type high-dimensional fit.",
               table(metrics["E09"][["task","model","rows","mae","r2"]]),table(effects["E09"]),
               "![Observed and predicted wiring](figures/wiring_observed_vs_predicted.png)"],
        "E10":[f"{counts['E10']['panel_contacts']} contacts have both endpoints in the panel and "
               f"{counts['E10']['usable_dual_side_contacts']} have linked mitochondria on both sides.",
               "Alternate postsynaptic mitochondria require matching source and target type, both ROIs and compartments, and target distance and volume bins. The available matches support a feasibility count only.",
               table(m10),"![Null match coverage](figures/matched_null_coverage.png)"],
    }
    overview = [
        "# FISH2 Phase II",
        f"The fixed panel contains {audit['neurons']} neurons and {audit['mitochondria']:,} mitochondria.",
        "No reliable held-out Concat increment over handcraft was observed for archived fan-out or rarefied wiring diversity. The central postsynaptic target-type test is blocked by label coverage. E10 is a feasibility analysis.",
        "All fitted models use the same five presynaptic-neuron folds and identical rows per task. Concat PCA is fitted in training and matched to the unique handcraft dimension. Coverage models include the available proximity, count and volume covariates for each task. Per-neurite diameter was omitted because spacing conversion was unverified.",
        "Uncertainty intervals are paired neuron bootstraps conditional on fixed out-of-fold predictions. Positive gains favor Concat. Count and diversity gains are MAE differences; binary gains are log-loss differences. The archive measures observed contacts. The protocol does not establish completeness of a body's output or earlier encoder exposure.",
        "The learned representation is Concat. Independent image-background controls and cross-dataset replication are outside this FISH2 run.",
        "[Audit](inputs/audit.json) · [Feasibility](phase2_feasibility.csv)",
    ]
    for name,section in details.items():
        (root/name/"summary.md").write_text("\n\n".join([f"# {name}",*section])+"\n",encoding="utf-8")
        overview.append(f"## [{name}]({name}/summary.md)\n\n{section[0]}")
    (root/"phase2_results_summary.md").write_text("\n\n".join(overview)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
