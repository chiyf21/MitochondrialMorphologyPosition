"""Build the browsable FISH2 E01–E07 report from verified result tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fish2_downstream.common import read


def table(frame):
    def line(items):
        return "| " + " | ".join(map(str,items)) + " |"
    return "\n".join([line(frame.columns),line(["---"]*len(frame.columns)),
                      *(line(row) for row in frame.round(4).itertuples(index=False,name=None))])


def figures(root, recovery):
    fig, ax = plt.subplots(figsize=(8,7))
    ordered = recovery.sort_values("weighted_r2")
    ax.barh(ordered.target,ordered.weighted_r2,color=np.where(ordered.hull_definition.eq("convex_hull"),"#bb5535","#23788c"))
    ax.set_xlabel("Neuron-weighted out-of-fold R²")
    ax.axvline(0,color="black",linewidth=.7)
    fig.tight_layout()
    fig.savefig(root/"E01/morphometry_recovery.png",dpi=130)
    plt.close(fig)

    data = pd.read_csv(root/"E04/same_compartment_oof.csv")
    pick = data[(data.compartment=="axon") & (data.role=="pre") & (data.scale=="log1p") & (data.cohort=="all")]
    fig, axes = plt.subplots(2,2,figsize=(9,7))
    for i, model in enumerate(("geometry","geometry_concat")):
        ax = axes[0,i]
        ax.hexbin(pick.observed,pick[model],gridsize=35,mincnt=1,bins="log",cmap="viridis")
        high = max(pick.observed.max(),pick[model].max())
        ax.plot([0,high],[0,high],color="black",linewidth=.7)
        ax.set(xlabel="Observed log1p distance",ylabel="Out-of-fold prediction",title=model)
        ax = axes[1,i]
        ax.hist(pick[model]-pick.observed,bins=45,color="#23788c")
        ax.axvline(0,color="black",linewidth=.7)
        ax.set(xlabel="Prediction minus observed",ylabel="Instances")
    fig.tight_layout()
    fig.savefig(root/"E04/axon_pre_oof.png",dpi=130)
    plt.close(fig)


def contrast(frame, context):
    base = frame[frame.model.eq("geometry")].set_index(context).r2
    keep = frame[frame.model.eq("geometry_concat")].copy()
    keep["geometry_r2"] = [base.loc[key] for key in keep[context].itertuples(index=False,name=None)]
    cols = ["instances","bodies","geometry_r2","r2","mae_gain_vs_geometry",
            "mae_gain_vs_geometry_lo","mae_gain_vs_geometry_hi"]
    return keep[context+cols]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    root = parser.parse_args().root
    audit = read(root/"data_audit.json")
    recovery = pd.read_csv(root/"E01/recovery.csv")
    stratum = pd.read_csv(root/"E01/solidity_strata.csv")
    cls = pd.read_csv(root/"E03/classification.csv")
    e04 = pd.read_csv(root/"E04/same_compartment_scores.csv")
    e05 = pd.read_csv(root/"E05/scores.csv")
    unseen = pd.read_csv(root/"E05_not_refine_train/scores.csv")
    e06 = pd.read_csv(root/"E06/concat_shift.csv").sort_values("p")
    e06["holm_p"] = np.minimum(1,np.maximum.accumulate(e06.p.to_numpy()*np.arange(len(e06),0,-1)))
    e06.to_csv(root/"E06/concat_shift_adjusted.csv",index=False)
    fan = pd.read_csv(root/"E07/site_labels.csv")
    fan_scores = pd.read_csv(root/"E07/scores.csv")
    figures(root,recovery)

    statuses = [
        ("E01","Complete","Morphometric recovery without soma and low-solidity stratum"),
        ("E02","Complete, exploratory","Matched EM texture contrast and blinded gallery"),
        ("E03","Prior results collated","Existing LR runs and confusion matrices"),
        ("E04","Complete","Same-compartment pre/post nearest distance, with Refine exclusion sensitivity"),
        ("E05","Complete","Local pre/post counts and density at 1, 2, and 5 µm"),
        ("E06","Complete, exploratory","Same-neuron and same-compartment proximity contrasts"),
        ("E07","Complete for observed fan-out","Coordinate-defined pre sites and archived partners")]
    pd.DataFrame(statuses,columns=["experiment","status","result"]).to_csv(root/"experiment_feasibility.csv",index=False)
    notes = [
        "# FISH2 E01–E07",
        f"The reviewed handoff contains {audit['instances']:,} mitochondria across {audit['neurons']} neurons.",
        f"Embedding inputs are {audit['concat_dimension']}-dimensional Concat and {audit['handcrafted_dimension']}-column handcraft.",
        "The predictions use fixed neuron-disjoint five-fold splits. MAE gains below are baseline minus Concat MAE.",
        "Intervals resample neurons conditional on the fixed out-of-fold predictions. Refine sensitivity excludes E50 Refine train bodies; earlier encoder exposure remains unverified.",
        table(pd.DataFrame(statuses,columns=["Experiment","Status","Result"])),
        "## E01 Morphometric recovery",
        "[All targets and neuron-bootstrap intervals](E01/recovery.csv) · [Solidity strata](E01/solidity_strata.csv)",
        f"The low-solidity stratum uses the bottom quartile of mask voxel volume over convex hull volume ({stratum.solidity_cutoff.iloc[0]:.3f}).",
        "The handcraft field named convex hull compactness is a hull surface area in the feature implementation.",
        "![Morphometric recovery](E01/morphometry_recovery.png)",
        "## E02 Geometry-matched EM texture",
        "Each matched triple uses one anchor and two mitochondria from other neurons of the same type and compartment. The two candidates are close in handcraft space with volume and PC1 length ratios within 1.25. Effects are the absolute texture gap for high Concat distance minus the gap for low Concat distance.",
        "Intervals summarize anchor-neuron variability. Candidate reuse and crop/background differences can affect this exploratory comparison. Texture is derived from EM intensities and there are no cristae labels.",
        table(pd.read_csv(root/"E02/effects.csv")),
        "[Triples](E02/matched_triples.csv) · [Blinded PNG and OBJ files](E02/blind_gallery/) · [Gallery key](E02/gallery_key.csv)",
        "## E03 Neuron type and compartment classification",
        "Existing logistic-regression results. R1 per-group refers to merged neuron predictions. R2 per-mito is the instance-level axon/dendrite result. Confusions and other scopes are saved under E03.",
        table(cls[((cls.task=="R1") & (cls.scope=="per_group")) | ((cls.task=="R2") & (cls.scope=="per_mito"))].drop(columns="source")),
        "## E04 Nearest synapse along the skeleton",
        "This run restricts mask seeds, paths and projected sites to the reviewed compartment. The earlier E04 used the full original skeleton, allowing paths across compartments.",
        table(contrast(e04[e04.scale.eq("log1p")],["cohort","compartment","role"])),
        "[All raw/log scores](E04/same_compartment_scores.csv) · [Coverage](E04/same_compartment_coverage.csv)",
        "![Axon pre predicted versus observed and residuals](E04/axon_pre_oof.png)",
        "## E05 Local synaptic load",
        "Windows are centered on skeleton samples covered by each mitochondrial mask. Counts use distinct recorded sites; density divides count by the measured accessible cable length. Predictors control for cell type, volume, PC1 length and accessible cable length. Axon pre results on the original scale follow.",
        table(contrast(e05[(e05.compartment=="axon") & (e05.role=="pre") & (e05.scale=="raw")],["radius_um","target"])),
        "Refit after excluding E50 Refine training neurons.",
        table(contrast(unseen[(unseen.compartment=="axon") & (unseen.role=="pre") & (unseen.scale=="raw")],["radius_um","target"])),
        "[All targets](E05/scores.csv) · [Coverage and zero rates](E05/coverage.csv) · [Refine exclusion](E05_not_refine_train/scores.csv)",
        "## E06 Pre/post proximity comparisons",
        "Near is ≤1 µm and far is ≥2 µm along the reviewed compartment skeleton. Both near labels can hold for an instance. The pre-versus-post comparison uses pre-near/post-far and post-near/pre-far pairs. Matching is within neuron and compartment and within 1.25-fold volume and PC1 length. The Concat statistic is a standardized paired shift with neuron sign-flip p-values and Holm correction across six contrasts.",
        table(e06),
        "[Handcraft and EM effects](E06/effects.csv) · [Matched instances](E06/matched_pairs.csv) · [Blinded gallery](E06/blind_gallery/)",
        "## E07 Observed presynaptic fan-out",
        f"{len(fan):,} distinct pre coordinates were reconstructed from the connection archive. {fan.eligible.sum():,} had a same-compartment projected mitochondrial centroid within 5 µm. {(fan.partner_sites>1).sum():,} observed sites had multiple distinct post endpoints.",
        "The targets are distinct observed post endpoints and distinct partner bodies per pre coordinate. Site keys are reconstructed from coordinates, and fan-out coverage is limited to recorded connections. The nearest mitochondrial centroid is a geodesic approximation.",
        table(contrast(fan_scores,["compartment","target"])),
        "[Site labels](E07/site_labels.csv) · [Coverage](E07/coverage.csv)",
    ]
    (root/"report.md").write_text("\n\n".join(notes)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
