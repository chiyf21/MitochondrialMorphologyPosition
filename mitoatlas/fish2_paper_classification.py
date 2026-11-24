"""Compare classifiers under the LC paper's scope-specific and grouped protocol."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from fish2_downstream.common import dump
from .figures.confusion import render
from .handoff import inputs
from .lc import classify_lc
from .synapse_review import summarize_e03

SEED = 20261124


def paper_folds(frame, task):
    scope, target = ("compartment", "cell_type") if task == "R1" else ("cell_type", "compartment")
    labels = set(frame[target])
    folds = np.full(len(frame), -1)
    rng = np.random.default_rng(SEED)
    for name, indices in frame.groupby(scope, sort=True).indices.items():
        rows = frame.iloc[indices]
        ids = np.sort(rows.body_id.unique())
        if len(ids) < 5:
            raise ValueError(f"{task} {name}: fewer than five bodies")
        for _ in range(100):
            perm = rng.permutation(ids)
            width = len(perm) // 5
            groups = [list(perm[i*width:(i+1)*width]) for i in range(5)]
            for i, body in enumerate(perm[5*width:][::-1]):
                groups[i].append(body)
            if all(set(rows.loc[~rows.body_id.isin(test), target]) == labels for test in groups):
                break
        else:
            raise ValueError(f"{task} {name}: a training fold lacks a class")
        for i, test in enumerate(groups):
            folds[indices[rows.body_id.isin(test).to_numpy()]] = i
    assert (folds >= 0).all()
    return folds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("handoff", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--scaling", choices=["full_cohort","training_fold"], default="full_cohort")
    a = p.parse_args()
    frame, reps, hashes = inputs(a.handoff)
    hand = reps["handcrafted"]
    if not hand["mito CA"].equals(hand["PC1 CA"]):
        raise ValueError("Expected identical FISH2 cross-sectional area columns")
    reps["handcrafted"] = hand.drop(columns="mito CA")
    a.out.mkdir(parents=True, exist_ok=False)
    dump(a.out/"run.json", dict(arguments=vars(a), inputs=hashes, seed=SEED,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        compartments=["axon", "dendrite"], primary_scoring="R1 and R2 per_group",
        folds="seeded body split by compartment for R1 or cell type for R2",
        handcrafted="21D, exact duplicate dropped", probability_pool="mean log10 probability"))
    cohort = frame[frame.compartment_eligible].reset_index(drop=True)
    x = {k:v[frame.compartment_eligible].reset_index(drop=True) for k,v in reps.items()}
    folds = {task:paper_folds(cohort,task) for task in ("R1","R2")}
    for classifier in ("rf","lr"):
        for task, fold in folds.items():
            dest = a.out/"axon_dendrite"/classifier/task
            dest.mkdir(parents=True)
            cols = ["record_id","body_id","cell_type","compartment"]
            cohort[cols].assign(fold=fold).to_csv(dest/"cohort.csv",index=False)
            classify_lc(cohort,x,task,SEED,dest,fold,classifier=classifier,
                        scaling=a.scaling,scope_local_folds=True)
    export_results(a.out)


def export_results(out):
    scores = []
    for classifier in ("rf","lr"):
        root = out/"axon_dendrite"/classifier
        summarize_e03(root,root,classifier)
        render(root,root/"figures",individual=True,tasks=(("R1","per_group"),("R2","per_group")))
        data = pd.read_csv(root/"classification.csv")
        for row in data.itertuples():
            filename = "group_oof.csv" if row.scope == "per_group" else "instance_oof.csv"
            pred = pd.read_csv(root/row.task/f"{row.representation}_{classifier}"/filename)
            data.loc[row.Index,"n_scored"] = len(pred)
            data.loc[row.Index,"zero_product_ties"] = (pred.all_classes_have_zero_product.sum()
                                                      if row.scope == "per_group" else 0)
        scores.append(data.assign(cohort="axon_dendrite",classifier=classifier))
    pd.concat(scores,ignore_index=True).to_csv(out/"comparison.csv",index=False)


if __name__ == "__main__":
    main()
