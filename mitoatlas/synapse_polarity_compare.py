"""Compare 21D-only and Concat-only on the saved full FISH2 polarity folds."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from fish2_downstream.common import dump, sha
from .synapse_polarity import fit_oof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oof", type=Path)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = json.loads(args.dataset_manifest.read_text())["handcrafted_columns"]
    keys = ["record_id", "body_id", "is_pre", "cell_type_fold", "concat_only"]
    oof = pd.read_csv(args.oof, usecols=keys + names + ["cohort", "compartment", "near_nm", "opposite_far_nm"],
                      dtype={"body_id": str})
    data = oof.query('cohort == "full" and compartment == "pooled" and near_nm == 1000 and opposite_far_nm == 2000').copy()
    if (not data.record_id.is_unique or set(data.cell_type_fold) != set(range(5))
            or not data.groupby("body_id").cell_type_fold.nunique().eq(1).all()
            or not data["mito CA"].equals(data["PC1 CA"])):
        raise ValueError("Expected aligned pooled predictions, neuron-disjoint folds and duplicated CA")
    names.remove("mito CA")
    feature = data[names].to_numpy()
    if len(names) != 21 or not np.isfinite(feature).all():
        raise ValueError("Expected 21 finite handcrafted measurements")
    data["handcrafted_only"] = fit_oof(feature, data.is_pre.to_numpy(dtype=int), data.cell_type_fold.to_numpy())
    models = ["handcrafted_only", "concat_only"]
    if not data[models].apply(lambda series: series.between(0, 1)).to_numpy().all():
        raise ValueError("Nonfinite or invalid out-of-fold probabilities")
    within = pd.DataFrame([
        dict(body_id=body, **{model: roc_auc_score(rows.is_pre, rows[model]) for model in models})
        for body, rows in data.groupby("body_id") if rows.is_pre.nunique() == 2
    ])
    rng = np.random.default_rng(20261117)
    sample = rng.integers(len(within), size=(1999, len(within)))
    scores = []
    for model in models:
        auc = within[model].to_numpy()
        lo, hi = np.quantile(auc[sample].mean(axis=1), [.025, .975])
        scores.append(dict(model=model, instances=len(data), bodies=data.body_id.nunique(),
                           pooled_auroc=roc_auc_score(data.is_pre, data[model]),
                           balanced_accuracy=balanced_accuracy_score(data.is_pre, data[model] >= .5),
                           within_bodies=len(within), within_auroc=auc.mean(), within_lo=lo, within_hi=hi))
    delta = (within.concat_only - within.handcrafted_only).to_numpy()
    lo, hi = np.quantile(delta[sample].mean(axis=1), [.025, .975])
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(scores).to_csv(args.output / "scores.csv", index=False)
    data[keys + ["handcrafted_only"]].to_csv(args.output / "oof.csv", index=False)
    dump(args.output / "run.json", dict(
        source=str(args.oof), source_sha256=sha(args.oof),
        dataset_manifest=str(args.dataset_manifest), dataset_manifest_sha256=sha(args.dataset_manifest),
        handcrafted_columns=names, near_nm=1000, opposite_far_nm=2000,
        within_auc_delta_concat_vs_handcrafted=float(delta.mean()),
        within_auc_delta_lo=float(lo), within_auc_delta_hi=float(hi),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    ))
    print(pd.DataFrame(scores).round(6).to_string(index=False))
    print(f"Paired mean AUC delta {delta.mean():.6f} [{lo:.6f}, {hi:.6f}]")


if __name__ == "__main__":
    main()
