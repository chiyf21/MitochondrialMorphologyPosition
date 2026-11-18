"""Probe whether FISH2 mitochondria distinguish exclusive pre/post proximity."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import sha
from mitoatlas.handoff import inputs
from mitoatlas.synapse_load import predictors


def evaluate(models, cohorts):
    oof, scores, within = [], [], []
    for cohort, eligible in cohorts.items():
        for compartment, group in eligible.groupby("compartment"):
            label = group.is_pre.to_numpy(dtype=int)
            fold = group.cell_type_fold.to_numpy(dtype=int)
            idx = group.embedding_row.to_numpy()
            group = group.copy().assign(cohort=cohort)
            context = dict(cohort=cohort, compartment=compartment, pre=int(label.sum()),
                           post=int((1-label).sum()), bodies=group.body_id.nunique())
            observed = []
            for name, x in models.items():
                p = np.full(len(group), np.nan)
                for f in range(5):
                    train = fold != f
                    test = ~train
                    fit = make_pipeline(StandardScaler(), LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=2000
                    ))
                    fit.fit(x[idx[train]], label[train])
                    p[test] = fit.predict_proba(x[idx[test]])[:, 1]
                if not np.isfinite(p).all():
                    raise ValueError(f"Invalid predictions for {name}")
                group[name] = p
                for body, block in group.groupby("body_id"):
                    if block.is_pre.nunique() == 2:
                        observed.append(dict(model=name, body_id=body,
                                             auc=roc_auc_score(block.is_pre, block[name])))
                scores.append(context | dict(model=name, pooled_auroc=roc_auc_score(label, p),
                                             pooled_balanced_accuracy=balanced_accuracy_score(label, p>=.5)))
            stats = pd.DataFrame(observed)
            table = stats.pivot(index="body_id", columns="model", values="auc")
            sampled = np.random.default_rng(20261117).integers(len(table), size=(1999, len(table)))
            for score in scores[-len(models):]:
                values = table[score["model"]].to_numpy()
                delta_hand = values - table.geometry_handcrafted.to_numpy()
                delta_geometry = values - table.geometry.to_numpy()
                low, high = np.quantile(values[sampled].mean(axis=1), [.025,.975])
                dlo, dhi = np.quantile(delta_hand[sampled].mean(axis=1), [.025,.975])
                glo, ghi = np.quantile(delta_geometry[sampled].mean(axis=1), [.025,.975])
                score.update(within_bodies=len(values), within_auroc=values.mean(), within_lo=low,
                             within_hi=high, delta_vs_hand=delta_hand.mean(),
                             delta_vs_hand_lo=dlo, delta_vs_hand_hi=dhi,
                             delta_vs_geometry=delta_geometry.mean(),
                             delta_vs_geometry_lo=glo, delta_vs_geometry_hi=ghi)
            within.append(stats.assign(cohort=cohort, compartment=compartment))
            oof.append(group)
    return pd.DataFrame(scores), pd.concat(oof, ignore_index=True), pd.concat(within, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("windows_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame, reps, hashes = inputs(args.handoff)
    cols = ["body_id", "record_id", "embedding_row", "cell_type", "compartment"]
    windows = pd.read_csv(args.windows_csv, dtype={"body_id": str})
    rows = windows.loc[windows.radius_um.eq(1)].merge(
        frame[cols + ["cell_type_fold", "refine_split"]], on=cols, validate="one_to_one", how="left"
    )
    if rows.cell_type_fold.isna().any():
        raise ValueError("Window and handoff identity mismatch")
    pre = rows.nearest_pre_um.le(1) & rows.nearest_post_um.ge(2)
    post = rows.nearest_post_um.le(1) & rows.nearest_pre_um.ge(2)
    rows = rows.assign(is_pre=pre).loc[pre | post]
    models = {name: x for name, x in predictors(frame, reps).items() if x is not None}
    models["concat_only"] = reps["concat"].to_numpy()
    summary, oof, within = evaluate(models, {
        "all": rows,
        "not_refine_train": rows[rows.refine_split.ne("train")],
    })
    args.output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output / "scores.csv", index=False)
    oof.to_csv(args.output / "oof.csv", index=False)
    within.to_csv(args.output / "within_neuron.csv", index=False)
    (args.output / "run.json").write_text(json.dumps(dict(
        handoff=str(args.handoff), windows=str(args.windows_csv), windows_sha256=sha(args.windows_csv),
        input_sha256=hashes, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        near_um=1, opposite_far_um=2, logistic_c=1, bootstrap=1999, seed=20261117), indent=2) + "\n")
    cols = ["cohort","compartment","model","pre","post","within_bodies","pooled_auroc",
            "within_auroc","within_lo","within_hi","delta_vs_hand","delta_vs_hand_lo","delta_vs_hand_hi",
            "delta_vs_geometry","delta_vs_geometry_lo","delta_vs_geometry_hi"]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
