"""LC notebook classification and log-probability aggregation with LR."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from fish2_downstream.common import dump
from fish2_downstream.evaluate import metrics

SECTIONS = ("axon", "dendrite")


def group_predictions(frame, probability, classes, group, minimum):
    records = []
    with np.errstate(divide="ignore"):
        logp = np.log10(probability)
    for _, indices in frame.groupby(group, sort=True).indices.items():
        rows = frame.iloc[indices]
        if len(rows) < minimum:
            continue
        if rows.label.nunique() != 1:
            raise ValueError("Group contains conflicting true labels")
        score = logp[indices].mean(axis=0)
        row = rows.iloc[0]
        records.append(dict(body_id=row.body_id, cell_type=row.cell_type,
                            compartment=row.compartment if "compartment" in group else "all",
                            label=int(row.label), true_class=classes[int(row.label)],
                            prediction=int(np.argmax(score)), n_mito=len(rows),
                            all_classes_have_zero_product=bool(np.isneginf(score).all())))
    return pd.DataFrame(records)


def summarize_groups(groups, classes):
    if groups.empty:
        return None
    return metrics(groups.label, groups.prediction, classes)


def classify_lc(frame, representations, task, seed, out, fixed_fold):
    frame = frame.copy().reset_index(drop=True)
    frame["fold"] = np.asarray(fixed_fold)
    if (not frame.compartment.isin(SECTIONS).all() or set(frame.fold) != set(range(5))
            or frame.groupby("body_id").fold.nunique().ne(1).any()):
        raise ValueError("Expected accepted axon/dendrite instances and fixed by-neuron folds")
    scope = "compartment" if task == "R1" else "cell_type"
    target = "cell_type" if task == "R1" else "compartment"
    classes = sorted(frame[target].unique()) if task == "R1" else ["dendrite", "axon"]
    frame["label"] = pd.Categorical(frame[target], categories=classes).codes
    y = frame.label.to_numpy()
    summaries = []
    for name, representation in representations.items():
        x = representation.to_numpy(dtype=np.float64)
        # The original LC notebook z-scores the full analysis cohort.
        std = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(std > 0, std, 1)
        oof = np.full((len(frame), len(classes)), np.nan)
        training = []
        for category, indices in frame.groupby(scope, sort=True).indices.items():
            for fold in range(5):
                local = frame.fold.iloc[indices].to_numpy() == fold
                test, train = indices[local], indices[~local]
                if not len(test):
                    continue
                if (set(y[train]) != set(range(len(classes)))
                        or set(frame.body_id.iloc[train]) & set(frame.body_id.iloc[test])):
                    raise ValueError(f"{task} {category} fold {fold}: invalid train/test coverage")
                model = LogisticRegression(C=1, class_weight="balanced", solver="lbfgs",
                                           max_iter=3000, random_state=seed + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x[train], y[train])
                oof[np.ix_(test, model.classes_)] = model.predict_proba(x[test])
                training.append(dict(scope=category, fold=fold, train_instances=len(train),
                                     test_instances=len(test), iterations=int(model.n_iter_[0])))
        if not np.isfinite(oof).all():
            raise ValueError(f"{task} {name}: incomplete out-of-fold probabilities")
        np.testing.assert_allclose(oof.sum(axis=1), 1, rtol=1e-12)
        dest = out / f"{name}_lr"
        dest.mkdir()
        grouped = group_predictions(frame, oof, classes, ["body_id"] if task == "R1"
                                    else ["body_id", "compartment"], 5 if task == "R1" else 1)
        group_metrics = summarize_groups(grouped, classes)
        if group_metrics is None:
            raise ValueError("No eligible test groups")
        scored = dict(per_mito=metrics(y, oof.argmax(axis=1), classes), per_group=group_metrics)
        probability = pd.DataFrame(oof, columns=[f"p_{label}" for label in classes])
        pd.concat([frame, probability], axis=1).assign(prediction=oof.argmax(axis=1)).to_csv(
            dest / "instance_oof.csv", index=False)
        grouped.to_csv(dest / "group_oof.csv", index=False)
        dump(dest / "metrics.json", scored)
        dump(dest / "training.json", training)
        if task == "R1":
            within = group_predictions(frame, oof, classes, ["body_id", "compartment"], 5)
            within.to_csv(dest / "group_oof_by_compartment.csv", index=False)
            partitions = {"all": grouped}
            partitions.update({section: within.loc[within.compartment.eq(section)] for section in SECTIONS}
                              if not within.empty else {})
            detail = {}
            for section, part in partitions.items():
                result = summarize_groups(part, classes)
                if result is not None:
                    detail[section] = result
                    summaries.append(dict(representation=name, section=section,
                                          neurons=len(part), instances=int(part.n_mito.sum()),
                                          balanced_accuracy=result["balanced_accuracy"],
                                          macro_f1=result["macro_f1"]))
            dump(dest / "metrics_by_compartment.json", detail)
        print(f"{task} LC LR {name} complete", flush=True)
    if summaries:
        pd.DataFrame(summaries).to_csv(out / "section_scores.csv", index=False)
