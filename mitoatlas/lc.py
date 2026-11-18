"""LC notebook classification and log-probability aggregation with LR or RF."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def classify_lc(frame, representations, task, seed, out, fixed_fold, classifier="lr", include_cable=False, scaling="full_cohort"):
    if classifier not in {"lr", "rf"}:
        raise ValueError(f"Unsupported LC classifier: {classifier}")
    if scaling not in {"training_fold", "full_cohort"}:
        raise ValueError(f"Unsupported LC scaling: {scaling}")
    frame = frame.copy().reset_index(drop=True)
    frame["fold"] = np.asarray(fixed_fold)
    sections = (*SECTIONS, "connecting_cable") if task == "R2" and include_cable else SECTIONS
    if (not frame.compartment.isin(sections).all() or set(frame.fold) != set(range(5))
            or frame.groupby("body_id").fold.nunique().ne(1).any()):
        raise ValueError("Expected accepted LC compartments and fixed by-neuron folds")
    scope = "compartment" if task == "R1" else "cell_type"
    target = "cell_type" if task == "R1" else "compartment"
    classes = sorted(frame[target].unique()) if task == "R1" else list(reversed(SECTIONS))
    if task == "R2" and include_cable:
        classes.insert(1, "connecting_cable")
    frame["label"] = pd.Categorical(frame[target], categories=classes).codes
    y = frame.label.to_numpy()
    for name, representation in representations.items():
        x = representation.to_numpy(dtype=np.float64)
        # The original LC notebook z-scores the full analysis cohort.
        if scaling == "full_cohort":
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
                train_x, test_x = x[train], x[test]
                if scaling == "training_fold":
                    mean = train_x.mean(axis=0)
                    std = train_x.std(axis=0)
                    scale = np.where(std > 0, std, 1)
                    train_x, test_x = (train_x-mean)/scale, (test_x-mean)/scale
                if classifier == "rf":
                    counts = np.bincount(y[train], minlength=len(classes))
                    model = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=.01,
                                                   random_state=seed + fold, n_jobs=8)
                    model.fit(train_x, y[train], sample_weight=1 / counts[y[train]])
                    iterations = None
                else:
                    model = LogisticRegression(C=1, class_weight="balanced", solver="lbfgs",
                                               max_iter=3000, random_state=seed + fold)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(train_x, y[train])
                    iterations = int(model.n_iter_[0])
                oof[np.ix_(test, model.classes_)] = model.predict_proba(test_x)
                training.append(dict(scope=category, fold=fold, train_instances=len(train),
                                     test_instances=len(test), iterations=iterations, scaling=scaling))
        if not np.isfinite(oof).all():
            raise ValueError(f"{task} {name}: incomplete out-of-fold probabilities")
        np.testing.assert_allclose(oof.sum(axis=1), 1, rtol=1e-12)
        dest = out / f"{name}_{classifier}"
        dest.mkdir()
        grouped = group_predictions(frame, oof, classes, ["body_id"] if task == "R1"
                                    else ["body_id", "compartment"], 5 if task == "R1" else 1)
        group_metrics = summarize_groups(grouped, classes)
        if group_metrics is None:
            raise ValueError("No eligible test groups")
        scored = dict(per_group=group_metrics, per_mito=metrics(y, oof.argmax(axis=1), classes))
        probability = pd.DataFrame(oof, columns=[f"p_{label}" for label in classes])
        pd.concat([frame, probability], axis=1).assign(prediction=oof.argmax(axis=1)).to_csv(
            dest / "instance_oof.csv", index=False)
        grouped.to_csv(dest / "group_oof.csv", index=False)
        dump(dest / "metrics.json", scored)
        dump(dest / "training.json", training)
        print(f"{task} LC {classifier.upper()} {name} complete", flush=True)
