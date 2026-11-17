"""Neuron-disjoint classification and regression probes."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import dump
from fish2_downstream.evaluate import evaluate


def folds(frame, seed):
    bodies = frame[["body_id", "cell_type"]].drop_duplicates().sort_values("body_id")
    if bodies.cell_type.value_counts().min() < 5:
        raise ValueError("Five-fold benchmark requires at least five neurons per included type")
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    assigned = {}
    for fold, (_, test) in enumerate(split.split(bodies, bodies.cell_type)):
        assigned.update(dict.fromkeys(bodies.body_id.iloc[test], fold))
    return frame.body_id.map(assigned).to_numpy()


def classify(frame, representations, task, classifiers, seed, out, fixed_fold=None):
    if task == "R1":
        label, scope, group = "cell_type", "pooled", ["body_id"]
    else:
        label, scope, group = "compartment", "by_type", ["body_id", "compartment"]
    frame = frame.copy()
    labels = sorted(frame[label].unique())
    if len(labels) < 2:
        raise ValueError(f"{task} requires at least two classes")
    frame["fold"] = folds(frame, seed) if fixed_fold is None else np.asarray(fixed_fold)
    if (set(frame.fold) != set(range(5))
            or frame.groupby("body_id").fold.nunique().ne(1).any()):
        raise ValueError("Invalid fixed neuron folds")
    frame["label"] = pd.Categorical(frame[label], categories=labels).codes
    config = dict(label=label, scope=scope, group=group, classes=labels,
                  groups=len(frame[group].drop_duplicates()))
    for classifier in classifiers:
        for name, x in representations.items():
            report, instance, grouped, training = evaluate(x.to_numpy(dtype=np.float64), frame, config, classifier, 1)
            destination = out / f"{name}_{classifier}"
            destination.mkdir()
            dump(destination / "metrics.json", report)
            dump(destination / "training.json", training)
            instance.to_csv(destination / "instance_oof.csv", index=False)
            grouped.to_csv(destination / "group_oof.csv", index=False)


def predict_oof(x, y, fold, alpha):
    predictions = np.full(y.shape, np.nan)
    for value in np.unique(fold):
        train = fold != value
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], y[train])
        predictions[~train] = model.predict(x[~train])
    return predictions


def regress(frame, representations, responses, task, seed, alpha, out, fixed_fold=None):
    fold = folds(frame, seed) if fixed_fold is None else np.asarray(fixed_fold)
    if set(fold) != set(range(5)) or frame.assign(fold=fold).groupby("body_id").fold.nunique().ne(1).any():
        raise ValueError("Invalid fixed neuron folds")
    records, predictions = [], []
    hand = representations["handcrafted"]
    learned_oof = {}
    if task == "R3":
        learned_oof = {name: predict_oof(x.to_numpy(dtype=float), responses.to_numpy(dtype=float), fold, alpha)
                       for name, x in representations.items() if name != "handcrafted"}
    for j, (target, response) in enumerate(responses.items()):
        y = response.to_numpy(dtype=float)
        if np.ptp(y) == 0:
            records.append(dict(target=target, status="constant_target"))
            continue
        if task == "R3":
            predictors = dict(representations)
            predictors["handcrafted"] = hand.drop(columns=target)
        else:
            predictors = {"handcrafted": hand}
            for name, x in representations.items():
                if name != "handcrafted":
                    predictors[f"handcrafted+{name}"] = pd.concat([hand, x], axis=1)
        for model, x in predictors.items():
            yhat = (learned_oof[model][:, j] if model in learned_oof
                    else predict_oof(x.to_numpy(dtype=float), y, fold, alpha))
            # Equal total weight per held-out neuron.
            weight = 1 / frame.groupby("body_id").body_id.transform("size").to_numpy()
            records.append(dict(target=target, predictors=model, status="complete",
                                r2=r2_score(y, yhat, sample_weight=weight),
                                mae=mean_absolute_error(y, yhat, sample_weight=weight)))
            predictions.append(frame[["body_id", "record_id"]].assign(
                target=target, predictors=model, fold=fold, observed=y, predicted=yhat))
    scores = pd.DataFrame(records)
    if task == "A3" and predictions:
        baseline = scores.loc[scores.predictors.eq("handcrafted")].set_index("target")
        scores["delta_r2_vs_geometry"] = scores.r2 - scores.target.map(baseline.r2)
        scores["delta_mae_vs_geometry"] = scores.mae - scores.target.map(baseline.mae)
    scores.to_csv(out / "scores.csv", index=False)
    if predictions:
        oof = pd.concat(predictions, ignore_index=True)
        oof["residual"] = oof.observed - oof.predicted
        oof.to_csv(out / "oof.csv", index=False)
