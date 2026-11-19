"""Compare per-mitochondrion distribution pooling and learned score pooling."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from fish2_downstream.common import dump, sha
from .e07_embedding_refresh import aligned_concat
from .handoff import inputs
from .phase2_models import SEED
from .synapse_branch_counts import load_units


WIDTH = 21
HIDDEN = 12
EPOCHS = 240
SEEDS = (SEED, SEED + 1, SEED + 2)
MODELS = ("geometry", "mean", "mean_sd", "mean_sd_quantiles", "learned_scores")


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))
        self.geometry = nn.Linear(3, 2)

    def forward(self, embeddings, owners, geometry, counts):
        pooled = geometry.new_zeros((len(geometry), 2)).index_add(0, owners, self.scorer(embeddings))
        return self.geometry(geometry) + pooled / counts[:, None]


def learned_predictions(pc, bags, geometry, targets, train, test):
    def pack(selected):
        groups = [bags[i] for i in np.flatnonzero(selected)]
        counts = np.array([len(group) for group in groups])
        owners = np.repeat(np.arange(len(groups)), counts)
        return (torch.tensor(pc[np.concatenate(groups)], dtype=torch.float32),
                torch.tensor(owners, dtype=torch.long),
                torch.tensor(geometry[selected], dtype=torch.float32),
                torch.tensor(counts, dtype=torch.float32))

    train_data, test_data = pack(train), pack(test)
    target_scaler = StandardScaler().fit(targets[train])
    standardized = torch.tensor(target_scaler.transform(targets[train]), dtype=torch.float32)
    prediction = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SetRegressor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(*train_data), standardized)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            estimate = target_scaler.inverse_transform(model(*test_data).numpy())
            prediction.append(estimate)
    return np.mean(prediction, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "source", "concat_export", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame, reps, hashes = inputs(args.handoff)
    reps["concat"], provenance = aligned_concat(frame, args.concat_export)
    units, audit, members = load_units(args.source, frame, reps, "compartment")
    keys = ["body_id", "unit"]
    groups = members.groupby(keys).embedding_row.agg(list)
    related = pd.MultiIndex.from_frame(units[keys])
    bags = [np.asarray(rows, dtype=int) for rows in groups.loc[related]]
    if not np.array_equal([len(bag) for bag in bags], units.mito_count) or not members.embedding_row.is_unique:
        raise ValueError("Mitochondrial memberships do not match count regression cohort")

    x = reps["concat"].to_numpy()
    target_cols = ["pre_count", "post_count"]
    y = units[target_cols].to_numpy()
    geo = np.log1p(units[["length_um", "mito_count", "mito_volume_um3"]].to_numpy())
    folds = units.cell_type_fold.to_numpy()
    fits = {name: np.full(y.shape, np.nan) for name in MODELS}
    for f in range(5):
        test, train = folds == f, folds != f
        instance_train = np.concatenate([bags[i] for i in np.flatnonzero(train)])
        if set(units.body_id[train]) & set(units.body_id[test]):
            raise ValueError("A neuron appears in the training and test sets")
        scaled = StandardScaler().fit(x[instance_train]).transform(x)
        projector = PCA(n_components=WIDTH, svd_solver="randomized", random_state=SEED)
        pc = projector.fit(scaled[instance_train]).transform(scaled)
        pc = StandardScaler().fit(pc[instance_train]).transform(pc)
        geom = StandardScaler().fit(geo[train]).transform(geo)
        values = [pc[bag] for bag in bags]
        means = np.stack([part.mean(axis=0) for part in values])
        sds = np.stack([part.std(axis=0) for part in values])
        quantiles = np.stack([np.quantile(part, [0.1, 0.9], axis=0).ravel() for part in values])
        for name, features in zip(MODELS[:4], (None, means, np.c_[means, sds], np.c_[means, sds, quantiles])):
            design = geom if features is None else np.c_[geom, StandardScaler().fit(features[train]).transform(features)]
            fits[name][test] = Ridge(alpha=10).fit(design[train], y[train]).predict(design[test])
        fits["learned_scores"][test] = learned_predictions(pc, bags, geom, y, train, test)
        print(f"Fold {f}: {test.sum()} held-out compartments", flush=True)

    for name, estimate in fits.items():
        if not np.isfinite(estimate).all():
            raise ValueError(f"Incomplete or nonfinite predictions for {name}")
        fits[name] = np.maximum(estimate, 0)
    scores, gains = [], []
    oof = units[keys + ["compartment", "cell_type_fold", "length_um", "mito_count", *target_cols]].copy()
    for i, role in enumerate(("pre", "post")):
        body_losses = {}
        for name, estimate in fits.items():
            pred = estimate[:, i]
            scores.append(dict(target=role, model=name, r2=r2_score(y[:, i], pred),
                               mae=mean_absolute_error(y[:, i], pred)))
            body_losses[name] = pd.Series(abs(y[:, i] - pred)).groupby(units.body_id).mean().to_numpy()
            oof[f"{role}_{name}"] = pred
        n = len(body_losses["mean"])
        draws = np.random.default_rng(SEED).integers(n, size=(1999, n))
        for name in MODELS[2:]:
            improvement = body_losses["mean"] - body_losses[name]
            lo, hi = np.quantile(improvement[draws].mean(axis=1), [0.025, 0.975])
            gains.append(dict(target=role, model=name, reference="mean", mae_gain=improvement.mean(), lo=lo, hi=hi))
    args.output.mkdir(parents=True, exist_ok=False)
    oof.to_csv(args.output / "oof.csv", index=False)
    pd.DataFrame(scores).to_csv(args.output / "scores.csv", index=False)
    pd.DataFrame(gains).to_csv(args.output / "paired_mae_gains.csv", index=False)
    dump(args.output / "run.json", dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source=str(args.source), handoff=str(args.handoff), handoff_hashes=hashes, cohort=audit,
        inputs={name:sha(args.source/name) for name in ("sites.parquet", "segments.parquet", "mito_segments.parquet")},
        packages={name:version(name) for name in ("numpy", "scikit-learn", "torch")},
        pc_width=WIDTH, neural_width=HIDDEN, neural_epochs=EPOCHS, neural_seeds=SEEDS, **provenance))
    print(pd.DataFrame(scores).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
