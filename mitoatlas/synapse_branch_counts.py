"""Predict observed pre and post endpoint counts from pooled FISH2 mitochondria."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, r2_score
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import dump, sha
from .e07_embedding_refresh import aligned_concat
from .handoff import inputs
from .phase2_models import SEED, blocks


def load_units(source, frame, reps, scale):
    by = ["body_id", "unit"]
    keys = ["body_id", "segment"]
    segments = pd.read_parquet(source / "segments.parquet").rename(columns={"segment": "unit"})
    if not segments.length_um.gt(0).all():
        raise ValueError("Expected positive arclength for each skeleton bin")
    index = segments.rename(columns={"unit": "segment"})[keys + ["compartment"]]
    folds = frame[["body_id", "cell_type_fold"]].assign(body_id=frame.body_id.astype(int)).drop_duplicates()
    if not folds.body_id.is_unique or set(folds.cell_type_fold) != set(range(5)):
        raise ValueError("Expected the five frozen neuron-disjoint folds")
    mito = pd.read_parquet(source / "mito_segments.parquet").dropna(subset=["segment"])
    mito = mito.assign(body_id=mito.body_id.astype(int))
    if not mito.embedding_row.is_unique or not np.array_equal(
        frame.body_id.iloc[mito.embedding_row].astype(int).to_numpy(), mito.body_id
    ):
        raise ValueError("Expected one valid segment per mitochondrial embedding row")
    sites = pd.read_parquet(source / "sites.parquet").query("accepted").dropna(subset=["segment"])
    if sites.duplicated(["body_id", "role", "endpoint_key"]).any() or not set(sites.role) <= {"pre", "post"}:
        raise ValueError("Expected distinct owned biological endpoints")
    mito = mito.merge(index, on=keys, how="left", validate="many_to_one", indicator=True)
    sites = sites.drop(columns="compartment").merge(index, on=keys, how="left",
                                                    validate="many_to_one", indicator=True)
    if not mito._merge.eq("both").all() or not sites._merge.eq("both").all():
        raise ValueError("A mapped endpoint or mitochondrion lacks a skeleton segment")
    field = "compartment" if scale == "compartment" else "segment"
    mito["unit"], sites["unit"] = mito[field], sites[field]
    units = segments
    unmatched_mito = 0
    if scale == "compartment":
        units = segments.groupby(["body_id", "compartment"], as_index=False).length_um.sum()
        units["unit"] = units.compartment
        reviewed = frame.loc[frame.compartment_eligible, ["body_id", "compartment", "embedding_row"]]
        reviewed = reviewed.assign(body_id=reviewed.body_id.astype(int), unit=reviewed.compartment)
        mito = reviewed.merge(units[by], on=by, how="inner", validate="many_to_one")
        unmatched_mito = len(reviewed) - len(mito)
    counts = sites.groupby(by + ["role"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    counts = counts.rename(columns={"pre":"pre_count", "post":"post_count"}).reset_index()
    totals = units.merge(counts, on=by, how="left", validate="one_to_one")
    totals[["pre_count","post_count"]] = totals[["pre_count","post_count"]].fillna(0).astype(int)
    if not np.array_equal(totals[["pre_count", "post_count"]].sum(),
                          sites.role.value_counts().reindex(["pre","post"]).to_numpy()):
        raise ValueError("Endpoint counts were lost during aggregation")
    pooled = blocks(reps, mito, by)
    data = (totals.merge(pooled, on=by, validate="one_to_one")
            .merge(folds, on="body_id", validate="many_to_one"))
    audit = dict(units_total=len(totals), units_with_mito=len(data), bodies=data.body_id.nunique(),
                 mitochondria=int(data.mito_count.sum()), reviewed_mito_without_geometry=unmatched_mito,
                 **{f"{role}_mapped": int(totals[f"{role}_count"].sum()) for role in ("pre","post")},
                 **{f"{role}_in_eligible_units": int(data[f"{role}_count"].sum()) for role in ("pre","post")})
    return data, audit, mito[by + ["embedding_row"]]


def evaluate(data, role, regressor):
    target = f"{role}_count"
    fold, y = data.cell_type_fold.to_numpy(), data[target].to_numpy()
    geo = np.log1p(data[["length_um","mito_count","mito_volume_um3"]].to_numpy())
    hand = data.filter(regex=r"^handcraft_\d+$").to_numpy()
    concat = data.filter(regex=r"^concat_\d+$").to_numpy()
    names = ["train_mean", "geometry", "21D_only", "concat_only", "geometry_21D", "geometry_concat"]
    fitted = {name: np.empty(len(data)) for name in names}
    for f in range(5):
        test, train = fold == f, fold != f
        if not test.any() or not y[train].sum():
            raise ValueError("A fold lacks evaluation units or training endpoints")
        geom = StandardScaler().fit(geo[train]).transform(geo)
        hand_z = StandardScaler().fit(hand[train]).transform(hand)
        scaled = StandardScaler().fit(concat[train]).transform(concat)
        pca = PCA(n_components=hand.shape[1], svd_solver="randomized", random_state=SEED)
        pc = pca.fit(scaled[train]).transform(scaled)
        pc = StandardScaler().fit(pc[train]).transform(pc)
        fitted["train_mean"][test] = y[train].mean()
        designs = (geom, hand_z, pc, np.c_[geom,hand_z], np.c_[geom,pc])
        for name, x in zip(names[1:], designs):
            model = PoissonRegressor(alpha=1, max_iter=500) if regressor == "poisson" else Ridge(alpha=10)
            model.fit(x[train], y[train])
            fitted[name][test] = np.maximum(model.predict(x[test]), 0)
    by_neuron = data.body_id.to_numpy()
    scores, losses = [], {}
    for name, pred in fitted.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"Nonfinite OOF predictions for {name}")
        scores.append(dict(model=name, units=len(y), neurons=len(set(by_neuron)), endpoints=y.sum(),
                           zeros=int((y == 0).sum()), mean_count=y.mean(),
                           r2=r2_score(y, pred), mae=mean_absolute_error(y, pred),
                           poisson_deviance=mean_poisson_deviance(y, np.maximum(pred, 1e-8))))
        losses[name] = pd.Series(abs(y-pred)).groupby(by_neuron).mean().to_numpy()
    n = len(losses["geometry"])
    draws = np.random.default_rng(SEED).integers(n, size=(1999,n))
    effects = []
    for reference in ("geometry", "geometry_21D"):
        gain = losses[reference] - losses["geometry_concat"]
        lo, hi = np.quantile(gain[draws].mean(axis=1), [0.025,0.975])
        effects.append(dict(reference=reference, gain=gain.mean(), lo=lo, hi=hi))
    points = data[["body_id","unit","compartment","cell_type_fold","length_um","mito_count",target]]
    return pd.DataFrame(scores), pd.DataFrame(effects), points.assign(**fitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "source", "concat_export", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frame, reps, hashes = inputs(args.handoff)
    reps["concat"], provenance = aligned_concat(frame, args.concat_export)
    args.output.mkdir(parents=True, exist_ok=False)
    audit, all_scores, effects = {}, [], []
    for scale in ("compartment", "bin10um"):
        data, audit[scale], _ = load_units(args.source, frame, reps, scale)
        print(scale, audit[scale], flush=True)
        for role, fallback in (("pre","axon"), ("post","dendrite")):
            for scope, rows in (("pooled",data), (fallback,data[data.compartment.eq(fallback)])):
                for regressor in ("poisson", "ridge"):
                    scores, gains, predictions = evaluate(rows, role, regressor)
                    meta = dict(scale=scale, target=role, scope=scope, regressor=regressor)
                    all_scores.append(scores.assign(**meta))
                    effects.append(gains.assign(**meta))
                    predictions.to_csv(args.output / f"oof_{scale}_{scope}_{role}_{regressor}.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(args.output / "scores.csv", index=False)
    pd.concat(effects, ignore_index=True).to_csv(args.output / "paired_mae_gains.csv", index=False)
    dump(args.output / "run.json", dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        seed=SEED, handoff=str(args.handoff), source=str(args.source), handoff_hashes=hashes,
        input_hashes={name:sha(args.source/name) for name in ("segments.parquet","mito_segments.parquet","sites.parquet")},
        packages={name:version(name) for name in ("numpy","pandas","scikit-learn")},
        models="Poisson alpha=1 and nonnegative-clipped Ridge alpha=10; training-fold scaling and Concat PCA to 21D",
        aggregation="mean of QC-reviewed mitochondria (compartment) or projected mitochondria (10 um bins); observed owned endpoints; at least one mitochondrion per unit",
        counts=audit, **provenance))


if __name__ == "__main__":
    main()
