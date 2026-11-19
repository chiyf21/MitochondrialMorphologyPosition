"""Audit how much pooled synapse count prediction survives known compartment controls."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import dump, sha
from .e07_embedding_refresh import aligned_concat
from .handoff import inputs
from .phase2_models import SEED
from .synapse_branch_counts import load_units


GEOMETRY = ["length_um", "mito_count", "mito_volume_um3"]
COMPARISONS = (
    ("geometry", "geometry_concat"),
    ("geometry_compartment", "geometry_compartment_concat"),
    ("flex_geometry_compartment", "flex_geometry_compartment_concat"),
)


def fit(data):
    counts = data[["pre_count","post_count"]].to_numpy()
    raw_geometry = data[GEOMETRY].to_numpy()
    log_geometry = np.log1p(raw_geometry)
    axon = data.compartment.eq("axon").to_numpy(dtype=float)[:,None]
    fold = data.cell_type_fold.to_numpy()
    concat = data.filter(regex=r"^concat_\d+$").to_numpy()
    hand = data.filter(regex=r"^handcraft_\d+$").to_numpy()
    if hand.shape[1] != 21 or set(data.compartment) != {"axon","dendrite"}:
        raise ValueError("Expected accepted axon/dendrite units and 21 morphology features")
    predictions = {name: np.full(counts.shape, np.nan) for pair in COMPARISONS for name in pair}
    predictions.update({name: np.full(counts.shape, np.nan) for name in
                        ("geometry_compartment_21D", "flex_geometry_compartment_21D")})
    for i in sorted(set(fold)):
        test, train = fold==i, fold!=i
        geom = StandardScaler().fit(log_geometry[train]).transform(log_geometry)
        expanded = StandardScaler().fit(np.c_[raw_geometry,log_geometry][train]).transform(
            np.c_[raw_geometry,log_geometry])
        with_compartment = np.c_[geom, axon]
        flexible = np.c_[expanded, axon, expanded*axon]
        normalized = StandardScaler().fit(concat[train]).transform(concat)
        pca = PCA(n_components=21, svd_solver="randomized", random_state=SEED)
        pc = pca.fit(normalized[train]).transform(normalized)
        pc = StandardScaler().fit(pc[train]).transform(pc)
        hand_z = StandardScaler().fit(hand[train]).transform(hand)
        designs = {
            "geometry": geom,
            "geometry_concat": np.c_[geom,pc],
            "geometry_compartment": with_compartment,
            "geometry_compartment_concat": np.c_[with_compartment,pc],
            "geometry_compartment_21D": np.c_[with_compartment,hand_z],
            "flex_geometry_compartment": flexible,
            "flex_geometry_compartment_concat": np.c_[flexible,pc],
            "flex_geometry_compartment_21D": np.c_[flexible,hand_z],
        }
        for name, x in designs.items():
            predictions[name][test] = np.maximum(Ridge(alpha=10).fit(x[train], counts[train]).predict(x[test]),0)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Nonfinite or missing OOF predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff","source","concat_export","output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frame, reps, hashes = inputs(args.handoff)
    reps["concat"], provenance = aligned_concat(frame,args.concat_export)
    units, audit, _ = load_units(args.source, frame, reps, "compartment")
    predictions = fit(units)
    meta = ["body_id","unit","compartment","cell_type_fold",*GEOMETRY,"pre_count","post_count"]
    oof = units[meta].copy()
    scores, gains = [], []
    for i, role in enumerate(("pre","post")):
        y = oof[f"{role}_count"].to_numpy()
        for name, estimate in predictions.items():
            oof[f"{role}_{name}"] = estimate[:,i]
        for scope in ("pooled","axon","dendrite"):
            use = np.ones(len(y), dtype=bool) if scope=="pooled" else units.compartment.eq(scope).to_numpy()
            bodies = oof.body_id.to_numpy()[use]
            errors = {}
            for name, estimate in predictions.items():
                p = estimate[use,i]
                scores.append(dict(target=role,scope=scope,model=name,units=int(use.sum()),
                                   r2=r2_score(y[use],p),mae=mean_absolute_error(y[use],p)))
                errors[name] = pd.Series(abs(y[use]-p)).groupby(bodies).mean().to_numpy()
            n = len(set(bodies))
            draws = np.random.default_rng(SEED).integers(n,size=(1999,n))
            for baseline, candidate in COMPARISONS:
                delta = errors[baseline]-errors[candidate]
                lo,hi = np.quantile(delta[draws].mean(axis=1),[.025,.975])
                gains.append(dict(target=role,scope=scope,baseline=baseline,model=candidate,
                                  neurons=n,gain=delta.mean(),lo=lo,hi=hi))
    args.output.mkdir(parents=True, exist_ok=False)
    oof.to_csv(args.output/"oof.csv",index=False)
    pd.DataFrame(scores).to_csv(args.output/"scores.csv",index=False)
    pd.DataFrame(gains).to_csv(args.output/"paired_mae_gains.csv",index=False)
    dump(args.output/"run.json",dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        seed=SEED, cohort=audit, handoff=str(args.handoff), handoff_hashes=hashes, source=str(args.source),
        input_hashes={name:sha(args.source/name) for name in ("segments.parquet","mito_segments.parquet","sites.parquet")},
        model="Nonnegative-clipped Ridge alpha 10; training-fold PCA of mean Concat to 21D", **provenance))


if __name__ == "__main__":
    main()
