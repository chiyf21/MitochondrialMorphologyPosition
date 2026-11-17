"""E04: predict physical synapse distances on fixed, held-out FISH2 neurons."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import dump, sha
from .handoff import inputs


def evaluate(frame, y, predictors, args):
    fold = frame.cell_type_fold.to_numpy()
    if set(fold) != set(range(5)) or frame.groupby("body_id").cell_type_fold.nunique().ne(1).any():
        raise ValueError("Expected five fixed neuron-disjoint folds")
    output = frame[["body_id", "record_id", "embedding_row", "cell_type_fold"]].copy()
    output["observed"] = y
    errors, summaries = {}, []
    weights = 1 / frame.groupby("body_id").body_id.transform("size").to_numpy()
    for name, x in predictors.items():
        pred = np.full(len(y), np.nan)
        for i in range(5):
            train = fold != i
            if x is None:
                pred[~train] = np.average(y[train], weights=weights[train])
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
                model.fit(x[train], y[train])
                pred[~train] = model.predict(x[~train])
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite predictions: {name}")
        output[name] = pred
        absolute = abs(y-pred)
        errors[name] = pd.Series(absolute).groupby(frame.body_id.reset_index(drop=True)).mean()
        summaries.append(dict(model=name, instances=len(frame), bodies=frame.body_id.nunique(),
                              r2=r2_score(y,pred,sample_weight=weights),
                              mae=np.average(absolute,weights=weights)))
    rng = np.random.default_rng(args.seed)
    n = frame.body_id.nunique()
    sample = rng.integers(n, size=(args.bootstrap, n))
    for row in summaries:
        error = errors[row["model"]]
        for key, ref in [("mean", "train_mean"), ("hand", "geometry_handcrafted"), ("geometry", "geometry")]:
            delta = (errors[ref] - error).to_numpy()
            lo,hi = np.quantile(delta[sample].mean(axis=1), [.025,.975])
            row[f"mae_gain_vs_{key}"] = float(delta.mean())
            row[f"mae_gain_vs_{key}_lo"] = float(lo)
            row[f"mae_gain_vs_{key}_hi"] = float(hi)
    return pd.DataFrame(summaries), output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "distances", "out"):
        p.add_argument(name,type=Path)
    p.add_argument("--alpha",type=float,default=10.)
    p.add_argument("--bootstrap",type=int,default=999)
    p.add_argument("--seed",type=int,default=20261117)
    args=p.parse_args()
    if args.bootstrap < 1 or not np.isfinite(args.alpha) or args.alpha <= 0:
        p.error("Expect positive alpha and bootstrap count")
    frame, reps, hashes = inputs(args.handoff)
    dist_path = args.distances / "mitochondria_distances.csv"
    distances = pd.read_csv(dist_path, dtype={"body_id": str})
    key = ["record_id","body_id","embedding_row","cell_type","compartment"]
    wanted = frame.loc[frame.compartment_eligible,key+["cell_type_fold","refine_split"]]
    merged = wanted.merge(distances,on=key,how="left",validate="one_to_one",indicator=True)
    if len(distances)!=len(wanted) or not merged.pop("_merge").eq("both").all():
        raise ValueError("Distances and eligible handoff instances do not match exactly")
    hand = reps["handcrafted"].to_numpy(dtype=float)
    concat = reps["concat"].to_numpy(dtype=float)
    names = list(reps["handcrafted"].columns)
    # PC1 CA and mito CA are exactly duplicated in this handoff.
    if np.array_equal(hand[:,names.index("mito CA")],hand[:,names.index("PC1 CA")]):
        i = names.index("mito CA")
        hand = np.delete(hand,i,axis=1)
        names.remove("mito CA")
    basic = hand[:,[names.index("size"),names.index("PC1 Length")]]
    if (basic <= 0).any():
        raise ValueError("Volume and PC1 length must be positive")
    basic = np.log(basic)
    types = pd.get_dummies(frame.cell_type).to_numpy(dtype=float)
    all_models = {"train_mean":None,"cell_type":types,"geometry":np.c_[types,basic],
                  "geometry_handcrafted":np.c_[types,basic,hand],
                  "geometry_concat":np.c_[types,basic,concat]}
    args.out.mkdir(parents=True, exist_ok=False)
    scores, labels, presence = [], [], []
    for role in ("pre","post"):
        for definition in ("occupied","center"):
            target = f"nearest_{role}_{definition}_um"
            for compartment in ("axon","dendrite"):
                cohort = merged.loc[merged.compartment.eq(compartment)].copy()
                available = cohort[target].notna()
                presence.append(dict(role=role,definition=definition,compartment=compartment,
                                     available=int(available.sum()),eligible=len(cohort),
                                     bodies_available=cohort.loc[available,"body_id"].nunique()))
                cohort = cohort.loc[available].reset_index(drop=True)
                y = cohort[target].to_numpy()
                if not len(y) or not np.isfinite(y).all() or (y < 0).any():
                    raise ValueError(f"Invalid distances for {role} {definition} {compartment}")
                idx = cohort.embedding_row.to_numpy()
                predictors = {k:None if x is None else x[idx] for k,x in all_models.items()}
                for transform,response in [("um",y),("log1p_um",np.log1p(y))]:
                    metrics,predictions=evaluate(cohort,response,predictors,args)
                    context=dict(role=role,definition=definition,compartment=compartment,scale=transform)
                    scores.append(metrics.assign(**context))
                    labels.append(predictions.assign(**context))
                print(role,definition,compartment,len(cohort),cohort.body_id.nunique(),flush=True)
    pd.concat(scores,ignore_index=True).to_csv(args.out/"scores.csv",index=False)
    pd.concat(labels,ignore_index=True).to_csv(args.out/"oof.csv",index=False)
    pd.DataFrame(presence).to_csv(args.out/"coverage.csv",index=False)
    dump(args.out/"run.json",dict(arguments=vars(args), input_sha256=hashes|{
        str(p):sha(p) for p in (dist_path,args.distances/"run.json")},
        handcrafted_columns=names, score_weighting="equal total weight per test neuron",
        bootstrap_unit="paired neuron; fixed out-of-fold predictions",
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        status="COMPLETE"))


if __name__=="__main__":
    main()
