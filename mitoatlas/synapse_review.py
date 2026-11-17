"""Finalize the FISH2 E01–E07 review from fixed handoff and completed run files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from fish2_downstream.common import dump, read, sha
from .handoff import inputs
from .synapse_load import predict, predictors
from .synapse_regression import evaluate


def mask_solidity(path):
    mask = np.load(path,mmap_mode="r",allow_pickle=False) != 0
    z,y = np.where(mask.any(axis=2))
    rgb = mask[z,y]
    first = rgb.argmax(axis=1)
    last = mask.shape[2]-1-rgb[:,::-1].argmax(axis=1)
    points = np.unique(np.c_[np.r_[z,z],np.r_[y,y],np.r_[first,last]],axis=0)
    # Each occupied x run supplies its two extremes. The convex hull of
    # occupied voxel boxes is the hull of the extreme centers plus unit cube corners.
    if len(points) > 4:
        try:
            points = points[ConvexHull(points).vertices]
        except QhullError:
            pass
    corners = np.array(list(product((0,1),repeat=3)))
    hull = ConvexHull((points[:,None,:]+corners).reshape(-1,3))
    return float(mask.sum()/hull.volume)


def e01(export, out, frame):
    source = export/"mitoatlas_e50_r3_concat_no_soma_20261117_01"
    oof = pd.read_csv(source/"R3/oof.csv",dtype={"body_id":str})
    groups = pd.read_csv(source/"R3_feature_groups/target_groups.csv")
    out.mkdir(exist_ok=True)
    selected = frame[frame.record_id.isin(oof.record_id.unique())]
    with ProcessPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(mask_solidity,selected.mask_path,chunksize=24))
    solidity = pd.Series(values,index=selected.record_id,name="mask_solidity")
    solidity.to_csv(out/"mask_solidity.csv")
    cutoff = solidity.quantile(.25)
    oof["low_solidity"] = oof.record_id.map(solidity).le(cutoff)
    rng = np.random.default_rng(20261117)
    rows = []
    for target, data in oof.groupby("target"):
        for subset, group in data.groupby("low_solidity"):
            weights = 1 / group.groupby("body_id").body_id.transform("size")
            mean = np.average(group.observed,weights=weights)
            var = np.average((group.observed-mean)**2,weights=weights)
            mse = np.average(group.residual**2,weights=weights)
            rows.append(dict(target=target,low_solidity=subset,solidity_cutoff=cutoff,
                             instances=len(group),bodies=group.body_id.nunique(),
                             weighted_r2=1-mse/var if var > 0 else np.nan,
                             weighted_mae=np.average(abs(group.residual),weights=weights)))
    pd.DataFrame(rows).to_csv(out/"solidity_strata.csv",index=False)
    rows = []
    for target, data in oof.groupby("target"):
        data = data.assign(y2=data.observed**2,absolute=abs(data.residual),squared=data.residual**2)
        moments = data.groupby("body_id")[["observed","y2","absolute","squared"]].mean().to_numpy()
        n = len(moments)
        y,y2,mae,mse = moments.mean(0)
        by,by2,bmae,bmse = moments[rng.integers(n,size=(999,n))].mean(1).T
        low,high = np.nanquantile(1-bmse/np.where(by2>by**2,by2-by**2,np.nan),[.025,.975])
        mlo,mhi = np.quantile(bmae,[.025,.975])
        rows.append(dict(target=target,instances=len(data),bodies=n,weighted_r2=1-mse/(y2-y*y),
                         r2_low=low,r2_high=high,weighted_mae=mae,mae_low=mlo,mae_high=mhi))
    pd.DataFrame(rows).merge(groups,on="target",validate="one_to_one").to_csv(out/"recovery.csv",index=False)
    dump(out/"run.json",dict(source=str(source),source_hash=sha(source/"R3/oof.csv")))


def e03(export, out):
    out.mkdir(exist_ok=True)
    rows = []
    src = export/"mitoatlas_e50_lc_lr_20261117_01"
    for task in ("R1","R2"):
        for rep in ("handcrafted","concat"):
            filename = src/task/f"{rep}_lr/metrics.json"
            for scope, result in read(filename).items():
                if not isinstance(result,dict) or "macro_f1" not in result:
                    continue
                rows.append(dict(task=task,representation=rep,scope=scope,macro_f1=result["macro_f1"],
                                 balanced_accuracy=result["balanced_accuracy"],source=str(filename)))
                if "confusion" in result:
                    names = list(result["recall"])
                    pd.DataFrame(result["confusion"],index=names,columns=names).to_csv(
                        out/f"{task}_{rep}_{scope}_confusion.csv",index_label="truth")
    pd.DataFrame(rows).to_csv(out/"classification.csv",index=False)


def e04(export, out, frame, reps, args):
    windows = pd.read_csv(out.parent/"E05/windows.csv",dtype={"body_id":str})
    site = windows[windows.radius_um.eq(1)].merge(frame[["record_id","cell_type_fold","refine_split"]],
                                                on="record_id",validate="one_to_one")
    all_models = predictors(frame,reps)
    out.mkdir(exist_ok=True)
    coverage,metrics,predictions = [],[],[]
    for comp, rows in site.groupby("compartment"):
        for role in ("pre","post"):
            col = f"nearest_{role}_um"
            for cohort, wanted in [("all",rows[col].notna()),("encoder_not_refine_train",
                                                            rows[col].notna() & ~rows.refine_split.eq("train"))]:
                group = rows[wanted].copy()
                ctx = dict(compartment=comp,role=role,cohort=cohort)
                n = group.cell_type_fold.nunique()
                coverage.append(ctx|dict(instances=len(group),bodies=group.body_id.nunique(),observed_folds=n))
                if n != 5:
                    continue
                ix = group.embedding_row.to_numpy()
                x = {k:None if v is None else v[ix] for k,v in all_models.items()}
                y = group[col].to_numpy()
                for name, response in (("raw",y),("log1p",np.log1p(y))):
                    m,p = evaluate(group,response,x,args)
                    metrics.append(m.assign(**ctx,scale=name))
                    predictions.append(p.assign(**ctx,scale=name))
    pd.DataFrame(coverage).to_csv(out/"same_compartment_coverage.csv",index=False)
    pd.concat(metrics,ignore_index=True).to_csv(out/"same_compartment_scores.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"same_compartment_oof.csv",index=False)
    dump(out/"run.json",dict(source_sha256=sha(out.parent/"E05/windows.csv"),
         original_run=str(export/"e04_fish2_full_20261117_01"),
         sensitivity="occupied samples, target sites and paths within the same reviewed compartment",
         encoder_not_refine_train="only excludes E50 Refine train; earlier pretraining exposure unverified",
         ridge_alpha=args.alpha,bootstrap=args.bootstrap,seed=args.seed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    args.alpha,args.bootstrap,args.seed = 10.,999,20261117
    frame,reps,_ = inputs(args.export/"handoff_e50_12068")
    e01(args.export,args.out/"E01",frame)
    e03(args.export,args.out/"E03")
    e04(args.export,args.out/"E04",frame,reps,args)
    windows = pd.read_csv(args.out/"E05/windows.csv",dtype={"body_id":str})
    excluded = set(frame.loc[frame.refine_split.eq("train"),"body_id"])
    destination = args.out/"E05_not_refine_train"
    destination.mkdir(exist_ok=True)
    analysis = argparse.Namespace(**(vars(args)|{"out":destination}))
    predict(windows[~windows.body_id.isin(excluded)],frame,reps,analysis)
    dump(destination/"run.json",dict(source_sha256=sha(args.out/"E05/windows.csv"),
                                     criterion="Exclude bodies marked E50 Refine train",
                                     ridge_alpha=args.alpha,bootstrap=args.bootstrap,seed=args.seed))
    audit = dict(neurons=frame.body_id.nunique(),instances=len(frame),
                 eligible_compartment=int(frame.compartment_eligible.sum()),
                 refine_train_neurons=frame.loc[frame.refine_split.eq("train"),"body_id"].nunique(),
                 concat_dimension=reps["concat"].shape[1],handcrafted_dimension=reps["handcrafted"].shape[1],
                 voxel_size_xyz_nm=[8,8,30],array_axis="ZYX",
                 synapse_partners="observed pre/post coordinate pairs from 150 archived connection tables")
    dump(args.out/"data_audit.json",audit)


if __name__=="__main__":
    main()
