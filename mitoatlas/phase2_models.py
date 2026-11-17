"""FISH2 connectivity probes with fixed neuron folds and a matched PCA comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_poisson_deviance, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from fish2_downstream.common import dump, read
from .handoff import inputs

SEED = 20261117
FOLDS = set(range(5))
MODELS = ("type","anatomy","coverage","handcraft","concat")


def blocks(mito, groups, cols):
    """Return aligned mean representations, dropping the duplicate CA column."""
    hand = mito["handcrafted"]
    if np.array_equal(hand["PC1 CA"],hand["mito CA"]):
        hand = hand.drop(columns="mito CA")
    grouped = groups.reset_index(drop=True)
    unique = grouped[cols].drop_duplicates().reset_index(drop=True)
    code = grouped.merge(unique.assign(code=np.arange(len(unique))),on=cols,
                         validate="many_to_one").code.to_numpy()
    count = np.bincount(code)
    ids = grouped.embedding_row.to_numpy()
    pooled = {}
    for name,x in (("handcraft",hand),("concat",mito["concat"])):
        matrix = x.to_numpy()
        totals = np.zeros((len(unique),matrix.shape[1]))
        np.add.at(totals,code,matrix[ids])
        for column, values in enumerate((totals/count[:,None]).T):
            pooled[f"{name}_{column}"] = values
    unique["mito_count"] = count
    volume = hand["size"].to_numpy()[ids]
    unique["mito_volume_um3"] = np.bincount(code,weights=volume)/1e9
    return pd.concat([unique,pd.DataFrame(pooled)],axis=1)


def get_array(data, cols):
    values = data[cols].copy()
    for c in values.columns:
        if not pd.api.types.is_numeric_dtype(values[c]):
            values[c] = values[c].astype("string").fillna("__missing__")
        else:
            value = values[c].astype(float)
            values[c] = value.where(np.isfinite(value),0)
            if not np.isfinite(value).all():
                values[c+"_missing"] = (~np.isfinite(value)).astype(float)
    return pd.get_dummies(values,dtype=float).to_numpy()


def folded(data, target, anatomy, coverage, kind, keys):
    """All models predict exactly the same rows; PCA is fit only in training."""
    fold = data.fold.to_numpy()
    if set(fold)!=FOLDS or data.groupby("body_id").fold.nunique().ne(1).any():
        raise ValueError("Expected five presynaptic-body-disjoint folds")
    y = data[target].to_numpy()
    if kind != "class":
        y = y.astype(float)
        if y.shape[1] == 1:
            y = y.ravel()
    body = data.body_id.to_numpy()
    weight = 1/data.groupby("body_id").body_id.transform("size").to_numpy()
    classes = np.unique(y) if kind=="class" else None
    shape = (len(data),len(classes)) if kind=="class" else y.shape
    predictions = {k:np.empty(shape) for k in MODELS}
    xx = [get_array(data,["cell_type"]),
          get_array(data,["cell_type",*anatomy]),
          get_array(data,["cell_type",*anatomy,*coverage])]
    hc = data.filter(regex=r"^handcraft_\d+$").to_numpy()
    concat = data.filter(regex=r"^concat_\d+$").to_numpy()
    for i in sorted(FOLDS):
        test = fold==i
        train = ~test
        covariates = [StandardScaler().fit(a[train]).transform(a) for a in xx]
        hand = StandardScaler().fit(hc[train]).transform(hc)
        learned = StandardScaler().fit(concat[train]).transform(concat)
        pca = PCA(n_components=min(hc.shape[1],int(train.sum())-1),svd_solver="randomized",random_state=SEED)
        pc = pca.fit(learned[train]).transform(learned)
        pc = StandardScaler().fit(pc[train]).transform(pc)
        x = covariates+[np.c_[covariates[-1],hand],np.c_[covariates[-1],pc]]
        w = weight[train]/weight[train].mean()
        for name, design in zip(MODELS,x):
            if kind=="class":
                estimator = LogisticRegression(C=.1,max_iter=500)
                estimator.fit(design[train],y[train].ravel(),sample_weight=w)
                if not np.array_equal(estimator.classes_,classes):
                    raise ValueError("A held-out class is absent in training")
                predictions[name][test] = estimator.predict_proba(design[test])
            elif kind=="count":
                estimator = PoissonRegressor(alpha=.1,max_iter=500).fit(design[train],y[train],sample_weight=w)
                predictions[name][test] = np.clip(estimator.predict(design[test]),1e-5,None)
            else:
                estimator = Ridge(alpha=10).fit(design[train],y[train],sample_weight=w)
                estimate = estimator.predict(design[test])
                if kind=="composition":
                    estimate = np.maximum(0,estimate)
                    total = estimate.sum(axis=1,keepdims=True)
                    estimate = np.divide(estimate,total,out=np.broadcast_to(y[train].mean(axis=0),estimate.shape).copy(),where=total>0)
                predictions[name][test] = estimate
    metrics, outputs, losses, confusion = [], [], {}, []
    meta = data[["body_id","fold",*keys]].reset_index(drop=True)
    for name,pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"Nonfinite held-out prediction for {name}")
        stats = dict(model=name,rows=len(y),bodies=len(set(body)))
        if kind=="class":
            truth = y.ravel()
            chosen = classes[pred.argmax(axis=1)]
            stats |= dict(macro_f1=f1_score(truth,chosen,average="macro",sample_weight=weight),
                          balanced_accuracy=balanced_accuracy_score(truth,chosen,sample_weight=weight))
            if len(classes)==2:
                positive = truth==classes[1]
                stats |= dict(auroc=roc_auc_score(positive,pred[:,1],sample_weight=weight),
                              average_precision=average_precision_score(positive,pred[:,1],sample_weight=weight),
                              positive_prevalence=np.average(positive,weights=weight))
            mat = confusion_matrix(truth,chosen,labels=classes)
            confusion.extend(dict(model=name,truth=a,predicted=b,count=mat[i,j])
                             for i,a in enumerate(classes) for j,b in enumerate(classes))
            err = -np.log(np.clip(pred[np.arange(len(y)),np.searchsorted(classes,truth)],1e-7,1))
            stats["log_loss"] = np.average(err,weights=weight)
            values = meta.assign(model=name,observed=truth,predicted=chosen)
            for i,c in enumerate(classes):
                values[f"p_{c}"] = pred[:,i]
            outputs.append(values)
        else:
            if kind=="composition":
                err = abs(y-pred).sum(axis=1)/2
                stats["total_variation"] = np.average(err,weights=weight)
                for i,label in enumerate(target):
                    outputs.append(meta.assign(model=name,target=label,observed=y[:,i],predicted=pred[:,i]))
            else:
                err = abs(y-pred)
                stats["mae"] = np.average(err,weights=weight)
                stats["r2"] = r2_score(y,pred,sample_weight=weight)
                if kind=="count":
                    stats["poisson_deviance"] = mean_poisson_deviance(y,pred,sample_weight=weight)
                outputs.append(meta.assign(model=name,observed=y,predicted=pred))
        losses[name] = pd.Series(err).groupby(body).mean().to_numpy()
        metrics.append(stats)
    delta = np.stack([losses["coverage"]-losses["concat"],losses["handcraft"]-losses["concat"]])
    n = delta.shape[1]
    draws = np.random.default_rng(SEED).integers(n,size=(1999,n))
    low,high = np.quantile(delta[:,draws].mean(axis=2),[.025,.975],axis=1)
    effects = pd.DataFrame(dict(reference=["coverage","handcraft"],gain=delta.mean(1),lo=low,hi=high))
    return pd.DataFrame(metrics),pd.concat(outputs,ignore_index=True),effects,pd.DataFrame(confusion)


def retain_types(data, col, min_contacts=20, min_source_bodies=5):
    """Fix the label set to types that satisfy the rule in every training fold."""
    counts = []
    for fold in sorted(FOLDS):
        c = data[data.fold.ne(fold)].groupby(col).body_id.agg(["count","nunique"])
        counts.append(c.assign(test_fold=fold))
    audit = pd.concat(counts).reset_index()
    usable = audit[audit["count"].ge(min_contacts)
                   & audit["nunique"].ge(min_source_bodies)].groupby(col).size()
    return usable[usable.eq(5)].index.tolist(),audit


def write_task(root, tasks, audit):
    root.mkdir(exist_ok=False)
    scores, predictions, effects, confusion = [],[],[],[]
    for name,data,target,anatomy,coverage,kind,keys in tasks:
        if len(data)<25 or data.body_id.nunique()<15 or set(data.fold)!=FOLDS:
            audit[name] = {"status":"BLOCKED","rows":len(data),"bodies":data.body_id.nunique()}
            continue
        a,b,c,d = folded(data,target,anatomy,coverage,kind,keys)
        scores.append(a.assign(task=name))
        predictions.append(b.assign(task=name))
        effects.append(c.assign(task=name))
        if not d.empty:
            confusion.append(d.assign(task=name))
        audit[name] = {"status":"COMPLETE","rows":len(data),"bodies":data.body_id.nunique()}
    if scores:
        pd.concat(scores,ignore_index=True).to_csv(root/"metrics.csv",index=False)
        output = pd.concat(predictions,ignore_index=True)
        output["observed"] = output.observed.astype(str)
        output["predicted"] = output.predicted.astype(str)
        output.to_parquet(root/"predictions.parquet",index=False)
        pd.concat(effects,ignore_index=True).to_csv(root/"paired_effects.csv",index=False)
    if confusion:
        pd.concat(confusion,ignore_index=True).to_csv(root/"confusion.csv",index=False)
    dump(root/"sample_counts.json",audit)


def load_tables(source, frame):
    sites = pd.read_parquet(source/"sites.parquet")
    link = pd.read_parquet(source/"associations.parquet")
    contacts = pd.read_parquet(source/"contacts.parquet")
    mito = frame.assign(body_id=frame.body_id.astype(int))
    identity = mito[["body_id","cell_type","cell_type_fold","refine_split"]].drop_duplicates()
    identity = identity.rename(columns={"cell_type_fold":"fold"})
    if not identity.body_id.is_unique:
        raise ValueError("Refine split varies within neuron")
    lookup = sites[["body_id","role","endpoint_key","site_index"]]
    for role in ("pre","post"):
        cols = [f"bodyId_{role}",f"{role}_endpoint_key"]
        role_sites = lookup[lookup.role.eq(role)].drop(columns="role")
        mapped = role_sites.rename(columns=dict(zip(["body_id","endpoint_key","site_index"],cols+[f"{role}_site"])))
        contacts = contacts.merge(mapped,on=cols,how="left",validate="many_to_one")
    if contacts.pre_site.isna().any():
        raise ValueError("Missing output site mapping")
    contacts["pre_site"] = contacts.pre_site.astype(int)
    min_link = link.sort_values(["distance_um","embedding_row"]).drop_duplicates("site_index")
    return sites,link,contacts,mito,identity,min_link


def site_data(sites,link,contact,identity,reps,nearest):
    sites = sites[sites.role.eq("pre")].copy()
    group = contact.groupby("pre_site")
    roi = group.roi_pre.agg(lambda s: s.dropna().iloc[0] if s.nunique()==1 else "__mixed_or_unknown__")
    fan = group.size()
    sites = sites.join(fan.rename("fanout"),on="site_index").join(roi.rename("roi"),on="site_index")
    sites["high_fanout"] = np.where(sites.fanout.gt(1),"multi","single")
    sites = sites.merge(identity,on="body_id",validate="many_to_one")
    nearest_fields = nearest[["site_index","embedding_row","distance_um"]]
    neighbors = blocks(reps,link[link.site_index.isin(sites.site_index)],["site_index"])
    common = sites.merge(nearest_fields,on="site_index",validate="one_to_one")
    common = common.merge(neighbors[["site_index","mito_count","mito_volume_um3"]],on="site_index",validate="one_to_one")
    single = blocks(reps,nearest_fields[nearest_fields.site_index.isin(sites.site_index)],["site_index"]).drop(
        columns=["mito_count","mito_volume_um3"])
    return (common.merge(single,on="site_index",validate="one_to_one"),
            common.merge(neighbors.drop(columns=["mito_count","mito_volume_um3"]),on="site_index",validate="one_to_one"))


def rarefied(counts, depth=10):
    n = int(counts.sum())
    if n<depth:
        return np.nan
    missing = n-counts
    probs = np.zeros(len(counts))
    ok = missing>=depth
    def logchoose(a):
        return gammaln(a+1)-gammaln(a-depth+1)
    probs[ok] = np.exp(logchoose(missing[ok])-logchoose(n))
    return float(np.sum(1-probs))


def e07(root, sites, models):
    tasks = []
    anatomy = ["compartment","roi","soma_distance_um"]
    coverage = ["distance_um","mito_count","mito_volume_um3"]
    for name,data in zip(("nearest","pooled5um"),models):
        tasks.extend([(f"{name}_count",data,["fanout"],anatomy,coverage,"count",["site_index"]),
                      (f"{name}_multi_contact",data,["high_fanout"],anatomy,coverage,"class",["site_index"])])
    write_task(root,tasks,{"status":"LIMITED","observed_sites":len(sites[sites.role.eq("pre")]),
                           "reason":"Original archive coordinate endpoint keys; independent site IDs unavailable"})


def e08(root, source, sites, contacts, identity, models, reps):
    linked = contacts.merge(models[0].drop(columns=["roi"]),left_on="pre_site",right_on="site_index",validate="many_to_one")
    labeled = linked[linked.target_type.notna()]
    labels, coverage = retain_types(labeled,"target_type")
    tasks = []
    anatomy = ["compartment","roi_pre","soma_distance_um"]
    cov = ["distance_um","mito_count","mito_volume_um3"]
    if len(labels)>=2:
        tasks.append(("A_target_type",labeled[labeled.target_type.isin(labels)],["target_type"],anatomy,cov,
                      "class",["contact_id","site_index"]))
    segment = pd.read_parquet(source/"segments.parquet")
    mito = pd.read_parquet(source/"mito_segments.parquet").dropna(subset=["segment"])
    mito["body_id"] = mito.body_id.astype(int)
    features = blocks(reps,mito,["body_id","segment"])
    seglink = sites.loc[sites.role.eq("pre") & sites.accepted & sites.segment.notna(),
                        ["site_index","body_id","segment","soma_distance_um"]]
    outputs = contacts.merge(seglink,left_on="pre_site",right_on="site_index",validate="many_to_one")
    x = outputs[outputs.target_type.notna()].copy()
    audit = dict(A_status="READY" if len(labels)>=2 else "BLOCKED",
                 observed_contacts=len(contacts),mapped_contacts=int(contacts.target_type.notna().sum()),
                 linked_typed_contacts=len(labeled),A_types=labels,B_labeled_contacts=len(x),
                 B_min_contacts_per_segment=3)
    by = ["body_id","segment"]
    if len(labels)>=2:
        x["target"] = x.target_type.where(x.target_type.isin(labels),"other_labeled")
        count = pd.crosstab([x.body_id,x.segment],x.target).add_prefix("target_")
        profile = count.div(count.sum(axis=1),axis=0)
        grouped = outputs.groupby(by)
        meta = grouped.agg(contact_count=("contact_id","size"),roi_count=("roi_pre","nunique"),
                           roi=("roi_pre","first"),soma_distance_um=("soma_distance_um","mean")).join(
                               count.sum(axis=1).rename("typed_contacts")).join(profile)
        sheet = (meta.reset_index().merge(segment,on=by,validate="one_to_one")
                 .merge(features,on=by,validate="one_to_one").merge(identity,on="body_id",validate="many_to_one"))
        sheet = sheet.loc[sheet.typed_contacts.ge(3)&sheet.roi_count.eq(1)]
        targets = list(profile)
        tasks.append(("B_segment_composition",sheet,targets,
                      ["compartment","length_um","roi","soma_distance_um","contact_count"],
                      ["mito_count","mito_volume_um3"],"composition",["segment"]))
    else:
        audit["B_status"] = "BLOCKED"
    write_task(root,tasks,audit)
    coverage.to_csv(root/"training_target_type_coverage.csv",index=False)
    contacts.groupby("target_type",dropna=False).agg(contacts=("contact_id","size"),
        targets=("bodyId_post","nunique"),presynaptic_neurons=("bodyId_pre","nunique")).to_csv(root/"target_type_coverage.csv")
    outputs.groupby(by).agg(contacts=("contact_id","size"),known_types=("target_type","count")).reset_index().to_csv(
        root/"segment_target_coverage.csv",index=False)


def e09(root, source, contacts, sites, mito, identity, reps):
    anatomy = pd.read_csv(source/"anatomy.csv")
    tasks = []
    targets = []
    output = contacts.merge(sites[["site_index","compartment","accepted"]],left_on="pre_site",
                            right_on="site_index",validate="many_to_one")
    for scope,c in [("all",output),("axon",output[output.accepted & output.compartment.eq("axon")])]:
        for body,group in c.groupby("bodyId_pre"):
            n = group.bodyId_post.value_counts().to_numpy()
            targets.append(dict(body_id=body,scope=scope,observed_contacts=len(group),partners=len(n),
                                rarefied_partners=rarefied(n)))
    diversity = pd.DataFrame(targets)
    groupcols = ["body_id"]
    for name,pooling,outcome in (("A_all","all","all"),("A_axon","axon","axon"),("D_dendrite_to_axon","dendrite","axon")):
        subset = mito if pooling=="all" else mito[mito.compartment_eligible & mito.compartment.eq(pooling)]
        pooled = blocks(reps,subset,groupcols)
        fit = (diversity[diversity.scope.eq(outcome)].merge(identity,on="body_id",validate="one_to_one")
               .merge(anatomy,on="body_id",validate="one_to_one").merge(pooled,on="body_id",validate="one_to_one"))
        fit = fit[fit.rarefied_partners.notna()]
        # The dendrite predictor uses no axonal connectivity-derived variables.
        tasks.append((name,fit,["rarefied_partners"],[f"{pooling}_length_um"],
                      ["mito_count","mito_volume_um3"],"ridge",[]))
    known = contacts[contacts.target_type.notna()].merge(identity,left_on="bodyId_pre",right_on="body_id",validate="many_to_one")
    types,_ = retain_types(known,"target_type")
    n = known.groupby("body_id").size()
    audit = dict(observed_output_bodies=output.bodyId_pre.nunique(),mapped_target_bodies=len(n),
                 bodies_with_10_typed_contacts=int(n.ge(10).sum()),B_types=types,
                 C_status="LIMITED",C_reason="15 or fewer bodies per fine-grained source type; use pooled OOF anatomy baseline")
    if len(types)>=2:
        known["target"] = known.target_type.where(known.target_type.isin(types),"other_labeled")
        counts = pd.crosstab(known.body_id,known.target).add_prefix("target_")
        profile = counts.div(counts.sum(axis=1),axis=0).join(n.rename("typed_contacts")).reset_index()
        fit = (profile.merge(identity,on="body_id",validate="one_to_one")
               .merge(anatomy,on="body_id",validate="one_to_one")
               .merge(blocks(reps,mito,groupcols),on="body_id",validate="one_to_one"))
        tasks.append(("B_output_composition",fit[fit.typed_contacts.ge(10)],list(counts),
                      ["all_length_um"],["mito_count","mito_volume_um3"],"composition",[]))
    else:
        audit["B_status"] = "BLOCKED"
    write_task(root,tasks,audit)
    diversity.to_csv(root/"wiring_profiles.csv",index=False)
    output.groupby("source_type").bodyId_pre.nunique().rename("body_count").to_csv(root/"source_type_counts.csv")


def e10(root, contacts, sites, nearest, reps):
    root.mkdir(exist_ok=False)
    x = contacts.loc[contacts.post_site.notna() & contacts.bodyId_pre.ne(contacts.bodyId_post)].copy()
    for role in ("pre","post"):
        local = nearest.merge(sites[["site_index","compartment"]],on="site_index",validate="one_to_one")
        names = {"site_index":f"{role}_site","distance_um":f"{role}_distance_um",
                 "embedding_row":f"{role}_embedding_row","compartment":f"{role}_compartment"}
        x = x.merge(local.rename(columns=names),on=f"{role}_site",validate="many_to_one")
    x.to_csv(root/"paired_contacts.csv",index=False)
    feasibility = []
    for threshold in (1,2,5):
        pair = x[x.pre_distance_um.le(threshold)&x.post_distance_um.le(threshold)].drop_duplicates(
            ["pre_embedding_row","post_embedding_row"])
        strata = ["source_type","target_type","roi_pre","roi_post","pre_compartment","post_compartment"]
        # Alternate postsynaptic mitochondria must be from a different body, with
        # compatible ROI, compartment, type, size and synapse-distance bins.
        pair = pair.dropna(subset=strata).copy()
        pair["distance_bin"] = np.floor(pair.post_distance_um/.5)
        volume = reps["handcrafted"]["size"]
        pair["volume_bin"] = np.floor(np.log2(volume.iloc[pair.post_embedding_row].to_numpy()))
        strata += ["distance_bin","volume_bin"]
        matched = pair.groupby(strata).filter(lambda g: g.bodyId_post.nunique()>=2)
        feasibility.append(dict(threshold_um=threshold,dual_side_pairs=len(pair),matched_pairs=len(matched),
                                source_bodies=matched.bodyId_pre.nunique(),target_bodies=matched.bodyId_post.nunique()))
    pd.DataFrame(feasibility).to_csv(root/"match_feasibility.csv",index=False)
    dump(root/"sample_counts.json",dict(panel_contacts=int(contacts.post_site.notna().sum()),
         usable_dual_side_contacts=len(x),functional_null=strata,
         status="LIMITED",reason="Matched-null feasibility first; avoid row-wise significance with shared bodies"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff","source","out"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    frame,reps,_ = inputs(args.handoff)
    sites,link,contacts,mito,identity,nearest = load_tables(args.source,frame)
    args.out.mkdir(exist_ok=True)
    models = site_data(sites,link,contacts,identity,reps,nearest)
    e07(args.out/"E07",sites,models)
    e08(args.out/"E08",args.source,sites,contacts,identity,models,reps)
    e09(args.out/"E09",args.source,contacts,sites,mito,identity,reps)
    e10(args.out/"E10",contacts,sites,nearest,reps)
    entries = []
    for task,state in (("E07","LIMITED"),("E08","LIMITED"),("E09","READY"),("E10","LIMITED")):
        counts = read(args.out/task/"sample_counts.json")
        if task=="E08" and counts.get("A_status")==counts.get("B_status")=="BLOCKED":
            state = "BLOCKED"
        entries.append(dict(experiment=task,status=state,details=str(counts)))
    pd.DataFrame(entries).to_csv(args.out/"phase2_feasibility.csv",index=False)


if __name__ == "__main__":
    main()
