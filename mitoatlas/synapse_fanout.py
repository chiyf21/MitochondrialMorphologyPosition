"""E07: observed FISH2 site fan-out and prediction on fixed neuron folds."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from fish2_downstream.common import dump, read, sha
from .handoff import inputs
from .synapse_distance import CENTROID, SPACING, Skeleton
from .synapse_load import predictors
from .synapse_regression import evaluate

PRE = ["x_pre", "y_pre", "z_pre"]
POST = ["x_post", "y_post", "z_post"]


def nearest_centers(graph, sources, sites):
    edges = graph.pairs
    u, v = edges.T
    m = len(graph.xyz)
    root = m + np.arange(len(sources))
    a, b = edges[sources.edge].T
    length = graph.length[sources.edge]
    offset = sources.frac.to_numpy() * length
    sparse = coo_matrix((np.r_[graph.length, graph.length, offset, length-offset],
                         (np.r_[u, v, root, root], np.r_[v, u, a, b])),
                        shape=(m+len(root), m+len(root))).tocsr()
    dist, _, owner = dijkstra(sparse, indices=root, min_only=True, return_predecessors=True)
    near = []
    by_edge = dict(tuple(sources.reset_index(drop=True).groupby("edge")))
    for r in sites.itertuples():
        e, t = r.projected_edge_row, r.edge_fraction
        a,b = edges[e]
        span = graph.length[e]
        d, ix = min((dist[a]+t*span, owner[a]-m), (dist[b]+(1-t)*span, owner[b]-m))
        if e in by_edge:
            group = by_edge[e]
            same = abs(group.frac-t)*span
            best = same.idxmin()
            d, ix = min((d, ix), (same.loc[best], best))
        near.append((d / 1000, sources.iloc[ix].embedding_row) if np.isfinite(d) else (np.nan, -1))
    return pd.DataFrame(near, index=sites.index, columns=["distance_um", "embedding_row"])


def body_sites(body, raw, sites, rows, config):
    raw = raw[raw.bodyId_pre.astype(str).eq(body)]
    if raw[[*PRE, *POST, "pre_endpoint_key", "post_endpoint_key", "bodyId_post"]].isna().any().any():
        raise ValueError(f"Missing raw endpoint fields: {body}")
    key = raw[PRE].astype(str).agg("_".join, axis=1)
    if (pd.DataFrame({"xyz":key,"site":raw.pre_endpoint_key}).groupby("site").xyz.nunique().gt(1).any()
            or pd.DataFrame({"xyz":key,"site":raw.pre_endpoint_key}).groupby("xyz").site.nunique().gt(1).any()):
        raise ValueError(f"Ambiguous presynaptic site keys: {body}")
    grouped = raw.groupby(PRE)
    counts = grouped.agg(partner_bodies=("bodyId_post","nunique"),
                         pre_endpoint_key=("pre_endpoint_key","first")).reset_index()
    posts = raw.drop_duplicates([*PRE,"bodyId_post","post_endpoint_key"]).groupby(PRE).size().rename("partner_sites").reset_index()
    counts = counts.merge(posts,on=PRE,validate="one_to_one")
    sites = sites.merge(counts.rename(columns=dict(zip(PRE,["x_seg","y_seg","z_seg"]))),
                        on=["x_seg","y_seg","z_seg"], how="outer", validate="one_to_one", indicator=True)
    if not sites._merge.eq("both").all():
        raise ValueError(f"Raw and E04 presynaptic coordinates differ: {body}")
    sites = sites.drop(columns="_merge")
    snap = Path(config["snapshot"])
    marker = read(snap/"corrected"/body/"review.json")
    edges = pd.read_parquet(snap/Path(marker["files"]["compartment_edges.parquet"]).relative_to(config["original_review_root"]))
    nodes = pd.read_parquet(snap/"automatic"/body/"skeleton_nodes.parquet")
    xyz = nodes[["x_em","y_em","z_em"]].to_numpy()*SPACING
    if rows.skeleton_xyz_rescaled.iloc[0]:
        xyz *= [16,16,15]
    labels = edges.compartment.to_numpy()
    sites["compartment"] = labels[sites.projected_edge_row.to_numpy()]
    sites["body_id"] = body
    sites["distance_um"] = np.nan
    sites["embedding_row"] = -1
    rows = rows.copy()
    rows["edge"] = rows.center_skeleton_edge_id.map(pd.Series(edges.index,index=edges.edge_id))
    for comp in ("axon","dendrite"):
        selected = np.flatnonzero(labels == comp)
        portion = sites[sites.accepted & sites.compartment.eq(comp)]
        source = rows[rows.compartment.eq(comp) & rows.center_projection_nm.le(config["max_projection_nm"])
                      & rows.edge.isin(selected)].copy()
        if portion.empty or source.empty:
            continue
        local = pd.Series(np.arange(len(selected)),index=selected)
        graph = Skeleton(xyz,edges.loc[selected,["u","v"]].to_numpy(int))
        source["edge"] = source.edge.map(local)
        e = source.edge.to_numpy()
        offset = source[CENTROID].to_numpy()*SPACING - graph.start[e]
        source["frac"] = np.clip(np.einsum("ij,ij->i",offset,graph.delta[e])/graph.dd[e],0,1)
        near = nearest_centers(graph,source,portion.assign(projected_edge_row=portion.projected_edge_row.map(local)))
        sites.loc[near.index,near.columns] = near
    return sites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff","distances","connections","out"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    args.alpha, args.bootstrap, args.seed = 10.,999,20261117
    frame,reps,hashes = inputs(args.handoff)
    args.out.mkdir(parents=True,exist_ok=False)
    config = read(args.distances/"run.json")["arguments"]
    site = pd.read_csv(args.distances/"synapse_projection_qc.csv",dtype={"body_id":str})
    centers = pd.read_csv(args.distances/"mitochondria_distances.csv",dtype={"body_id":str})
    rows = frame.merge(centers[["body_id","record_id","center_skeleton_edge_id","center_projection_nm"]],
                       on=["body_id","record_id"],validate="one_to_one")
    grouped = dict(tuple(site[site.role.eq("pre")].groupby("body_id")))
    all_sites = []
    for body, group in rows.groupby("body_id"):
        path = args.connections/body/"neuron_synapse_connections.parquet"
        raw = pd.read_parquet(path)
        hashes[str(path)] = sha(path)
        all_sites.append(body_sites(body,raw,grouped[body],group,config))
    data = pd.concat(all_sites,ignore_index=True)
    data["embedding_row"] = data.embedding_row.astype(int)
    data["record_id"] = data.embedding_row.map(frame.set_index("embedding_row").record_id)
    data["eligible"] = data.distance_um.le(5)
    data.to_csv(args.out/"site_labels.csv",index=False)
    data.groupby(["compartment","eligible","partner_sites","partner_bodies"]).size().rename("sites").reset_index().to_csv(
        args.out/"coverage.csv",index=False)
    models = predictors(frame,reps)
    fold = frame.drop_duplicates("body_id").set_index("body_id").cell_type_fold
    scores,oof = [],[]
    for comp, group in data[data.eligible].groupby("compartment"):
        group = group.copy()
        group["cell_type_fold"] = group.body_id.map(fold)
        idx = group.embedding_row.to_numpy()
        x = {k:None if arr is None else np.c_[arr[idx],np.log1p(group.distance_um)] for k,arr in models.items()}
        for target in ("partner_sites","partner_bodies"):
            y = group[target].to_numpy()
            if len(np.unique(y)) < 2 or group.cell_type_fold.nunique() < 5:
                continue
            metric,pred = evaluate(group,np.log1p(y),x,args)
            scores.append(metric.assign(compartment=comp,target=target))
            oof.append(pred.assign(compartment=comp,target=target,pre_endpoint_key=group.pre_endpoint_key))
    if scores:
        pd.concat(scores,ignore_index=True).to_csv(args.out/"scores.csv",index=False)
        pd.concat(oof,ignore_index=True).to_csv(args.out/"oof.csv",index=False)
    dump(args.out/"run.json",dict(arguments=vars(args),input_sha256=hashes|{
        str(args.distances/f):sha(args.distances/f) for f in
        ("run.json","mitochondria_distances.csv","synapse_projection_qc.csv")},
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        status="COMPLETE_OBSERVED" if scores else "SKIPPED_NO_VARIABLE_LABEL",
        label="observed distinct post endpoints or partner bodies per coordinate-defined pre site",
        linkage="nearest eligible same-compartment centroid along original skeleton within 5 um"))


if __name__=="__main__":
    main()
