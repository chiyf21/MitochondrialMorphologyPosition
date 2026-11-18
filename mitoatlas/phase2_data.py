"""Frozen FISH2 Phase II inputs: output contacts and dual-side skeleton links."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from fish2_downstream.common import dump, read, sha
from .handoff import inputs
from .synapse_distance import CENTROID, SPACING, Skeleton

CAP = 5000.
SIDES = ("pre", "post")
CONTACT = ["bodyId_pre", "pre_endpoint_key", "bodyId_post", "post_endpoint_key"]
XYZ = ["x_seg", "y_seg", "z_seg"]
COMPS = ("axon", "dendrite")


def chains(graph, labels, width=10000.):
    """Return nonoverlapping arclength bins on degree-two runs of each compartment."""
    chain, offset, direction = {}, {}, {}
    summary = []
    for comp in COMPS:
        adjacent = defaultdict(list)
        for e in np.flatnonzero(labels == comp):
            u, v = graph.pairs[e]
            adjacent[u].append((e, v))
            adjacent[v].append((e, u))
        for start in sorted(adjacent):
            if len(adjacent[start]) == 2:
                continue
            for initial in adjacent[start]:
                if initial[0] in chain:
                    continue
                name = f"{comp}_{initial[0]}"
                length, node, (e, next_node) = 0., start, initial
                while e not in chain:
                    chain[e], offset[e], direction[e] = name, length, graph.pairs[e, 0] == node
                    length += graph.length[e]
                    node = next_node
                    options = [item for item in adjacent[node] if item[0] != e]
                    if len(options) != 1:
                        break
                    e, next_node = options[0]
                summary.extend(dict(segment=f"{name}_{i}", compartment=comp,
                                    length_um=min(width, length-i*width)/1000)
                               for i in range(int(np.ceil(length/width))))
        if set(np.flatnonzero(labels == comp)) - chain.keys():
            raise ValueError("A reviewed original skeleton compartment contains a cycle")
    return chain, offset, direction, summary


def segments_for(graph, chain, offset, direction, edge, frac):
    return [f"{chain[e]}_{int(max(0,offset[e]+(t if direction[e] else 1-t)*graph.length[e]-1e-8)//10000)}"
            if e in chain else None for e,t in zip(edge,frac)]


def links(graph, selected, sites, sources):
    """Geodesic centroid-to-site pairs within 5 µm on original compartment edges."""
    if sites.empty or sources.empty:
        return []
    edges = graph.pairs
    a,b = edges[selected].T
    m = len(graph.xyz)
    roots = m + np.arange(len(sources))
    se = sources.edge.to_numpy()
    u,v = edges[se].T
    length = graph.length[se]
    f = sources.frac.to_numpy()*length
    matrix = coo_matrix((np.r_[graph.length[selected],graph.length[selected],f,length-f],
                        (np.r_[a,b,roots,roots],np.r_[b,a,u,v])),
                       shape=(m+len(roots),m+len(roots))).tocsr()
    ee = sites.projected_edge_row.to_numpy()
    u,v = edges[ee].T
    t = sites.edge_fraction.to_numpy()
    weighted = t*graph.length[ee]
    endings = graph.length[ee]-weighted
    sf = sources.frac.to_numpy()
    src = sources.embedding_row.to_numpy()
    site_idx = sites.site_index.to_numpy()
    output = []
    for first in range(0,len(roots),24):
        stop = min(first+24,len(roots))
        d = dijkstra(matrix,indices=roots[first:stop],limit=CAP)
        dist = np.minimum(d[:,u]+weighted,d[:,v]+endings)
        same = se[first:stop,None] == ee[None,:]
        local = abs(sf[first:stop,None]-t[None,:])*graph.length[ee]
        dist = np.where(same,np.minimum(dist,local),dist)
        i,j = np.where(dist<=CAP)
        output.extend(zip(site_idx[j],src[first+i],dist[i,j]/1000))
    return output


def body_data(job, cfg):
    body, sites, centers = job
    snapshot = Path(cfg["snapshot"])
    marker = read(snapshot/"corrected"/str(body)/"review.json")
    ep = snapshot/Path(marker["files"]["compartment_edges.parquet"]).relative_to(cfg["original_review_root"])
    npth = snapshot/"automatic"/str(body)/"skeleton_nodes.parquet"
    edges,nodes = pd.read_parquet(ep),pd.read_parquet(npth)
    if centers.skeleton_xyz_rescaled.nunique()!=1:
        raise ValueError(f"Ambiguous coordinate scale: {body}")
    xyz = nodes[["x_em","y_em","z_em"]].to_numpy()*SPACING
    if centers.skeleton_xyz_rescaled.iloc[0]:
        xyz *= [16,16,15]
    pairs = edges[["u","v"]].to_numpy(int)
    graph = Skeleton(xyz,pairs)
    labels = edges.compartment.to_numpy()
    chain,offset,orientation,summ = chains(graph,labels)
    geometry = {"body_id":body}
    for comp in ("all",*COMPS):
        take = np.ones(len(labels),dtype=bool) if comp == "all" else labels==comp
        geometry[f"{comp}_length_um"] = graph.length[take].sum()/1000
    centers = centers.copy()
    centers["edge"] = centers.center_skeleton_edge_id.map(pd.Series(edges.index,index=edges.edge_id))
    if centers.edge.isna().any():
        raise ValueError(f"Missing center edge for {body}")
    ce = centers.edge.to_numpy()
    displacement = centers[CENTROID].to_numpy()*SPACING-graph.start[ce]
    centers["frac"] = np.clip(np.einsum("ij,ij->i",displacement,graph.delta[ce])/graph.dd[ce],0,1)
    good = centers.center_projection_nm.le(cfg["max_projection_nm"]) & centers.compartment.eq(labels[ce])
    centers = centers.loc[good]
    centers["segment"] = segments_for(graph,chain,offset,orientation,centers.edge,centers.frac)
    sites = sites.copy()
    se = sites.projected_edge_row.to_numpy()
    if not np.array_equal(edges.edge_id.to_numpy()[se],sites.skeleton_edge_id):
        raise ValueError(f"Cached projection edge mismatch: {body}")
    sites["compartment"] = labels[se]
    sites["segment"] = segments_for(graph,chain,offset,orientation,se,sites.edge_fraction)
    a,b = pairs[se].T
    soma = nodes.distance_to_soma_geodesic_nm.to_numpy()
    t = sites.edge_fraction.to_numpy()
    sites["soma_distance_um"] = np.minimum(soma[a]+t*graph.length[se],soma[b]+(1-t)*graph.length[se])/1000
    linked = []
    for comp in COMPS:
        selected = np.flatnonzero(labels == comp)
        linked.extend(links(graph,selected,sites[sites.accepted & sites.compartment.eq(comp)],
                            centers[centers.compartment.eq(comp)]))
    return sites,centers[["body_id","embedding_row","segment"]],summ,geometry,linked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff","distances","phase1_sites","connections","roster","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=12)
    args = p.parse_args()
    frame,_,hashes = inputs(args.handoff)
    args.out.mkdir(parents=True,exist_ok=False)
    roster = pd.read_csv(args.roster,dtype={"body_id":int},usecols=["body_id","cell_type"])
    if not roster.body_id.is_unique:
        raise ValueError("Target roster must have unique bodies")
    labels = roster.set_index("body_id").cell_type
    labels = labels[labels.notna() & ~labels.str.lower().isin(["unlabeled","unknown"])]
    if not frame.drop_duplicates("body_id").set_index("body_id").cell_type.eq(
            labels.reindex(frame.body_id.astype(int).unique()).set_axis(frame.body_id.unique())).all():
        raise ValueError("Panel and roster disagree on types")
    panel = set(frame.body_id.astype(int))
    all_contacts, keys, coverage = [], [], []
    for body in sorted(panel):
        path = args.connections/str(body)/"neuron_synapse_connections.parquet"
        raw = pd.read_parquet(path)
        hashes[str(path)] = sha(path)
        owned = raw[raw.bodyId_pre.eq(body)].copy()
        coverage.append(dict(body_id=body,rows=len(raw),output_rows=len(owned),
                             complete_output_rows=int(owned[CONTACT].notna().all(axis=1).sum())))
        all_contacts.append(owned.dropna(subset=CONTACT))
        for role in SIDES:
            xyz = [f"{axis}_{role}" for axis in "xyz"]
            name = f"{role}_endpoint_key"
            k = raw.loc[raw[f"bodyId_{role}"].eq(body),[name,*xyz]].dropna().drop_duplicates()
            if not k[name].is_unique or k.duplicated(xyz).any():
                raise ValueError(f"Raw endpoint keys and coordinates are not one-to-one: {body} {role}")
            keys.append(k.rename(columns={name:"endpoint_key",**dict(zip(xyz,XYZ))})
                        .assign(body_id=body,role=role))
    contacts = pd.concat(all_contacts,ignore_index=True)
    duplicate = contacts.duplicated(CONTACT)
    contacts = contacts.loc[~duplicate].reset_index(drop=True)
    contacts["target_type"] = contacts.bodyId_post.map(labels)
    contacts["source_type"] = contacts.bodyId_pre.map(labels)
    contacts["contact_id"] = np.arange(len(contacts))
    contacts.to_parquet(args.out/"contacts.parquet",index=False)
    pd.DataFrame(coverage).to_csv(args.out/"archive_coverage.csv",index=False)
    src = args.distances/"run.json"
    cfg = read(src)["arguments"]
    projected = pd.read_csv(args.distances/"synapse_projection_qc.csv")
    sitekey = ["body_id","role",*XYZ]
    projected = projected.merge(pd.concat(keys,ignore_index=True),on=sitekey,validate="one_to_one",how="outer",indicator=True)
    if not projected.pop("_merge").eq("both").all():
        raise ValueError("Raw own-endpoint coordinates and cached projections differ")
    projected["site_index"] = np.arange(len(projected))
    nearest = pd.read_csv(args.phase1_sites,usecols=["body_id","pre_endpoint_key","eligible","distance_um","embedding_row",
                                                    "partner_sites","partner_bodies"])
    rawcounts = contacts.groupby(["bodyId_pre","pre_endpoint_key"]).agg(
        partner_sites=("contact_id","size"),partner_bodies=("bodyId_post","nunique"))
    merged = nearest.set_index(["body_id","pre_endpoint_key"])[["partner_sites","partner_bodies"]]
    if not rawcounts.sort_index().equals(merged.sort_index()):
        raise ValueError("Phase I fan-out counts disagree with archived output contact keys")
    centers = pd.read_csv(args.distances/"mitochondria_distances.csv",dtype={"body_id":str})
    centers = frame.merge(centers[["record_id","center_skeleton_edge_id","center_projection_nm"]],
                          on="record_id",validate="one_to_one")
    jobs = [(int(body),projected[projected.body_id.eq(int(body))],rows)
            for body,rows in centers.groupby("body_id")]
    site_parts, mito_parts, segments, geometry, associations = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, (s,m,b,g,a) in zip(jobs,pool.map(partial(body_data,cfg=cfg),jobs)):
            site_parts.append(s)
            mito_parts.append(m)
            segments.extend(dict(body_id=job[0],**r) for r in b)
            geometry.append(g)
            associations.extend(a)
    sites = pd.concat(site_parts,ignore_index=True)
    association = pd.DataFrame(associations,columns=["site_index","embedding_row","distance_um"])
    sites.to_parquet(args.out/"sites.parquet",index=False)
    association.to_parquet(args.out/"associations.parquet",index=False)
    pd.concat(mito_parts,ignore_index=True).to_parquet(args.out/"mito_segments.parquet",index=False)
    pd.DataFrame(segments).to_parquet(args.out/"segments.parquet",index=False)
    pd.DataFrame(geometry).to_csv(args.out/"anatomy.csv",index=False)
    audit = dict(mitochondria=len(frame),neurons=len(panel),neurons_with_type=frame.loc[frame.cell_type.notna(),"body_id"].nunique(),
                 observed_contacts_with_partner_ids=len(contacts),raw_duplicate_contacts=int(duplicate.sum()),
                 observed_pre_sites=len(nearest),observed_multi_contact_sites=int(nearest.partner_sites.gt(1).sum()),
                 pre_sites_linked_phase1=int(nearest.eligible.sum()),target_type_contacts=int(contacts.target_type.notna().sum()),
                 target_type_count=contacts.target_type.nunique(),
                 source_bodies_with_multiple_mapped_types=int(contacts.groupby("bodyId_pre").target_type.nunique().ge(2).sum()),
                 contact_source_neurons=contacts.bodyId_pre.nunique(),
                 both_bodies_in_panel=int(contacts.bodyId_post.isin(panel).sum()))
    dump(args.out/"audit.json",audit)
    dump(args.out/"run.json",{"inputs":hashes|{str(path):sha(path) for path in
         (src,args.roster,args.phase1_sites,args.distances/"synapse_projection_qc.csv",
          args.distances/"mitochondria_distances.csv")},
         "sites":"archive endpoint keys, one-to-one with native endpoint coordinates; independent site IDs unavailable",
         "linkage":"same-compartment graph centroid geodesics to accepted projections, 5 µm cap",
         "segments":"10 µm nonoverlapping bins along original degree-two compartment runs"})


if __name__ == "__main__":
    main()
