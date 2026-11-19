"""Benchmark graph, density and hierarchical clustering on the fixed PC30 panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
import subprocess
from time import perf_counter

from finch import FINCH
import igraph
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from fish2_downstream.common import dump, sha


SEED = 50000
GRAPH_N = 30
GAMMAS = (.2, .5, 1., 1.5)
MIN_SIZES = (100, 300)


def neighbors(x, k):
    return NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(x).kneighbors()


def graph(dist, ind):
    n, k = ind.shape
    row = np.repeat(np.arange(n), k)
    col = ind.ravel()
    sigma = np.maximum(dist[:, -1], 1e-8)
    weights = np.exp(-dist.ravel() ** 2 / (sigma[row]*sigma[col]))
    adj = csr_matrix((weights, (row, col)), shape=(n,n))
    edge = triu(adj.maximum(adj.T), k=1, format="coo")
    result = igraph.Graph(n=n, edges=list(zip(edge.row, edge.col)), directed=False)
    result.es["weight"] = edge.data.tolist()
    return result


def leiden(g, resolution, seed):
    return np.array(leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, n_iterations=2, seed=seed).membership)


def hdb(x, size):
    return HDBSCAN(min_cluster_size=size, min_samples=15, metric="euclidean",
                   cluster_selection_method="eom", n_jobs=2).fit_predict(x)


def finch(x, ind, seed):
    labels, _, _ = FINCH(x, initial_rank=ind[:, 0].astype(np.int32, copy=True),
                        distance="euclidean", random_state=seed, verbose=False)
    return labels


def fit_time(function, *args):
    start = perf_counter()
    labels = function(*args)
    return labels, perf_counter()-start


def describe(method, setting, fit_seconds, build_seconds, labels):
    valid=labels[labels >= 0]
    sizes=np.unique(valid, return_counts=True)[1]
    data=dict(method=method, setting=str(setting), clusters=len(sizes), assigned=len(valid)/len(labels),
              largest_share=(sizes.max()/len(labels) if len(sizes) else 0),
              small_cluster_share=sizes[sizes<100].sum()/len(labels),
              fit_seconds=fit_seconds, neighbor_seconds=build_seconds, total_seconds=fit_seconds+build_seconds)
    print(data,flush=True)
    return data


def repeat(job):
    path, ids, settings, seed = job
    x = np.load(path, mmap_mode="r", allow_pickle=False)[ids]
    d, i = neighbors(x, GRAPH_N)
    labels = {"Leiden": leiden(graph(d,i), float(settings["Leiden"]), seed),
              "HDBSCAN": hdb(x, int(settings["HDBSCAN"])),
              "K-means": KMeans(n_clusters=9,n_init=5,random_state=seed).fit_predict(x)}
    f = finch(x,i,seed)
    counts = [len(np.unique(f[:, j])) for j in range(f.shape[1])]
    j = int(np.argmin(abs(np.asarray(counts)-int(settings["FINCH_target"]))))
    labels["FINCH"] = f[:,j]
    print(f"Subsample {seed-SEED} completed",flush=True)
    return ids,labels


def paired_scores(repeats):
    out=[]
    for (i, (a, first)), (j, (b, second)) in combinations(enumerate(repeats,1),2):
        common, u,v=np.intersect1d(a,b,return_indices=True)
        for name in first:
            x,y=first[name][u],second[name][v]
            assigned=(x>=0)&(y>=0)
            use=assigned.sum()
            ari=np.nan
            if use>=2 and min(len(np.unique(x[assigned])),len(np.unique(y[assigned])))>=2:
                ari=adjusted_rand_score(x[assigned],y[assigned])
            out.append(dict(method=name, repeat_a=i,repeat_b=j,overlap=len(common),
                            jointly_assigned_share=use/len(common), ari_jointly_assigned=ari,
                            ari_including_noise=adjusted_rand_score(x,y)))
    return pd.DataFrame(out)


def draw(rows, labels, settings, output):
    fig, axes = plt.subplots(2,2,figsize=(9.6,8),layout="constrained",sharex=True,sharey=True)
    x,y=rows.umap_1.to_numpy(),rows.umap_2.to_numpy()
    order=np.random.default_rng(SEED).permutation(len(rows))
    cmap=plt.get_cmap("tab20")
    for ax,(method,values) in zip(axes.ravel(),labels.items()):
        colors=np.array([cmap(int(i)%20) if i>=0 else (.72,.74,.77,.45) for i in values])
        ax.scatter(x[order],y[order],c=colors[order],s=2.25,lw=0,alpha=.65)
        n=len(np.unique(values[values>=0]))
        ax.set(title=f"{method} · {n} clusters",aspect="equal")
        ax.spines[["top","right"]].set_visible(False)
        if method=="HDBSCAN":
            ax.text(.02,.02,"Grey: noise",transform=ax.transAxes,fontsize=9)
    for ax in axes[-1]: ax.set_xlabel("UMAP 1")
    for ax in axes[:,0]: ax.set_ylabel("UMAP 2")
    fig.savefig(output/"PC30_method_comparison.png",dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--reuse-full-fit", action="store_true")
    args=parser.parse_args()
    pc=args.cluster_run/"pc30.npy"
    assignment=args.cluster_run/"assignments.csv.gz"
    rows=pd.read_csv(assignment,dtype={"body_id":str})
    x=np.load(pc,mmap_mode="r",allow_pickle=False)
    if x.shape!=(len(rows),30) or not rows.embedding_row.eq(np.arange(len(rows))).all() or not np.isfinite(x).all():
        raise ValueError("PC30 row alignment failed")
    if args.reuse_full_fit:
        saved = pd.read_csv(args.output/"assignments.csv.gz", dtype={"body_id": str})
        keys = ["record_id", "body_id", "embedding_row"]
        if not rows[keys].equals(saved[keys]):
            raise ValueError("Cached output does not match PC30 identities")
        benchmark = pd.read_csv(args.output/"benchmark.csv", dtype={"setting": str})
        results = {name:saved[name].to_numpy() for name in saved if name not in keys}
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        observations, results=[],{}
        labels, secs=fit_time(KMeans(n_clusters=9,n_init=5,random_state=SEED).fit_predict,x)
        observations.append(describe("K-means",9,secs,0,labels))
        results["K-means_9"]=labels
        (dist, ind), ns=fit_time(neighbors,x,GRAPH_N)
        g, gs=fit_time(graph,dist,ind)
        print(f"Exact {GRAPH_N}-NN: {ns:.2f}s; graph assembly: {gs:.2f}s; edges: {g.ecount()}",flush=True)
        for setting in GAMMAS:
            labels,secs=fit_time(leiden,g,setting,SEED)
            observations.append(describe("Leiden",setting,secs,ns+gs,labels))
            results[f"Leiden_{setting}"]=labels
        fh, secs=fit_time(finch,x,ind,SEED)
        for j in range(fh.shape[1]):
            observations.append(describe("FINCH",j+1,secs,ns,fh[:,j]))
            results[f"FINCH_{j+1}"]=fh[:,j]
        for setting in MIN_SIZES:
            labels,secs=fit_time(hdb,x,setting)
            observations.append(describe("HDBSCAN",setting,secs,0,labels))
            results[f"HDBSCAN_{setting}"]=labels
        benchmark=pd.DataFrame(observations)
        benchmark.to_csv(args.output/"benchmark.csv",index=False)
        pd.concat([rows[["record_id","body_id","embedding_row"]],pd.DataFrame(results)],axis=1).to_csv(
            args.output/"assignments.csv.gz",index=False)
    search=benchmark.assign(k_gap=(benchmark.clusters-9).abs())
    g_row=search.query('method=="Leiden"').sort_values(["k_gap","small_cluster_share"]).iloc[0]
    f_row=search.query('method=="FINCH"').sort_values(["k_gap","small_cluster_share"]).iloc[0]
    h_rows=search.query('method=="HDBSCAN" and clusters>=2')
    if h_rows.empty:
        h_rows=search.query('method=="HDBSCAN"')
    h_row=h_rows.sort_values(["assigned","k_gap"],ascending=[False,True]).iloc[0]
    settings={"Leiden":g_row.setting,"FINCH":f_row.setting,"FINCH_target":int(f_row.clusters),
              "HDBSCAN":h_row.setting}
    mapping={m:results[f"{m}_{s}"] for m,s in [("K-means",9),("Leiden",g_row.setting),
                                              ("FINCH",f_row.setting),("HDBSCAN",h_row.setting)]}
    draw(rows,mapping,settings,args.output)
    rng=np.random.default_rng(SEED)
    bodies=rows.body_id.unique()
    jobs=[]
    for i in range(4):
        selected=rng.choice(bodies, size=int(.8*len(bodies)), replace=False)
        jobs.append((pc,np.flatnonzero(rows.body_id.isin(selected)),settings,SEED+i+1))
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        replicates=list(pool.map(repeat,jobs))
    stability=paired_scores(replicates)
    stability.to_csv(args.output/"resample_stability.csv",index=False)
    print(stability.groupby("method")[["jointly_assigned_share","ari_jointly_assigned","ari_including_noise"]]
          .median().round(4).to_string())
    dump(args.output/"run.json",dict(instances=len(rows),neurons=len(bodies),space="fixed full-data PC30 (unwhitened)",
        seed=SEED, subsamples=4, subsample_fraction=.8, exact_graph_k=GRAPH_N, graph="undirected union; local Gaussian",
        leiden_resolutions=GAMMAS, hdbscan_min_cluster_sizes=MIN_SIZES, hdbscan_min_samples=15,
        settings=settings, pc30_sha256=sha(pc), assignments_sha256=sha(assignment),
        packages={p:version(p) for p in ["igraph","leidenalg","finch-clust","scikit-learn"]},
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))


if __name__=="__main__":
    main()
