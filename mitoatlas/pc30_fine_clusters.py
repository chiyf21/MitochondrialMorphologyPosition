"""Compare Leiden partitions and FINCH hierarchies on fixed PC30."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import subprocess

from finch import FINCH
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

from fish2_downstream.common import dump, sha
from mitoatlas.pc30_unsupervised_comparison import SEED, graph, leiden, neighbors, paired_scores


GRAPH_SIZES = (15, 30, 60)
RESOLUTIONS = (1.5, 2.0, 3.0, 4.0)
METRICS = ("euclidean", "cosine")
FINCH_TARGETS = (25, 40, 60)
COARSE_GRAPH_SIZES = (30, 60, 90)
COARSE_RESOLUTIONS = (0.25, 0.5, 0.75, 1.0)
COARSE_FINCH_TARGETS = (5, 8, 12)
REPEATS = 4
FRACTION = 0.8


def fit(job):
    path, ids, seed, full, coarse = job
    graph_sizes = COARSE_GRAPH_SIZES if coarse else GRAPH_SIZES
    resolutions = COARSE_RESOLUTIONS if coarse else RESOLUTIONS
    targets = COARSE_FINCH_TARGETS if coarse else FINCH_TARGETS
    x = np.load(path, mmap_mode="r", allow_pickle=False)[ids]
    distances, indices = neighbors(x, max(graph_sizes))
    labels = {}
    for k in graph_sizes:
        network = graph(distances[:, :k], indices[:, :k])
        for resolution in resolutions:
            labels[f"Leiden k={k} gamma={resolution}"] = leiden(network, resolution, seed)
    first_neighbors = {
        "euclidean": indices[:, 0],
        "cosine": NearestNeighbors(n_neighbors=1, algorithm="brute", metric="cosine")
        .fit(x).kneighbors(return_distance=False)[:, 0],
    }
    for metric, rank in first_neighbors.items():
        for j, target in enumerate(targets):
            hierarchy, counts, target_labels = FINCH(
                x, initial_rank=rank.astype(np.int32, copy=True), distance=metric,
                req_clust=target, ann_threshold=len(x) + 1, random_state=seed, verbose=False,
            )
            labels[f"FINCH {metric} target={target}"] = target_labels
            if coarse and j == 0:
                labels[f"FINCH {metric} coarsest automatic level"] = hierarchy[:, -1]
            if full and j == 0:
                for level, count in enumerate(counts):
                    labels[f"FINCH {metric} level={level + 1} ({count} clusters)"] = hierarchy[:, level]
    if coarse:
        for k in (3, 9):
            labels[f"K-means K={k}"] = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
    print(f"Fit completed for {len(ids)} instances (seed {seed})", flush=True)
    return ids, labels


def cluster_recovery(full_labels, repeats):
    """Match each full-data cluster to the subcluster with highest Jaccard on shared rows."""
    records = []
    for repeat, (ids, fitted) in enumerate(repeats, 1):
        for setting, labels in fitted.items():
            reference = full_labels[setting][ids]
            found = contingency_matrix(reference, labels)
            union = found.sum(axis=1, keepdims=True) + found.sum(axis=0) - found
            jaccard = found / union
            matches = jaccard.argmax(axis=1)
            for row, cluster in enumerate(np.unique(reference)):
                records.append(dict(setting=setting, repeat=repeat, cluster=cluster,
                                    cluster_size=found[row].sum(),
                                    matched_cluster=np.unique(labels)[matches[row]],
                                    jaccard=jaccard[row, matches[row]]))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coarse", action="store_true", help="Try lower Leiden resolutions and smaller FINCH targets.")
    args = parser.parse_args()
    pc = args.cluster_run / "pc30.npy"
    identity = args.cluster_run / "assignments.csv.gz"
    rows = pd.read_csv(identity, dtype={"body_id": str})
    x = np.load(pc, mmap_mode="r", allow_pickle=False)
    n = len(rows)
    if x.shape != (n, 30) or not rows.embedding_row.eq(np.arange(n)).all() or not np.isfinite(x).all():
        raise ValueError("PC30 row alignment failed")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    jobs = [(pc, np.arange(n), SEED, True, args.coarse)]
    for i in range(REPEATS):
        ids = np.sort(rng.choice(n, size=int(FRACTION * n), replace=False))
        jobs.append((pc, ids, SEED + i + 1, False, args.coarse))
    with ProcessPoolExecutor(max_workers=5, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(fit, jobs))
    _, full_labels = results[0]
    stability = paired_scores(results[1:])
    ari = stability.groupby("method").ari_jointly_assigned.agg(["median", "min", "max"])
    recovery = cluster_recovery(full_labels, results[1:])
    recovery.to_csv(args.output / "cluster_recovery.csv", index=False)
    cluster_medians = recovery.groupby(["setting", "cluster"]).jaccard.median()
    worst_cluster = cluster_medians.groupby("setting").min()
    summary = []
    for setting, labels in full_labels.items():
        sizes = np.unique(labels, return_counts=True)[1]
        summary.append(dict(setting=setting, clusters=len(sizes), median_size=np.median(sizes),
                            min_size=sizes.min(), clusters_below_100=(sizes < 100).sum(),
                            largest_share=sizes.max() / n,
                            median_ari=ari["median"].get(setting, np.nan),
                            min_ari=ari["min"].get(setting, np.nan),
                            max_ari=ari["max"].get(setting, np.nan),
                            worst_cluster_median_jaccard=worst_cluster.get(setting, np.nan)))
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    stability.to_csv(args.output / "resample_stability.csv", index=False)
    pd.concat([rows[["embedding_row", "record_id", "body_id"]], pd.DataFrame(full_labels)], axis=1).to_csv(
        args.output / "assignments.csv.gz", index=False)
    dump(args.output / "run.json", dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        pc30_sha256=sha(pc), assignments_sha256=sha(identity), seed=SEED, instances=n,
        resample_unit="instance", fraction=FRACTION, replicates=REPEATS, coarse=args.coarse,
        graph_sizes=COARSE_GRAPH_SIZES if args.coarse else GRAPH_SIZES,
        resolutions=COARSE_RESOLUTIONS if args.coarse else RESOLUTIONS,
        finch_metrics=METRICS, finch_targets=COARSE_FINCH_TARGETS if args.coarse else FINCH_TARGETS))


if __name__ == "__main__":
    main()
