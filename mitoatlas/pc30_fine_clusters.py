"""Compare finer Leiden partitions and FINCH hierarchies on fixed PC30."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import subprocess

from finch import FINCH
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from fish2_downstream.common import dump, sha
from mitoatlas.pc30_unsupervised_comparison import SEED, graph, leiden, neighbors, paired_scores


GRAPH_SIZES = (15, 30, 60)
RESOLUTIONS = (1.5, 2.0, 3.0, 4.0)
METRICS = ("euclidean", "cosine")
FINCH_TARGETS = (25, 40, 60)
REPEATS = 4
FRACTION = 0.8


def fit(job):
    path, ids, seed, full = job
    x = np.load(path, mmap_mode="r", allow_pickle=False)[ids]
    distances, indices = neighbors(x, max(GRAPH_SIZES))
    labels = {}
    for k in GRAPH_SIZES:
        network = graph(distances[:, :k], indices[:, :k])
        for resolution in RESOLUTIONS:
            labels[f"Leiden k={k} gamma={resolution}"] = leiden(network, resolution, seed)
    first_neighbors = {
        "euclidean": indices[:, 0],
        "cosine": NearestNeighbors(n_neighbors=1, algorithm="brute", metric="cosine")
        .fit(x).kneighbors(return_distance=False)[:, 0],
    }
    for metric, rank in first_neighbors.items():
        for j, target in enumerate(FINCH_TARGETS):
            hierarchy, counts, target_labels = FINCH(
                x, initial_rank=rank.astype(np.int32, copy=True), distance=metric,
                req_clust=target, ann_threshold=len(x) + 1, random_state=seed, verbose=False,
            )
            labels[f"FINCH {metric} target={target}"] = target_labels
            if full and j == 0:
                for level, count in enumerate(counts):
                    labels[f"FINCH {metric} level={level + 1} ({count} clusters)"] = hierarchy[:, level]
    print(f"Fit completed for {len(ids)} instances (seed {seed})", flush=True)
    return ids, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_run", type=Path)
    parser.add_argument("output", type=Path)
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
    jobs = [(pc, np.arange(n), SEED, True)]
    for i in range(REPEATS):
        ids = np.sort(rng.choice(n, size=int(FRACTION * n), replace=False))
        jobs.append((pc, ids, SEED + i + 1, False))
    with ProcessPoolExecutor(max_workers=5, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(fit, jobs))
    _, full_labels = results[0]
    stability = paired_scores(results[1:])
    medians = stability.groupby("method").ari_jointly_assigned.median()
    summary = []
    for setting, labels in full_labels.items():
        sizes = np.unique(labels, return_counts=True)[1]
        summary.append(dict(setting=setting, clusters=len(sizes), median_size=np.median(sizes),
                            min_size=sizes.min(), clusters_below_100=(sizes < 100).sum(),
                            largest_share=sizes.max() / n, median_ari=medians.get(setting, np.nan)))
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    stability.to_csv(args.output / "resample_stability.csv", index=False)
    pd.concat([rows[["embedding_row", "record_id", "body_id"]], pd.DataFrame(full_labels)], axis=1).to_csv(
        args.output / "assignments.csv.gz", index=False)
    dump(args.output / "run.json", dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        pc30_sha256=sha(pc), assignments_sha256=sha(identity), seed=SEED, instances=n,
        resample_unit="instance", fraction=FRACTION, replicates=REPEATS, graph_sizes=GRAPH_SIZES,
        resolutions=RESOLUTIONS, finch_metrics=METRICS, finch_targets=FINCH_TARGETS))


if __name__ == "__main__":
    main()
