"""Evaluate FISH2 PC30 K-means stability using independent neuron blocks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from fish2_downstream.common import dump, sha


SEED = 50000
KS = tuple(range(2, 13))
REPEATS = 16
SHAPE = ["convex hull compactness", "PC1 Length", "PC2 Length", "PC3 Length",
         "mito SA", "mito diameter", "length_along_skeleton_nm", "SBI"]


def pca():
    return PCA(n_components=30, svd_solver="randomized", random_state=SEED)


def kmeans(k, seed, starts):
    return KMeans(n_clusters=k, n_init=starts, random_state=seed)


def refit(job):
    path, train, hold, seed = job
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    x = vectors[train]
    fitted = pca().fit(x)
    train_pc = fitted.transform(x)
    test_pc = fitted.transform(vectors[hold])
    result = {k: kmeans(k, seed, 5).fit(train_pc).predict(test_pc) for k in KS}
    return seed, result


def matching(ref, other, k):
    matrix = np.zeros((k, k), dtype=int)
    np.add.at(matrix, (ref, other), 1)
    total = matrix.sum(0)[None, :] + matrix.sum(1)[:, None] - matrix
    jaccard = np.divide(matrix, total, out=np.zeros_like(matrix, dtype=float), where=total > 0)
    ref_ids, other_ids = linear_sum_assignment(jaccard, maximize=True)
    align = np.empty(k, dtype=int)
    align[other_ids] = ref_ids
    return align, jaccard[ref_ids, other_ids]


def save_figure(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def figures(summary, assignments, candidates, output):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.fill_between(summary.k, summary.ari_p10, summary.ari_p90, color="#b4d9d1", label="10th–90th percentile")
    ax.plot(summary.k, summary.ari_median, "o-", color="#125859", label="Median pairwise ARI")
    ax.set(xlabel="Number of clusters (K)", ylabel="ARI on held-out neurons", xticks=list(KS), ylim=(0, 1.02))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, output, "PC30_stability")
    fig, axes = plt.subplots(1, len(candidates), figsize=(4.3*len(candidates), 4.3),
                             layout="constrained", sharex=True, sharey=True)
    order = np.random.default_rng(SEED).permutation(len(assignments))
    for ax, k in zip(axes, candidates):
        for cluster in range(k):
            data = assignments.iloc[order]
            points = data.loc[data[f"k{k}"].eq(cluster)]
            ax.scatter(points.umap_1, points.umap_2, s=2.3, linewidth=0, alpha=.67,
                       color=plt.get_cmap("tab20")(cluster), label=f"C{cluster + 1}", rasterized=True)
        ax.set(title=f"K = {k}", xlabel="UMAP 1", aspect="equal")
        ax.legend(markerscale=2, loc="best", frameon=False, fontsize=8, ncol=2)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("UMAP 2")
    save_figure(fig, output, "PC30_candidate_clusters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("umap", type=Path)
    parser.add_argument("handcraft", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.export / "embedding_manifest.json").read_text())
    index = args.export / manifest["matrix_metadata"]
    descriptor = manifest["representations"]["concat"]
    source = args.export / descriptor["path"]
    digest = sha(source)
    rows = pd.read_csv(index, dtype={"body_id": str})
    coord = pd.read_csv(args.umap, dtype={"body_id": str})
    vectors = np.load(source, mmap_mode="r", allow_pickle=False)
    keys = ["record_id", "body_id", "embedding_row"]
    if (digest != descriptor["sha256"] or not rows.record_id.is_unique
            or not rows.embedding_row.eq(np.arange(len(rows))).all()
            or vectors.shape != (manifest["instances"], descriptor["dimension"])
            or len(rows) != len(vectors) or not rows[keys].equals(coord[keys])
            or not np.isfinite(vectors).all()):
        raise ValueError("Concat and UMAP row identity or checksum mismatch")
    holdout = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    dev, hold = next(holdout.split(rows, groups=rows.body_id))
    dev_bodies = np.sort(rows.iloc[dev].body_id.unique())
    rng = np.random.default_rng(SEED)
    jobs, draws = [], []
    for i in range(REPEATS):
        seed = SEED + i + 1
        selected = rng.choice(dev_bodies, size=int(.8*len(dev_bodies)), replace=False)
        train = dev[rows.iloc[dev].body_id.isin(selected)]
        jobs.append((source, train, hold, seed))
        draws.extend(dict(seed=seed, body_id=body) for body in selected)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        repeats = []
        for seed, record in pool.map(refit, jobs):
            repeats.append(record)
            print(f"Completed neuron subsample {seed - SEED}/{REPEATS}", flush=True)

    dev_pca = pca().fit(vectors[dev])
    dev_pc, hold_pc = dev_pca.transform(vectors[dev]), dev_pca.transform(vectors[hold])
    full_pca = pca().fit(vectors)
    full_pc = full_pca.transform(vectors)
    volume = (rows.voxel_count.to_numpy() * (8*8*30 / 1e9))
    volume_x = np.log10(volume).reshape(-1, 1)
    hand = pd.read_csv(args.handcraft, usecols=["record_id"] + SHAPE).set_index("record_id")
    if not hand.index.is_unique:
        raise ValueError("Duplicate handcrafted instance IDs")
    assignments = coord[keys + ["cell_type", "volume_um3", "umap_1", "umap_2"]].copy()
    assignments["held_out"] = False
    assignments.loc[hold, "held_out"] = True
    summaries, clusters, representatives, per_draw, morphology = [], [], [], [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for k in KS:
        reference = kmeans(k, SEED, 10).fit(dev_pc)
        ref = reference.predict(hold_pc)
        if len(np.unique(ref)) != k:
            raise ValueError(f"Development reference has missing clusters on holdout for K={k}")
        bootstrap_labels = [r[k] for r in repeats]
        aris = [adjusted_rand_score(a,b) for a,b in combinations(bootstrap_labels, 2)]
        jaccards = []
        repeat_columns = {}
        for i, predicted in enumerate(bootstrap_labels):
            align, jaccard = matching(ref, predicted, k)
            jaccards.append(jaccard)
            repeat_columns[f"repeat_{i+1:02}"] = align[predicted] + 1
            per_draw.append(dict(k=k, seed=SEED+i+1, ari_to_reference=adjusted_rand_score(ref,predicted)))
        hold_labels = assignments.iloc[hold][keys].copy().assign(reference_cluster=ref+1)
        hold_labels.assign(**repeat_columns).to_csv(args.output / f"holdout_k{k:02}.csv.gz", index=False)
        vol_baseline = DecisionTreeClassifier(max_leaf_nodes=k, min_samples_leaf=30, random_state=SEED)
        vol_baseline.fit(volume_x[dev], reference.labels_)
        vol_acc = balanced_accuracy_score(ref, vol_baseline.predict(volume_x[hold]))
        full = kmeans(k, SEED, 10).fit(full_pc)
        name_map, full_match = matching(ref, full.labels_[hold], k)
        final = name_map[full.labels_]
        assignments[f"k{k}"] = final+1
        jaccards = np.asarray(jaccards)
        low, mid, high = np.quantile(aris, [.1,.5,.9])
        summaries.append(dict(k=k, ari_p10=low, ari_median=mid, ari_p90=high,
                              worst_cluster_median_jaccard=np.median(jaccards,axis=0).min(),
                              volume_only_balanced_accuracy=vol_acc,
                              full_vs_reference_ari=adjusted_rand_score(ref,final[hold])))
        for c in range(k):
            ids = np.flatnonzero(final == c)
            local = rows.iloc[ids]
            body_counts = local.body_id.value_counts()
            q = np.quantile(volume[ids], [.1,.5,.9])
            clusters.append(dict(k=k, cluster=c+1, instances=len(ids), neurons=len(body_counts),
                                 largest_neuron_share=body_counts.iloc[0]/len(ids), heldout_instances=int((ref==c).sum()),
                                 volume_p10_um3=q[0], volume_median_um3=q[1], volume_p90_um3=q[2],
                                 median_jaccard=np.median(jaccards[:,c]), jaccard_p10=np.quantile(jaccards[:,c],.1),
                                 full_ref_jaccard=full_match[c]))
            centers = full.cluster_centers_[full.labels_[ids]]
            distance = np.linalg.norm(full_pc[ids]-centers, axis=1)
            nearest = np.argsort(distance, kind="stable")
            chosen = local.iloc[nearest].drop_duplicates("body_id").head(5)
            for rank, row in enumerate(chosen.itertuples(), 1):
                representatives.append(dict(k=k, cluster=c+1, rank=rank, record_id=row.record_id,
                                            body_id=row.body_id, volume_um3=volume[row.Index],
                                            em_path=row.em_path, mask_path=row.mask_path))
            features = hand.reindex(local.record_id)
            morphology.append(dict(k=k, cluster=c+1, **features.median().to_dict()))
    summary = pd.DataFrame(summaries)
    # The coarse partition is always shown. Additional panels are exploratory candidates.
    top = summary[summary.k.gt(2)].sort_values(["ari_p10", "worst_cluster_median_jaccard"], ascending=False)
    candidates = [2, *sorted(top.k.head(2).tolist())]
    summary["display_candidate"] = summary.k.isin(candidates)
    for name, df in (("stability", summary), ("clusters", pd.DataFrame(clusters)),
                     ("repeat_stability", pd.DataFrame(per_draw)), ("subsample_bodies", pd.DataFrame(draws)),
                     ("representatives", pd.DataFrame(representatives)),
                     ("morphology", pd.DataFrame(morphology)), ("assignments", assignments)):
        suffix = ".csv.gz" if name == "assignments" else ".csv"
        df.to_csv(args.output / (name+suffix), index=False)
    np.save(args.output / "pc30.npy", full_pc)
    dump(args.output / "run.json", dict(
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        instances=len(rows), bodies=rows.body_id.nunique(), heldout_instances=len(hold),
        heldout_bodies=rows.iloc[hold].body_id.nunique(), development_bodies=len(dev_bodies),
        training_bodies_per_repeat=int(.8*len(dev_bodies)), seed=SEED, repeats=REPEATS, k=list(KS),
        pca_variance=float(full_pca.explained_variance_ratio_.sum()), kmeans_starts_repeat=5, kmeans_starts_reference=10,
        candidate_k=candidates, concat_sha256=digest, map_sha256=sha(index),
        umap_sha256=sha(args.umap), handcraft_sha256=sha(args.handcraft),
    ))
    figures(summary, assignments, candidates, args.output)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
