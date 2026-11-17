"""Descriptive state geometry and geometry-matched EM candidate pairs."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from .morphology import KEY, standardize


def atlas(frame, representations, seed, out, umap=False):
    for name, features in representations.items():
        x, _ = standardize(features)
        destination = out / name
        destination.mkdir()
        pca = PCA(svd_solver="full").fit(x)
        ratio = pca.explained_variance_ratio_
        positive = ratio[ratio > 0]
        effective_rank = np.exp(-(positive * np.log(positive)).sum())
        pd.DataFrame(dict(pc=np.arange(1, len(ratio) + 1), variance_ratio=ratio,
                          cumulative_variance=ratio.cumsum(), effective_rank=effective_rank
                          )).to_csv(destination / "spectrum.csv", index=False)
        pcs = pca.transform(x)[:, :min(len(ratio), 20)]
        coords = frame[KEY + ["cell_type", "compartment"]].reset_index(drop=True).copy()
        coords = pd.concat([coords, pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])], axis=1)
        if umap:
            from umap import UMAP
            coords[["UMAP1", "UMAP2"]] = UMAP(
                random_state=seed, n_jobs=1, n_neighbors=min(15, len(pcs) - 1)
            ).fit_transform(pcs)
        stability, gallery, composition, profiles = [], [], [], []
        for k in (2, 3, 4, 6):
            if len(np.unique(pcs, axis=0)) < k:
                continue
            primary = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pcs)
            labels = primary.labels_
            coords[f"k{k}"] = labels
            for offset in (1, 2, 3):
                other = KMeans(n_clusters=k, n_init=10, random_state=seed + offset).fit_predict(pcs)
                stability.append(dict(k=k, seed=seed + offset, ari=adjusted_rand_score(labels, other)))
            composition.append(coords.assign(cluster=labels).groupby(
                ["cluster", "cell_type", "compartment"], dropna=False).size().rename("instances")
                .reset_index().assign(k=k))
            profiles.append(representations["handcrafted"].reset_index(drop=True)
                            .assign(cluster=labels).groupby("cluster").median().reset_index().assign(k=k))
            for cluster, center in enumerate(primary.cluster_centers_):
                members = np.flatnonzero(labels == cluster)
                distance = np.linalg.norm(pcs[members] - center, axis=1)
                chosen = members[np.argsort(distance, kind="stable")[:5]]
                gallery.append(coords.iloc[chosen][KEY].assign(k=k, cluster=cluster))
        coords.to_csv(destination / "states.csv", index=False)
        pd.DataFrame(stability, columns=["k", "seed", "ari"]).to_csv(destination / "stability.csv", index=False)
        if gallery:
            pd.concat(gallery, ignore_index=True).to_csv(destination / "gallery_ids.csv", index=False)
            pd.concat(composition, ignore_index=True).to_csv(destination / "composition.csv", index=False)
            pd.concat(profiles, ignore_index=True).to_csv(destination / "cluster_morphometrics.csv", index=False)


def matched_pairs(frame, representations, neighbors, out):
    hand, _ = standardize(representations["handcrafted"])
    learned = {name: standardize(x)[0] for name, x in representations.items()
               if name != "handcrafted"}
    rows = []
    # Query only other neurons of the same type and compartment.
    for (typ, comp), group in frame.groupby(["cell_type", "compartment"]):
        by_body = list(group.groupby("body_id").indices.values())
        absolute = group.index.to_numpy()
        for i, members in enumerate(by_body):
            a = absolute[members]
            other = [absolute[ix] for j, ix in enumerate(by_body) if j != i]
            if not other:
                continue
            b = np.concatenate(other)
            nn = NearestNeighbors(n_neighbors=min(neighbors, len(b)), algorithm="brute").fit(hand[b])
            distances, indices = nn.kneighbors(hand[a])
            for j, anchor in enumerate(a):
                for rank, (dist, target) in enumerate(zip(distances[j], b[indices[j]]), start=1):
                    row = dict(cell_type=typ, compartment=comp, body_a=frame.body_id.iloc[anchor],
                               record_a=frame.record_id.iloc[anchor], body_b=frame.body_id.iloc[target],
                               record_b=frame.record_id.iloc[target], geometry_rank=rank,
                               geometry_distance=dist)
                    for name, x in learned.items():
                        row[f"{name}_distance"] = np.linalg.norm(x[anchor] - x[target])
                    rows.append(row)
    if not rows:
        raise ValueError("A2 needs at least two neurons in a type and compartment")
    pairs = pd.DataFrame(rows)
    for name in learned:
        pairs[f"{name}_distance_percentile"] = pairs.groupby(
            ["cell_type", "compartment"]
        )[f"{name}_distance"].rank(pct=True)
    pairs.to_csv(out / "candidate_pairs.csv", index=False)


def make_review_shortlist(frame, handcrafted, out):
    """Retain geometrically close but representation-distant pairs for EM review."""
    pairs = pd.read_csv(out / "candidate_pairs.csv", dtype={"body_a": str, "body_b": str})
    pairs["geometry_percentile"] = pairs.groupby(
        ["cell_type", "compartment"]
    ).geometry_distance.rank(pct=True)
    metadata = frame.set_index("record_id")
    hand = handcrafted.set_axis(frame.record_id, axis=0)
    for side in ("a", "b"):
        ids = pairs[f"record_{side}"]
        for name in ("em_path", "mask_path"):
            pairs[f"{name}_{side}"] = ids.map(metadata[name])
        pairs[f"size_{side}"] = ids.map(hand["size"])
        pairs[f"pc1_length_{side}"] = ids.map(hand["PC1 Length"])
    for feature in ("size", "pc1_length"):
        x, y = pairs[f"{feature}_a"], pairs[f"{feature}_b"]
        pairs[f"{feature}_ratio"] = np.maximum(x, y) / np.minimum(x, y).where(np.minimum(x, y) > 0)
    close = pairs.loc[
        pairs.geometry_percentile.le(.25) & pairs.size_ratio.le(1.25)
        & pairs.pc1_length_ratio.le(1.25)].copy()
    close["pair_key"] = ["|".join(sorted(pair)) for pair in zip(close.record_a, close.record_b)]
    picks = []
    for col in pairs.columns:
        if not col.endswith("_distance_percentile"):
            continue
        name = col.removesuffix("_distance_percentile")
        eligible = close.loc[close[col].ge(.90)].sort_values(
            [col, "geometry_distance", "record_a", "record_b"],
            ascending=[False, True, True, True]).drop_duplicates("pair_key")
        sample = eligible.groupby(["cell_type", "compartment"], sort=False).head(2)
        picks.append(sample.assign(selection_representation=name, selection_distance_percentile=sample[col]))
    if picks:
        pd.concat(picks, ignore_index=True).drop(columns="pair_key").to_csv(
            out / "em_review_shortlist.csv", index=False)
