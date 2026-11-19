"""Test 21D hand-crafted features across the finalized FISH2 PC30 clusters.

The primary heatmap is neuron-balanced: each neuron contributes one mean per
cluster, so a neuron with many mitochondria cannot dominate the profile.  The
CSV tables also retain instance-weighted summaries and both instance-level
Kruskal-Wallis and neuron-block Friedman tests.
"""

import argparse
from itertools import combinations
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu

from fish2_downstream.common import dump, sha


FEATURES_22 = ["size", "mito SA", "mito CA", "convex hull compactness"] + [
    f"PC{axis} {suffix}" for axis in (1, 2, 3)
    for suffix in ["Length", "inertia moment", "symmetry", "CA", "Circum"]
] + ["mito diameter", "length_along_skeleton_nm", "SBI"]
# mito CA and PC1 CA are the same cross-sectional quantity in this export.
FEATURES_21 = [name for name in FEATURES_22 if name != "mito CA"]


def bh(values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=1.0)
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def log_transform(values):
    values = np.asarray(values, dtype=float)
    return np.log1p(values) if np.nanmin(values) >= 0 else values


def transform_name(values):
    return "log1p" if np.nanmin(values) >= 0 else "raw"


def cluster_label_map(labels):
    raw = np.sort(pd.unique(labels))
    return {value: f"C{i + 1}" for i, value in enumerate(raw)}, raw


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix, row_order, q_values, out, name, title, colorbar_label):
    matrix = matrix.loc[row_order]
    n_rows = len(matrix)
    fig_h = max(4.3, 0.34 * n_rows + 1.7)
    fig, ax = plt.subplots(figsize=(8.2, fig_h), layout="constrained")
    values = matrix.to_numpy(dtype=float)
    limit = max(1.0, float(np.nanmax(np.abs(values))))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                      interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, fontsize=13)
    labels = []
    for feature in matrix.index:
        q = q_values.get(feature, np.nan)
        labels.append(f"{feature}  (q={q:.1e})" if np.isfinite(q) else feature)
    ax.set_yticks(np.arange(n_rows), labels, fontsize=10)
    ax.tick_params(length=0)
    ax.set_xlabel("Leiden cluster", fontsize=15, labelpad=10)
    ax.set_ylabel("21D handcrafted feature", fontsize=15, labelpad=10)
    ax.set_title(title, fontsize=15, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(colorbar_label, fontsize=12, rotation=90, labelpad=12)
    save(fig, out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("handcraft", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cluster-column", default="Leiden k=90 gamma=0.5")
    parser.add_argument("--min-bodies", type=int, default=2)
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments, dtype={"record_id": str, "body_id": str})
    hand = pd.read_csv(args.handcraft, dtype={"record_id": str, "body_id": str})
    required = ["record_id", "body_id", args.cluster_column]
    if not set(required).issubset(assignments.columns):
        raise ValueError(f"Assignment table is missing {set(required) - set(assignments.columns)}")
    if not set(["record_id", "body_id"] + FEATURES_21).issubset(hand.columns):
        raise ValueError("Handcrafted table does not contain the 21D schema")
    if not assignments.record_id.is_unique or not hand.record_id.is_unique:
        raise ValueError("record_id must be unique")
    keys = ["record_id", "body_id"]
    data = assignments[keys + [args.cluster_column]].merge(
        hand[keys + FEATURES_21], on=keys, how="inner", validate="one_to_one")
    if data.empty or data[FEATURES_21].isna().any().any():
        raise ValueError("The joined handcraft table is empty or contains missing features")
    if not np.isfinite(data[FEATURES_21].to_numpy(dtype=float)).all():
        raise ValueError("Handcrafted features must be finite")
    data = data.rename(columns={args.cluster_column: "cluster_raw"})
    label_map, raw_labels = cluster_label_map(data.cluster_raw)
    data["cluster"] = data.cluster_raw.map(label_map)
    clusters = [label_map[x] for x in raw_labels]
    if len(clusters) < 2:
        raise ValueError("At least two clusters are required")

    args.output.mkdir(parents=True, exist_ok=False)
    rows, cluster_rows, pair_rows = [], [], []
    body_cluster = data.groupby(["body_id", "cluster"], sort=True)[FEATURES_21].mean()
    complete = body_cluster.reset_index().pivot(index="body_id", columns="cluster", values=FEATURES_21)
    complete = complete.dropna(axis=0, how="any")
    body_profiles = {}
    for feature in FEATURES_21:
        transformed = log_transform(data[feature].to_numpy(dtype=float))
        mean, std = transformed.mean(), transformed.std(ddof=0)
        if std <= 0:
            raise ValueError(f"Constant feature: {feature}")
        body_values = complete[feature]
        body_profiles[feature] = body_values.mean(axis=0)
        groups = [data.loc[data.cluster.eq(cluster), feature].to_numpy(dtype=float) for cluster in clusters]
        h_stat, h_p = kruskal(*groups)
        body_groups = [body_values[cluster].to_numpy(dtype=float) for cluster in clusters]
        if len(complete) >= args.min_bodies:
            f_stat, f_p = friedmanchisquare(*body_groups)
        else:
            f_stat, f_p = np.nan, np.nan
        rows.append(dict(feature=feature, instance_n=len(data), body_n_complete=len(complete),
                         kruskal_h=float(h_stat), kruskal_p=float(h_p),
                         friedman_chi2=float(f_stat), friedman_p=float(f_p),
                         transform=transform_name(data[feature].to_numpy(dtype=float))))
        for cluster in clusters:
            selected = data.cluster.eq(cluster)
            cluster_rows.append(dict(cluster=cluster, feature=feature,
                                     instance_n=int(selected.sum()),
                                     body_n=int(data.loc[selected, "body_id"].nunique()),
                                     raw_mean=float(data.loc[selected, feature].mean()),
                                     raw_median=float(data.loc[selected, feature].median()),
                                     transformed_z_mean=float(((transformed - mean) / std)[selected.to_numpy()].mean()),
                                     neuron_balanced_mean=float(body_profiles[feature].get(cluster, np.nan))))
        for a, b in combinations(clusters, 2):
            x = data.loc[data.cluster.eq(a), feature].to_numpy(dtype=float)
            y = data.loc[data.cluster.eq(b), feature].to_numpy(dtype=float)
            u, p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            pair_rows.append(dict(feature=feature, cluster_a=a, cluster_b=b,
                                  n_a=len(x), n_b=len(y), mannwhitney_u=float(u), p=float(p),
                                  rank_biserial=float(2 * u / (len(x) * len(y)) - 1)))

    stats = pd.DataFrame(rows)
    stats["kruskal_q"] = bh(stats.kruskal_p)
    stats["friedman_q"] = bh(stats.friedman_p)
    pairs = pd.DataFrame(pair_rows)
    pairs["q"] = bh(pairs.p.to_numpy())
    cluster_summary = pd.DataFrame(cluster_rows)
    balanced = pd.DataFrame(body_profiles).T.reindex(index=FEATURES_21, columns=clusters)
    balanced.index.name = "feature"
    balanced.columns.name = "cluster"
    balanced_z = balanced.sub(balanced.mean(axis=1), axis=0).div(balanced.std(axis=1, ddof=0), axis=0).fillna(0.0)
    instance_profile = cluster_summary.pivot(index="feature", columns="cluster", values="transformed_z_mean").reindex(FEATURES_21)
    instance_profile.columns.name = "cluster"
    instance_profile = instance_profile.reindex(columns=clusters)
    order = stats.sort_values(["friedman_q", "kruskal_q", "feature"], kind="stable").feature.tolist()
    significant = stats.loc[stats.friedman_q < 0.05, "feature"].tolist()
    if not significant:
        significant = stats.loc[stats.kruskal_q < 0.05, "feature"].tolist()

    stats.to_csv(args.output / "feature_statistics.csv", index=False)
    pairs.to_csv(args.output / "pairwise_statistics.csv", index=False)
    cluster_summary.to_csv(args.output / "cluster_feature_summary.csv", index=False)
    balanced.to_csv(args.output / "neuron_balanced_cluster_means.csv")
    balanced_z.to_csv(args.output / "neuron_balanced_cluster_zscores.csv")
    instance_profile.to_csv(args.output / "instance_weighted_cluster_zscores.csv")
    data[keys + ["cluster_raw", "cluster"] + FEATURES_21].to_csv(args.output / "joined_assignments_features.csv", index=False)

    q_map = stats.set_index("feature").friedman_q.to_dict()
    heatmap(balanced_z, order, q_map, args.output, "FISH2_Leiden6_21D_heatmap",
            "FISH2 PC30 Leiden six-cluster morphology", "Neuron-balanced mean z-score")
    heatmap(balanced_z, [x for x in order if x in significant], q_map, args.output,
            "FISH2_Leiden6_21D_heatmap_significant",
            "FISH2 PC30 Leiden six-cluster significant morphology", "Neuron-balanced mean z-score")

    dump(args.output / "run.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        assignments=str(args.assignments.resolve()), handcrafted=str(args.handcraft.resolve()),
        assignments_sha256=sha(args.assignments), handcrafted_sha256=sha(args.handcraft),
        cluster_column=args.cluster_column, raw_clusters=[int(x) for x in raw_labels],
        cluster_labels=clusters, instances_joined=len(data), bodies_joined=int(data.body_id.nunique()),
        dropped_assignment_rows=int(len(assignments) - len(data)),
        complete_bodies_for_friedman=int(len(complete)), features=FEATURES_21,
        primary_test="Friedman test on per-neuron per-cluster means",
        screening_test="Kruskal-Wallis test on individual mitochondria",
        heatmap_scale="Within-feature z-score after log1p for nonnegative features; per-neuron cluster means equally weighted",
        significant_features_friedman_q_lt_0p05=significant,
    ))
    print(stats.sort_values("friedman_q")[["feature","kruskal_q","friedman_q"]].to_string(index=False))
    print(f"joined={len(data)} bodies={data.body_id.nunique()} complete_bodies={len(complete)} clusters={clusters}")


if __name__ == "__main__":
    main()
