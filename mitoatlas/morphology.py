"""R4 morphology distributions and R5 cross-compartment coordination."""
import argparse
from itertools import combinations
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist

from fish2_downstream.common import FEATURES, dump, sha, table, unique

KEY = ["body_id", "record_id"]
COMPARTMENTS = ("axon", "dendrite")


def load_cohort(path, min_bodies, rng, smoke):
    data = table(path)
    unique(data)
    if data[KEY + ["cell_type"]].isna().any().any():
        raise ValueError("Missing instance identity or labels")
    if data.groupby("body_id").cell_type.nunique().gt(1).any():
        raise ValueError("Each neuron must have a single cell type")
    selected = data[data.compartment.isin(COMPARTMENTS)].copy()
    if "compartment_qc" in selected:
        selected = selected[selected.compartment_qc.eq("ok")]
    selected = selected.sort_values(KEY)
    coverage = selected.groupby(["cell_type", "body_id"]).compartment.nunique()
    paired = coverage[coverage.eq(len(COMPARTMENTS))].reset_index()[["cell_type", "body_id"]]
    grouped = paired.groupby("cell_type", sort=True)
    eligible = paired[grouped.body_id.transform("size").ge(min_bodies)]
    if smoke and not eligible.empty:
        picked = []
        for _, group in list(eligible.groupby("cell_type"))[:3]:
            picked.extend(rng.choice(group.body_id, min(3, len(group)), replace=False))
        eligible = eligible[eligible.body_id.isin(picked)]
    selected = selected[selected.body_id.isin(eligible.body_id)]
    if selected.cell_type.nunique() < 2:
        raise ValueError("At least two cell types need enough neurons with both compartments")
    if smoke:
        selected = selected.groupby(["body_id", "compartment"], sort=False).head(12)
    return data, selected.reset_index(drop=True)


def aligned_embedding(path, cohort):
    data = table(path)
    unique(data)
    features = [col for col in data if col.startswith("embedding_")]
    if not features:
        raise ValueError("Embedding CSV needs embedding_ feature columns")
    indexed = data.set_index(KEY)[features]
    keys = pd.MultiIndex.from_frame(cohort[KEY])
    if not keys.isin(indexed.index).all():
        raise ValueError("Embedding is missing exact (body_id, record_id) matches")
    return indexed.reindex(keys)


def standardize(features):
    x = features.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite on every selected instance")
    std = x.std(axis=0)
    use = std > 0
    if not use.any():
        raise ValueError("All features are constant")
    return (x[:, use] - x[:, use].mean(axis=0)) / std[use], list(features.columns[use])


def distribution_distance(a, b, block=256):
    """Mean of the pooled bidirectional nearest-neighbor distances (LC notebook)."""
    near_a = np.full(len(a), np.inf)
    near_b = np.full(len(b), np.inf)
    for i in range(0, len(a), block):
        aa = slice(i, i + block)
        for j in range(0, len(b), block):
            bb = slice(j, j + block)
            distance = cdist(a[aa], b[bb])
            near_a[aa] = np.minimum(near_a[aa], distance.min(axis=1))
            near_b[bb] = np.minimum(near_b[bb], distance.min(axis=0))
    return (near_a.sum() + near_b.sum()) / (len(a) + len(b))


def r4(cohort, x):
    groups = cohort.groupby(["cell_type", "compartment"]).indices
    rows = []
    for a, b in combinations(groups, 2):
        if a[0] == b[0]:
            contrast = "across_compartments"
        elif a[1] == b[1]:
            contrast = "across_types"
        else:
            continue
        ia, ib = groups[a], groups[b]
        rows.append(dict(contrast=contrast, type_a=a[0], compartment_a=a[1],
                         type_b=b[0], compartment_b=b[1], instances_a=len(ia),
                         instances_b=len(ib), bodies_a=cohort.body_id.iloc[ia].nunique(),
                         bodies_b=cohort.body_id.iloc[ib].nunique(),
                         distance=distribution_distance(x[ia], x[ib])))
    return pd.DataFrame(rows)


def r5(cohort, x, seed, replicates):
    rng = np.random.default_rng(seed)
    centers, draws, counts = {}, {}, {}
    for typ, group in cohort.groupby("cell_type"):
        bodies = sorted(group.body_id.unique())
        sums, sizes = [], []
        for body in bodies:
            indexes = [group.index[group.body_id.eq(body) & group.compartment.eq(c)]
                       for c in COMPARTMENTS]
            sums.append([x[i].sum(axis=0) for i in indexes])
            sizes.append([len(i) for i in indexes])
        sums, sizes = np.asarray(sums), np.asarray(sizes)
        # Each draw uses the same resampled neurons in axon and dendrite.
        weights = rng.multinomial(len(bodies), np.full(len(bodies), 1 / len(bodies)),
                                  size=replicates)
        total = weights @ sums.reshape(len(bodies), -1)
        draws[typ] = total.reshape(replicates, 2, x.shape[1]) / (weights @ sizes)[:, :, None]
        centers[typ] = sums.sum(axis=0) / sizes.sum(axis=0)[:, None]
        counts[typ] = len(bodies)

    def cosine(delta):
        a, b = delta[..., 0, :], delta[..., 1, :]
        norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        dot = (a * b).sum(axis=-1)
        return np.divide(dot, norm, where=norm > 0, out=np.full_like(dot, np.nan))

    rows = []
    for a, b in combinations(centers, 2):
        values = cosine(draws[a] - draws[b])
        finite = values[np.isfinite(values)]
        lo, hi = np.quantile(finite, [.025, .975]) if finite.size else (np.nan, np.nan)
        rows.append(dict(type_a=a, type_b=b, bodies_a=counts[a], bodies_b=counts[b],
                         cosine=cosine(centers[a] - centers[b]).item(),
                         ci_low=lo, ci_high=hi, valid_bootstraps=len(finite)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name for provenance")
    parser.add_argument("--handcrafted", required=True, type=Path)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--embedding-name", default="learned")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--min-bodies", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=999)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.min_bodies < 2 or args.bootstrap < 1:
        parser.error("Require at least two bodies per type and one bootstrap draw")
    if args.embedding_name in {"handcrafted", ""}:
        parser.error("Choose a distinct embedding name")
    data, cohort = load_cohort(args.handcrafted, args.min_bodies,
                               np.random.default_rng(args.seed), args.smoke)
    hand = cohort.rename(columns={"length along skeleton": "length_along_skeleton_nm"})
    inputs = {"handcrafted": args.handcrafted}
    matrices = {"handcrafted": standardize(hand[FEATURES])}
    if args.embedding:
        inputs[args.embedding_name] = args.embedding
        matrices[args.embedding_name] = standardize(aligned_embedding(args.embedding, cohort))
    distances, coordination = [], []
    for name, (x, _) in matrices.items():
        distances.append(r4(cohort, x).assign(representation=name))
        coordination.append(r5(cohort, x, args.seed, args.bootstrap).assign(representation=name))
    args.out.mkdir(parents=True, exist_ok=False)
    cohort[KEY + ["cell_type", "compartment"]].to_csv(args.out / "cohort.csv", index=False)
    pd.concat(distances).to_csv(args.out / "r4_distribution.csv", index=False)
    pd.concat(coordination).to_csv(args.out / "r5_coordination.csv", index=False)
    dump(args.out / "run.json", dict(
        arguments=vars(args), instances_source=len(data), instances=len(cohort),
        bodies=cohort.body_id.nunique(), types=cohort.cell_type.nunique(),
        inputs={name: dict(path=str(path.resolve()), sha256=sha(path)) for name, path in inputs.items()},
        columns={name: columns for name, (_, columns) in matrices.items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    print(f"{args.dataset}: {len(cohort)} instances, {cohort.body_id.nunique()} neurons -> {args.out}")


if __name__ == "__main__":
    main()
