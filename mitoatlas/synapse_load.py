"""E05: FISH2 same-compartment skeleton windows around occupied mask samples."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from heapq import heappop, heappush
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from fish2_downstream.common import dump, read, sha
from .handoff import inputs
from .synapse_distance import Skeleton, SPACING, SiteDistances
from .synapse_regression import evaluate

RADII = (1000, 2000, 5000)
ROLES = ("pre", "post")
COMPS = ("axon", "dendrite")
KEY = ["body_id", "record_id", "embedding_row", "cell_type", "compartment"]


def window(graph, adjacent, seeds, cap):
    """Find endpoint geodesics from occupied points within cap nm."""
    distances, queue, active = {}, [], set(seeds)
    for edge, samples in seeds.items():
        u, v = graph.pairs[edge]
        length = graph.length[edge]
        for node, d in ((u, samples[0]), (v, length - samples[-1])):
            if d < distances.get(node, np.inf) and d <= cap:
                distances[node] = d
                heappush(queue, (d, node))
    while queue:
        d, node = heappop(queue)
        if d != distances[node]:
            continue
        for other, edge in adjacent[node]:
            active.add(edge)
            cost = d + graph.length[edge]
            if cost <= cap and cost < distances.get(other, np.inf):
                distances[other] = cost
                heappush(queue, (cost, other))
    return distances, active


def intervals(length, samples, du, dv, radius):
    spans = []
    if du <= radius:
        spans.append((0., min(length, radius - du)))
    if dv <= radius:
        spans.append((max(0., length - radius + dv), length))
    if samples is not None:
        for block in np.split(samples, np.flatnonzero(np.diff(samples) > 2 * radius) + 1):
            spans.append((max(0., block[0] - radius), min(length, block[-1] + radius)))
    union = []
    for start, stop in sorted(spans):
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], stop)
        else:
            union.append([start, stop])
    return union


def measure(graph, seeds, adjacent, sites):
    distances, active = window(graph, adjacent, seeds, max(RADII))
    counts = np.zeros((len(RADII), 2), dtype=int)
    lengths = np.zeros(len(RADII))
    for e in sorted(active):
        u, v = graph.pairs[e]
        du, dv = distances.get(u, np.inf), distances.get(v, np.inf)
        for i, r in enumerate(RADII):
            for low, high in intervals(graph.length[e], seeds.get(e), du, dv, r):
                lengths[i] += high - low
                if e in sites:
                    for j, positions in enumerate(sites[e]):
                        counts[i, j] += (np.searchsorted(positions, high + 1e-8, side="right")
                                         - np.searchsorted(positions, low - 1e-8))
    return lengths, counts


def process(job, provenance):
    _, rows, site = job
    body = rows.body_id.iloc[0]
    cfg = provenance["arguments"]
    snapshot = Path(cfg["snapshot"])
    marker = snapshot / "corrected" / body / "review.json"
    edge_path = snapshot / Path(read(marker)["files"]["compartment_edges.parquet"]).relative_to(cfg["original_review_root"])
    node_path = snapshot / "automatic" / body / "skeleton_nodes.parquet"
    for path in (marker, edge_path, node_path):
        if sha(path) != provenance["input_sha256"][str(path)]:
            raise ValueError(f"E04 skeleton provenance changed: {path}")
    edges, nodes = pd.read_parquet(edge_path), pd.read_parquet(node_path)
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * SPACING
    if rows.skeleton_xyz_rescaled.iloc[0]:
        xyz *= [16, 16, 15]
    graph = Skeleton(xyz, edges[["u", "v"]].to_numpy(int))
    labels = edges.compartment.to_numpy()
    if not np.array_equal(edges.edge_id.to_numpy()[site.projected_edge_row], site.skeleton_edge_id):
        raise ValueError(f"E04 projected edges changed for {body}")
    site = site[site.accepted]
    site_groups = site.groupby("projected_edge_row")
    sites = {e: [np.sort(group.loc[group.role.eq(role), "edge_fraction"].to_numpy() * graph.length[e])
                 for role in ROLES] for e, group in site_groups}
    results = []
    for comp in COMPS:
        adjacent = defaultdict(list)
        selected = np.flatnonzero(labels == comp)
        for e in selected:
            u, v = graph.pairs[e]
            adjacent[u].append((v, e))
            adjacent[v].append((u, e))
        selected_sites = site[labels[site.projected_edge_row] == comp]
        restricted = Skeleton(xyz, graph.pairs[selected])
        # Original node indices stay intact. The adjacency and seed set are both restricted by compartment.
        remap = {e: i for i, e in enumerate(selected)}
        nearest = {role: SiteDistances(restricted, [(remap[r.projected_edge_row], r.edge_fraction, 0.)
                    for r in selected_sites[selected_sites.role.eq(role)].itertuples()]) for role in ROLES}
        for row in rows[rows.compartment.eq(comp)].itertuples():
            bbox = np.asarray(json.loads(row.bbox_xyz), dtype=int)
            mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False)
            if mask.shape != tuple((bbox[3:] - bbox[:3])[::-1]):
                raise ValueError(f"Mask shape disagrees with bbox: {row.record_id}")
            occ, t = graph.occupied(bbox, mask, cfg["step_nm"])
            keep = labels[occ] == comp
            occ, t = occ[keep], t[keep]
            if not len(occ):
                continue
            seeds = {e: np.unique(t[occ == e]) * graph.length[e] for e in np.unique(occ)}
            lengths, counts = measure(graph, seeds, adjacent, sites)
            near_dist = {f"nearest_{role}_um": dist.nearest(np.array([remap[e] for e in occ]), t)
                         for role, dist in nearest.items()}
            for radius, length, (pre, post) in zip(RADII, lengths / 1000, counts):
                if length:
                    results.append({k: getattr(row, k) for k in KEY} | near_dist |
                                   dict(radius_um=radius/1000, length_um=length,
                                        pre_count=pre, post_count=post,
                                        pre_density=pre/length, post_density=post/length))
    print(body, len(results), flush=True)
    return results


def predictors(frame, reps):
    hand = reps["handcrafted"]
    if np.array_equal(hand["mito CA"],hand["PC1 CA"]):
        hand = hand.drop(columns="mito CA")
    basic = np.log(hand[["size", "PC1 Length"]].to_numpy())
    types = pd.get_dummies(frame.cell_type).to_numpy(dtype=float)
    return {"train_mean": None, "geometry": np.c_[types, basic],
            "geometry_handcrafted": np.c_[types, basic, hand],
            "geometry_concat": np.c_[types, basic, reps["concat"]]}


def predict(labels, frame, reps, args):
    scores, oof, coverage = [], [], []
    models = predictors(frame, reps)
    labels = labels.merge(frame[["record_id", "cell_type_fold"]], on="record_id", validate="many_to_one")
    for (comp, radius), rows in labels.groupby(["compartment", "radius_um"]):
        idx = rows.embedding_row.to_numpy()
        for role in ROLES:
            for kind in ("count", "density"):
                y = rows[f"{role}_{kind}"].to_numpy()
                x = {k: None if v is None else np.c_[v[idx], np.log(rows.length_um)] for k,v in models.items()}
                ctx = dict(compartment=comp, role=role, radius_um=radius, target=kind)
                coverage.append(ctx | dict(instances=len(y), bodies=rows.body_id.nunique(),
                                           zero_fraction=np.mean(y == 0), mean=y.mean()))
                for scale, target in (("raw", y), ("log1p", np.log1p(y))):
                    metric, pred = evaluate(rows, target, x, args)
                    scores.append(metric.assign(**ctx, scale=scale))
                    oof.append(pred.assign(**ctx, scale=scale))
    for name, data in (("scores", scores), ("oof", oof), ("coverage", [pd.DataFrame(coverage)])):
        pd.concat(data, ignore_index=True).to_csv(args.out / f"{name}.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "distances", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--bodies", nargs="+")
    args = p.parse_args()
    args.alpha, args.bootstrap, args.seed = 10., 999, 20261117
    frame, reps, hashes = inputs(args.handoff)
    source = args.distances / "run.json"
    site_path = args.distances / "synapse_projection_qc.csv"
    provenance = read(source)
    if provenance["input_sha256"][str(args.handoff/"metadata/instances.csv")] != hashes["metadata/instances.csv"]:
        raise ValueError("E04 and E05 instance manifests differ")
    site = pd.read_csv(site_path, dtype={"body_id": str})
    grouped = dict(tuple(site.groupby("body_id")))
    cohort = frame[frame.compartment_eligible]
    if args.bodies:
        cohort = cohort[cohort.body_id.isin(args.bodies)]
    jobs = [(body, rows, grouped[body]) for body, rows in cohort.groupby("body_id")]
    args.out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        labels = pd.DataFrame([row for block in pool.map(partial(process, provenance=provenance), jobs) for row in block])
    labels.to_csv(args.out / "windows.csv", index=False)
    if not args.bodies:
        predict(labels, frame, reps, args)
    dump(args.out/"run.json", dict(arguments=vars(args), input_sha256=hashes|{
        str(p): sha(p) for p in (source, site_path)},
        commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        radii_nm=RADII, allowed_edges="reviewed same-compartment original edges",
        instances=labels.record_id.nunique(), bodies=labels.body_id.nunique()))


if __name__ == "__main__":
    main()
