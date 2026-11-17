"""E04 FISH2 pilot: nearest pre/post distances on reviewed original skeleton edges."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from fish2_downstream.common import dump, read, sha

SPACING = np.array([8., 8., 30.])
SITE_XYZ = ["x_em_nm", "y_em_nm", "z_em_nm"]
CENTROID = ["centroid_x_voxel", "centroid_y_voxel", "centroid_z_voxel"]


class Skeleton:
    def __init__(self, xyz, pairs):
        self.xyz = xyz
        self.pairs = pairs
        self.start, self.end = xyz[pairs].transpose(1, 0, 2)
        self.delta = self.end - self.start
        self.length = np.linalg.norm(self.delta, axis=1)
        self.dd = np.where(self.length > 0, self.length**2, 1.)
        self.lower = np.minimum(self.start, self.end)
        self.upper = np.maximum(self.start, self.end)

    def project(self, points):
        """Exact nearest segment projection in bounded batches."""
        result = []
        for start in range(0, len(points), 32):
            offset = points[start:start+32, None, :] - self.start
            t = np.clip(np.einsum("qed,ed->qe", offset, self.delta) / self.dd, 0, 1)
            delta = offset - t[..., None] * self.delta
            dist = np.einsum("qed,qed->qe", delta, delta)
            idx = np.arange(len(t))
            edges = dist.argmin(axis=1)
            result.extend(zip(edges, t[idx, edges], np.sqrt(dist[idx, edges])))
        return result

    def occupied(self, bbox, mask, step):
        """Sample original edges at at most step nm, retain points inside mask voxels."""
        lo, hi = (bbox[:3] - .5) * SPACING, (bbox[3:] - .5) * SPACING
        eligible = np.flatnonzero(((self.upper >= lo) & (self.lower < hi)).all(axis=1))
        edges, fractions = [], []
        for e in eligible:
            fractions_edge = np.linspace(0., 1., max(1, int(np.ceil(self.length[e] / step))) + 1)
            coords = self.start[e] + fractions_edge[:, None] * self.delta[e]
            voxels = np.floor(coords / SPACING + .5).astype(int) - bbox[:3]
            inside = ((voxels >= 0) & (voxels < np.array(mask.shape[::-1]))).all(axis=1)
            idx = np.flatnonzero(inside)
            idx = idx[mask[tuple(voxels[idx, ::-1].T)] != 0]
            if len(idx):
                edges.extend([e] * len(idx))
                fractions.extend(fractions_edge[idx])
        return np.asarray(edges, dtype=int), np.asarray(fractions)


class SiteDistances:
    def __init__(self, graph, events):
        self.graph = graph
        self.same_edge = {}
        root = len(graph.xyz)
        starts, ends = graph.pairs.T
        reach = np.full(root, np.inf)
        for edge, t, _ in events:
            length = graph.length[edge]
            a, b = graph.pairs[edge]
            reach[a] = min(reach[a], t * length)
            reach[b] = min(reach[b], (1-t) * length)
            self.same_edge.setdefault(edge, []).append(t)
        self.same_edge = {key: np.sort(values) for key, values in self.same_edge.items()}
        nodes = np.flatnonzero(np.isfinite(reach))
        matrix = coo_matrix((np.r_[graph.length, graph.length, reach[nodes]],
                            (np.r_[starts, ends, np.full(len(nodes), root)], np.r_[ends, starts, nodes])),
                            shape=(root+1, root+1)).tocsr()
        self.distance = dijkstra(matrix, indices=root, directed=True)[:root]

    def nearest(self, edges, t):
        if not len(edges):
            return np.nan
        a, b = self.graph.pairs[edges].T
        length = self.graph.length[edges]
        best = min(np.min(self.distance[a] + t * length),
                   np.min(self.distance[b] + (1-t) * length))
        for e in np.unique(edges):
            sites = self.same_edge.get(e)
            if sites is None:
                continue
            q = t[edges == e]
            idx = np.searchsorted(sites, q)
            nearest = np.minimum(abs(sites[np.minimum(idx, len(sites)-1)] - q),
                                 abs(sites[np.maximum(idx-1, 0)] - q))
            best = min(best, nearest.min() * self.graph.length[e])
        return best / 1000 if np.isfinite(best) else np.nan


def process(body, rows, args):
    snap = args.snapshot
    marker = snap / "corrected" / body / "review.json"
    review = read(marker)
    if review["state"] != "corrected" or str(review["body_id"]) != body:
        raise ValueError(f"Invalid corrected review: {body}")
    edges_path = snap / Path(review["files"]["compartment_edges.parquet"]).relative_to(args.original_review_root)
    nodes_path = snap / "automatic" / body / "skeleton_nodes.parquet"
    sites_path = args.synapses / "automatic" / body / "synapse_attachments.parquet"
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    expected = edges[["parent_id_original", "child_id_original"]].to_numpy()
    pairs = edges[["u", "v"]].to_numpy(int)
    if not np.array_equal(nodes.rowId.to_numpy()[pairs], expected):
        raise ValueError(f"Reviewed edges disagree with node IDs: {body}")
    if len(rows.skeleton_xyz_rescaled.unique()) != 1:
        raise ValueError(f"Inconsistent coordinate scale: {body}")
    xyz = nodes[["x_em", "y_em", "z_em"]].to_numpy() * SPACING
    if rows.skeleton_xyz_rescaled.iloc[0]:
        xyz *= [16, 16, 15]
    graph = Skeleton(xyz, pairs)
    site = pd.read_parquet(sites_path)
    site = site[site.role.isin(["pre", "post"])].drop_duplicates(["role", *SITE_XYZ]).copy()
    site = site.reset_index(drop=True)
    if not np.isfinite(site[SITE_XYZ].to_numpy()).all():
        raise ValueError(f"Invalid synapse coordinates: {body}")
    projected = graph.project(site[SITE_XYZ].to_numpy())
    site[["projected_edge_row", "edge_fraction", "projection_nm"]] = pd.DataFrame(projected, index=site.index, columns=range(3))
    site["accepted"] = site.projection_nm.le(args.max_projection_nm)
    site["body_id"] = body
    site["skeleton_edge_id"] = edges.edge_id.to_numpy()[site.projected_edge_row.to_numpy(dtype=int)]
    role_graph = {}
    for role in ("pre", "post"):
        selected = site.index[site.role.eq(role) & site.accepted]
        role_graph[role] = SiteDistances(graph, [projected[j] for j in selected])

    results = []
    projected_centers = graph.project(rows[CENTROID].to_numpy() * SPACING)
    for row, (edge, t, dist) in zip(rows.itertuples(), projected_centers):
        bbox = np.asarray(__import__("json").loads(row.bbox_xyz), dtype=int)
        mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False)
        if mask.shape != tuple((bbox[3:] - bbox[:3])[::-1]):
            raise ValueError(f"Mask crop does not agree with bbox: {row.record_id}")
        occ_edge, occ_t = graph.occupied(bbox, mask, args.step_nm)
        result = dict(body_id=body, record_id=row.record_id, cell_type=row.cell_type,
                      compartment=row.compartment, embedding_row=row.embedding_row,
                      center_skeleton_edge_id=edges.edge_id.iloc[edge], center_projection_nm=dist,
                      occupied_edges=len(np.unique(occ_edge)), occupied_samples=len(occ_edge),
                      skeleton_xyz_rescaled=bool(row.skeleton_xyz_rescaled))
        for role, distances in role_graph.items():
            center = distances.nearest(np.array([edge]), np.array([t])) if dist <= args.max_projection_nm else np.nan
            occupied = distances.nearest(occ_edge, occ_t)
            result[f"nearest_{role}_center_um"] = center
            result[f"nearest_{role}_occupied_um"] = occupied
        results.append(result)
    hashes = {str(p): sha(p) for p in (marker, nodes_path, edges_path, sites_path)}
    return results, site, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "snapshot", "original_review_root", "synapses", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--bodies", required=True, nargs="+")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--step-nm", type=float, default=4.)
    p.add_argument("--max-projection-nm", type=float, default=1000.)
    args = p.parse_args()
    if args.workers < 1 or not all(np.isfinite(x) and x > 0 for x in (args.step_nm, args.max_projection_nm)):
        p.error("Sampling and QC limits must be finite and positive")
    manifest = args.handoff / "metadata/instances.csv"
    table = pd.read_csv(manifest, dtype={"body_id": str})
    table = table[table.body_id.isin(args.bodies) & table.compartment_eligible]
    if set(table.body_id) != set(args.bodies) or not table.record_id.is_unique:
        raise ValueError("Requested bodies must all have QC-passed axon/dendrite instances")
    rows, sites, hashes = [], [], {str(manifest): sha(manifest)}
    bodies, blocks = zip(*table.groupby("body_id"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for body, (values, events, digests) in zip(bodies, pool.map(process, bodies, blocks, repeat(args))):
            rows.extend(values)
            sites.append(events)
            hashes.update(digests)
            print(f"{body}: {len(values)} mitochondria and {len(events)} deduplicated synapse endpoints", flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(rows)
    result.to_csv(args.out / "mitochondria_distances.csv", index=False)
    pd.concat(sites, ignore_index=True).to_csv(args.out / "synapse_projection_qc.csv", index=False)
    dump(args.out / "run.json", dict(arguments=vars(args), input_sha256=hashes,
         skeleton="original reviewed SWC edges only; no artificial bridges",
         commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         bodies=result.body_id.nunique(), instances=len(result)))
    print(args.out)


if __name__ == "__main__":
    main()
