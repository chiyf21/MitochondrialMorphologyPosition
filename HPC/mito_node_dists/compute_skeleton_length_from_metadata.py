#!/usr/bin/env python3
"""Per-mito local SWC-tangent extent plus a mapped-node geodesic diagnostic."""
import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def load_swc(path, voxel_nm):
    rows, declared_scale = [], None
    pat = re.compile(r'(?:"scale"\s*:\s*|scale\s*[:=]\s*)([0-9.eE+-]+)', re.I)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            m = pat.search(line)
            if m:
                declared_scale = float(m.group(1))
            continue
        f = line.split()
        if len(f) >= 7:
            rows.append((int(f[0]), int(f[1]), *map(float, f[2:6]), int(float(f[6]))))
    if not rows:
        raise ValueError(f"No SWC nodes: {path}")
    if declared_scale is None or declared_scale <= 0:
        raise ValueError(f"Missing SWC scale in header: {path}")
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate SWC node IDs")
    idx = {int(n): i for i, n in enumerate(ids)}
    # The supplied manifest declares EM global XYZ and the SWC header says nm.
    # Dataset bbox coordinates are voxels, so convert only the mask vertices.
    xyz_nm = np.asarray([r[2:5] for r in rows], dtype=np.float64)
    xyz = xyz_nm / voxel_nm
    adj = [[] for _ in rows]
    for i, r in enumerate(rows):
        parent = r[6]
        if parent < 0:
            continue
        if parent not in idx:
            raise ValueError(f"Missing parent node {parent}")
        j = idx[parent]
        w = float(np.linalg.norm(xyz_nm[i] - xyz_nm[j]))
        adj[i].append((j, w))
        adj[j].append((i, w))
    comps, seen = [], set()
    for s in range(len(rows)):
        if s in seen:
            continue
        comp, stack, degree_sum = [], [s], 0
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            degree_sum += len(adj[u])
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if degree_sum // 2 != len(comp) - 1:
            raise ValueError("SWC graph is not a forest")
        comps.append(comp)
    return ids, xyz, xyz_nm, adj, comps, float(declared_scale)


def farthest(start, candidates, adj, ids):
    dist, stack = {start: 0.0}, [(start, -1)]
    while stack:
        u, parent = stack.pop()
        for v, w in adj[u]:
            if v != parent:
                dist[v] = dist[u] + w
                stack.append((v, u))
    reachable = candidates.intersection(dist)
    end = min(reachable, key=lambda i: (-dist[i], int(ids[i])))
    return end, dist[end]


def mapped_path_diameter(mapped, comps, adj, ids):
    best = None
    for comp in comps:
        candidates = mapped.intersection(comp)
        if not candidates:
            continue
        start = min(candidates, key=lambda i: int(ids[i]))
        a, _ = farthest(start, candidates, adj, ids)
        b, length = farthest(a, candidates, adj, ids)
        if best is None or length > best[0]:
            best = (length, a, b)
    return best


def read_neuron(neuron_root, body):
    folder = neuron_root / body
    mp = folder / "neuron_pointcloud_manifest.json"
    m = json.loads(mp.read_text())
    if str(m.get("bodyId", m.get("body_id", ""))) != body:
        raise ValueError(f"body ID mismatch: {mp}")
    voxel = np.asarray(m.get("voxel_size_nm"), dtype=np.float64)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError(f"voxel_size_nm must be positive XYZ: {mp}")
    swc = Path(m["skeleton_swc"])
    if not swc.is_file():
        swc = folder / "skeleton" / f"{body}.swc"
    if not swc.is_file():
        raise FileNotFoundError(swc)
    return load_swc(swc, voxel), voxel, swc, mp


def surface_points(mask, seed, max_points=200_000):
    if mask.size <= 100_000_000:
        verts_zyx, _, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), 0.5)
        return verts_zyx[:, ::-1] - 1, "marching_cubes"
    # Huge, sparse stored crops can make marching cubes allocate tens of GB.
    # Sample foreground voxels and retain only 6-connected boundary voxels.
    coords = np.argwhere(mask)
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        coords = coords[np.sort(rng.choice(len(coords), max_points, replace=False))]
    boundary = np.zeros(len(coords), dtype=bool)
    for axis in range(3):
        for step in (-1, 1):
            neighbor = coords.copy()
            neighbor[:, axis] += step
            valid = (neighbor[:, axis] >= 0) & (neighbor[:, axis] < mask.shape[axis])
            edge = ~valid
            neighbor_valid = neighbor[valid]
            edge[valid] = ~mask[tuple(neighbor_valid.T)]
            boundary |= edge
    points = coords[boundary]
    return points[:, ::-1].astype(np.float64), "sampled_boundary_voxels"


def process(row, ds, neuron_root, cache):
    body = str(row["body_id"])
    if body not in cache:
        cache[body] = read_neuron(neuron_root, body)
    (swc, voxel, swc_path, neuron_manifest) = cache[body]
    ids, xyz, xyz_nm, adj, comps, scale = swc
    meta = json.loads((ds / row["metadata_path"]).read_text())
    bbox = np.asarray(meta["bbox_xyz"], dtype=np.int64)
    mask = np.load(ds / row["mask_path"], allow_pickle=False).astype(bool)
    if bbox.shape != (6,) or mask.ndim != 3 or not mask.any():
        raise ValueError("Invalid bbox or empty/non-3D mask")
    extent = bbox[3:] - bbox[:3]
    shape_xyz = np.asarray(mask.shape[::-1])
    if not (np.array_equal(extent, shape_xyz) or np.array_equal(extent + 1, shape_xyz)):
        raise ValueError(f"bbox/mask shape mismatch: extent={extent}, shapeXYZ={shape_xyz}")

    points_xyz, mapping_method = surface_points(mask, int(row["global_instance"]))
    verts_xyz = points_xyz + bbox[:3]
    verts_nm = verts_xyz * voxel
    dists, nearest = cKDTree(xyz_nm).query(verts_nm, k=1, workers=1)
    k = int(np.argmin(dists))
    nearest_i = int(nearest[k])
    mapped = set(map(int, nearest.tolist()))
    diameter = mapped_path_diameter(mapped, comps, adj, ids)
    if diameter is None:
        length, a_id, b_id = math.nan, "", ""
    else:
        length, a, b = diameter
        a_id, b_id = int(ids[a]), int(ids[b])
    nxyz = xyz[nearest_i]
    peers = [int(edge[0]) for edge in adj[nearest_i]]
    if not peers:
        _, nearby = cKDTree(xyz_nm).query(xyz_nm[nearest_i], k=min(4, len(xyz_nm)))
        peers = [int(i) for i in np.atleast_1d(nearby) if int(i) != nearest_i]
    if peers:
        tangent = xyz_nm[max(peers, key=lambda j: float(np.linalg.norm(xyz_nm[j] - xyz_nm[nearest_i])))] - xyz_nm[nearest_i]
        tangent_norm = float(np.linalg.norm(tangent))
    else:
        tangent_norm = 0.0
    if tangent_norm > 0:
        tangent /= tangent_norm
        length_along = float(np.ptp(verts_nm @ tangent))
    else:
        length_along = math.nan
    graph_span = float(length) if math.isfinite(length) else math.nan
    return {
        "record_id": row["record_id"], "source_run_id": row["source_run_id"],
        "body_id": body, "cell_type": row.get("cell_type", ""),
        "global_instance": row["global_instance"],
        "nearest_skeleton_node_id": int(ids[nearest_i]),
        "nearest_node_x_global_vox": nxyz[0], "nearest_node_y_global_vox": nxyz[1],
        "nearest_node_z_global_vox": nxyz[2],
        "nearest_surface_distance_nm": float(dists[k]),
        "mapped_skeleton_node_count": len(mapped),
        "skeleton_endpoint_a_node_id": a_id, "skeleton_endpoint_b_node_id": b_id,
        "length_along_skeleton_nm": length_along,
        "length_along_skeleton_um": float(length_along / 1000) if math.isfinite(length_along) else "",
        "mapped_skeleton_geodesic_span_nm": graph_span,
        "skeleton_length_method": "mitochondrial_surface_extent_projected_onto_nearest_SWC_tangent",
        "swc_header_scale": scale, "swc_scale_applied": 1.0,
        "swc_coordinate_units": "EM_global_XYZ_nm", "surface_mapping_method": mapping_method,
        "surface_point_count": len(verts_xyz), "voxel_size_xyz_nm": ",".join(map(str, voxel)),
        "qc": "ok", "swc_path": str(swc_path), "neuron_manifest_path": str(neuron_manifest),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--neuron-root", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--run-id", action="append")
    ap.add_argument("--body-id", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start-index", type=int, default=0)
    a = ap.parse_args()
    with (a.dataset_root / "metadata/instances.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.run_id:
        rows = [r for r in rows if r["source_run_id"] in set(a.run_id)]
    if a.body_id:
        rows = [r for r in rows if r["body_id"] in set(a.body_id)]
    if a.start_index < 0:
        ap.error("--start-index must be nonnegative")
    rows = rows[a.start_index:]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        ap.error("No instances selected")
    cache, out = {}, []
    for i, row in enumerate(rows, 1):
        try:
            out.append(process(row, a.dataset_root, a.neuron_root, cache))
        except Exception as e:
            out.append({k: row.get(k, "") for k in ("record_id", "source_run_id", "body_id", "cell_type", "global_instance")} | {"qc": f"error: {type(e).__name__}: {e}"})
        if i % 100 == 0:
            print(f"processed {i}/{len(rows)}", flush=True)
    fields = ["record_id","source_run_id","body_id","cell_type","global_instance",
        "nearest_skeleton_node_id","nearest_node_x_global_vox","nearest_node_y_global_vox",
        "nearest_node_z_global_vox","nearest_surface_distance_nm","mapped_skeleton_node_count",
        "skeleton_endpoint_a_node_id","skeleton_endpoint_b_node_id",
        "length_along_skeleton_nm","length_along_skeleton_um","mapped_skeleton_geodesic_span_nm",
        "skeleton_length_method","swc_scale_applied","swc_header_scale","swc_coordinate_units","surface_mapping_method","surface_point_count",
        "voxel_size_xyz_nm","qc","swc_path","neuron_manifest_path"]
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(f"WROTE {a.output_csv}; rows={len(out)}; errors={sum(r['qc'] != 'ok' for r in out)}")


if __name__ == "__main__":
    main()
