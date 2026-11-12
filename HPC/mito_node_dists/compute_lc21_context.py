"""Calculate paper-style context measures with raw or author-cleaned skeletons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from paper_neuron_context import get_CrossSection_info, get_down_idxs, overlay_mito_on_skeleton
from paper_voxel_geometry import cart_2_spherical, get_foreground_stats

FIELDS = ["record_id", "body_id", "skeleton_source", "mito CA",
          "length along skeleton", "section_status", "length_status"]


@lru_cache(maxsize=64)
def load_skeleton(path, kind):
    if kind == "author_cleaned":
        with path.open(newline="") as stream:
            names = ["rowId", "x", "y", "z", "radius", "link", "distance", "theta", "phi"]
            skeleton = np.array([[float(row[k]) for k in names] for row in csv.DictReader(stream)])
    else:
        raw = np.loadtxt(path, comments="#", ndmin=2)
        xyz = raw[:, 2:5]
        nodes = {int(node): i for i, node in enumerate(raw[:, 0])}
        if len(nodes) != len(raw):
            raise ValueError("Duplicate SWC nodes")
        parent = np.array([nodes[int(node)] if node >= 0 else -1 for node in raw[:, 6]])
        vec = xyz[np.maximum(parent, 0)] - xyz
        distance = np.linalg.norm(vec, axis=1)
        for child in np.flatnonzero(parent >= 0):
            root = parent[child]
            if parent[root] == -1 and distance[root] == 0:
                vec[root] = -vec[child]
                distance[root] = np.linalg.norm(vec[root])
        directions = [cart_2_spherical(*v)[1:] if n else (0.0, 0.0)
                      for v, n in zip(vec, distance)]
        distance[parent == -1] = 0
        skeleton = np.column_stack((raw[:, 0], xyz, raw[:, 5], raw[:, 6],
                                    distance, directions))
    if not len(skeleton) or not np.isfinite(skeleton).all():
        raise ValueError("Invalid skeleton")
    return skeleton, cKDTree(skeleton[:, 1:4])


def compute(item):
    row, geom, root, kind, max_radius = item
    body = row["body_id"]
    out = {"record_id": row["record_id"], "body_id": body, "skeleton_source": kind}
    folder = Path(row["relative_dir"])
    path = root / (f"{body}.swc" if kind == "raw_neuprint" else
                   f"s_pandas_{body}_{row['cell_type']}_200nm.csv")
    try:
        meta = json.loads((folder / "metadata.json").read_text())
        center = np.asarray(meta["center_xyz"])
        skeleton, tree = load_skeleton(path, kind)
        nearest = tree.query(center)[1]
        radius = max(1, int(2 * skeleton[nearest, 4]))
        radius = min(radius, max_radius)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return dict(out, section_status=message, length_status=message)
    try:
        _, plane = get_CrossSection_info(int(body), center, radius, return_mito=True,
                                        max_dr=max_radius)
        if plane is None:
            raise ValueError("No uncropped section within radius limit")
        area = get_foreground_stats(plane, None, return_COM=False)[0]
        out.update({"mito CA": int(area), "section_status": "ok" if area else "zero_section"})
    except Exception as exc:
        out["section_status"] = f"{type(exc).__name__}: {exc}"
        out["length_status"] = "requires mito CA"
        return out
    try:
        if not area:
            raise ValueError("Zero mito CA")
        length = float(geom["PC1 Length"])
        mask = np.load(folder / "mask.npy", mmap_mode="r", allow_pickle=False)
        coords = np.argwhere(mask)[:, ::-1] + np.asarray(meta["bbox_xyz"][:3])
        edges = overlay_mito_on_skeleton(skeleton, coords, length, np.sqrt(area / np.pi))
        first, last = edges[0], edges[-1]
        if first[3] != "up" or last[3] != "down":
            raise ValueError("Invalid author edge sequence")
        status = "pc1_fallback_between_nodes"
        if first[1] != last[0]:
            path = get_down_idxs(skeleton, first[1], skeleton[:, 0] == last[1], count_start=True)
            if skeleton[path[-1], 0] != last[1]:
                raise ValueError("Skeleton endpoints have no common downstream path")
            candidate = first[2] + skeleton[path[:-1], 6].sum() + last[2]
            if candidate >= 0:
                length, status = float(candidate), "ok"
            else:
                status = "pc1_fallback_negative_path"
        out.update({"length along skeleton": length, "length_status": status})
    except Exception as exc:
        out["length_status"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--skeletons", type=Path, required=True)
    p.add_argument("--skeleton-kind", choices=["author_cleaned", "raw_neuprint"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-radius", type=int, default=300)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text())["records"][:args.limit]
    with args.geometry.open(newline="") as stream:
        geom = {r["record_id"]: r for r in csv.DictReader(stream)}
    items = [(r, geom[r["record_id"]], args.skeletons, args.skeleton_kind, args.max_radius)
             for r in rows]
    partial = args.output.with_suffix(".partial.csv")
    if args.output.exists():
        raise FileExistsError(args.output)
    sections = lengths = 0
    with partial.open("x", newline="") as stream, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        for index, row in enumerate(pool.map(compute, items), 1):
            writer.writerow(row)
            sections += row["section_status"] == "ok"
            lengths += "length along skeleton" in row
            if index % 25 == 0 or index == len(items):
                stream.flush()
                print(f"{index}/{len(items)} processed; {sections} sections, {lengths} lengths", flush=True)
    partial.replace(args.output)


if __name__ == "__main__":
    main()
