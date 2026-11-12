from __future__ import annotations

import heapq
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================================================
# V11: SCIENCE-LIKE CONNECTED COMPARTMENT ANNOTATION
#
# Inspired by the compartment logic in Sager et al., Science (2026):
#   dendritic arbor base -> dendrite
#   axonal arbor base    -> axon
#   path between bases   -> connecting cable
#   soma -> cable        -> cell body fiber
#
# fish2 approximation:
#   POST synapses = dendritic evidence
#   PRE synapses  = axonal evidence
#
# PRE/POST are used ONLY to locate the two large arbors. Once an arbor is
# defined, topology dominates local synapse sparsity. There is deliberately
# NO ambiguous output class.
# ============================================================

BODY_ID = int(sys.argv[1]) if len(sys.argv) > 1 else int(globals().get("BODY_ID", 100011685))
ROOT = Path(globals().get("ROOT", "/home/cyf/JaneliaEM"))

NEURON_DIR = ROOT / "data" / "neuron_downloads" / str(BODY_ID)
SKELETON_DIR = NEURON_DIR / "skeleton"
SKELETON_CANDIDATES = [
    SKELETON_DIR / f"{BODY_ID}_skeleton_em.parquet",
    SKELETON_DIR / f"{BODY_ID}_skeleton.parquet",
    SKELETON_DIR / f"{BODY_ID}_em_coordinates.parquet",
    SKELETON_DIR / f"{BODY_ID}.parquet",
]
SKELETON_PATH = next((p for p in SKELETON_CANDIDATES if p.exists()), SKELETON_CANDIDATES[-1])
SYNAPSE_PATH = NEURON_DIR / "neuron_synapse_connections.parquet"

SEG_TO_EM = np.asarray([2.0, 2.0, 0.5], dtype=float)
EM_RESOLUTION_NM = np.asarray([8.0, 8.0, 30.0], dtype=float)
SYNAPSE_MAX_NODE_DISTANCE_NM = 2500.0
SOMA_PREDICTION_SEARCH_RADIUS_NM = 20000.0
CAPTURE_FRACTIONS = (0.90, 0.85, 0.80, 0.75)
MIN_TARGET_SYNAPSES_FOR_BASE = 3
FALLBACK_CABLE_HALF_WIDTH_NM = 5000.0

COMPARTMENT_CODES = {
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "cell_body_fiber": 4,
    "connecting_cable": 5,
}


def find_first_soma_location(obj) -> Optional[np.ndarray]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in {"somalocation", "soma_location"}:
                try:
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if len(arr) >= 3 and np.all(np.isfinite(arr[:3])):
                        return arr[:3]
                except Exception:
                    pass
        for value in obj.values():
            found = find_first_soma_location(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_first_soma_location(value)
            if found is not None:
                return found
    return None


def discover_soma_prediction(neuron_dir: Path) -> Tuple[Optional[np.ndarray], str]:
    external_neuron = globals().get("neuron")
    if isinstance(external_neuron, dict):
        found = find_first_soma_location(external_neuron)
        if found is not None:
            return found, "global_neuron.somaLocation"

    for path in sorted(neuron_dir.rglob("*.json")):
        if "compartment_summary" in path.name:
            continue
        try:
            if path.stat().st_size > 10_000_000:
                continue
            obj = json.loads(path.read_text())
        except Exception:
            continue
        found = find_first_soma_location(obj)
        if found is not None:
            return found, str(path)
    return None, "none"


def connected_components(adjacency):
    n = len(adjacency)
    component_id = np.full(n, -1, dtype=int)
    components = []
    for start in range(n):
        if component_id[start] >= 0:
            continue
        cid = len(components)
        q = deque([start])
        component_id[start] = cid
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _e, _l in adjacency[u]:
                if component_id[v] < 0:
                    component_id[v] = cid
                    q.append(int(v))
        components.append(comp)
    return component_id, components


def orient_tree(adjacency, root: int, allowed_nodes: Set[int]):
    n = len(adjacency)
    parent = np.full(n, -1, dtype=int)
    parent_edge = np.full(n, -1, dtype=int)
    depth = np.full(n, -1, dtype=int)
    order = []
    q = deque([int(root)])
    depth[root] = 0
    while q:
        u = q.popleft()
        order.append(u)
        for v, e, _l in adjacency[u]:
            if v not in allowed_nodes or depth[v] >= 0:
                continue
            parent[v] = int(u)
            parent_edge[v] = int(e)
            depth[v] = depth[u] + 1
            q.append(int(v))
    return parent, parent_edge, depth, order


def children_from_parent(parent: np.ndarray, nodes: Iterable[int]):
    children = [[] for _ in range(len(parent))]
    for n in nodes:
        p = int(parent[n])
        if p >= 0:
            children[p].append(int(n))
    return children


def is_ancestor(a: int, b: int, parent: np.ndarray) -> bool:
    cur = int(b)
    while cur >= 0:
        if cur == int(a):
            return True
        cur = int(parent[cur])
    return False


def path_to_root(node: int, parent: np.ndarray) -> List[int]:
    out = []
    cur = int(node)
    while cur >= 0:
        out.append(cur)
        cur = int(parent[cur])
    return out


def path_between_nodes(a: int, b: int, parent: np.ndarray) -> List[int]:
    ancestors = set(path_to_root(a, parent))
    b_side = []
    cur = int(b)
    while cur >= 0 and cur not in ancestors:
        b_side.append(cur)
        cur = int(parent[cur])
    lca = cur
    if lca < 0:
        return []
    a_side = []
    cur = int(a)
    while cur != lca:
        a_side.append(cur)
        cur = int(parent[cur])
    a_side.append(lca)
    return a_side + list(reversed(b_side))


def path_length_nm(path_nodes: Sequence[int], pos_nm: np.ndarray) -> float:
    if len(path_nodes) < 2:
        return 0.0
    pts = pos_nm[np.asarray(path_nodes, dtype=int)]
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def descendant_nodes(base: int, children) -> Set[int]:
    out = set()
    q = deque([int(base)])
    while q:
        u = q.popleft()
        if u in out:
            continue
        out.add(u)
        q.extend(int(v) for v in children[u])
    return out


def multisource_dijkstra(adjacency, sources, allowed_nodes: Set[int]) -> np.ndarray:
    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=float)
    heap = []
    for s in set(map(int, sources)):
        if s not in allowed_nodes:
            continue
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, s))
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, _e, l in adjacency[u]:
            if v not in allowed_nodes:
                continue
            nd = d + float(l)
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, int(v)))
    return dist


# ============================================================
# Load
# ============================================================
if not SKELETON_PATH.exists():
    raise FileNotFoundError(f"Skeleton file not found: {SKELETON_PATH}")
if not SYNAPSE_PATH.exists():
    raise FileNotFoundError(f"Synapse table not found: {SYNAPSE_PATH}")

skeleton = pd.read_parquet(SKELETON_PATH).copy().reset_index(drop=True)
synapses = pd.read_parquet(SYNAPSE_PATH).copy().reset_index(drop=True)

required_skeleton = {"rowId", "x", "y", "z", "radius", "link"}
missing = required_skeleton - set(skeleton.columns)
if missing:
    raise ValueError(f"Skeleton missing columns: {sorted(missing)}")
required_synapse = {
    "bodyId_pre", "bodyId_post",
    "x_pre", "y_pre", "z_pre",
    "x_post", "y_post", "z_post",
}
missing = required_synapse - set(synapses.columns)
if missing:
    raise ValueError(f"Synapse table missing columns: {sorted(missing)}")

skeleton["rowId"] = pd.to_numeric(skeleton["rowId"]).astype(int)
skeleton["link"] = pd.to_numeric(skeleton["link"]).fillna(-1).astype(int)
skeleton["radius"] = pd.to_numeric(skeleton["radius"], errors="coerce").fillna(0.0).astype(float)

n_nodes = len(skeleton)
node_index = {int(rid): i for i, rid in enumerate(skeleton["rowId"])}
if {"x_em", "y_em", "z_em"}.issubset(skeleton.columns):
    node_em_voxels = skeleton[["x_em", "y_em", "z_em"]].to_numpy(float)
else:
    node_em_voxels = skeleton[["x", "y", "z"]].to_numpy(float) * SEG_TO_EM
node_positions_nm = node_em_voxels * EM_RESOLUTION_NM

# ============================================================
# Graph
# ============================================================
adjacency = [[] for _ in range(n_nodes)]
edge_rows = []
for child_idx, row in skeleton.iterrows():
    parent_id = int(row["link"])
    if parent_id <= 0 or parent_id not in node_index:
        continue
    parent_idx = int(node_index[parent_id])
    child_idx = int(child_idx)
    length_nm = float(np.linalg.norm(node_positions_nm[child_idx] - node_positions_nm[parent_idx]))
    e = len(edge_rows)
    edge_rows.append({
        "edge_id": e + 1,
        "u": parent_idx,
        "v": child_idx,
        "parent_id_original": parent_id,
        "child_id_original": int(row["rowId"]),
        "length_nm": length_nm,
    })
    adjacency[parent_idx].append((child_idx, e, length_nm))
    adjacency[child_idx].append((parent_idx, e, length_nm))

edges = pd.DataFrame(edge_rows)
if edges.empty:
    raise ValueError("No valid skeleton edges found")
n_edges = len(edges)
edge_u = edges["u"].to_numpy(int)
edge_v = edges["v"].to_numpy(int)

component_id, components = connected_components(adjacency)

# ============================================================
# Soma/root: max radius near soft soma prediction; otherwise global max radius.
# ============================================================
radii = skeleton["radius"].to_numpy(float)
soma_prediction_seg, soma_prediction_source = discover_soma_prediction(NEURON_DIR)
if soma_prediction_seg is not None:
    soma_prediction_nm = soma_prediction_seg * SEG_TO_EM * EM_RESOLUTION_NM
    d = np.linalg.norm(node_positions_nm - soma_prediction_nm, axis=1)
    nearby = np.where(d <= SOMA_PREDICTION_SEARCH_RADIUS_NM)[0]
    if len(nearby) == 0:
        nearby = np.asarray([int(np.argmin(d))], dtype=int)
    soma_seed = int(nearby[int(np.argmax(radii[nearby]))])
    soma_method = "largest_radius_near_soma_prediction"
    soma_prediction_to_seed_nm = float(d[soma_seed])
else:
    soma_seed = int(np.argmax(radii))
    soma_method = "global_largest_radius"
    soma_prediction_to_seed_nm = None

main_component = int(component_id[soma_seed])
main_nodes = set(map(int, components[main_component]))
parent, parent_edge, depth, tree_order = orient_tree(adjacency, soma_seed, main_nodes)
children = children_from_parent(parent, main_nodes)

# ============================================================
# Attach PRE/POST to closest skeleton NODES. Raw counts only.
# ============================================================
node_tree = cKDTree(node_positions_nm)
node_pre_count = np.zeros(n_nodes, dtype=int)
node_post_count = np.zeros(n_nodes, dtype=int)
attachment_rows = []
pre_positions_nm, post_positions_nm = [], []

for row_idx, row in synapses.iterrows():
    synapse_key = row.get("synapse_key", row_idx)
    if pd.isna(synapse_key):
        synapse_key = row_idx
    for role in ("pre", "post"):
        try:
            if int(row[f"bodyId_{role}"]) != BODY_ID:
                continue
        except Exception:
            continue
        xyz_seg = np.asarray([row[f"x_{role}"], row[f"y_{role}"], row[f"z_{role}"]], dtype=float)
        if not np.all(np.isfinite(xyz_seg)):
            continue
        xyz_nm = xyz_seg * SEG_TO_EM * EM_RESOLUTION_NM
        dist_nm, node_idx = node_tree.query(xyz_nm, k=1)
        node_idx = int(node_idx)
        dist_nm = float(dist_nm)
        attached = dist_nm <= SYNAPSE_MAX_NODE_DISTANCE_NM and node_idx in main_nodes
        attachment_rows.append({
            "synapse_key": str(synapse_key),
            "row_index": int(row_idx),
            "role": role,
            "x_seg": float(xyz_seg[0]), "y_seg": float(xyz_seg[1]), "z_seg": float(xyz_seg[2]),
            "x_nm": float(xyz_nm[0]), "y_nm": float(xyz_nm[1]), "z_nm": float(xyz_nm[2]),
            "nearest_node_index": node_idx,
            "nearest_rowId": int(skeleton.iloc[node_idx]["rowId"]),
            "distance_nm": dist_nm,
            "attachment_status": "attached" if attached else "too_far_or_outside_main_component",
        })
        if not attached:
            continue
        if role == "pre":
            node_pre_count[node_idx] += 1
            pre_positions_nm.append(xyz_nm.copy())
        else:
            node_post_count[node_idx] += 1
            post_positions_nm.append(xyz_nm.copy())

attachments = pd.DataFrame(attachment_rows)
total_pre = int(np.sum(node_pre_count[list(main_nodes)]))
total_post = int(np.sum(node_post_count[list(main_nodes)]))
if total_pre == 0 and total_post == 0:
    raise RuntimeError("No PRE/POST synapses attached to main skeleton component. Check coordinates.")

# ============================================================
# Subtree synapse counts
# ============================================================
subtree_pre = node_pre_count.copy()
subtree_post = node_post_count.copy()
for u in reversed(tree_order):
    p = int(parent[u])
    if p >= 0:
        subtree_pre[p] += subtree_pre[u]
        subtree_post[p] += subtree_post[u]

main_list = np.asarray(sorted(main_nodes), dtype=int)
main_node_tree = cKDTree(node_positions_nm[main_list])

def representative_node(positions_nm: Sequence[np.ndarray]) -> Optional[int]:
    if not positions_nm:
        return None
    median_xyz = np.median(np.asarray(positions_nm, dtype=float), axis=0)
    _d, local_idx = main_node_tree.query(median_xyz, k=1)
    return int(main_list[int(local_idx)])

pre_rep = representative_node(pre_positions_nm)
post_rep = representative_node(post_positions_nm)

def first_base(rep: Optional[int], subtree_target: np.ndarray, total_target: int, capture: float) -> Optional[int]:
    if rep is None or total_target <= 0:
        return None
    minimum_count = min(MIN_TARGET_SYNAPSES_FOR_BASE, total_target)
    cur = int(rep)
    while cur >= 0:
        count = int(subtree_target[cur])
        if count >= minimum_count and count / total_target >= capture:
            return cur
        cur = int(parent[cur])
    return None

selected_capture = None
axon_base = None
dendrite_base = None
for capture in CAPTURE_FRACTIONS:
    a = first_base(pre_rep, subtree_pre, total_pre, capture)
    d = first_base(post_rep, subtree_post, total_post, capture)
    if a is None or d is None or a == d:
        continue
    if is_ancestor(a, d, parent) or is_ancestor(d, a, parent):
        continue
    axon_base, dendrite_base = int(a), int(d)
    selected_capture = float(capture)
    break

science_like_mode = axon_base is not None and dendrite_base is not None
node_label = np.asarray([None] * n_nodes, dtype=object)
node_reason = np.asarray(["unassigned"] * n_nodes, dtype=object)
soma_nodes, axon_nodes, dendrite_nodes, cbf_nodes, cable_nodes = set(), set(), set(), set(), set()
main_bifurcation = None
fallback_info = {}

# ============================================================
# Main Science-like mode
# ============================================================
if science_like_mode:
    axon_nodes = descendant_nodes(axon_base, children)
    dendrite_nodes = descendant_nodes(dendrite_base, children)
    if axon_nodes & dendrite_nodes:
        raise RuntimeError("Unexpected axon/dendrite subtree overlap")

    for n in axon_nodes:
        node_label[n] = "axon"
        node_reason[n] = "distal_subtree_of_axon_base"
    for n in dendrite_nodes:
        node_label[n] = "dendrite"
        node_reason[n] = "distal_subtree_of_dendrite_base"

    ad_path = path_between_nodes(axon_base, dendrite_base, parent)
    # internal path nodes = cable; endpoints remain A/D
    for n in ad_path[1:-1]:
        cable_nodes.add(int(n))
        node_label[n] = "connecting_cable"
        node_reason[n] = "path_between_arbor_bases"

    if ad_path:
        main_bifurcation = min(ad_path, key=lambda n: int(depth[n]))

    # If soma is itself on the A-D path, do not create a CBF; soma sits at the split.
    soma_nodes.add(int(soma_seed))
    node_label[soma_seed] = "soma"
    node_reason[soma_seed] = "largest_radius_soma_root"

    initial_cbf_path = []
    if main_bifurcation is not None and main_bifurcation != soma_seed:
        root_to_bif = list(reversed(path_to_root(main_bifurcation, parent)))
        if root_to_bif and root_to_bif[0] == soma_seed and root_to_bif[-1] == main_bifurcation:
            initial_cbf_path = root_to_bif[1:-1]

    # Science-like soma boundary: median radius of initial CBF, then walk outward
    # from max-radius soma node until radius drops below that median.
    cbf_median_radius = None
    if initial_cbf_path:
        vals = radii[np.asarray(initial_cbf_path, dtype=int)]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            cbf_median_radius = float(np.median(vals))
            full_path = [soma_seed] + initial_cbf_path
            for n in full_path[1:]:
                if float(radii[n]) < cbf_median_radius:
                    break
                soma_nodes.add(int(n))
                node_label[n] = "soma"
                node_reason[n] = "science_like_soma_radius_boundary"

        for n in initial_cbf_path:
            if n not in soma_nodes and node_label[n] is None:
                cbf_nodes.add(int(n))
                node_label[n] = "cell_body_fiber"
                node_reason[n] = "soma_to_main_bifurcation_path"

    if main_bifurcation is not None and main_bifurcation not in soma_nodes and main_bifurcation not in axon_nodes and main_bifurcation not in dendrite_nodes:
        cable_nodes.add(int(main_bifurcation))
        node_label[main_bifurcation] = "connecting_cable"
        node_reason[main_bifurcation] = "main_bifurcation"

# ============================================================
# Fallback: connected PRE-core / POST-core split, still no ambiguous
# ============================================================
else:
    cbf_median_radius = None
    if pre_rep is not None and post_rep is not None and pre_rep != post_rep:
        dist_pre = multisource_dijkstra(adjacency, [pre_rep], main_nodes)
        dist_post = multisource_dijkstra(adjacency, [post_rep], main_nodes)
        core_path = path_between_nodes(pre_rep, post_rep, parent)

        for n in main_nodes:
            if n == soma_seed:
                continue
            if dist_pre[n] <= dist_post[n]:
                node_label[n] = "axon"
                node_reason[n] = "fallback_nearer_pre_core"
            else:
                node_label[n] = "dendrite"
                node_reason[n] = "fallback_nearer_post_core"

        for n in core_path:
            if n != soma_seed and abs(float(dist_pre[n] - dist_post[n])) <= FALLBACK_CABLE_HALF_WIDTH_NM:
                cable_nodes.add(int(n))
                node_label[n] = "connecting_cable"
                node_reason[n] = "fallback_AD_boundary_band"

        soma_nodes.add(int(soma_seed))
        node_label[soma_seed] = "soma"
        node_reason[soma_seed] = "fallback_soma_root"

        if cable_nodes:
            main_bifurcation = min(cable_nodes, key=lambda n: int(depth[n]))
            if main_bifurcation != soma_seed:
                root_to_bif = list(reversed(path_to_root(main_bifurcation, parent)))
                for n in root_to_bif[1:-1]:
                    if n not in cable_nodes:
                        cbf_nodes.add(int(n))
                        node_label[n] = "cell_body_fiber"
                        node_reason[n] = "fallback_soma_to_boundary"
        fallback_info = {"mode": "two_core_voronoi"}

    elif pre_rep is not None:
        for n in main_nodes:
            node_label[n] = "soma" if n == soma_seed else "axon"
            node_reason[n] = "pre_only"
        soma_nodes.add(int(soma_seed))
        fallback_info = {"mode": "pre_only"}

    elif post_rep is not None:
        for n in main_nodes:
            node_label[n] = "soma" if n == soma_seed else "dendrite"
            node_reason[n] = "post_only"
        soma_nodes.add(int(soma_seed))
        fallback_info = {"mode": "post_only"}
    else:
        raise RuntimeError("Could not construct PRE/POST representatives")

# ============================================================
# Side branches: inherit the compartment they attach to.
# This is the key continuity rule. No local ambiguous islands.
# ============================================================
def fill_unlabeled_components_by_boundary():
    unlabeled = {int(n) for n in main_nodes if node_label[n] is None}
    seen = set()
    stats = {"components": 0, "nodes": 0}
    priority = {"axon": 0, "dendrite": 1, "connecting_cable": 2, "cell_body_fiber": 3, "soma": 4}

    for start in list(unlabeled):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        boundary = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _e, _l in adjacency[u]:
                if v not in main_nodes:
                    continue
                if node_label[v] is None:
                    if v not in seen:
                        seen.add(int(v))
                        q.append(int(v))
                else:
                    boundary.append(str(node_label[v]))
        if boundary:
            counts = {lab: boundary.count(lab) for lab in set(boundary)}
            m = max(counts.values())
            tied = [lab for lab, c in counts.items() if c == m]
            target = sorted(tied, key=lambda x: priority.get(x, 99))[0]
        else:
            target = "soma"
        for n in comp:
            node_label[n] = target
            node_reason[n] = f"side_branch_inherits_{target}"
        stats["components"] += 1
        stats["nodes"] += len(comp)
    return stats

inheritance_stats = fill_unlabeled_components_by_boundary()

# Disconnected skeleton components cannot be made connected by labeling.
# Assign deterministic polarity but report them as QC.
global_default = "axon" if total_pre >= total_post else "dendrite"
disconnected_component_stats = []
for cid, comp in enumerate(components):
    if cid == main_component:
        continue
    pre = int(np.sum(node_pre_count[np.asarray(comp, dtype=int)]))
    post = int(np.sum(node_post_count[np.asarray(comp, dtype=int)]))
    target = "axon" if pre > post else "dendrite" if post > pre else global_default
    for n in comp:
        node_label[n] = target
        node_reason[n] = "disconnected_component_polarity"
    disconnected_component_stats.append({"component_id": cid, "nodes": len(comp), "pre": pre, "post": post, "assigned": target})

for n in range(n_nodes):
    if node_label[n] is None:
        node_label[n] = global_default
        node_reason[n] = "defensive_fill"

# ============================================================
# Edges inherit child-node labels on rooted tree; other edges use endpoints.
# ============================================================
edge_label = np.asarray([""] * n_edges, dtype=object)
edge_reason = np.asarray([""] * n_edges, dtype=object)
for n in main_nodes:
    e = int(parent_edge[n])
    if e >= 0:
        edge_label[e] = str(node_label[n])
        edge_reason[e] = "child_node_label"
for e in range(n_edges):
    if edge_label[e]:
        continue
    u, v = int(edge_u[e]), int(edge_v[e])
    lu, lv = str(node_label[u]), str(node_label[v])
    if lu == lv:
        edge_label[e] = lu
    elif "connecting_cable" in {lu, lv}:
        edge_label[e] = "connecting_cable"
    elif "cell_body_fiber" in {lu, lv}:
        edge_label[e] = "cell_body_fiber"
    elif "soma" in {lu, lv}:
        edge_label[e] = "soma"
    else:
        edge_label[e] = lv
    edge_reason[e] = "endpoint_transition"

# ============================================================
# Connectivity QC in main skeleton
# ============================================================
def label_component_count(label: str):
    selected = {n for n in main_nodes if str(node_label[n]) == label}
    if not selected:
        return 0, 0.0, []
    seen, sizes = set(), []
    for start in list(selected):
        if start in seen:
            continue
        q = deque([start]); seen.add(start); size = 0
        while q:
            u = q.popleft(); size += 1
            for v, _e, _l in adjacency[u]:
                if v in selected and v not in seen:
                    seen.add(int(v)); q.append(int(v))
        sizes.append(size)
    total = sum(sizes)
    return len(sizes), (max(sizes) / total if total else 0.0), sizes

connectivity_qc = {}
for label in COMPARTMENT_CODES:
    c, f, sizes = label_component_count(label)
    connectivity_qc[label] = {
        "components_in_main_skeleton": int(c),
        "largest_component_node_fraction": float(f),
        "component_sizes": list(map(int, sizes)),
    }

# ============================================================
# Annotated tables
# ============================================================
annotated = skeleton.copy()
annotated["compartment"] = node_label
annotated["compartment_code"] = [COMPARTMENT_CODES[str(x)] for x in node_label]
annotated["compartment_reason"] = node_reason
annotated["pre_count_at_node"] = node_pre_count
annotated["post_count_at_node"] = node_post_count
annotated["subtree_pre_count"] = subtree_pre
annotated["subtree_post_count"] = subtree_post
annotated["tree_depth_from_soma"] = depth

edges["compartment"] = edge_label
edges["compartment_code"] = [COMPARTMENT_CODES[str(x)] for x in edge_label]
edges["compartment_reason"] = edge_reason

# ============================================================
# Summary helpers
# ============================================================
def base_diag(base, target):
    if base is None:
        return None
    pre, post = int(subtree_pre[base]), int(subtree_post[base])
    return {
        "rowId": int(skeleton.iloc[base]["rowId"]),
        "node_index": int(base),
        "depth": int(depth[base]),
        "subtree_pre": pre,
        "subtree_post": post,
        "pre_fraction_of_all_pre": pre / total_pre if total_pre else None,
        "post_fraction_of_all_post": post / total_post if total_post else None,
        "target": target,
    }

axon_base_diag = base_diag(axon_base, "pre/axon")
dendrite_base_diag = base_diag(dendrite_base, "post/dendrite")
final_ad_path = path_between_nodes(axon_base, dendrite_base, parent) if science_like_mode else []

# ============================================================
# Save
# ============================================================
SKELETON_DIR.mkdir(parents=True, exist_ok=True)
output_skeleton = SKELETON_DIR / f"{BODY_ID}_annotated_skeleton_v11_science_like.parquet"
output_edges = SKELETON_DIR / f"{BODY_ID}_annotated_edges_v11_science_like.parquet"
output_attachments = SKELETON_DIR / f"{BODY_ID}_synapse_node_attachments_v11_science_like.parquet"
output_swc = SKELETON_DIR / f"{BODY_ID}_annotated_v11_science_like.swc"
output_summary = SKELETON_DIR / f"{BODY_ID}_compartment_summary_v11_science_like.json"

annotated.to_parquet(output_skeleton, index=False)
edges.to_parquet(output_edges, index=False)
attachments.to_parquet(output_attachments, index=False)

with output_swc.open("w") as f:
    f.write("# V11 codes: soma=1 axon=2 dendrite=3 cell_body_fiber=4 connecting_cable=5\n")
    f.write("# No ambiguous class. PRE/POST locate arbors; topology defines compartments.\n")
    for row in annotated.itertuples(index=False):
        f.write(
            f"{int(row.rowId)} {int(row.compartment_code)} "
            f"{float(row.x):.6f} {float(row.y):.6f} {float(row.z):.6f} "
            f"{float(row.radius):.6f} {int(row.link)}\n"
        )

summary = {
    "bodyId": BODY_ID,
    "version": "V11_science_like_connected",
    "mode": "science_like_subtree_arbor" if science_like_mode else "connected_fallback",
    "principle": "PRE/POST locate large arbor bases; topology creates connected compartments; no ambiguous class.",
    "selected_capture_fraction": selected_capture,
    "soma": {
        "rowId": int(skeleton.iloc[soma_seed]["rowId"]),
        "method": soma_method,
        "prediction_source": soma_prediction_source,
        "prediction_to_seed_nm": soma_prediction_to_seed_nm,
        "cbf_median_radius": cbf_median_radius,
    },
    "synapses": {
        "attached_pre": total_pre,
        "attached_post": total_post,
        "uses_confidence_weighting": False,
        "pre_representative_rowId": int(skeleton.iloc[pre_rep]["rowId"]) if pre_rep is not None else None,
        "post_representative_rowId": int(skeleton.iloc[post_rep]["rowId"]) if post_rep is not None else None,
    },
    "arbor_bases": {
        "axon": axon_base_diag,
        "dendrite": dendrite_base_diag,
        "AD_path_length_nm": path_length_nm(final_ad_path, node_positions_nm) if final_ad_path else None,
        "main_bifurcation_rowId": int(skeleton.iloc[main_bifurcation]["rowId"]) if main_bifurcation is not None else None,
    },
    "node_compartments": annotated["compartment"].value_counts().to_dict(),
    "connectivity_qc": connectivity_qc,
    "side_branch_inheritance": inheritance_stats,
    "fallback": fallback_info,
    "disconnected_components": disconnected_component_stats,
    "outputs": {
        "annotated_skeleton": str(output_skeleton),
        "annotated_edges": str(output_edges),
        "attachments": str(output_attachments),
        "swc": str(output_swc),
    },
}
output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("=" * 80)
print("V11 SCIENCE-LIKE CONNECTED COMPARTMENT ANNOTATION")
print("=" * 80)
print(f"BODY_ID={BODY_ID}")
print(f"mode={summary['mode']}")
print(f"attached PRE={total_pre}, POST={total_post}")
print(f"capture={selected_capture}")
print(f"axon_base={axon_base_diag['rowId'] if axon_base_diag else None}")
print(f"dendrite_base={dendrite_base_diag['rowId'] if dendrite_base_diag else None}")
print("node compartments:", annotated["compartment"].value_counts().to_dict())
print("connectivity QC:")
for label, info in connectivity_qc.items():
    print(f"  {label:18s}: components={info['components_in_main_skeleton']}, largest_fraction={info['largest_component_node_fraction']:.4f}")
print("saved:", output_swc)
