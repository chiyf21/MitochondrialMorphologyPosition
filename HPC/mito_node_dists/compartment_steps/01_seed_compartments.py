from __future__ import annotations

import heapq
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================================================
# Topology-first neuron compartment annotation (V3)
#
# Uses ONLY:
#   1) skeleton topology + radius
#   2) pre/post synapse coordinates and polarity
#
# Main ideas:
#   - classify connected arbors, not individual edges independently
#   - treat predicted somaLocation as a soft prior and correct it using local radius/topology
#   - use subtree-level synapse polarity to find main dendritic/axonal arbors
#   - define connecting cable as the graph path between arbor bases
#   - explicitly identify the cell-body fiber from soma to the main neurite backbone
#   - fill ambiguous gaps only along non-branching segments using continuity
#   - use geodesic (graph) support only for ambiguous residual subtrees
#   - allow soma-missing / ambiguous cases
#
# Edge labels are authoritative. Node/SWC labels are derived from incident edges.
# ============================================================

BODY_ID = int(sys.argv[1]) if len(sys.argv) > 1 else int(globals().get("BODY_ID", 107583207))
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

# Fish2 coordinates: segmentation voxel -> EM voxel -> nm.
SEG_TO_EM = np.asarray([2.0, 2.0, 0.5], dtype=float)
EM_RESOLUTION_NM = np.asarray([8.0, 8.0, 30.0], dtype=float)

# ---------- Synapse attachment ----------
ATTACHMENT_MAX_DISTANCE_NM = 2500.0
ATTACHMENT_CANDIDATES = 32

# ---------- Soma detection ----------
# If no curated somaLocation is available, accept the largest-radius node as soma
# only when it is a strong radius outlier. This prevents forcing a soma into
# cropped neurons that do not contain the cell body.
SOMA_MIN_MAX_TO_MEDIAN_RATIO = 2.5
SOMA_MIN_ROBUST_Z = 5.0
# A predicted soma coordinate is only a prior. Search nearby skeleton nodes and
# prefer a node that is both close to the prediction and locally thick.
SOMA_PREDICTION_SEARCH_RADIUS_NM = 20000.0
SOMA_PREDICTION_DISTANCE_SIGMA_NM = 6000.0
SOMA_PREDICTION_MAX_ACCEPTED_OFFSET_NM = 20000.0
SOMA_PREDICTION_MIN_CONFIDENCE = 0.30
SOMA_EXPANSION_MAX_GEODESIC_NM = 12000.0
SOMA_EXPANSION_MIN_RADIUS_RATIO_TO_SEED = 0.30
SOMA_EXPANSION_MIN_RADIUS_RATIO_TO_MEDIAN = 1.75

# ---------- Main arbor inference ----------
# Candidate subtree must explain a meaningful fraction of one synapse polarity
# and be reasonably pure.
MIN_ARBOR_CAPTURE = 0.25
MIN_ARBOR_PURITY = 0.65
MIN_ARBOR_SUPPORT = 3.0
TOP_CANDIDATES_PER_CLASS = 80

# ---------- Residual subtree labeling ----------
RESIDUAL_MIN_SUPPORT = 2.0
RESIDUAL_MIN_FRACTION = 0.67

# ---------- Geodesic local support (diagnostic + fallback) ----------
GEODESIC_RADIUS_NM = 8000.0
GEODESIC_SIGMA_NM = 3000.0

COMPARTMENT_CODES = {
    "ambiguous": 0,
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "cell_body_fiber": 4,
    "connecting_cable": 5,
}


# ============================================================
# Helpers
# ============================================================



def find_first_soma_location(obj) -> Optional[np.ndarray]:
    """Recursively find a somaLocation-like XYZ triplet in JSON-like metadata."""
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
    """
    Find an existing predicted soma coordinate without requiring new data.

    Priority:
      1) notebook/global neuron['somaLocation']
      2) JSON metadata already stored under the neuron directory
    """
    external_neuron = globals().get("neuron")
    if isinstance(external_neuron, dict):
        found = find_first_soma_location(external_neuron)
        if found is not None:
            return found, "global_neuron.somaLocation"

    # Search small metadata JSON files only. Ignore our own compartment summaries.
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


def build_nonbranching_segments(
    adjacency: Sequence[List[Tuple[int, int, float]]],
    soma_nodes: Set[int],
    n_edges: int,
) -> List[Tuple[List[int], Tuple[int, int]]]:
    """Split a tree/forest into maximal non-branching edge segments."""
    degree = np.asarray([len(x) for x in adjacency], dtype=int)
    stop_nodes = {i for i, d in enumerate(degree) if d != 2} | set(map(int, soma_nodes))
    visited: Set[int] = set()
    segments: List[Tuple[List[int], Tuple[int, int]]] = []

    for start_node in sorted(stop_nodes):
        for nbr, edge_idx, _length in adjacency[start_node]:
            if edge_idx in visited:
                continue
            seg = [int(edge_idx)]
            visited.add(int(edge_idx))
            prev = int(start_node)
            cur = int(nbr)
            while cur not in stop_nodes:
                nxt = [(v, e, l) for v, e, l in adjacency[cur] if int(e) not in visited]
                if not nxt:
                    break
                # In a tree, a non-stop degree-2 node has exactly one forward edge.
                v, e, _l = nxt[0]
                seg.append(int(e))
                visited.add(int(e))
                prev, cur = cur, int(v)
            segments.append((seg, (int(start_node), int(cur))))

    # Defensive coverage for malformed components/cycles.
    for e in range(n_edges):
        if e not in visited:
            segments.append(([int(e)], (-1, -1)))
            visited.add(int(e))
    return segments


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, direction) / denominator)
    fraction = min(1.0, max(0.0, fraction))
    closest = start + fraction * direction
    return float(np.linalg.norm(point - closest))


def robust_zscore_of_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 5:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return 0.0
    return float(0.67448975 * (np.max(values) - median) / mad)


def build_node_path(parent: np.ndarray, a: int, b: int) -> List[int]:
    """Return node path a -> b in a rooted tree using parent pointers."""
    ancestors_a = {}
    cur = int(a)
    dist = 0
    while cur >= 0:
        ancestors_a[cur] = dist
        cur = int(parent[cur])
        dist += 1

    path_b = []
    cur = int(b)
    while cur not in ancestors_a and cur >= 0:
        path_b.append(cur)
        cur = int(parent[cur])
    lca = cur
    if lca < 0:
        return []

    path_a = []
    cur = int(a)
    while cur != lca:
        path_a.append(cur)
        cur = int(parent[cur])
    path_a.append(lca)
    return path_a + list(reversed(path_b))


def subtree_disjoint(tin: np.ndarray, tout: np.ndarray, a: int, b: int) -> bool:
    def is_ancestor(u: int, v: int) -> bool:
        return int(tin[u]) <= int(tin[v]) and int(tout[v]) <= int(tout[u])

    return not is_ancestor(a, b) and not is_ancestor(b, a)


def geodesic_distance_from_sources(
    adjacency: Sequence[List[Tuple[int, int, float]]],
    sources: Iterable[int],
    max_distance_nm: float,
) -> np.ndarray:
    """Shortest graph distance from a set of source nodes, truncated at max distance."""
    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=float)
    heap: List[Tuple[float, int]] = []
    for s in sources:
        s = int(s)
        if 0 <= s < n and dist[s] > 0:
            dist[s] = 0.0
            heapq.heappush(heap, (0.0, s))
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u] or d > max_distance_nm:
            continue
        for v, _edge_idx, length_nm in adjacency[u]:
            nd = d + float(length_nm)
            if nd < dist[v] and nd <= max_distance_nm:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def add_local_geodesic_support(
    adjacency: Sequence[List[Tuple[int, int, float]]],
    sources: Iterable[int],
    source_weight: float,
    output: np.ndarray,
    radius_nm: float,
    sigma_nm: float,
) -> None:
    """
    Add one source's Gaussian support along the skeleton graph.

    Unlike Euclidean smoothing, this cannot jump between two neurites that are
    spatially close but far apart along the neuron. The search is truncated and
    stores distances only for visited nodes, avoiding an O(N) allocation per source.
    """
    if source_weight <= 0:
        return
    best: Dict[int, float] = {}
    heap: List[Tuple[float, int]] = []
    for s in sources:
        s = int(s)
        if s not in best or 0.0 < best[s]:
            best[s] = 0.0
            heapq.heappush(heap, (0.0, s))
    while heap:
        d, u = heapq.heappop(heap)
        if d != best.get(u) or d > radius_nm:
            continue
        output[u] += float(source_weight) * math.exp(-0.5 * (d / sigma_nm) ** 2)
        for v, _edge_idx, length_nm in adjacency[u]:
            nd = d + float(length_nm)
            if nd > radius_nm:
                continue
            if nd < best.get(v, np.inf):
                best[v] = nd
                heapq.heappush(heap, (nd, v))


# ============================================================
# Load data
# ============================================================
if not SKELETON_PATH.exists():
    raise FileNotFoundError(f"Skeleton file not found: {SKELETON_PATH}")
if not SYNAPSE_PATH.exists():
    raise FileNotFoundError(f"Synapse table not found: {SYNAPSE_PATH}")

skeleton = pd.read_parquet(SKELETON_PATH).copy().reset_index(drop=True)
synapses = pd.read_parquet(SYNAPSE_PATH).copy().reset_index(drop=True)

required_skeleton = {"rowId", "x", "y", "z", "radius", "link"}
missing_skeleton = required_skeleton - set(skeleton.columns)
if missing_skeleton:
    raise ValueError(f"Skeleton is missing columns: {sorted(missing_skeleton)}")

required_synapse = {
    "bodyId_pre", "bodyId_post",
    "x_pre", "y_pre", "z_pre",
    "x_post", "y_post", "z_post",
}
missing_synapse = required_synapse - set(synapses.columns)
if missing_synapse:
    raise ValueError(f"Synapse table is missing columns: {sorted(missing_synapse)}")

skeleton["rowId"] = pd.to_numeric(skeleton["rowId"]).astype(int)
skeleton["link"] = pd.to_numeric(skeleton["link"]).fillna(-1).astype(int)
skeleton["radius"] = pd.to_numeric(skeleton["radius"], errors="coerce").fillna(0.0).astype(float)

node_index: Dict[int, int] = {int(row_id): i for i, row_id in enumerate(skeleton["rowId"])}

if {"x_em", "y_em", "z_em"}.issubset(skeleton.columns):
    node_em_voxels = skeleton[["x_em", "y_em", "z_em"]].to_numpy(float)
else:
    node_em_voxels = skeleton[["x", "y", "z"]].to_numpy(float) * SEG_TO_EM
node_positions_nm = node_em_voxels * EM_RESOLUTION_NM
n_nodes = len(skeleton)

# ============================================================
# Build undirected skeleton tree/forest
# ============================================================
edge_rows = []
adjacency: List[List[Tuple[int, int, float]]] = [[] for _ in range(n_nodes)]

for child_index, row in skeleton.iterrows():
    child_id = int(row["rowId"])
    parent_id = int(row["link"])
    if parent_id <= 0 or parent_id not in node_index:
        continue
    parent_index = int(node_index[parent_id])
    start = node_positions_nm[parent_index]
    end = node_positions_nm[child_index]
    length_nm = float(np.linalg.norm(end - start))
    edge_idx = len(edge_rows)
    edge_rows.append({
        "edge_id": edge_idx + 1,
        "parent_id_original": parent_id,
        "child_id_original": child_id,
        "u": parent_index,
        "v": int(child_index),
        "length_nm": length_nm,
        "midpoint_x_nm": float((start[0] + end[0]) / 2.0),
        "midpoint_y_nm": float((start[1] + end[1]) / 2.0),
        "midpoint_z_nm": float((start[2] + end[2]) / 2.0),
    })
    adjacency[parent_index].append((int(child_index), edge_idx, length_nm))
    adjacency[int(child_index)].append((parent_index, edge_idx, length_nm))

edges = pd.DataFrame(edge_rows)
if edges.empty:
    raise ValueError("No valid skeleton edges were found.")

n_edges = len(edges)
edge_u = edges["u"].to_numpy(int)
edge_v = edges["v"].to_numpy(int)
edge_length = edges["length_nm"].to_numpy(float)
edge_start = node_positions_nm[edge_u]
edge_end = node_positions_nm[edge_v]
edge_midpoints = edges[["midpoint_x_nm", "midpoint_y_nm", "midpoint_z_nm"]].to_numpy(float)
edge_tree = cKDTree(edge_midpoints)

# ============================================================
# Attach synapse endpoints to nearest skeleton edge
# ============================================================
endpoint_rows = []
for row_index, row in synapses.iterrows():
    synapse_key = row.get("synapse_key", row_index)
    if pd.isna(synapse_key):
        synapse_key = row_index
    for role in ("pre", "post"):
        try:
            body_value = int(row[f"bodyId_{role}"])
        except Exception:
            continue
        if body_value != BODY_ID:
            continue
        xyz = np.asarray([row[f"x_{role}"], row[f"y_{role}"], row[f"z_{role}"]], dtype=float)
        if not np.all(np.isfinite(xyz)):
            continue
        confidence = row.get(f"confidence_{role}", 1.0)
        confidence = 1.0 if pd.isna(confidence) else max(0.0, float(confidence))
        endpoint_rows.append({
            "synapse_key": str(synapse_key),
            "row_index": int(row_index),
            "role": role,
            "x_seg": float(xyz[0]),
            "y_seg": float(xyz[1]),
            "z_seg": float(xyz[2]),
            "confidence": confidence,
        })

endpoints = pd.DataFrame(endpoint_rows)
if not endpoints.empty:
    endpoints = endpoints.drop_duplicates(
        subset=["synapse_key", "role", "x_seg", "y_seg", "z_seg"]
    ).reset_index(drop=True)

edge_pre_weight = np.zeros(n_edges, dtype=float)
edge_post_weight = np.zeros(n_edges, dtype=float)
edge_pre_count = np.zeros(n_edges, dtype=int)
edge_post_count = np.zeros(n_edges, dtype=int)
attachment_rows = []

if not endpoints.empty:
    for endpoint in endpoints.itertuples(index=False):
        seg_position = np.asarray([endpoint.x_seg, endpoint.y_seg, endpoint.z_seg], dtype=float)
        point_nm = seg_position * SEG_TO_EM * EM_RESOLUTION_NM
        k = min(ATTACHMENT_CANDIDATES, n_edges)
        _, candidate_indices = edge_tree.query(point_nm, k=k)
        candidate_indices = np.atleast_1d(candidate_indices).astype(int)
        exact = np.asarray([
            point_segment_distance(point_nm, edge_start[i], edge_end[i])
            for i in candidate_indices
        ])
        best_pos = int(np.argmin(exact))
        edge_idx = int(candidate_indices[best_pos])
        best_distance = float(exact[best_pos])
        attached = best_distance <= ATTACHMENT_MAX_DISTANCE_NM

        attachment_rows.append({
            "synapse_key": endpoint.synapse_key,
            "row_index": int(endpoint.row_index),
            "role": endpoint.role,
            "x_seg": float(endpoint.x_seg),
            "y_seg": float(endpoint.y_seg),
            "z_seg": float(endpoint.z_seg),
            "x_em_nm": float(point_nm[0]),
            "y_em_nm": float(point_nm[1]),
            "z_em_nm": float(point_nm[2]),
            "confidence": float(endpoint.confidence),
            "edge_id": int(edge_idx + 1) if attached else None,
            "attachment_distance_nm": best_distance,
            "attachment_status": "attached" if attached else "too_far_from_skeleton",
        })
        if not attached:
            continue
        if endpoint.role == "pre":
            edge_pre_count[edge_idx] += 1
            edge_pre_weight[edge_idx] += float(endpoint.confidence)
        else:
            edge_post_count[edge_idx] += 1
            edge_post_weight[edge_idx] += float(endpoint.confidence)

attachments = pd.DataFrame(attachment_rows)

# ============================================================
# Detect soma: predicted coordinate is a SOFT prior, not ground truth
# ============================================================
radii = skeleton["radius"].to_numpy(float)
positive_radii = radii[np.isfinite(radii) & (radii > 0)]
median_radius = float(np.median(positive_radii)) if len(positive_radii) else 0.0
max_radius = float(np.max(positive_radii)) if len(positive_radii) else 0.0
max_to_median = max_radius / median_radius if median_radius > 0 else np.inf
robust_z = robust_zscore_of_max(positive_radii)

soma_seed: Optional[int] = None
soma_method = "none"
soma_confidence = 0.0
soma_prediction_source = "none"
soma_prediction_seg: Optional[np.ndarray] = None
soma_prediction_nm: Optional[np.ndarray] = None
soma_prediction_to_seed_nm: Optional[float] = None

# Existing soma coordinate may itself be predicted/inexact. Use it only to define
# a search neighborhood, then snap to the most plausible thick skeleton node.
soma_prediction_seg, soma_prediction_source = discover_soma_prediction(NEURON_DIR)
if soma_prediction_seg is not None:
    soma_prediction_nm = soma_prediction_seg * SEG_TO_EM * EM_RESOLUTION_NM
    node_distance_to_prediction = np.linalg.norm(node_positions_nm - soma_prediction_nm, axis=1)
    nearby = np.where(node_distance_to_prediction <= SOMA_PREDICTION_SEARCH_RADIUS_NM)[0]
    if len(nearby) == 0:
        nearby = np.asarray([int(np.argmin(node_distance_to_prediction))], dtype=int)

    med = max(median_radius, 1e-9)
    radius_term = np.log1p(np.maximum(radii[nearby], 0.0) / med)
    distance_term = np.exp(
        -0.5 * (node_distance_to_prediction[nearby] / SOMA_PREDICTION_DISTANCE_SIGMA_NM) ** 2
    )
    # The coordinate is a prior, but radius is independent anatomical evidence.
    score = 1.25 * distance_term + 0.75 * radius_term
    predicted_candidate = int(nearby[int(np.argmax(score))])
    offset = float(node_distance_to_prediction[predicted_candidate])
    seed_radius_ratio = float(radii[predicted_candidate] / med) if med > 0 else 0.0
    pred_conf = 0.60 * math.exp(-offset / max(SOMA_PREDICTION_DISTANCE_SIGMA_NM, 1.0))
    pred_conf += 0.40 * min(1.0, max(0.0, (seed_radius_ratio - 1.0) / 2.0))

    if offset <= SOMA_PREDICTION_MAX_ACCEPTED_OFFSET_NM and pred_conf >= SOMA_PREDICTION_MIN_CONFIDENCE:
        soma_seed = predicted_candidate
        soma_method = "soft_soma_prediction_plus_radius"
        soma_confidence = float(min(1.0, pred_conf))
        soma_prediction_to_seed_nm = offset

# Independent fallback/competitor: a very strong global radius outlier.
radius_candidate: Optional[int] = None
radius_confidence = 0.0
if len(positive_radii):
    candidate = int(np.nanargmax(radii))
    if max_to_median >= SOMA_MIN_MAX_TO_MEDIAN_RATIO and robust_z >= SOMA_MIN_ROBUST_Z:
        radius_candidate = candidate
        ratio_margin = min(1.0, max(0.0, (max_to_median - SOMA_MIN_MAX_TO_MEDIAN_RATIO) / SOMA_MIN_MAX_TO_MEDIAN_RATIO))
        z_margin = min(1.0, max(0.0, (robust_z - SOMA_MIN_ROBUST_Z) / SOMA_MIN_ROBUST_Z))
        radius_confidence = float(0.50 + 0.25 * ratio_margin + 0.25 * z_margin)

# Radius-only detection is primarily a fallback when the predicted soma prior
# cannot be reconciled with the skeleton. If a predicted seed was accepted, keep
# it; the local search above has already allowed radius evidence to move the seed
# away from the raw predicted coordinate.
if radius_candidate is not None and soma_seed is None:
    soma_seed = int(radius_candidate)
    soma_method = "radius_outlier_fallback" if soma_prediction_seg is not None else "radius_outlier"
    soma_confidence = radius_confidence
    if soma_prediction_nm is not None:
        soma_prediction_to_seed_nm = float(
            np.linalg.norm(node_positions_nm[soma_seed] - soma_prediction_nm)
        )

soma_nodes: Set[int] = set()
if soma_seed is not None:
    seed_radius = max(float(radii[soma_seed]), 1e-12)
    min_soma_radius = max(
        SOMA_EXPANSION_MIN_RADIUS_RATIO_TO_SEED * seed_radius,
        SOMA_EXPANSION_MIN_RADIUS_RATIO_TO_MEDIAN * median_radius,
    )
    # Radius-constrained geodesic flood fill keeps soma as one connected region.
    best_dist = {int(soma_seed): 0.0}
    heap = [(0.0, int(soma_seed))]
    while heap:
        d, u = heapq.heappop(heap)
        if d != best_dist[u] or d > SOMA_EXPANSION_MAX_GEODESIC_NM:
            continue
        if u != soma_seed and radii[u] < min_soma_radius:
            continue
        soma_nodes.add(u)
        for v, _edge_idx, length_nm in adjacency[u]:
            nd = d + float(length_nm)
            if nd > SOMA_EXPANSION_MAX_GEODESIC_NM:
                continue
            if radii[v] < min_soma_radius:
                continue
            if nd < best_dist.get(v, np.inf):
                best_dist[v] = nd
                heapq.heappush(heap, (nd, v))

# ============================================================
# Choose a root and orient the undirected tree
# ============================================================
# Prefer soma seed; otherwise use an original skeleton root. If multiple roots,
# choose the one belonging to the largest connected component by node count.
original_roots = [
    i for i, link in enumerate(skeleton["link"].to_numpy(int))
    if int(link) <= 0 or int(link) not in node_index
]

# Connected components of node graph.
component_id = np.full(n_nodes, -1, dtype=int)
components: List[List[int]] = []
for start in range(n_nodes):
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
                q.append(v)
    components.append(comp)

largest_component = int(np.argmax([len(c) for c in components]))
if soma_seed is not None:
    root = int(soma_seed)
else:
    roots_in_largest = [r for r in original_roots if component_id[r] == largest_component]
    root = int(roots_in_largest[0] if roots_in_largest else components[largest_component][0])

parent = np.full(n_nodes, -1, dtype=int)
parent_edge = np.full(n_nodes, -1, dtype=int)
depth = np.full(n_nodes, -1, dtype=int)
path_distance = np.full(n_nodes, np.nan, dtype=float)
order = []

# Orient every connected component. Main component starts at selected root.
component_roots = [root]
for cid, comp in enumerate(components):
    if cid == component_id[root]:
        continue
    component_roots.append(int(comp[0]))

for r in component_roots:
    if depth[r] >= 0:
        continue
    depth[r] = 0
    path_distance[r] = 0.0
    stack = [r]
    while stack:
        u = stack.pop()
        order.append(u)
        for v, edge_idx, length_nm in adjacency[u]:
            if v == parent[u] or depth[v] >= 0:
                continue
            parent[v] = u
            parent_edge[v] = edge_idx
            depth[v] = depth[u] + 1
            path_distance[v] = path_distance[u] + float(length_nm)
            stack.append(v)

# Euler intervals for subtree containment/disjointness.
tin = np.full(n_nodes, -1, dtype=int)
tout = np.full(n_nodes, -1, dtype=int)
children: List[List[int]] = [[] for _ in range(n_nodes)]
for v in range(n_nodes):
    if parent[v] >= 0:
        children[parent[v]].append(v)

timer = 0
for r in component_roots:
    stack = [(r, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            tin[u] = timer
            timer += 1
            stack.append((u, 1))
            for v in reversed(children[u]):
                stack.append((v, 0))
        else:
            tout[u] = timer - 1

# Re-orient edge -> child node according to rooted tree.
edge_child = np.full(n_edges, -1, dtype=int)
edge_parent_node = np.full(n_edges, -1, dtype=int)
for v in range(n_nodes):
    e = int(parent_edge[v])
    if e >= 0:
        edge_child[e] = v
        edge_parent_node[e] = int(parent[v])

# ============================================================
# Subtree-level pre/post statistics
# ============================================================
node_direct_pre = np.zeros(n_nodes, dtype=float)
node_direct_post = np.zeros(n_nodes, dtype=float)
# Assign each edge's direct synapse evidence to the distal/child node.
for e in range(n_edges):
    child = edge_child[e]
    if child >= 0:
        node_direct_pre[child] += edge_pre_weight[e]
        node_direct_post[child] += edge_post_weight[e]

subtree_pre = node_direct_pre.copy()
subtree_post = node_direct_post.copy()
for u in reversed(order):
    p = parent[u]
    if p >= 0:
        subtree_pre[p] += subtree_pre[u]
        subtree_post[p] += subtree_post[u]

total_pre = float(np.sum(edge_pre_weight))
total_post = float(np.sum(edge_post_weight))

# Main component totals are more relevant than detached skeleton fragments.
main_nodes_mask = component_id == component_id[root]
main_pre = float(sum(node_direct_pre[i] for i in np.where(main_nodes_mask)[0]))
main_post = float(sum(node_direct_post[i] for i in np.where(main_nodes_mask)[0]))
if main_pre > 0:
    total_pre_for_capture = main_pre
else:
    total_pre_for_capture = total_pre
if main_post > 0:
    total_post_for_capture = main_post
else:
    total_post_for_capture = total_post

# Candidate subtree scores.
axon_candidates = []
dendrite_candidates = []
for e in range(n_edges):
    child = int(edge_child[e])
    if child < 0 or component_id[child] != component_id[root]:
        continue
    if child in soma_nodes:
        continue

    pre = float(subtree_pre[child])
    post = float(subtree_post[child])
    support = pre + post
    pre_capture = pre / total_pre_for_capture if total_pre_for_capture > 0 else 0.0
    post_capture = post / total_post_for_capture if total_post_for_capture > 0 else 0.0
    pre_purity = pre / support if support > 0 else 0.0
    post_purity = post / support if support > 0 else 0.0

    ax_score = pre_capture * pre_purity * (1.0 - min(1.0, post_capture))
    de_score = post_capture * post_purity * (1.0 - min(1.0, pre_capture))

    if support >= MIN_ARBOR_SUPPORT and pre_capture >= MIN_ARBOR_CAPTURE and pre_purity >= MIN_ARBOR_PURITY:
        axon_candidates.append((ax_score, e, child, pre_capture, pre_purity, post_capture))
    if support >= MIN_ARBOR_SUPPORT and post_capture >= MIN_ARBOR_CAPTURE and post_purity >= MIN_ARBOR_PURITY:
        dendrite_candidates.append((de_score, e, child, post_capture, post_purity, pre_capture))

axon_candidates.sort(reverse=True, key=lambda x: x[0])
dendrite_candidates.sort(reverse=True, key=lambda x: x[0])
axon_candidates = axon_candidates[:TOP_CANDIDATES_PER_CLASS]
dendrite_candidates = dendrite_candidates[:TOP_CANDIDATES_PER_CLASS]

# Find best disjoint axon/dendrite subtree pair.
best_pair = None
for ax in axon_candidates:
    for de in dendrite_candidates:
        if subtree_disjoint(tin, tout, int(ax[2]), int(de[2])):
            pair_score = float(ax[0] + de[0])
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, ax, de)

edge_label = np.asarray(["ambiguous"] * n_edges, dtype=object)
edge_reason = np.asarray(["unassigned"] * n_edges, dtype=object)

# Soma edges are fixed first.
for e in range(n_edges):
    if edge_u[e] in soma_nodes and edge_v[e] in soma_nodes:
        edge_label[e] = "soma"
        edge_reason[e] = "soma_radius_region"

axon_base_edge = None
dendrite_base_edge = None
axon_base_node = None
dendrite_base_node = None
pair_confidence = "failed"

main_backbone_nodes: List[int] = []
main_backbone_edges: List[int] = []
cell_body_attachment_node: Optional[int] = None
cell_body_fiber_edges: List[int] = []
connecting_cable_edges: List[int] = []

if best_pair is not None:
    _pair_score, ax, de = best_pair
    axon_base_edge = int(ax[1])
    axon_base_node = int(ax[2])
    dendrite_base_edge = int(de[1])
    dendrite_base_node = int(de[2])
    pair_confidence = "high"

    # Label whole connected main arbors first.
    for e in range(n_edges):
        child = int(edge_child[e])
        if child < 0 or edge_label[e] == "soma":
            continue
        if tin[axon_base_node] <= tin[child] <= tout[axon_base_node]:
            edge_label[e] = "axon"
            edge_reason[e] = "main_axon_subtree"
        elif tin[dendrite_base_node] <= tin[child] <= tout[dendrite_base_node]:
            edge_label[e] = "dendrite"
            edge_reason[e] = "main_dendrite_subtree"

    # The path between the two arbor bases defines the main neurite backbone.
    main_backbone_nodes = build_node_path(parent, axon_base_node, dendrite_base_node)
    for i in range(1, len(main_backbone_nodes)):
        a = int(main_backbone_nodes[i - 1])
        b = int(main_backbone_nodes[i])
        if parent[b] == a:
            e = int(parent_edge[b])
        elif parent[a] == b:
            e = int(parent_edge[a])
        else:
            continue
        if e >= 0:
            main_backbone_edges.append(e)

    # If soma is off the axon-dendrite backbone, the unique path from soma to the
    # nearest backbone node is the cell-body fiber. This is topologically distinct
    # from the dendrite-axon connecting cable even though both may be synapse-poor.
    if soma_seed is not None and main_backbone_nodes:
        soma_dist = geodesic_distance_from_sources(adjacency, [soma_seed], np.inf)
        finite_backbone = [n for n in main_backbone_nodes if np.isfinite(soma_dist[n])]
        if finite_backbone:
            cell_body_attachment_node = int(min(finite_backbone, key=lambda n: soma_dist[n]))
            soma_to_backbone_nodes = build_node_path(parent, soma_seed, cell_body_attachment_node)
            backbone_edge_set = set(main_backbone_edges)
            for i in range(1, len(soma_to_backbone_nodes)):
                a = int(soma_to_backbone_nodes[i - 1])
                b = int(soma_to_backbone_nodes[i])
                if parent[b] == a:
                    e = int(parent_edge[b])
                elif parent[a] == b:
                    e = int(parent_edge[a])
                else:
                    continue
                if e < 0 or e in backbone_edge_set:
                    continue
                if edge_label[e] == "ambiguous":
                    edge_label[e] = "cell_body_fiber"
                    edge_reason[e] = "path_soma_to_main_backbone"
                    cell_body_fiber_edges.append(e)

    # Remaining unlabeled portions of the axon-dendrite backbone are the
    # connecting cable. It is therefore a topology-defined path, not a fallback.
    for e in main_backbone_edges:
        if edge_label[e] == "ambiguous":
            edge_label[e] = "connecting_cable"
            edge_reason[e] = "path_between_main_arbors"
            connecting_cable_edges.append(e)

# ============================================================
# Geodesic local pre/post support for diagnostics/fallback
# ============================================================
# Use edge endpoints as graph sources for edges carrying synapses. This avoids
# Euclidean leakage between spatially close but topologically distant branches.
pre_source_nodes = []
post_source_nodes = []
for e in range(n_edges):
    if edge_pre_weight[e] > 0:
        pre_source_nodes.extend([int(edge_u[e]), int(edge_v[e])])
    if edge_post_weight[e] > 0:
        post_source_nodes.extend([int(edge_u[e]), int(edge_v[e])])

# For weighted local support, propagate each synapse-bearing edge's evidence over
# the tree within a geodesic radius. This is O(K * local neighborhood), where K
# is the number of synapse-bearing edges, and is usually tractable for a neuron.
node_geo_pre = np.zeros(n_nodes, dtype=float)
node_geo_post = np.zeros(n_nodes, dtype=float)

source_edges = np.where((edge_pre_weight + edge_post_weight) > 0)[0]
for e in source_edges:
    sources = [int(edge_u[e]), int(edge_v[e])]
    if edge_pre_weight[e] > 0:
        add_local_geodesic_support(
            adjacency, sources, float(edge_pre_weight[e]), node_geo_pre,
            GEODESIC_RADIUS_NM, GEODESIC_SIGMA_NM,
        )
    if edge_post_weight[e] > 0:
        add_local_geodesic_support(
            adjacency, sources, float(edge_post_weight[e]), node_geo_post,
            GEODESIC_RADIUS_NM, GEODESIC_SIGMA_NM,
        )

edge_geo_pre = 0.5 * (node_geo_pre[edge_u] + node_geo_pre[edge_v])
edge_geo_post = 0.5 * (node_geo_post[edge_u] + node_geo_post[edge_v])

# ============================================================
# Branch continuity interpolation (V3)
# ============================================================
# Fill only along maximal non-branching segments. This uses the anatomical prior
# that a single branch should not flicker A-D-A-D because of sparse synapses.
node_to_edges: List[List[int]] = [[] for _ in range(n_nodes)]
for e in range(n_edges):
    node_to_edges[int(edge_u[e])].append(e)
    node_to_edges[int(edge_v[e])].append(e)

segments = build_nonbranching_segments(adjacency, soma_nodes, n_edges)
branch_fill_count = 0
for seg_edges, endpoints_nodes in segments:
    ambiguous_seg = [e for e in seg_edges if edge_label[e] == "ambiguous"]
    if not ambiguous_seg:
        continue

    fixed = {
        str(edge_label[e]) for e in seg_edges
        if edge_label[e] != "ambiguous" and edge_label[e] != "soma"
    }
    fill_label: Optional[str] = None
    reason = ""

    # D-?-D / A-?-A / cable-?-cable / CBF-?-CBF inside one branch.
    if len(fixed) == 1:
        fill_label = next(iter(fixed))
        reason = "same_nonbranching_segment_continuity"
    elif len(fixed) == 0:
        # If the segment itself is entirely ambiguous, inspect labels just beyond
        # its two endpoints. Do not cross a branch point during interpolation.
        boundary = set()
        seg_set = set(seg_edges)
        for node in endpoints_nodes:
            if node < 0:
                continue
            for nbr_e in node_to_edges[node]:
                if nbr_e in seg_set:
                    continue
                lab = str(edge_label[nbr_e])
                if lab not in {"ambiguous", "soma"}:
                    boundary.add(lab)
        if len(boundary) == 1:
            fill_label = next(iter(boundary))
            reason = "same_boundary_label_nonbranching_segment"
        elif boundary == {"axon", "dendrite"}:
            fill_label = "connecting_cable"
            reason = "nonbranching_gap_between_axon_and_dendrite"

    if fill_label is not None:
        for e in ambiguous_seg:
            edge_label[e] = fill_label
            edge_reason[e] = reason
            branch_fill_count += 1

# ============================================================
# Label residual connected components, not individual edges
# ============================================================
unassigned_edges = {e for e in range(n_edges) if edge_label[e] == "ambiguous"}
visited_residual: Set[int] = set()

# Edge adjacency via shared nodes was built above for branch interpolation.
for start_edge in list(unassigned_edges):
    if start_edge in visited_residual:
        continue
    comp_edges = []
    q = deque([start_edge])
    visited_residual.add(start_edge)
    boundary_labels = set()

    while q:
        e = q.popleft()
        comp_edges.append(e)
        for node in (int(edge_u[e]), int(edge_v[e])):
            for nbr_e in node_to_edges[node]:
                if nbr_e == e:
                    continue
                if edge_label[nbr_e] == "ambiguous":
                    if nbr_e not in visited_residual:
                        visited_residual.add(nbr_e)
                        q.append(nbr_e)
                else:
                    boundary_labels.add(str(edge_label[nbr_e]))

    pre = float(np.sum(edge_pre_weight[comp_edges]))
    post = float(np.sum(edge_post_weight[comp_edges]))
    support = pre + post
    # Add a small amount of geodesic context to sparse components.
    geo_pre = float(np.mean(edge_geo_pre[comp_edges])) if comp_edges else 0.0
    geo_post = float(np.mean(edge_geo_post[comp_edges])) if comp_edges else 0.0

    label = "ambiguous"
    reason = "residual_ambiguous"
    if support >= RESIDUAL_MIN_SUPPORT:
        pre_frac = pre / support
        post_frac = post / support
        if pre_frac >= RESIDUAL_MIN_FRACTION:
            label = "axon"
            reason = "residual_component_pre_dominant"
        elif post_frac >= RESIDUAL_MIN_FRACTION:
            label = "dendrite"
            reason = "residual_component_post_dominant"
    else:
        geo_total = geo_pre + geo_post
        if geo_total > 0:
            if geo_pre / geo_total >= RESIDUAL_MIN_FRACTION:
                label = "axon"
                reason = "residual_geodesic_pre_context"
            elif geo_post / geo_total >= RESIDUAL_MIN_FRACTION:
                label = "dendrite"
                reason = "residual_geodesic_post_context"

    # If synapse evidence is weak, inherit a unique anatomical boundary label.
    if label == "ambiguous":
        informative_boundaries = boundary_labels & {"axon", "dendrite", "connecting_cable", "cell_body_fiber"}
        if len(informative_boundaries) == 1:
            label = next(iter(informative_boundaries))
            reason = "residual_inherited_unique_boundary"
        elif informative_boundaries == {"axon", "dendrite"}:
            label = "connecting_cable"
            reason = "residual_between_axon_and_dendrite"

    for e in comp_edges:
        edge_label[e] = label
        edge_reason[e] = reason

# Remaining low-evidence edges leaving the soma are most naturally part of the
# cell-body fiber. Strong local synapse polarity can still override this.
for e in range(n_edges):
    if edge_label[e] != "ambiguous":
        continue
    touches_soma = (edge_u[e] in soma_nodes) or (edge_v[e] in soma_nodes)
    if not touches_soma:
        continue
    support = float(edge_pre_weight[e] + edge_post_weight[e])
    if support >= RESIDUAL_MIN_SUPPORT:
        if edge_pre_weight[e] / support >= RESIDUAL_MIN_FRACTION:
            edge_label[e] = "axon"
            edge_reason[e] = "soma_boundary_strong_pre"
            continue
        if edge_post_weight[e] / support >= RESIDUAL_MIN_FRACTION:
            edge_label[e] = "dendrite"
            edge_reason[e] = "soma_boundary_strong_post"
            continue
    edge_label[e] = "cell_body_fiber"
    edge_reason[e] = "low_evidence_edge_leaving_soma"

# ============================================================
# Derived diagnostics per edge
# ============================================================
edges["pre_count"] = edge_pre_count
edges["post_count"] = edge_post_count
edges["pre_weight"] = edge_pre_weight
edges["post_weight"] = edge_post_weight
edges["geodesic_pre_support"] = edge_geo_pre
edges["geodesic_post_support"] = edge_geo_post
edges["compartment"] = edge_label
edges["label_reason"] = edge_reason

# Oriented-tree diagnostics.
edges["rooted_parent_node_index"] = edge_parent_node
edges["rooted_child_node_index"] = edge_child
edges["subtree_pre_weight"] = [float(subtree_pre[c]) if c >= 0 else np.nan for c in edge_child]
edges["subtree_post_weight"] = [float(subtree_post[c]) if c >= 0 else np.nan for c in edge_child]
edges["subtree_pre_fraction"] = [
    float(subtree_pre[c] / total_pre_for_capture) if c >= 0 and total_pre_for_capture > 0 else 0.0
    for c in edge_child
]
edges["subtree_post_fraction"] = [
    float(subtree_post[c] / total_post_for_capture) if c >= 0 and total_post_for_capture > 0 else 0.0
    for c in edge_child
]

# ============================================================
# Derive node labels from incident edge labels
# ============================================================
node_label = []
node_confidence = []
for i in range(n_nodes):
    if i in soma_nodes:
        node_label.append("soma")
        node_confidence.append(1.0)
        continue

    votes = defaultdict(float)
    for _nbr, e, length_nm in adjacency[i]:
        label = str(edge_label[e])
        # Longer incident segments contribute more to a junction's display label.
        votes[label] += max(float(length_nm), 1.0)
    if not votes:
        node_label.append("ambiguous")
        node_confidence.append(0.0)
        continue
    best_label, best_weight = max(votes.items(), key=lambda kv: kv[1])
    total_weight = float(sum(votes.values()))
    node_label.append(best_label)
    node_confidence.append(float(best_weight / total_weight) if total_weight > 0 else 0.0)

annotated = skeleton.copy()
annotated["compartment"] = node_label
annotated["compartment_code"] = [COMPARTMENT_CODES.get(x, 0) for x in node_label]
annotated["compartment_confidence"] = node_confidence
annotated["distance_from_root_nm"] = path_distance
annotated["tree_depth"] = depth
annotated["geodesic_pre_support"] = node_geo_pre
annotated["geodesic_post_support"] = node_geo_post
annotated["is_soma_node"] = [i in soma_nodes for i in range(n_nodes)]

if soma_seed is not None:
    dist_to_soma = geodesic_distance_from_sources(adjacency, [soma_seed], np.inf)
    annotated["distance_to_soma_geodesic_nm"] = dist_to_soma
else:
    annotated["distance_to_soma_geodesic_nm"] = np.nan


# ============================================================
# Connectivity / polarity diagnostics
# ============================================================
def compartment_edge_diagnostics(label_name: str) -> dict:
    selected = {e for e in range(n_edges) if str(edge_label[e]) == label_name}
    if not selected:
        return {
            "edge_count": 0,
            "connected_components": 0,
            "largest_component_length_fraction": 0.0,
            "total_length_um": 0.0,
            "pre_count": 0,
            "post_count": 0,
            "pre_fraction": None,
            "post_fraction": None,
        }

    seen: Set[int] = set()
    component_lengths = []
    for start in selected:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        length_sum = 0.0
        while q:
            e = q.popleft()
            length_sum += float(edge_length[e])
            for node in (int(edge_u[e]), int(edge_v[e])):
                for nbr_e in node_to_edges[node]:
                    if nbr_e in selected and nbr_e not in seen:
                        seen.add(nbr_e)
                        q.append(nbr_e)
        component_lengths.append(length_sum)

    total_length = float(sum(component_lengths))
    pre_n = int(np.sum(edge_pre_count[list(selected)]))
    post_n = int(np.sum(edge_post_count[list(selected)]))
    syn_total = pre_n + post_n
    return {
        "edge_count": int(len(selected)),
        "connected_components": int(len(component_lengths)),
        "largest_component_length_fraction": (
            float(max(component_lengths) / total_length) if total_length > 0 else 0.0
        ),
        "total_length_um": float(total_length / 1000.0),
        "pre_count": pre_n,
        "post_count": post_n,
        "pre_fraction": float(pre_n / syn_total) if syn_total > 0 else None,
        "post_fraction": float(post_n / syn_total) if syn_total > 0 else None,
    }

compartment_diagnostics = {
    label: compartment_edge_diagnostics(label)
    for label in ("soma", "cell_body_fiber", "dendrite", "connecting_cable", "axon", "ambiguous")
}

# ============================================================
# Save outputs
# ============================================================
output_edges = SKELETON_DIR / f"{BODY_ID}_annotated_edges_v3.parquet"
output_skeleton = SKELETON_DIR / f"{BODY_ID}_annotated_skeleton_v3.parquet"
output_attachments = SKELETON_DIR / f"{BODY_ID}_synapse_skeleton_attachments_v3.parquet"
output_swc = SKELETON_DIR / f"{BODY_ID}_annotated_v3.swc"
output_summary = SKELETON_DIR / f"{BODY_ID}_compartment_summary_v3.json"

edges.to_parquet(output_edges, index=False)
annotated.to_parquet(output_skeleton, index=False)
attachments.to_parquet(output_attachments, index=False)

with output_swc.open("w") as handle:
    handle.write("# compartment codes: ambiguous=0, soma=1, axon=2, dendrite=3, cell_body_fiber=4, connecting_cable=5\n")
    handle.write("# V3 topology-first annotation with soft soma prior + cell-body fiber + branch continuity; parquet edge labels are authoritative.\n")
    for row in annotated.itertuples(index=False):
        handle.write(
            f"{int(row.rowId)} {int(row.compartment_code)} "
            f"{float(row.x):.6f} {float(row.y):.6f} {float(row.z):.6f} "
            f"{float(row.radius):.6f} {int(row.link)}\n"
        )

attached_count = int((attachments["attachment_status"] == "attached").sum()) if len(attachments) else 0

summary = {
    "bodyId": BODY_ID,
    "status": "topology_first_inference_v3",
    "warning": (
        "Compartment labels are inferred from skeleton topology, radius, and pre/post synapse polarity only. "
        "No ROI, raw EM, mesh, mitochondria, or curated axon/dendrite labels are used."
    ),
    "coordinate_space": "EM physical nanometers for distance calculations; original skeleton XYZ retained.",
    "soma": {
        "detected": soma_seed is not None,
        "method": soma_method,
        "confidence": soma_confidence,
        "prediction_source": soma_prediction_source,
        "predicted_soma_seg_xyz": soma_prediction_seg.tolist() if soma_prediction_seg is not None else None,
        "prediction_to_final_seed_nm": soma_prediction_to_seed_nm,
        "seed_rowId": int(skeleton.iloc[soma_seed]["rowId"]) if soma_seed is not None else None,
        "n_soma_nodes": int(len(soma_nodes)),
        "max_radius": max_radius,
        "median_positive_radius": median_radius,
        "max_to_median_ratio": max_to_median,
        "robust_z_of_max": robust_z,
    },
    "main_arbors": {
        "pair_confidence": pair_confidence,
        "axon_base_edge_id": int(axon_base_edge + 1) if axon_base_edge is not None else None,
        "axon_base_rowId": int(skeleton.iloc[axon_base_node]["rowId"]) if axon_base_node is not None else None,
        "dendrite_base_edge_id": int(dendrite_base_edge + 1) if dendrite_base_edge is not None else None,
        "dendrite_base_rowId": int(skeleton.iloc[dendrite_base_node]["rowId"]) if dendrite_base_node is not None else None,
        "n_axon_candidates": int(len(axon_candidates)),
        "n_dendrite_candidates": int(len(dendrite_candidates)),
        "cell_body_attachment_rowId": (
            int(skeleton.iloc[cell_body_attachment_node]["rowId"])
            if cell_body_attachment_node is not None else None
        ),
        "n_cell_body_fiber_edges": int(len(cell_body_fiber_edges)),
        "n_connecting_cable_path_edges": int(len(connecting_cable_edges)),
        "branch_continuity_filled_edges": int(branch_fill_count),
    },
    "parameters": {
        "attachment_max_distance_nm": ATTACHMENT_MAX_DISTANCE_NM,
        "soma_min_max_to_median_ratio": SOMA_MIN_MAX_TO_MEDIAN_RATIO,
        "soma_min_robust_z": SOMA_MIN_ROBUST_Z,
        "soma_prediction_search_radius_nm": SOMA_PREDICTION_SEARCH_RADIUS_NM,
        "soma_prediction_distance_sigma_nm": SOMA_PREDICTION_DISTANCE_SIGMA_NM,
        "soma_prediction_min_confidence": SOMA_PREDICTION_MIN_CONFIDENCE,
        "soma_expansion_max_geodesic_nm": SOMA_EXPANSION_MAX_GEODESIC_NM,
        "min_arbor_capture": MIN_ARBOR_CAPTURE,
        "min_arbor_purity": MIN_ARBOR_PURITY,
        "min_arbor_support": MIN_ARBOR_SUPPORT,
        "residual_min_support": RESIDUAL_MIN_SUPPORT,
        "residual_min_fraction": RESIDUAL_MIN_FRACTION,
        "geodesic_radius_nm": GEODESIC_RADIUS_NM,
        "geodesic_sigma_nm": GEODESIC_SIGMA_NM,
    },
    "counts": {
        "skeleton_nodes": int(n_nodes),
        "skeleton_edges": int(n_edges),
        "connected_components": int(len(components)),
        "synapse_endpoints": int(len(attachments)),
        "attached_endpoints": attached_count,
        "unattached_endpoints": int(len(attachments) - attached_count),
        "total_pre_weight": total_pre,
        "total_post_weight": total_post,
        "edge_compartments": pd.Series(edge_label).value_counts().to_dict(),
        "node_compartments": pd.Series(node_label).value_counts().to_dict(),
    },
    "compartment_diagnostics": compartment_diagnostics,
    "outputs": {
        "annotated_edges_parquet": str(output_edges),
        "annotated_skeleton_parquet": str(output_skeleton),
        "synapse_attachments_parquet": str(output_attachments),
        "annotated_swc": str(output_swc),
    },
}
output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

# ============================================================
# Minimal self-checks
# ============================================================
assert len(annotated) == len(skeleton)
assert len(edges) == n_edges
assert set(annotated["compartment"]).issubset(COMPARTMENT_CODES)
assert set(edges["compartment"]).issubset(COMPARTMENT_CODES)

print(f"BODY_ID={BODY_ID}")
print(f"skeleton={SKELETON_PATH}")
print(f"nodes={n_nodes}, edges={n_edges}, components={len(components)}")
print(f"synapse endpoints={len(attachments)}, attached={attached_count}")
print(
    "soma:",
    {
        "detected": soma_seed is not None,
        "method": soma_method,
        "prediction_source": soma_prediction_source,
        "prediction_offset_nm": round(soma_prediction_to_seed_nm, 1) if soma_prediction_to_seed_nm is not None else None,
        "n_nodes": len(soma_nodes),
        "max/median_radius": round(max_to_median, 3) if np.isfinite(max_to_median) else max_to_median,
        "robust_z": round(robust_z, 3),
    },
)
print(
    "main arbors:",
    {
        "pair_confidence": pair_confidence,
        "axon_base_edge": axon_base_edge + 1 if axon_base_edge is not None else None,
        "dendrite_base_edge": dendrite_base_edge + 1 if dendrite_base_edge is not None else None,
    },
)
print("edge compartments:", pd.Series(edge_label).value_counts().to_dict())
print("node compartments:", pd.Series(node_label).value_counts().to_dict())
print("continuity diagnostics:")
for label, diag in compartment_diagnostics.items():
    print(f"  {label}: {diag}")
print("saved:", output_skeleton)
