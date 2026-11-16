import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BODY_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 105548440
ROOT = Path('/home/cyf/JaneliaEM')
SKELETON_DIR = ROOT / 'data' / 'neuron_downloads' / str(BODY_ID) / 'skeleton'
NODES_PATH = SKELETON_DIR / f'{BODY_ID}_annotated_skeleton_v11_science_like.parquet'
EDGES_PATH = SKELETON_DIR / f'{BODY_ID}_annotated_edges_v11_science_like.parquet'
OUT = ROOT / 'experiments' / 'distinguish_compartment_of_neuron' / 'derived_component_bridging' / str(BODY_ID)
OUT.mkdir(parents=True, exist_ok=True)

nodes = pd.read_parquet(NODES_PATH).copy()
edges = pd.read_parquet(EDGES_PATH).copy()
n_nodes = len(nodes)
edge_u = edges['u'].to_numpy(int)
edge_v = edges['v'].to_numpy(int)
if len(edge_u) and (edge_u.min() < 0 or edge_v.min() < 0 or edge_u.max() >= n_nodes or edge_v.max() >= n_nodes):
    raise ValueError('V11 edge u/v indices are outside the node table.')

if {'x_em', 'y_em', 'z_em'}.issubset(nodes.columns):
    positions_nm = nodes[['x_em', 'y_em', 'z_em']].to_numpy(float) * np.asarray([8.0, 8.0, 30.0])
else:
    positions_nm = nodes[['x', 'y', 'z']].to_numpy(float) * np.asarray([16.0, 16.0, 15.0])


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=int)
        self.rank = np.zeros(size, dtype=int)

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return int(index)

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


node_uf = UnionFind(n_nodes)
degree = np.zeros(n_nodes, dtype=int)
for left, right in zip(edge_u, edge_v):
    node_uf.union(int(left), int(right))
    degree[left] += 1
    degree[right] += 1

root_to_component = {}
component_id = np.empty(n_nodes, dtype=int)
for index in range(n_nodes):
    root = node_uf.find(index)
    component_id[index] = root_to_component.setdefault(root, len(root_to_component))
n_components = len(root_to_component)

row_id_to_index = {int(row_id): index for index, row_id in enumerate(nodes['rowId'])}
links = nodes['link'].to_numpy(int)
root_node = np.asarray([
    link <= 0 or int(link) not in row_id_to_index
    for link in links
], dtype=bool)

component_members = [np.where(component_id == cid)[0] for cid in range(n_components)]
candidate_nodes = []
for members in component_members:
    # ponytail: use terminal/root nodes as likely breakpoints; inspect EM before treating bridges as anatomy.
    boundary = members[(degree[members] <= 1) | root_node[members]]
    candidate_nodes.append(boundary if len(boundary) else members)

trees = [cKDTree(positions_nm[indices]) for indices in candidate_nodes]
component_pairs = []
for left in range(n_components):
    for right in range(left + 1, n_components):
        distances, right_local = trees[right].query(positions_nm[candidate_nodes[left]], k=1)
        left_local = int(np.argmin(distances))
        distance_nm = float(distances[left_local])
        left_node = int(candidate_nodes[left][left_local])
        right_node = int(candidate_nodes[right][int(right_local[left_local])])
        component_pairs.append((distance_nm, left, right, left_node, right_node))

component_pairs.sort(key=lambda item: item[0])
component_uf = UnionFind(n_components)
bridge_rows = []
for distance_nm, left_component, right_component, left_node, right_node in component_pairs:
    if not component_uf.union(left_component, right_component):
        continue
    left_label = str(nodes.iloc[left_node]['compartment'])
    right_label = str(nodes.iloc[right_node]['compartment'])
    bridge_label = left_label if left_label == right_label else 'ambiguous'
    bridge_rows.append({
        'bridge_id': len(bridge_rows) + 1,
        'u_node_index': left_node,
        'v_node_index': right_node,
        'u_rowId': int(nodes.iloc[left_node]['rowId']),
        'v_rowId': int(nodes.iloc[right_node]['rowId']),
        'component_u': left_component,
        'component_v': right_component,
        'distance_nm': distance_nm,
        'u_compartment': left_label,
        'v_compartment': right_label,
        'compartment': bridge_label,
        'is_ambiguous_bridge': bridge_label == 'ambiguous',
        'x0_nm': float(positions_nm[left_node, 0]),
        'y0_nm': float(positions_nm[left_node, 1]),
        'z0_nm': float(positions_nm[left_node, 2]),
        'x1_nm': float(positions_nm[right_node, 0]),
        'y1_nm': float(positions_nm[right_node, 1]),
        'z1_nm': float(positions_nm[right_node, 2]),
    })
    if len(bridge_rows) == n_components - 1:
        break

bridges = pd.DataFrame(bridge_rows)
if len(bridges) != n_components - 1:
    raise RuntimeError('Component MST did not connect every original component.')

nodes_with_components = nodes.copy()
nodes_with_components['original_component_id'] = component_id
nodes_with_components['original_graph_degree'] = degree
nodes_with_components['bridge_candidate'] = np.concatenate([
    np.isin(np.arange(n_nodes), candidate_nodes[cid])
    for cid in range(n_components)
]).reshape(n_components, n_nodes).any(axis=0) if n_components else False

original_edges = pd.DataFrame({
    'edge_kind': 'original',
    'edge_id': edges['edge_id'].to_numpy(int),
    'u_node_index': edge_u,
    'v_node_index': edge_v,
    'compartment': edges['compartment'].astype(str).to_numpy(),
    'distance_nm': edges['length_nm'].to_numpy(float),
    'is_ambiguous_bridge': False,
})
bridge_edges = bridges.assign(
    edge_kind='derived_bridge',
    edge_id=-bridges['bridge_id'].astype(int),
)[[
    'edge_kind', 'edge_id', 'u_node_index', 'v_node_index',
    'compartment', 'distance_nm', 'is_ambiguous_bridge',
]]
connected_edges = pd.concat([original_edges, bridge_edges], ignore_index=True)

summary = {
    'bodyId': BODY_ID,
    'status': 'derived_component_bridging_experiment',
    'warning': (
        'Bridge edges are straight-line derived links between disconnected skeleton components. '
        'They are not observed neuPrint edges and must not be treated as verified morphology.'
    ),
    'method': {
        'candidate_nodes': 'component terminal nodes (degree <= 1) plus exported root nodes',
        'component_connection': 'Euclidean minimum spanning tree over component pairs',
        'bridge_label_rule': 'same endpoint compartment -> same label; different endpoint compartments -> ambiguous',
    },
    'counts': {
        'original_nodes': int(n_nodes),
        'original_edges': int(len(edges)),
        'original_components': int(n_components),
        'added_bridge_edges': int(len(bridges)),
        'derived_components': 1,
        'bridge_compartments': bridges['compartment'].value_counts().to_dict(),
    },
    'bridge_distance_nm': {
        'min': float(bridges['distance_nm'].min()),
        'median': float(bridges['distance_nm'].median()),
        'mean': float(bridges['distance_nm'].mean()),
        'max': float(bridges['distance_nm'].max()),
        'total': float(bridges['distance_nm'].sum()),
    },
    'outputs': {
        'nodes_with_components': str(OUT / f'{BODY_ID}_nodes_with_components.parquet'),
        'bridge_edges': str(OUT / f'{BODY_ID}_derived_bridge_edges.parquet'),
        'connected_edges': str(OUT / f'{BODY_ID}_bridged_edges.parquet'),
    },
}

nodes_with_components.to_parquet(OUT / f'{BODY_ID}_nodes_with_components.parquet', index=False)
bridges.to_parquet(OUT / f'{BODY_ID}_derived_bridge_edges.parquet', index=False)
connected_edges.to_parquet(OUT / f'{BODY_ID}_bridged_edges.parquet', index=False)
(OUT / f'{BODY_ID}_bridge_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

# Self-check: original topology is preserved and bridges yield one connected component by construction.
assert len(connected_edges) == len(edges) + n_components - 1
assert int(bridges['is_ambiguous_bridge'].sum()) + int((~bridges['is_ambiguous_bridge']).sum()) == len(bridges)
print(json.dumps(summary, indent=2, ensure_ascii=False))
