"""Fill V3 bridge-aware ambiguous components from consistent labeled boundaries."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/cyf/JaneliaEM')
EXPERIMENT = ROOT / 'experiments' / 'distinguish_compartment_of_neuron'
SOURCE_ROOT = EXPERIMENT / 'v3_bridge_aware_fill'
RESULT_ROOT = EXPERIMENT / 'results'
OUT_ROOT = EXPERIMENT / 'v3_boundary_component_fill'
VALID = {'axon', 'dendrite', 'connecting_cable', 'cell_body_fiber'}
MIN_CONNECTED_SUPPORT_EDGES = 3


def build_adjacency(pairs, n_nodes):
    adjacency = [[] for _ in range(n_nodes)]
    for edge_id, (u, v) in enumerate(pairs):
        adjacency[u].append(edge_id); adjacency[v].append(edge_id)
    return adjacency


def other(pair, node): return pair[1] if pair[0] == node else pair[0]


def ambiguous_components(labels, pairs, adjacency):
    seen, components = set(), []
    for start, label in enumerate(labels):
        if label != 'ambiguous' or start in seen: continue
        queue, edges, nodes = deque([start]), [], set(); seen.add(start)
        while queue:
            edge = queue.popleft(); edges.append(edge); u, v = pairs[edge]; nodes.update((u, v))
            for node in (u, v):
                for neighbor in adjacency[node]:
                    if labels[neighbor] == 'ambiguous' and neighbor not in seen:
                        seen.add(neighbor); queue.append(neighbor)
        components.append((edges, nodes))
    return sorted(components, key=lambda item: len(item[0]), reverse=True)


def process(body_id):
    source = SOURCE_ROOT / str(body_id)
    if not source.exists(): return {'bodyId': body_id, 'status': 'skipped_no_bridge_aware_source'}
    raw = pd.read_parquet(source / f'{body_id}_annotated_edges_v3_bridge_aware.parquet').copy()
    bridges = pd.read_parquet(source / f'{body_id}_bridge_edges_v3_bridge_aware.parquet').copy()
    node_path = RESULT_ROOT / str(body_id) / f'{body_id}_annotated_skeleton_v3.parquet'
    if not node_path.exists():
        node_path = ROOT / 'data' / 'neuron_downloads' / str(body_id) / 'skeleton' / f'{body_id}_annotated_skeleton_v3.parquet'
    nodes = pd.read_parquet(node_path)
    raw_labels = raw.compartment_bridge_aware.astype(str).to_numpy(object)
    bridge_labels = bridges.compartment_bridge_aware.astype(str).to_numpy(object)
    labels = np.concatenate([raw_labels, bridge_labels])
    reasons = np.asarray(['inherited_bridge_aware_fill'] * len(labels), dtype=object)
    pairs = list(zip(raw.u.astype(int), raw.v.astype(int))) + list(zip(bridges.u_node_index.astype(int), bridges.v_node_index.astype(int)))
    adjacency = build_adjacency(pairs, len(nodes))
    soma_nodes = set(np.flatnonzero(nodes.compartment.astype(str).to_numpy() == 'soma'))
    total_filled, passes = 0, 0

    while True:
        passes += 1; changed = 0; run_cache = {}
        for component_edges, component_nodes in ambiguous_components(labels, pairs, adjacency):
            component_set = set(component_edges)
            boundary = set()
            for node in component_nodes:
                for edge in adjacency[node]:
                    if edge not in component_set and labels[edge] in VALID: boundary.add(edge)
            if not boundary: continue

            def same_label_run(start):
                label = str(labels[start]); key = (start, label)
                if key in run_cache: return run_cache[key]
                queue, seen = deque([start]), {start}
                while queue:
                    edge = queue.popleft()
                    for node in pairs[edge]:
                        for neighbor in adjacency[node]:
                            if neighbor not in seen and labels[neighbor] == label:
                                seen.add(neighbor); queue.append(neighbor)
                result = (label, len(seen));
                for edge in seen: run_cache[(edge, label)] = result
                return result

            qualified_labels = []
            for edge in boundary:
                label, support = same_label_run(edge)
                if support >= MIN_CONNECTED_SUPPORT_EDGES: qualified_labels.append(label)
            label_set = set(qualified_labels)
            if len(label_set) != 1: continue

            # Single-sided components are allowed only at a true outer boundary or soma.
            # A component with several neighboring runs is also safe only because all
            # qualified runs carry the same label.
            touches_soma = bool(component_nodes & soma_nodes)
            if len(boundary) == 1 or touches_soma or len(label_set) == 1:
                chosen = next(iter(label_set))
                for edge in component_edges:
                    labels[edge] = chosen; reasons[edge] = 'boundary_component_consistent_label'
                    changed += 1
        total_filled += changed
        if changed == 0 or passes >= 20: break

    raw_n = len(raw)
    raw['compartment_boundary_fill'] = labels[:raw_n]
    raw['label_reason_boundary_fill'] = reasons[:raw_n]
    bridges['compartment_boundary_fill'] = labels[raw_n:]
    bridges['label_reason_boundary_fill'] = reasons[raw_n:]
    output = OUT_ROOT / str(body_id); output.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output / f'{body_id}_annotated_edges_v3_boundary_fill.parquet', index=False)
    bridges.to_parquet(output / f'{body_id}_bridge_edges_v3_boundary_fill.parquet', index=False)
    summary = {'bodyId': body_id, 'status': 'success', 'min_connected_support_edges': MIN_CONNECTED_SUPPORT_EDGES, 'passes': passes, 'filled_edges_or_bridges': total_filled, 'raw_ambiguous_before': int((raw_labels == 'ambiguous').sum()), 'raw_ambiguous_after': int((raw.compartment_boundary_fill == 'ambiguous').sum()), 'bridge_label_counts': bridges.compartment_boundary_fill.value_counts().to_dict()}
    (output / f'{body_id}_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    results = [process(int(path.name)) for path in sorted(SOURCE_ROOT.iterdir()) if path.is_dir() and path.name.isdigit()]
    pd.DataFrame(results).to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == '__main__': main()
