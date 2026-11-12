"""Non-destructive V3 gap filling after adding derived skeleton bridges."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/cyf/JaneliaEM')
EXPERIMENT = ROOT / 'experiments' / 'distinguish_compartment_of_neuron'
RESULT_ROOT = EXPERIMENT / 'results'
BRIDGE_ROOT = EXPERIMENT / 'derived_component_bridging'
OUT_ROOT = EXPERIMENT / 'v3_bridge_aware_fill'
VALID = {'axon', 'dendrite', 'connecting_cable', 'cell_body_fiber'}


def chain_edges(edge_nodes, node_count):
    adjacency = [[] for _ in range(node_count)]
    for edge_id, (u, v) in enumerate(edge_nodes):
        adjacency[u].append((v, edge_id)); adjacency[v].append((u, edge_id))
    stops = {node for node, values in enumerate(adjacency) if len(values) != 2}
    seen, chains = set(), []
    for start in sorted(stops):
        for neighbor, edge_id in adjacency[start]:
            if edge_id in seen: continue
            chain, previous, current = [edge_id], start, neighbor; seen.add(edge_id)
            while current not in stops:
                choices = [(n, e) for n, e in adjacency[current] if n != previous and e not in seen]
                if not choices: break
                nxt, nxt_edge = choices[0]; chain.append(nxt_edge); seen.add(nxt_edge)
                previous, current = current, nxt
            chains.append(chain)
    for edge_id in range(len(edge_nodes)):
        if edge_id not in seen: chains.append([edge_id])
    return chains


def process(body_id):
    source = RESULT_ROOT / str(body_id)
    edge_path = source / f'{body_id}_annotated_edges_v3.parquet'
    node_path = source / f'{body_id}_annotated_skeleton_v3.parquet'
    if not edge_path.exists():
        source = ROOT / 'data' / 'neuron_downloads' / str(body_id) / 'skeleton'
        edge_path = source / f'{body_id}_annotated_edges_v3.parquet'
        node_path = source / f'{body_id}_annotated_skeleton_v3.parquet'
    bridge_path = BRIDGE_ROOT / str(body_id) / f'{body_id}_derived_bridge_edges.parquet'
    edges, nodes = pd.read_parquet(edge_path).copy(), pd.read_parquet(node_path)
    bridges = pd.read_parquet(bridge_path).copy() if bridge_path.exists() else pd.DataFrame()
    if bridges.empty:
        return {'bodyId': body_id, 'status': 'skipped_no_bridges'}

    raw_labels = edges['compartment'].astype(str).to_numpy(object)
    raw_reasons = edges['label_reason'].astype(str).to_numpy(object)
    bridge_labels, bridge_reasons = [], []
    for bridge in bridges.itertuples(index=False):
        # Use the V3 node labels, not the V11 labels recorded when bridges were created.
        left = str(nodes.iloc[int(bridge.u_node_index)].compartment)
        right = str(nodes.iloc[int(bridge.v_node_index)].compartment)
        if left == right and left in VALID:
            bridge_labels.append(left); bridge_reasons.append('same_endpoint_label_bridge')
        else:
            bridge_labels.append('ambiguous'); bridge_reasons.append('different_or_uninformative_endpoint_labels')
    bridges['compartment_bridge_aware'] = bridge_labels
    bridges['label_reason_bridge_aware'] = bridge_reasons

    raw_pairs = list(zip(edges.u.astype(int), edges.v.astype(int)))
    bridge_pairs = list(zip(bridges.u_node_index.astype(int), bridges.v_node_index.astype(int)))
    labels = np.concatenate([raw_labels, np.asarray(bridge_labels, dtype=object)])
    reasons = np.concatenate([raw_reasons, np.asarray(bridge_reasons, dtype=object)])
    chains = chain_edges(raw_pairs + bridge_pairs, len(nodes))
    changed = 0
    # Repeat because a newly-filled chain may create a valid boundary for another chain.
    while True:
        pass_changed = 0
        for chain in chains:
            fixed = [(i, str(labels[edge])) for i, edge in enumerate(chain) if str(labels[edge]) in VALID]
            for (left_i, left_label), (right_i, right_label) in zip(fixed[:-1], fixed[1:]):
                if left_label != right_label or right_i <= left_i + 1: continue
                middle = chain[left_i + 1:right_i]
                if all(str(labels[edge]) == 'ambiguous' for edge in middle):
                    for edge in middle:
                        labels[edge] = left_label; reasons[edge] = 'bridge_aware_same_label_gap_fill'; pass_changed += 1
        changed += pass_changed
        if pass_changed == 0: break

    raw_count = len(edges)
    edges['compartment_bridge_aware'] = labels[:raw_count]
    edges['label_reason_bridge_aware'] = reasons[:raw_count]
    bridges['compartment_bridge_aware'] = labels[raw_count:]
    bridges['label_reason_bridge_aware'] = reasons[raw_count:]
    output = OUT_ROOT / str(body_id); output.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(output / f'{body_id}_annotated_edges_v3_bridge_aware.parquet', index=False)
    bridges.to_parquet(output / f'{body_id}_bridge_edges_v3_bridge_aware.parquet', index=False)
    summary = {'bodyId': body_id, 'status': 'success', 'raw_edges': raw_count, 'bridges': len(bridges), 'newly_filled_edges_or_bridges': changed, 'raw_ambiguous_before': int((raw_labels == 'ambiguous').sum()), 'raw_ambiguous_after': int((edges.compartment_bridge_aware == 'ambiguous').sum()), 'bridge_labels': bridges.compartment_bridge_aware.value_counts().to_dict()}
    (output / f'{body_id}_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    results = [process(int(path.parent.name)) for path in sorted(RESULT_ROOT.glob('*/*_annotated_edges_v3.parquet'))]
    pd.DataFrame(results).to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__': main()
