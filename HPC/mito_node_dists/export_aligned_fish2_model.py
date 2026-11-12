#!/usr/bin/env python3
"""Write a model embedding CSV and neuron ID map aligned to a handcraft table."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["record_id", "body_id", "cell_type", "source_run_id", "global_instance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handcraft", type=Path)
    parser.add_argument("model_runs", type=Path, nargs="+")
    args = parser.parse_args()
    hand = pd.read_csv(args.handcraft, usecols=KEY + ["compartment", "compartment_qc"])
    if not hand.record_id.is_unique:
        raise ValueError("Handcraft record IDs must be unique")
    indices, arrays, checkpoints = [], [], set()
    for run_number, folder in enumerate(args.model_runs):
        done = json.loads((folder / "complete.json").read_text())
        checkpoints.add(done["checkpoint_sha256"])
        a = np.load(folder / "fused_12.npy", mmap_mode="r", allow_pickle=False)
        i = pd.read_csv(folder / "instance_map.csv", usecols=KEY + ["embedding_row"])
        if done["status"] != "complete" or a.shape != (len(i), 384) or len(i) != done["instances"]:
            raise ValueError(f"Incomplete model run: {folder}")
        if not np.array_equal(np.sort(i.embedding_row), np.arange(len(a))):
            raise ValueError(f"Invalid embedding row map: {folder}")
        indices.append(i.assign(run_number=run_number))
        arrays.append(a)
    index = pd.concat(indices, ignore_index=True)
    if len(checkpoints) != 1 or not index.record_id.is_unique:
        raise ValueError("Model runs must share one checkpoint and contain distinct instances")
    order = hand[KEY].merge(index, on=KEY, how="left", validate="one_to_one")
    if order.embedding_row.isna().any():
        raise ValueError(f"Missing exact model matches: {order.embedding_row.isna().sum()}")
    vectors = np.empty((len(order), 384), dtype=np.float32)
    for run_number, array in enumerate(arrays):
        selected = order.run_number.eq(run_number)
        vectors[selected] = array[order.loc[selected, "embedding_row"].to_numpy(int)]
    if not np.isfinite(vectors).all():
        raise ValueError("Model vectors contain invalid values")
    output = args.handcraft.parent
    table = pd.concat([hand, pd.DataFrame(vectors, columns=[f"embedding_{i:03d}" for i in range(384)])], axis=1)
    for name, df in [("fish2_instance_to_neuron_id.csv", hand), ("fish2_model_embedding_384.csv", table)]:
        target = output / name
        temp = target.with_suffix(".csv.tmp")
        df.to_csv(temp, index=False)
        temp.replace(target)
    print(f"{len(hand)} instances, {hand.body_id.nunique()} neurons, 384 dimensions")
    print("rows from model runs:", order.run_number.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
