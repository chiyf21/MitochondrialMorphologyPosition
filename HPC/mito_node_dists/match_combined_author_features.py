"""Match downloaded FlyEM instances to the published mitochondrial features."""
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
import json

from compute_lc21_paper_geometry import FEATURES
from export_flyem_annotated_features import coordinates, read_table, write_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    output = root / "author_dryad"
    _, neurons = read_table(root / "neuron_index.csv")
    if len({r["bodyId"] for r in neurons}) != len(neurons):
        raise ValueError("Duplicate body IDs in neuron index")
    published = {}
    for path in (output / "saved_mito_df").glob("*_mito_df.csv"):
        body = path.name.rsplit("_", 3)[1]
        if body in published:
            raise ValueError(f"Multiple author tables for {body}")
        _, rows = read_table(path)
        published[body] = defaultdict(list)
        for row in rows:
            if row["bodyId"] != body:
                raise ValueError(f"Author body ID mismatch in {path}")
            published[body][coordinates(row)].append(row)

    coverage = []
    matched = []
    for neuron in neurons:
        body = neuron["bodyId"]
        records = []
        for path in sorted((root / "neurons" / body / "instances").glob("*/metadata.json")):
            meta = json.loads(path.read_text())
            if str(meta["body_id"]) != body:
                raise ValueError(f"Metadata disagrees with neuron index: {path}")
            if not all((path.parent / meta[key]).is_file() for key in ("em_file", "mask_file")):
                continue
            records.append((path, meta, tuple(meta["center_xyz"])))

        table = published.get(body)
        # Use the neuron index for cell type. Retry metadata can carry a pipeline label.
        centers = Counter(center for _, _, center in records)
        counts = Counter()
        for path, meta, center in records:
            if table is None:
                counts["author_table_not_local"] += 1
                continue
            candidates = table.get(center, [])
            if not candidates:
                counts["no_exact_xyz"] += 1
                continue
            if centers[center] != 1 or len(candidates) != 1:
                counts["ambiguous_xyz"] += 1
                continue
            author = candidates[0]
            values = {name: float(author[name]) for name in FEATURES[:-1]}
            if not all(map(math.isfinite, values.values())) or min(values["size"], values["mito SA"]) <= 0:
                counts["invalid_features"] += 1
                continue
            radius = (3 * values["size"] / (4 * math.pi)) ** (1 / 3)
            matched.append({
                "body_id": body, "record_id": f"{body}_{path.parent.name.removeprefix('instance_')}",
                "cell_type": neuron["cell_type"], "source_mito_label": meta["source_mito_label"],
                "compartment": {2: "axon", 3: "dendrite", 5: "connecting cable"}[int(float(author["class"]))],
                "author_row_id": author["rowId"], **dict(zip(("x", "y", "z"), center)), **values,
                "SBI": math.log10(4 * math.pi * radius ** 2 / values["mito SA"]),
            })
            counts["matched"] += 1
        if sum(counts.values()) != len(records):
            raise ValueError(f"Coverage totals disagree for {body}")
        coverage.append({"body_id": body, "cell_type": neuron["cell_type"],
                         "planned": neuron["mitochondria_planned"], "ready": len(records),
                         "author_table_local": int(table is not None), **counts})

    id_columns = ("body_id", "record_id", "cell_type", "source_mito_label",
                  "compartment", "author_row_id", "x", "y", "z")
    status_columns = ("matched", "author_table_not_local", "no_exact_xyz", "ambiguous_xyz", "invalid_features")
    for row in coverage:
        for name in status_columns:
            row.setdefault(name, 0)
    if len({(r["body_id"], r["record_id"]) for r in matched}) != len(matched):
        raise ValueError("Duplicate matched instance IDs")
    write_table(output / "author_features22_exact.csv", [*id_columns, *FEATURES], matched)
    write_table(output / "neuron_coverage.csv",
                ["body_id", "cell_type", "planned", "ready", "author_table_local", *status_columns],
                coverage)
    print({key: sum(int(r[key]) for r in coverage) for key in
           ("planned", "ready", "author_table_local", *status_columns)})


if __name__ == "__main__":
    main()
