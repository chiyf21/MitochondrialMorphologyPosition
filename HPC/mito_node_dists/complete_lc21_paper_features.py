"""Join LC21 paper features and fill unavailable context values from the mask."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile


KEY = ("body_id", "record_id")
PROXIES = {
    "mito CA": ("PC1 CA", "section_status"),
    "length along skeleton": ("PC1 Length", "length_status"),
}
PROVENANCE = ("skeleton_source", "section_status", "length_status",
              "mito CA source", "length along skeleton source")


def read_table(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = {}
        for row in reader:
            key = tuple(row[column] for column in KEY)
            if not all(key) or key in rows:
                raise ValueError(f"Missing or duplicate instance key in {path}: {key}")
            rows[key] = row
        return reader.fieldnames, rows


def positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    folder = parser.parse_args().directory
    columns, geometry = read_table(folder / "paper22_geometry.csv")
    _, context = read_table(folder / "paper22_context_raw.csv")
    if not geometry or geometry.keys() != context.keys():
        raise ValueError("Geometry and context must have the same nonempty instance keys")

    features = columns[columns.index("size"):]
    rows = []
    counts = Counter()
    for key, geom in geometry.items():
        ctx = context[key]
        row = dict(geom)
        row.update({column: ctx[column] for column in PROVENANCE[:3]})
        for feature, (proxy, status) in PROXIES.items():
            if positive(ctx[feature]):
                row[feature] = ctx[feature]
                source = "raw_context" if ctx[status] == "ok" else ctx[status]
            else:
                row[feature] = geom[proxy]
                source = proxy
                if not positive(row[feature]):
                    raise ValueError(f"Invalid {proxy} for {key}")
            row[f"{feature} source"] = source
            counts[(feature, source)] += 1
        if len(features) != 22 or not all(math.isfinite(float(row[k])) for k in features):
            raise ValueError(f"Expected 22 finite features for {key}")
        rows.append(row)

    output = folder / "paper22_complete_pc1.csv"
    with NamedTemporaryFile(mode="w", newline="", dir=folder, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=[*columns, *PROVENANCE])
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(output)
    print(f"{len(rows)} rows, {len(features)} features: {output}")
    for (feature, source), count in counts.items():
        print(f"{feature} | {source}: {count}")


if __name__ == "__main__":
    main()
