"""Export the annotated cohort and its exact matches to published morphometrics."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


KEY = ("body_id", "record_id")
IDENTITY = (*KEY, "cell_type", "source_mito_label")
LABEL_FIELDS = ("compartment", "compartment_id", "compartment_source",
                "published_mito_class", "nearest_node_class", "nearest_node_id",
                "nearest_node_distance_voxels")


def read_table(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def indexed(rows):
    result = {tuple(row[column] for column in KEY): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate instance keys")
    return result


def coordinates(row, prefix=""):
    return tuple(float(row[prefix + axis]) for axis in ("x", "y", "z"))


def write_table(path, columns, rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    folder = args.directory
    author = folder / "author_dryad"
    columns, geometry = read_table(folder / "paper22_complete_pc1.csv")
    _, annotations = read_table(author / "paper22_author_compartments.csv")
    features = columns[columns.index("size"):columns.index("SBI") + 1]
    labels = indexed(annotations)
    if len(features) != 22 or indexed(geometry).keys() != labels.keys():
        raise ValueError("Expected the same cohort keys and 22 feature columns")

    published = defaultdict(list)
    for path in sorted((author / "saved_mito_df").glob("*_mito_df.csv")):
        _, rows = read_table(path)
        for row in rows:
            published[row["bodyId"], coordinates(row)].append(row)

    cohort = []
    baseline = []
    unmatched = defaultdict(int)
    for row in geometry:
        body, instance = (row[column] for column in KEY)
        label = labels[body, instance]
        if any(row[column] != label[column] for column in IDENTITY):
            raise ValueError(f"Conflicting metadata for {body}, {instance}")
        if not all(math.isfinite(float(row[column])) for column in features):
            raise ValueError(f"Nonfinite reconstructed feature for {body}, {instance}")
        cohort.append(dict(row, **{column: label[column] for column in LABEL_FIELDS}))
        if label["compartment_source"] != "published_mito_exact_center":
            continue
        matches = published[body, coordinates(row, "center_")]
        if len(matches) != 1:
            unmatched["ambiguous coordinates"] += 1
            continue
        original = matches[0]
        if int(float(original["class"])) != int(label["compartment_id"]):
            raise ValueError(f"Conflicting published class for {body}, {instance}")
        values = {column: float(original[column]) for column in features[:-1]}
        if (not all(math.isfinite(value) for value in values.values())
                or values["size"] <= 0 or values["mito SA"] <= 0):
            unmatched["invalid published features"] += 1
            continue
        radius = (values["size"] * 3 / (4 * math.pi)) ** (1 / 3)
        sbi = math.log10(4 * math.pi * radius ** 2 / values["mito SA"])
        baseline.append({
            **{column: row[column] for column in IDENTITY},
            "compartment": label["compartment"],
            "author_row_id": original["rowId"],
            **{column: original[column] for column in features[:-1]},
            "SBI": sbi,
        })

    if not baseline:
        raise ValueError("No usable exact matches to the published tables")
    metadata = columns[:columns.index("size")]
    context = columns[columns.index("SBI") + 1:]
    write_table(folder / "flyem_mito_handcrafted_annotated.csv",
                [*metadata, *LABEL_FIELDS, *features, *context], cohort)
    write_table(author / "flyem_mito_author_features_exact.csv",
                [*IDENTITY, "compartment", "author_row_id", *features], baseline)
    print(f"Annotated cohort: {len(cohort)} rows, {len(features)} features")
    print(f"Published features: {len(baseline)} exact matches; excluded: {dict(unmatched)}")


if __name__ == "__main__":
    main()
