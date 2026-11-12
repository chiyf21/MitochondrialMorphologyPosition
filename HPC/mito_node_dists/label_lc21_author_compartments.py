"""Attach published compartment labels to the instances in the feature table."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


CLASSES = dict(enumerate(("soma", "axon", "dendrite", "cell body fiber",
                          "connecting cable", "other"), start=1))
XYZ = ("x", "y", "z")
COLUMNS = ("body_id", "record_id", "cell_type", "source_mito_label",
           "compartment", "compartment_id", "compartment_source",
           "published_mito_class", "nearest_node_class", "nearest_node_id",
           "nearest_node_distance_voxels")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def xyz(row, prefix=""):
    return tuple(float(row[prefix + axis]) for axis in XYZ)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    folder = parser.parse_args().directory
    root = folder / "author_dryad"
    cohort = read_rows(folder / "paper22_geometry.csv")
    skeletons = {}
    for path in (root / "saved_clean_skeletons").glob("s_pandas_*_200nm.csv"):
        body = path.name.split("_")[2]
        if body in skeletons:
            raise ValueError(f"Multiple published skeletons for body {body}")
        nodes = read_rows(path)
        skeletons[body] = (nodes, cKDTree([xyz(node) for node in nodes]))
    mito = defaultdict(set)
    for path in (root / "saved_mito_df").glob("*_mito_df.csv"):
        body = path.name.rsplit("_", 3)[1]
        for row in read_rows(path):
            mito[body, xyz(row)].add(int(float(row["class"])))

    counts = Counter()
    agreements = Counter()
    output = root / "paper22_author_compartments.csv"
    temporary = output.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, COLUMNS)
        writer.writeheader()
        for row in cohort:
            body = row["body_id"]
            center = xyz(row, "center_")
            exact = mito[body, center]
            published = next(iter(exact)) if len(exact) == 1 else None
            node_label = node_id = distance = None
            if body in skeletons:
                nodes, tree = skeletons[body]
                distance, index = tree.query(center)
                node = nodes[index]
                node_label = int(float(node["node_classes"]))
                node_id = node["rowId"]
            if published is not None and node_label is not None:
                agreements[published == node_label] += 1
            if published is not None:
                label, source = published, "published_mito_exact_center"
            elif node_label is not None:
                label, source = node_label, "published_skeleton_nearest_node"
            else:
                label, source = None, "unavailable"
            counts[source, CLASSES.get(label, "")] += 1
            writer.writerow(dict(
                {key: row[key] for key in COLUMNS[:4]},
                compartment=CLASSES.get(label, ""), compartment_id=label,
                compartment_source=source, published_mito_class=published,
                nearest_node_class=node_label, nearest_node_id=node_id,
                nearest_node_distance_voxels=None if distance is None else float(distance),
            ))
    temporary.replace(output)
    print(f"{len(cohort)} instances; {len(skeletons)} skeletons; {output}")
    for (source, label), count in sorted(counts.items()):
        print(source, label, count)
    print("Published table and nearest node agree/disagree:", agreements[True], agreements[False])


if __name__ == "__main__":
    main()
