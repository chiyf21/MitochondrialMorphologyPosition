"""Stage the complete-data intersection of the combined FlyEM cohort."""

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

from export_flyem_annotated_features import read_table, write_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    author = source / "author_dryad"
    neuron_columns, coverage = read_table(author / "complete298_author_coverage.csv")
    feature_columns, features = read_table(author / "complete298_author_features22_exact.csv")
    numeric = feature_columns[feature_columns.index("size"):]
    if len(numeric) != 22:
        raise ValueError("Expected 22 hand-crafted features")
    by_body = defaultdict(list)
    for row in features:
        if row["compartment"] and all(math.isfinite(float(row[k])) for k in numeric):
            by_body[row["body_id"]].append(row)

    with tempfile.TemporaryDirectory(prefix=".complete_data_", dir=output.parent) as staging:
        stage = Path(staging) / output.name
        stage.mkdir()
        neurons, instances, selected = [], [], []
        for item in coverage:
            body = item["body_id"]
            neuron = source / "neurons" / body
            if (not by_body[body] or item["author_neuron_quality_present"] != "1"
                    or not item["author_skeleton_csv"] or not item["author_mito_csv"]
                    or not (neuron / "neuron_metadata.json").is_file()):
                continue
            target = stage / "neurons" / body
            (target / "instances").mkdir(parents=True)
            for path, name in (
                (neuron / "neuron_metadata.json", "neuron_metadata.json"),
                (Path(item["author_skeleton_csv"]), "author_clean_skeleton.csv"),
                (Path(item["author_mito_csv"]), "author_mito_df.csv"),
            ):
                (target / name).symlink_to(path.resolve(strict=True))

            for row in by_body[body]:
                record = row["record_id"]
                body_key, number = record.rsplit("_", 1)
                if body_key != body or not number.isdecimal():
                    raise ValueError(f"Unexpected instance ID: {record}")
                name = f"instance_{number}"
                folder = neuron / "instances" / name
                metadata = folder / "metadata.json"
                meta = json.loads(metadata.read_text())
                if (str(meta["body_id"]) != body
                        or str(meta["source_mito_label"]) != row["source_mito_label"]
                        or list(map(float, meta["center_xyz"])) != [float(row[k]) for k in ("x", "y", "z")]):
                    raise ValueError(f"Instance metadata mismatch: {record}")
                paths = (meta["em_file"], meta["mask_file"], "metadata.json")
                if not all((folder / name).is_file() for name in paths):
                    raise FileNotFoundError(f"Missing raw data for {record}")
                (target / "instances" / name).symlink_to(folder.resolve(strict=True), target_is_directory=True)
                relative = Path("neurons") / body / "instances" / name
                instances.append({k: row[k] for k in
                                  ("body_id", "record_id", "cell_type", "source_mito_label", "compartment")}
                                 | {"relative_dir": str(relative), **{
                                     k: str(output / relative / name) for k, name in
                                     zip(("em_path", "mask_path", "metadata_path"), paths)}})
                selected.append(row)

            base = Path("neurons") / body
            neurons.append(dict(item, selected_instances=len(by_body[body]),
                                results_path=str(output / base),
                                neuron_metadata=str(output / base / "neuron_metadata.json"),
                                author_mito_csv=str(output / base / "author_mito_df.csv"),
                                author_skeleton_csv=str(output / base / "author_clean_skeleton.csv")))

        if not instances or len({(r["body_id"], r["record_id"]) for r in instances}) != len(instances):
            raise ValueError("Expected a nonempty subset with unique instance keys")
        write_table(stage / "neuron_index.csv", [*neuron_columns, "selected_instances",
                    "results_path", "neuron_metadata"], neurons)
        write_table(stage / "instances.csv", list(instances[0]), instances)
        write_table(stage / "author_features22.csv", feature_columns, selected)
        (stage / "README.md").write_text(
            "# FlyEM complete-data subset\n\n"
            f"{len(neurons):,} neurons and {len(instances):,} matched mitochondria.\n\n"
            "Each instance has EM, mask, instance metadata, a finite 22-feature vector, "
            "and an author compartment label. Each neuron has source metadata, author "
            "quality fields, a cleaned skeleton, and the original author mitochondria table.\n\n"
            "instances.csv lists the selected instances and input paths. "
            "Join author_features22.csv on body_id and record_id. "
            "neurons/ contains only the selected instance directories as symbolic links. "
            "The original author per-neuron tables are linked in full. "
            "Keep the underlying source directories accessible.\n",
            encoding="utf-8",
        )
        stage.rename(output)
    print(f"{len(neurons)} neurons; {len(instances)} instances; {output}")


if __name__ == "__main__":
    main()
