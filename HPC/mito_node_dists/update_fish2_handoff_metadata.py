"""Update a FISH2 E50 handoff with verified reviewed compartments."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["record_id", "body_id", "cell_type"]
BINARY = ["axon", "dendrite"]


def fingerprint(path):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(bytes=path.stat().st_size, sha256=sha)


def load(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def update(handoff, reviewed, backup):
    manifest_path = handoff / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("compartment_review_update") or backup.exists():
        raise ValueError("Handoff has already been updated or backup exists")
    source = json.loads((reviewed / "manifest.json").read_text())
    for name, sha in source["output_sha256"].items():
        if fingerprint(reviewed / name)["sha256"] != sha:
            raise ValueError(f"Reviewed file digest mismatch: {name}")

    before = load(handoff / "metadata/instances.csv")
    labels = load(reviewed / "instance_compartments.csv")
    neurons = load(reviewed / "neuron_index_with_compartment.csv")
    hand = load(reviewed / "handcrafted_features22_with_compartment.csv")
    review_fields = labels.columns.difference(KEY).tolist()
    if not before.embedding_row.eq(np.arange(len(before))).all():
        raise ValueError("Expected canonical embedding row order")
    for name, dims in manifest["array_dimensions"].items():
        rel = f"embeddings/{name}.npy"
        matrix = np.load(handoff / rel, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (len(before), dims) or fingerprint(handoff / rel) != manifest["files"][rel]:
            raise ValueError(f"Existing matrix shape or digest mismatch: {rel}")
    pd.testing.assert_frame_equal(load(handoff / "embeddings/handcrafted_22.csv")[KEY[:2]], before[KEY[:2]])
    instances = before.drop(columns=before.columns.intersection(review_fields)).merge(
        labels, on=KEY, how="left", validate="one_to_one", sort=False, indicator=True)
    if len(labels) != len(before) or not instances.pop("_merge").eq("both").all():
        raise ValueError("Review labels do not match every embedding row")
    pd.testing.assert_frame_equal(instances[KEY + ["embedding_row"]], before[KEY + ["embedding_row"]])
    pd.testing.assert_frame_equal(neurons[load(handoff / "metadata/neuron_index.csv").columns],
                                  load(handoff / "metadata/neuron_index.csv"))
    hand = before[KEY + ["embedding_row"]].merge(hand, on=KEY, validate="one_to_one")
    if len(hand) != len(instances):
        raise ValueError("Handcrafted rows are missing")
    if not neurons.compartment_review_status.eq("corrected").all():
        raise ValueError("Review still incomplete")

    eligible = instances.compartment.isin(BINARY) & instances.compartment_qc.eq("ok")
    instances["compartment_eligible"] = eligible
    instances["compartment_fold"] = instances.cell_type_fold.astype("Int64").where(eligible)
    cohort = instances.loc[eligible]
    folds = cohort[["body_id", "cell_type", "compartment_fold"]].drop_duplicates().rename(
        columns={"compartment_fold": "fold"}).sort_values(["cell_type", "body_id"])
    if not folds.body_id.is_unique:
        raise ValueError("A neuron spans compartment folds")
    for cell_type, rows in cohort.groupby("cell_type"):
        for fold in range(5):
            if (set(rows.loc[rows.compartment_fold.ne(fold), "compartment"]) != set(BINARY)
                    or not rows.compartment_fold.eq(fold).any()):
                raise ValueError(f"Inadequate compartment coverage in {cell_type} fold {fold}")
    coverage = instances.groupby(["compartment", "compartment_qc"]).agg(
        instances=("record_id", "size"), bodies=("body_id", "nunique")).reset_index()
    protocol = json.loads((handoff / "metadata/protocol.json").read_text())
    facts = dict(instances=len(cohort), bodies=cohort.body_id.nunique(),
                 groups=len(cohort[["body_id", "compartment"]].drop_duplicates()),
                 classes=BINARY)
    protocol["legacy_reviewed_subset_label_source"] = protocol["label_source"]
    protocol["legacy_reviewed_subset_label_sha256"] = protocol["label_sha256"]
    protocol["label_source"] = "metadata/instance_compartments.csv"
    protocol["compartment_scope"] = ("Full manually reviewed panel; QC=ok and axon/dendrite. "
                                     "Fixed cell-type body folds; body × true compartment grouping.")
    protocol["cohorts"]["compartment"] = facts
    readme = (handoff / "README.md").read_text()
    readme = "\n".join(
        ("- 新版区室任务用 `compartment_eligible=true` 和 `compartment_fold`，"
         f"共有 {facts['instances']:,} 个通过 QC 的 axon/dendrite 实例，来自 {facts['bodies']} 个神经元。"
         "折号沿用全量 Cell Type 的神经元五折。旧 `compartment_reviewed_subset_fold` "
         "与对应折表留作历史对照。"
         if line.startswith("- `compartment_eligible=true`") else line)
        for line in readme.splitlines()) + "\n"
    readme += ("\n## 完整区室标注更新\n\n"
               "- `metadata/instance_compartments.csv` 包含匹配距离、骨架边和审核版本。\n"
               "- `metadata/handcrafted_features22_with_compartment.csv` 可直接用于下游分析。\n"
               "- `metadata/neuron_index.csv` 含审核状态与各区室实例数。\n"
               "- `metadata/compartment_body_folds.csv` 是新版区室折表。\n")

    with tempfile.TemporaryDirectory(prefix=".handoff-review-", dir=handoff) as work:
        stage = Path(work)
        (stage / "metadata").mkdir()
        (stage / "README.md").write_text(readme)
        tables = {
            "instances.csv": instances,
            "neuron_index.csv": neurons,
            "instance_compartments.csv": before[KEY + ["embedding_row"]].merge(labels, on=KEY, validate="one_to_one"),
            "handcrafted_features22_with_compartment.csv": hand,
            "compartment_body_folds.csv": folds,
            "reviewed_compartment_coverage.csv": coverage,
        }
        for name, table in tables.items():
            table.to_csv(stage / "metadata" / name, index=False)
        shutil.copyfile(reviewed / "manifest.json", stage / "metadata/compartment_review_manifest.json")
        protocol["label_sha256"] = fingerprint(stage / protocol["label_source"])["sha256"]
        write_json(stage / "metadata/protocol.json", protocol)
        changed = [path.relative_to(stage) for path in stage.rglob("*") if path.is_file()]
        for rel in changed:
            current = handoff / rel
            if current.exists() and fingerprint(current) != manifest["files"].get(rel.as_posix()):
                raise ValueError(f"Handoff file changed unexpectedly: {rel}")
            manifest["files"][rel.as_posix()] = fingerprint(stage / rel)
        manifest["compartment_review_update"] = dict(
            reviewed_neurons=len(neurons), labeled_instances=len(instances), eligible=facts,
            review_manifest="metadata/compartment_review_manifest.json")
        write_json(stage / "manifest.json", manifest)
        expected = {rel: fingerprint(stage / rel) for rel in [*changed, Path("manifest.json")]}
        backup.mkdir()
        for rel in [*changed, Path("manifest.json")]:
            current = handoff / rel
            if current.exists():
                saved = backup / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, saved)
        for rel in [*changed, Path("manifest.json")]:
            os.replace(stage / rel, handoff / rel)
            if fingerprint(handoff / rel) != expected[rel]:
                raise ValueError(f"Published digest mismatch: {rel}")
    print(json.dumps(facts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "reviewed", "backup"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    update(args.handoff, args.reviewed, args.backup)
