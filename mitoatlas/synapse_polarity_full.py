"""Evaluate FISH2 pre-versus-post prediction using full-cohort mask distances."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fish2_downstream.common import sha
from mitoatlas.synapse_load import predictors
from mitoatlas.synapse_polarity import evaluate


KEY = ["record_id", "body_id"]


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype={"body_id": str}, **kwargs)


def labels(frame, near_nm, far_nm):
    pre = frame.nearest_pre_distance_nm.le(near_nm) & frame.nearest_post_distance_nm.ge(far_nm)
    post = frame.nearest_post_distance_nm.le(near_nm) & frame.nearest_pre_distance_nm.ge(far_nm)
    return frame.loc[pre | post].assign(is_pre=pre.loc[pre | post])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("reviewed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = dict(instances=args.dataset / "instances.csv",
                 manifest=args.dataset / "dataset_manifest.json",
                 handcrafted=args.dataset / "handcrafted_features22.csv",
                 proximity=args.dataset / "synapse_proximity_v2/instance_synapse_proximity.csv",
                 embedding_rows=args.embeddings / "instance_map.csv",
                 embedding=args.embeddings / "embeddings/concat.npy",
                 training_provenance=args.embeddings / "training_provenance.json",
                 reviewed=args.reviewed)
    instance = read_csv(paths["instances"], usecols=KEY)
    frame = read_csv(paths["embedding_rows"], usecols=KEY + ["cell_type", "embedding_row", "refine_split"])
    concat = np.load(paths["embedding"], mmap_mode="r", allow_pickle=False)
    if (not frame[KEY].equals(instance) or not frame.record_id.is_unique
            or not np.array_equal(frame.embedding_row, np.arange(len(frame)))
            or concat.shape != (len(frame), 1536) or not np.isfinite(concat).all()):
        raise ValueError("Full-cohort instance map and E50 Concat are not aligned")
    names = json.loads(paths["manifest"].read_text())["handcrafted_columns"]
    hand = read_csv(paths["handcrafted"], usecols=KEY + names)
    if not hand["mito CA"].equals(hand["PC1 CA"]):
        raise ValueError("Full handcraft table does not have the expected duplicated CA")
    frame = frame.merge(hand, on=KEY, validate="one_to_one", how="left")
    annotated = read_csv(paths["proximity"], usecols=KEY + [
        "nearest_pre_distance_nm", "nearest_post_distance_nm", "annotation_status"])
    frame = frame.merge(annotated, on=KEY, validate="one_to_one", how="left")
    reviewed = read_csv(paths["reviewed"], usecols=KEY + ["compartment", "compartment_eligible"])
    if not reviewed.record_id.isin(frame.record_id).all():
        raise ValueError("Reviewed subset contains unknown instances")
    frame = frame.merge(reviewed, on=KEY, validate="one_to_one", how="left", indicator="review_panel")
    use = np.isfinite(frame[names]).all(axis=1) & frame.annotation_status.eq("ok")
    frame = frame.loc[use].reset_index(drop=True)
    frame["cell_type"] = frame.cell_type.fillna("unlabeled")
    reprs = {"handcrafted": frame[names], "concat": concat[frame.embedding_row.to_numpy()]}
    frame["embedding_row"] = np.arange(len(frame))
    models = {k: v for k, v in predictors(frame, reprs).items() if v is not None}
    models["concat_only"] = reprs["concat"]
    primary = labels(frame, 1000, 2000)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261118)
    folds = {}
    for fold, (_, held) in enumerate(splitter.split(primary, primary.is_pre, primary.body_id)):
        folds.update(dict.fromkeys(primary.iloc[held].body_id.unique(), fold))
    frame["cell_type_fold"] = frame.body_id.map(folds)
    args.output.mkdir(parents=True, exist_ok=False)
    all_scores, all_oof, all_within = [], [], []
    for near, far in ((1000, 2000), (500, 1000)):
        selected = labels(frame, near, far)
        # Bodies absent at the primary thresholds are assigned a fixed fold once.
        if selected.cell_type_fold.isna().any():
            extra = sorted(selected.loc[selected.cell_type_fold.isna(), "body_id"].unique())
            selected["cell_type_fold"] = selected.cell_type_fold.fillna(
                selected.body_id.map({body: i % 5 for i, body in enumerate(extra)})
            )
        pooled = selected.assign(compartment="pooled")
        cohorts = {"full": pooled, "full_not_refine_train": pooled[pooled.refine_split.ne("train")]}
        if near == 1000:
            review = selected[selected.compartment_eligible.eq(True)]
            independent_panel = pooled[pooled.review_panel.eq("left_only")]
            cohorts.update(reviewed=review, reviewed_not_refine_train=review[review.refine_split.ne("train")],
                           outside_review_panel=independent_panel,
                           outside_review_panel_not_refine_train=independent_panel[independent_panel.refine_split.ne("train")])
        scores, oof, within = evaluate(models, cohorts)
        for target, product in zip((all_scores, all_oof, all_within), (scores, oof, within)):
            target.append(product.assign(near_nm=near, opposite_far_nm=far))
        print(f"Completed near {near} nm / opposite far {far} nm: {len(selected)} instances", flush=True)
    for name, products in zip(("scores", "oof", "within_neuron"), (all_scores, all_oof, all_within)):
        pd.concat(products, ignore_index=True).to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "run.json").write_text(json.dumps(dict(
        inputs={name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        source_instances=len(instance), complete_features_and_proximity=len(frame), reviewed_instances=len(reviewed),
        distance="Euclidean distance from owned endpoint to occupied voxel center",
        grouping="StratifiedGroupKFold by body_id", fold_seed=20261118, bootstrap_seed=20261117,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()), indent=2) + "\n")
    summary = pd.concat(all_scores, ignore_index=True)
    cols = ["near_nm", "cohort", "compartment", "model", "pre", "post", "bodies", "within_bodies",
            "pooled_auroc", "within_auroc", "within_lo", "within_hi",
            "delta_vs_hand", "delta_vs_hand_lo", "delta_vs_hand_hi"]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
