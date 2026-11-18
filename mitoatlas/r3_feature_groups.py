"""Summarize existing R3 scores by FISH2 descriptor definition."""
import argparse
from pathlib import Path

import pandas as pd

from fish2_downstream.common import FEATURES, dump, sha

HULL = "convex hull compactness"
EXTENTS = {"mito diameter", *(f"PC{i} Length" for i in range(1, 4))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r3", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    scores = pd.read_csv(args.r3 / "scores.csv")
    if (set(scores.target) != set(FEATURES) or scores.duplicated(["predictors", "target"]).any()
            or not scores.status.eq("complete").all()
            or not scores.groupby("predictors").target.nunique().eq(len(FEATURES)).all()):
        raise ValueError("Expected one complete score per handcrafted feature and probe")
    oof = pd.read_csv(args.r3 / "oof.csv", usecols=["record_id", "target", "predictors", "observed"])
    area = oof.loc[oof.target.isin(["mito CA", "PC1 CA"])].pivot(
        index=["predictors", "record_id"], columns="target", values="observed")
    duplicate = area["mito CA"].notna().all() and area["mito CA"].eq(area["PC1 CA"]).all()
    scores["hull_definition"] = scores.target.map(lambda x: "convex_hull" if x == HULL else "no_convex_hull")
    scores["descriptor_family"] = scores.target.map(
        lambda x: "convex_hull" if x == HULL else "outer_extent" if x in EXTENTS else "other_no_convex_hull")
    scores["duplicate_target"] = duplicate & scores.target.eq("mito CA")
    sets = [("all_columns", scores), ("unique_targets", scores.loc[~scores.duplicate_target])]
    summaries = []
    for target_set, rows in sets:
        for scheme in ["hull_definition", "descriptor_family"]:
            summary = rows.groupby(["predictors", scheme]).r2.agg(
                n_targets="size", median_r2="median", min_r2="min", max_r2="max")
            summaries.append(summary.reset_index().rename(columns={scheme: "group"}).assign(
                target_set=target_set, scheme=scheme))
    args.out.mkdir(parents=True, exist_ok=False)
    scores.to_csv(args.out / "target_groups.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out / "group_scores.csv", index=False)
    dump(args.out / "provenance.json", dict(
        r3=str(args.r3.resolve()), input_sha256={n: sha(args.r3 / n) for n in ("scores.csv", "oof.csv")},
        duplicate_area_targets=bool(duplicate),
        hull_rule="The feature definition uses convex-hull surface area.",
        outer_extent_rule="Three PCA-axis projection spans and maximum point-to-point distance."))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
