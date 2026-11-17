"""Run the MitoAtlas R1–R5 and A1–A3 analysis pipeline for one dataset."""
import argparse
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import scipy
import sklearn

from fish2_downstream.common import FEATURES, dump, sha, table, unique
from .benchmark import classify, folds, regress
from .lc import classify_lc
from .morphology import (COMPARTMENTS, KEY, aligned_embedding, load_cohort,
                         r4, r5, standardize)
from .states import atlas, matched_pairs
from .report import write_report

TASKS = ("R1", "R2", "R3", "R4", "R5", "A1", "A2", "A3")


def load_analysis(args):
    frame = table(args.handcrafted).rename(columns={"length along skeleton": "length_along_skeleton_nm"})
    unique(frame)
    if frame.empty:
        raise ValueError("Empty handcrafted input")
    if frame[KEY + ["cell_type"]].isna().any().any():
        raise ValueError("Missing body, record, or cell type")
    if frame.groupby("body_id").cell_type.nunique().gt(1).any():
        raise ValueError("A body has conflicting cell types")
    if "compartment" not in frame:
        frame["compartment"] = pd.NA
    frame = frame.sort_values(KEY).reset_index(drop=True)
    reps = {"handcrafted": frame[FEATURES].astype(float)}
    paths = {"handcrafted": args.handcrafted}
    for name, path in args.embedding:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in reps or name == "texture_measurements":
            raise ValueError(f"Invalid or repeated representation name: {name}")
        paths[name] = Path(path)
        reps[name] = aligned_embedding(paths[name], frame).reset_index(drop=True).astype(float)
    for name, x in reps.items():
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError(f"{name} has nonfinite features; provide an audited common cohort")
    return frame, reps, paths


def select(frame, reps, index):
    return (frame.loc[index].reset_index(drop=True),
            {name: x.loc[index].reset_index(drop=True) for name, x in reps.items()})


def texture_inputs(path, frame):
    textures = table(path)
    unique(textures)
    cols = [name for name in textures if name.startswith("texture_")]
    if not cols:
        raise ValueError("Independent texture table needs texture_ columns")
    joined = frame[KEY].merge(textures[KEY + cols], on=KEY, how="left", validate="one_to_one")
    return joined[cols].astype(float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--handcrafted", required=True, type=Path)
    p.add_argument("--embedding", action="append", default=[], nargs=2, metavar=("NAME", "CSV"))
    p.add_argument("--texture", type=Path, help="Independent measurements, keyed CSV")
    p.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--classification-protocol", choices=("lc", "legacy"), default="lc")
    p.add_argument("--classifier", nargs="+", choices=("lr", "rf"), default=["lr"])
    p.add_argument("--alpha", type=float, default=10)
    p.add_argument("--bootstrap", type=int, default=999)
    p.add_argument("--neighbors", type=int, default=3)
    p.add_argument("--umap", action="store_true")
    args = p.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0 or min(args.bootstrap, args.neighbors) < 1:
        p.error("alpha, bootstrap and neighbors must be positive")
    frame, reps, paths = load_analysis(args)
    if args.texture and "A3" in args.tasks:
        paths["texture_measurements"] = args.texture
        texture = texture_inputs(args.texture, frame)
    else:
        texture = None
    accepted = frame.compartment.isin(COMPARTMENTS)
    if "compartment_qc" in frame:
        accepted &= frame.compartment_qc.eq("ok")
    compartment_frame, compartment_reps = select(frame, reps, frame.index[accepted])
    # R2 uses both compartments per body and a test group for every type/fold.
    by_body = compartment_frame.groupby("body_id").compartment.nunique()
    paired = compartment_frame[compartment_frame.body_id.isin(by_body[by_body.eq(2)].index)]
    type_counts = paired.groupby("cell_type").body_id.nunique()
    r2_index = paired[paired.cell_type.isin(type_counts[type_counts.ge(5)].index)].index

    args.out.mkdir(parents=True, exist_ok=False)
    source = dict(arguments=vars(args), instances=len(frame), bodies=frame.body_id.nunique(),
                  columns={name: list(x.columns) for name, x in reps.items()},
                  inputs={name: dict(path=str(path.resolve()), sha256=sha(path))
                          for name, path in paths.items()},
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                 cwd=Path(__file__).resolve().parent).strip(),
                  numpy=np.__version__, pandas=pd.__version__,
                  scipy=scipy.__version__, sklearn=sklearn.__version__, tasks={})
    dump(args.out / "run.json", source)
    failed = False
    for task in dict.fromkeys(args.tasks):
        dest = args.out / task
        dest.mkdir()
        selected, matrices = frame, reps
        try:
            reason = None
            if task == "R1" and args.classification_protocol == "lc":
                selected, matrices = compartment_frame, compartment_reps
                if not set(frame.cell_type) == set(selected.cell_type):
                    reason = "Some types have no accepted axon/dendrite instances"
            elif task == "R2":
                if args.classification_protocol == "lc":
                    selected, matrices = compartment_frame, compartment_reps
                else:
                    selected, matrices = select(compartment_frame, compartment_reps, r2_index)
                if selected.empty:
                    reason = "No cell type has five neurons with both accepted compartments"
            elif task in {"R4", "R5"}:
                if paired.groupby("cell_type").body_id.nunique().ge(2).sum() < 2:
                    reason = "Need two types with two paired-compartment neurons each"
                else:
                    _, selected = load_cohort(args.handcrafted, 2, np.random.default_rng(args.seed), False)
                    index = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(selected[KEY]))
                    selected, matrices = select(frame, reps, index)
            elif task == "R3":
                selected, matrices = select(frame, reps, frame.index[~frame.compartment.eq("soma").fillna(False)])
            elif task == "A2":
                selected, matrices = compartment_frame, compartment_reps
                if len(reps) == 1:
                    reason = "Need at least one learned representation"
                elif not selected.groupby(["cell_type", "compartment"]).body_id.nunique().ge(2).any():
                    reason = "Need two neurons in a shared type and compartment"
            elif task == "A3":
                if texture is None or len(reps) == 1:
                    reason = "Need independent texture_ measurements and a learned representation"
                else:
                    good = np.isfinite(texture.to_numpy()).all(axis=1)
                    measured = frame.loc[good]
                    counts = measured.groupby("cell_type").body_id.nunique()
                    index = measured[measured.cell_type.isin(counts[counts.ge(5)].index)].index
                    selected, matrices = select(frame, reps, index)
                    if selected.empty:
                        reason = "No type has five neurons with complete measurements"
            if task in {"R1", "R3"} and selected.groupby("cell_type").body_id.nunique().min() < 5:
                reason = "Need five neurons per included type"
            if task == "R1" and selected.cell_type.nunique() < 2:
                reason = "Need at least two cell types"
            if reason:
                source["tasks"][task] = dict(status="SKIPPED", reason=reason)
                continue

            selected[KEY + ["cell_type", "compartment"]].to_csv(dest / "cohort.csv", index=False)
            if task in {"R1", "R2"}:
                if args.classification_protocol == "lc":
                    fixed = frame.assign(fold=folds(frame, args.seed)).set_index("record_id")["fold"]
                    for classifier in dict.fromkeys(args.classifier):
                        classify_lc(selected, matrices, task, args.seed, dest,
                                    selected.record_id.map(fixed).to_numpy(), classifier=classifier)
                else:
                    classify(selected, matrices, task, dict.fromkeys(args.classifier), args.seed, dest)
            elif task in {"R3", "A3"}:
                y = matrices["handcrafted"] if task == "R3" else texture.loc[index].reset_index(drop=True)
                regress(selected, matrices, y, task, args.seed, args.alpha, dest)
            elif task in {"R4", "R5"}:
                results = []
                for name, features in matrices.items():
                    x, _ = standardize(features)
                    result = r4(selected, x) if task == "R4" else r5(selected, x, args.seed, args.bootstrap)
                    results.append(result.assign(representation=name))
                pd.concat(results).to_csv(dest / "results.csv", index=False)
            elif task == "A1":
                atlas(selected, matrices, args.seed, dest, args.umap)
            else:
                matched_pairs(selected, matrices, args.neighbors, dest)
            source["tasks"][task] = dict(status="COMPLETE", instances=len(selected),
                                         bodies=selected.body_id.nunique())
        except Exception as exc:
            source["tasks"][task] = dict(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            failed = True
        finally:
            dump(args.out / "run.json", source)
            print(task, source["tasks"][task], flush=True)
    print(f"Experiment reports: {write_report(args.out)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
