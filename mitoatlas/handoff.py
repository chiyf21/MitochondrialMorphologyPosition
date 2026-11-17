"""Run R1–R5, A1 and A2 from an E50 handoff with fixed NPY rows and by-neuron folds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from fish2_downstream.common import dump, read, sha, unique
from .benchmark import classify, regress
from .lc import classify_lc
from .morphology import r4, r5, standardize
from .report import write_report
from .states import atlas, matched_pairs, make_review_shortlist

CORE_TASKS = ("R1", "R2", "R3", "R4", "R5")
TASKS = (*CORE_TASKS, "A1", "A2")
IDENTITY = ["record_id", "body_id", "cell_type"]


def inputs(root):
    manifest = read(root / "manifest.json")
    protocol = read(root / "metadata/protocol.json")
    if not manifest.get("compartment_review_update"):
        raise ValueError("The full reviewed compartment handoff is required")
    names = ["metadata/instances.csv", "metadata/protocol.json", "metadata/handcrafted_columns.json"]
    names += [f"embeddings/{name}.npy" for name in manifest["array_dimensions"]]
    hashes = {name: sha(root / name) for name in names}
    if any(value != manifest["files"][name]["sha256"] for name, value in hashes.items()):
        raise ValueError("Handoff input checksum mismatch")
    frame = pd.read_csv(root / "metadata/instances.csv", dtype={"body_id": str})
    unique(frame)
    if (not frame.embedding_row.eq(np.arange(len(frame))).all()
            or frame[IDENTITY].isna().any().any()
            or frame.groupby("body_id").cell_type.nunique().ne(1).any()):
        raise ValueError("Invalid instance identity or NPY row order")
    matrices = {}
    for name, width in manifest["array_dimensions"].items():
        x = np.load(root / f"embeddings/{name}.npy", mmap_mode="r", allow_pickle=False)
        if x.shape != (len(frame), width) or not np.isfinite(x).all():
            raise ValueError(f"Invalid matrix: {name}")
        if name == "handcrafted_22":
            matrices["handcrafted"] = pd.DataFrame(x, columns=read(root / "metadata/handcrafted_columns.json"))
        else:
            matrices[name] = pd.DataFrame(x, copy=False)
    eligible = frame.compartment.isin(["axon", "dendrite"]) & frame.compartment_qc.eq("ok")
    if not eligible.equals(frame.compartment_eligible) or eligible.sum() != protocol["cohorts"]["compartment"]["instances"]:
        raise ValueError("Compartment eligibility disagrees with protocol")
    return frame, matrices, hashes


def three_compartment_cohort(frame):
    classes = {"axon", "dendrite", "connecting_cable"}
    eligible = frame.compartment.isin(classes) & frame.compartment_qc.eq("ok")
    types = []
    for cell_type, rows in frame.loc[eligible].groupby("cell_type"):
        if all(set(rows.loc[rows.cell_type_fold.ne(fold), "compartment"]) == classes
               and rows.cell_type_fold.eq(fold).any() for fold in range(5)):
            types.append(cell_type)
    if not types:
        raise ValueError("No cell type supports all three classes in each training fold")
    return eligible & frame.cell_type.isin(types)


def run_task(task, frame, matrices, args):
    dest = args.out / task
    dest.mkdir()
    select = pd.Series(True, index=frame.index)
    if task == "R3":
        select &= ~frame.compartment.eq("soma")
    elif task == "R2" and args.r2_include_cable:
        select = three_compartment_cohort(frame)
    elif task in {"R2", "R4", "R5", "A2"} or (task == "R1" and args.classification_protocol == "lc"):
        select &= frame.compartment_eligible
        if task in {"R4", "R5"}:
            coverage = frame.loc[select].groupby("body_id").compartment.nunique()
            select &= frame.body_id.isin(coverage[coverage.eq(2)].index)
    subset = frame.loc[select].reset_index(drop=True)
    reps = {name: x.loc[select].reset_index(drop=True) for name, x in matrices.items()}
    fold_name = "compartment_fold" if task == "R2" and not args.r2_include_cable else "cell_type_fold"
    cols = IDENTITY + ["compartment", "compartment_qc", "embedding_row"]
    if task in {"R1", "R2", "R3"}:
        cols.append(fold_name)
    if task == "R3":
        cols.append("refine_split")
    subset[cols].to_csv(dest / "cohort.csv", index=False)
    if task in {"R1", "R2"}:
        if args.classification_protocol == "lc":
            classify_lc(subset, reps, task, args.seed, dest, subset[fold_name].astype(int), classifier=args.classifier,
                        include_cable=task == "R2" and args.r2_include_cable)
        else:
            classify(subset, reps, task, [args.classifier], args.seed, dest, fixed_fold=subset[fold_name].astype(int))
    elif task == "R3":
        targets = reps["handcrafted"]
        probes = {name: reps[name] for name in args.r3_embeddings} if args.r3_embeddings else reps
        regress(subset, probes, targets, task, args.seed, 10, dest,
                fixed_fold=subset[fold_name].astype(int))
    elif task == "A1":
        atlas(subset, reps, args.seed, dest)
    elif task == "A2":
        matched_pairs(subset, reps, 3, dest)
        make_review_shortlist(subset, reps["handcrafted"], dest)
    else:
        results = []
        for name, features in reps.items():
            x, _ = standardize(features)
            result = r4(subset, x) if task == "R4" else r5(subset, x, args.seed, args.bootstrap)
            results.append(result.assign(representation=name))
            print(f"{task} {name} complete", flush=True)
        pd.concat(results).to_csv(dest / "results.csv", index=False)
    return dict(status="COMPLETE", instances=len(subset), bodies=subset.body_id.nunique())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("handoff", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=CORE_TASKS)
    p.add_argument("--classification-protocol", choices=("lc", "legacy"), default="lc")
    p.add_argument("--classifier", choices=("lr", "rf"), default="lr")
    p.add_argument("--r2-include-cable", action="store_true", help="LC R2 on types with all three classes across training folds")
    p.add_argument("--r3-embeddings", nargs="+", choices=("point", "image", "concat", "projection"),
                   help="Run only these learned probes; handcrafted values remain the targets")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--bootstrap", type=int, default=999)
    args = p.parse_args()
    if min(args.workers, args.bootstrap) < 1:
        p.error("workers and bootstrap must be positive")
    if args.r2_include_cable and ("R2" not in args.tasks or args.classification_protocol != "lc"):
        p.error("--r2-include-cable requires the LC R2 task")
    frame, matrices, hashes = inputs(args.handoff)
    if args.r3_embeddings and ("R3" not in args.tasks
                                or set(args.r3_embeddings) - set(matrices)):
        p.error("--r3-embeddings requires the R3 task and matching handoff arrays")
    args.out.mkdir(parents=True, exist_ok=False)
    meta = dict(arguments=dict(vars(args), dataset="FISH2 E50 primary 10x15"),
                instances=len(frame), bodies=frame.body_id.nunique(), input_sha256=hashes,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                tasks={task: dict(status="RUNNING") for task in dict.fromkeys(args.tasks)})
    dump(args.out / "run.json", meta)
    print(f"Verified {len(frame)} rows, {frame.body_id.nunique()} neurons, {len(matrices)} representations", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_task, task, frame, matrices, args): task for task in meta["tasks"]}
        for future in as_completed(pending):
            task = pending[future]
            try:
                meta["tasks"][task] = future.result()
            except Exception as exc:
                meta["tasks"][task] = dict(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            dump(args.out / "run.json", meta)
            print(task, meta["tasks"][task], flush=True)
    print(write_report(args.out))
    if any(record["status"] == "FAILED" for record in meta["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
