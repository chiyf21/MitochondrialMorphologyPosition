"""Run the FISH2 downstream experiment catalog with a new aligned Concat embedding."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from fish2_downstream.common import dump, read, sha, unique
from .handoff import inputs, run_task
from .synapse_comparisons import e02, e06
from .synapse_load import predict
from .synapse_review import evaluate_e04, summarize_e01, summarize_e03

MODULES = {
    "morphology": {"E01":"Recover handcraft morphology (formerly R3)",
                   "E03":"Cell type (R1) and compartment (R2)",
                   "R4":"Morphological distribution distance",
                   "R5":"Axon/dendrite coordination", "A1":"Descriptive representation atlas"},
    "em": {"E02":"Geometry-matched EM texture (includes the former A2 matching)"},
    "synapse": {"E04":"Nearest same-compartment synapse distance",
                "E05":"Local synapse counts and density", "E06":"Proximity contrasts"},
    "connectivity": {"E07":"Observed site fan-out", "E08":"Postsynaptic target specificity",
                     "E09":"Neuron wiring profile", "E10":"Dual-side matching feasibility"},
}
META = ("metadata/instances.csv","metadata/protocol.json","metadata/handcrafted_columns.json")
CACHE_KEYS = ["record_id","body_id","embedding_row","cell_type","compartment"]


def aligned_positions(record_ids, row_map):
    mapping = pd.read_csv(row_map,dtype={"record_id":str})
    unique(mapping)
    if "embedding_row" not in mapping:
        mapping["embedding_row"] = np.arange(len(mapping))
    row = mapping.embedding_row
    if (not pd.api.types.is_integer_dtype(row) or
            not np.array_equal(np.sort(row.to_numpy()),np.arange(len(mapping)))):
        raise ValueError("embedding_row must be a 0-based permutation of input NPY rows")
    if set(mapping.record_id) != set(record_ids):
        expected = set(record_ids)
        actual = set(mapping.record_id)
        raise ValueError(f"Embedding record IDs differ from panel: missing={len(expected-actual)}, extra={len(actual-expected)}")
    return mapping.set_index("record_id").loc[record_ids,"embedding_row"].to_numpy()


def prepare(args, frame, hand):
    position = aligned_positions(frame.record_id,args.row_map)
    src = np.load(args.embedding,mmap_mode="r",allow_pickle=False)
    if (src.ndim!=2 or src.shape[0]!=len(position) or src.shape[1]==0 or
            not np.issubdtype(src.dtype,np.floating) or not np.isfinite(src).all()):
        raise ValueError("Embedding must be a finite floating-point NPY matrix aligned to the supplied row map")
    if args.handcrafted:
        replacement = pd.read_csv(args.handcrafted,dtype={"record_id":str})
        unique(replacement)
        if set(replacement.record_id)!=set(frame.record_id):
            raise ValueError("Handcraft replacement must contain every panel record exactly once")
        hand = replacement.set_index("record_id").loc[frame.record_id,hand.columns]
        if not np.isfinite(hand.to_numpy(dtype=float)).all():
            raise ValueError("Handcraft replacement is incomplete or contains invalid values")
    dest = args.out/"inputs"
    (dest/"metadata").mkdir(parents=True)
    (dest/"embeddings").mkdir()
    for name in META:
        shutil.copyfile(args.panel/name,dest/name)
    np.save(dest/"embeddings/concat.npy",src[position])
    np.save(dest/"embeddings/handcrafted_22.npy",hand.to_numpy(dtype=float))
    files = {name:{"sha256":sha(dest/name)} for name in
             (*META,"embeddings/concat.npy","embeddings/handcrafted_22.npy")}
    dump(dest/"manifest.json",dict(compartment_review_update=True,files=files,
         array_dimensions={"concat":src.shape[1],"handcrafted_22":hand.shape[1]},
         panel_manifest=str(args.panel/"manifest.json"),
         source_embedding={"npy":str(args.embedding),"sha256":sha(args.embedding),
                           "row_map":str(args.row_map),"row_map_sha256":sha(args.row_map)},
         source_handcraft=str(args.handcrafted or args.panel/"embeddings/handcrafted_22.npy")))
    return dest


def validate_windows(path, frame, manifest_hash):
    provenance = read(path.parent/"run.json")
    if provenance["input_sha256"]["metadata/instances.csv"] != manifest_hash:
        raise ValueError("Synapse-window cache belongs to another panel or label revision")
    labels = pd.read_csv(path,dtype={"body_id":str})
    expected = frame.loc[frame.compartment_eligible,CACHE_KEYS]
    verified = labels[CACHE_KEYS].drop_duplicates().merge(expected,how="left",validate="one_to_one",indicator=True)
    if (not verified._merge.eq("both").all() or labels.duplicated(["record_id","radius_um"]).any() or
            set(labels.radius_um)!={1,2,5}):
        raise ValueError("Invalid synapse-window IDs, row indices, compartment labels or radii")
    return labels


def call(module,*arguments):
    subprocess.run([sys.executable,"-m",f"mitoatlas.{module}",*map(str,arguments)],check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel","embedding","row_map","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--modules",nargs="+",choices=MODULES,default=list(MODULES))
    parser.add_argument("--handcrafted",type=Path,help="Optional complete replacement handcraft CSV")
    parser.add_argument("--windows",type=Path,help="E05 windows.csv with adjacent run.json")
    parser.add_argument("--connectivity",type=Path,help="Existing phase2_data output directory")
    parser.add_argument("--solidity",type=Path,help="Optional cached E01 mask_solidity.csv for this panel")
    parser.add_argument("--train-bodies",type=Path,help="Optional CSV with body_id values used to train this embedding")
    parser.add_argument("--classifier",choices=["lr","rf"],default="lr")
    parser.add_argument("--lc-scaling",choices=["training_fold","full_cohort"],default="training_fold")
    args = parser.parse_args()
    args = argparse.Namespace(**{k:v.resolve() if isinstance(v,Path) else v for k,v in vars(args).items()})
    args.seed,args.alpha,args.bootstrap = 20261117,10.,999
    args.classification_protocol,args.r2_include_cable = "lc",False
    args.modules = list(dict.fromkeys(args.modules))
    frame, baseline, reference = inputs(args.panel)
    if args.out.exists():
        parser.error("Output must be a new directory")
    train = set()
    if args.train_bodies:
        path = pd.read_csv(args.train_bodies,dtype={"body_id":str})
        if path.body_id.isna().any() or path.empty:
            raise ValueError("Training body map has missing IDs or no rows")
        train = set(path.body_id) & set(frame.body_id)
    dest = prepare(args,frame,baseline["handcrafted"])
    frame,reps,hashes = inputs(dest)
    args.windows = args.windows or args.panel.parent/"mitoatlas_e01_e07_fish2_20261117_01/E05/windows.csv"
    args.connectivity = args.connectivity or args.panel.parent/"mitoatlas_phase2_fish2_20261117_01/inputs"
    default_solidity = args.panel.parent/"mitoatlas_e01_e07_fish2_20261117_01/E01/mask_solidity.csv"
    # A new panel without a supplied mask cache computes solidity directly.
    if args.solidity is None and default_solidity.is_file():
        args.solidity = default_solidity
    run = dict(arguments=vars(args),panel_sha256=reference,aligned_sha256=hashes,
               commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               train_body_overlap=len(train),
               tasks={k:{"status":"PENDING","description":value}
                      for m in args.modules for k,value in MODULES[m].items()})
    root = args.out

    def write():
        dump(root/"run.json",run)
        lines = ["# FISH2 downstream run","",
                 f"{len(frame)} mitochondria across {frame.body_id.nunique()} neurons, "
                 f"Concat width {reps['concat'].shape[1]}.","",
                 "| Experiment | Status | Output |","|---|---|---|"]
        for task,entry in run["tasks"].items():
            dest_path = entry.get("output")
            link = f"[{dest_path}]({dest_path}/)" if dest_path else ""
            if entry.get("reason"):
                link += " " + entry["reason"]
            lines.append(f"| {task} {entry['description']} | {entry['status']} | {link.replace('|','/')} |")
        (root/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    def mark(task,status,folder=None,reason=None):
        fields = run["tasks"][task]
        fields["status"] = status
        if folder is not None:
            fields["output"] = str(folder.relative_to(root))
        if reason is not None:
            fields["reason"] = str(reason)
        print(task,status,reason or "",flush=True)
        write()

    def stage(task, folder, operation):
        mark(task,"RUNNING",folder)
        try:
            reason = operation()
        except Exception as error:
            traceback.print_exc()
            mark(task,"FAILED",folder,f"{type(error).__name__}: {error}")
        else:
            mark(task,"BLOCKED" if reason else "COMPLETE",folder,reason)

    def probe(task, folder):
        folder.mkdir(parents=True,exist_ok=True)
        run_task(task,frame,reps,argparse.Namespace(**(vars(args)|{"out":folder})))

    write()
    if "morphology" in args.modules:
        morph = root/"01_morphology"
        def recovery():
            folder = morph/"E01"
            probe("R3",folder)
            call("r3_feature_groups",folder/"R3",folder/"feature_groups")
            summarize_e01(folder/"R3/oof.csv",folder/"feature_groups/target_groups.csv",
                          folder,frame,args.solidity)
            call("pipeline_figures","E01",folder)
        def classification():
            folder = morph/"E03"
            for task in ("R1","R2"):
                probe(task,folder)
            summarize_e03(folder,folder,args.classifier)
            call("pipeline_figures","E03",folder)
        stage("E01",morph/"E01",recovery)
        stage("E03",morph/"E03",classification)
        for task in ("R4","R5","A1"):
            stage(task,morph/task,lambda task=task:probe(task,morph))
    if "em" in args.modules:
        em = root/"02_em"/"E02"
        def texture():
            probe("A2",em)
            e02(frame,reps["handcrafted"],em/"A2/candidate_pairs.csv",em)
            if not (em/"effects.csv").is_file():
                return "No matched EM triples passed the preset criteria"
        stage("E02",em,texture)
    if "synapse" in args.modules:
        spatial = root/"03_synapse"
        if not args.windows.is_file():
            for task in MODULES["synapse"]:
                mark(task,"BLOCKED",reason=f"Missing cached windows: {args.windows}")
        else:
            try:
                windows = validate_windows(args.windows,frame,hashes["metadata/instances.csv"])
                run["synapse_windows_sha256"] = sha(args.windows)
            except Exception as error:
                for task in MODULES["synapse"]:
                    mark(task,"FAILED",reason=error)
            else:
                exclusions = {"exclude_reported_training_bodies":train} if train else {}
                def local_load():
                    folder = spatial/"E05"
                    folder.mkdir(parents=True)
                    predict(windows,frame,reps,argparse.Namespace(**(vars(args)|{"out":folder})))
                    if train:
                        subset = windows[~windows.body_id.isin(train)]
                        supported = subset.groupby(["compartment","radius_um"]).record_id.size()
                        folds = subset.merge(frame[["record_id","cell_type_fold"]],on="record_id").groupby(
                            ["compartment","radius_um"]).cell_type_fold.nunique()
                        if len(supported)==6 and folds.eq(5).all():
                            other = folder/"exclude_reported_training_bodies"
                            other.mkdir()
                            predict(subset,frame,reps,argparse.Namespace(**(vars(args)|{"out":other})))
                        else:
                            dump(folder/"training_exclusion.json",{"status":"BLOCKED","reason":"Requires five folds in all windows"})
                def proximity():
                    folder = spatial/"E06"
                    folder.mkdir(parents=True)
                    e06(frame,reps,args.windows,folder)
                    if not (folder/"effects.csv").is_file():
                        return "No proximity pairs passed the preset matching"
                stage("E04",spatial/"E04",lambda:evaluate_e04(args.windows,spatial/"E04",frame,reps,args,exclusions))
                stage("E05",spatial/"E05",local_load)
                stage("E06",spatial/"E06",proximity)
    if "connectivity" in args.modules:
        wiring = root/"04_connectivity"
        required = ["run.json","contacts.parquet","sites.parquet","associations.parquet",
                    "anatomy.csv","segments.parquet","mito_segments.parquet"]
        if not all((args.connectivity/p).is_file() for p in required):
            for task in MODULES["connectivity"]:
                mark(task,"BLOCKED",reason=f"Missing Phase II data cache: {args.connectivity}")
        else:
            try:
                manifest = read(args.connectivity/"run.json")
                if manifest["inputs"]["metadata/instances.csv"]!=hashes["metadata/instances.csv"]:
                    raise ValueError("Phase II cache differs from the selected panel")
                run["connectivity_sha256"] = {name:sha(args.connectivity/name) for name in required}
                call("phase2_models",dest,args.connectivity,wiring)
                for row in pd.read_csv(wiring/"phase2_feasibility.csv").itertuples():
                    status = {"READY":"COMPLETE","LIMITED":"LIMITED","BLOCKED":"BLOCKED"}[row.status]
                    mark(row.experiment,status,wiring/row.experiment)
            except Exception as error:
                for task in MODULES["connectivity"]:
                    if run["tasks"][task]["status"]=="PENDING":
                        mark(task,"FAILED",reason=f"{type(error).__name__}: {error}")
    write()
    print(root/"README.md",flush=True)
    if any(e["status"]=="FAILED" for e in run["tasks"].values()):
        raise SystemExit(1)


if __name__=="__main__":
    main()
