"""R1/R2 with Concat and the FISH2 21-feature handcraft baseline."""
import argparse
from pathlib import Path
import subprocess

from fish2_downstream.common import dump
from .figures.confusion import render
from .handoff import inputs, run_task
from .synapse_review import summarize_e03


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("handoff",type=Path,help="Reviewed frozen panel or prepared pipeline inputs")
    p.add_argument("out",type=Path)
    p.add_argument("--lc-scaling",choices=["training_fold","full_cohort"],default="training_fold")
    args = p.parse_args()
    frame, representations, hashes = inputs(args.handoff)
    hand = representations["handcrafted"]
    if hand.shape[1] != 22 or not hand["mito CA"].equals(hand["PC1 CA"]):
        p.error("The current 21D baseline requires 22 handcraft columns with duplicated mito CA and PC1 CA")
    representations["handcrafted"] = hand.drop(columns="mito CA")
    args.out.mkdir(parents=True,exist_ok=False)
    args.seed, args.classification_protocol = 20261117, "lc"
    args.classifier, args.r2_include_cable = "lr", False
    tasks = {}
    run = dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               arguments=vars(args),input_sha256=hashes,
               handcrafted=list(representations["handcrafted"]),concat_dim=representations["concat"].shape[1],
               tasks=tasks, publication_levels={"R1":"per_group","R2":"per_mito"})
    dump(args.out/"run.json",run)
    for task in ("R1","R2"):
        tasks[task] = run_task(task,frame,representations,args)
        dump(args.out/"run.json",run)
    summarize_e03(args.out,args.out,args.classifier)
    render(args.out,args.out/"figures",individual=True)


if __name__ == "__main__":
    main()
