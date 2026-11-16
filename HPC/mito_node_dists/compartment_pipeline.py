"""Prepare FISH2 inputs and replay the five original compartment steps."""

import hashlib
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd

STAGES = [
    "01_seed_compartments.py", "02_refine_compartments.py", "03_connect_components.py",
    "04_bridge_aware_fill.py", "05_boundary_fill.py",
]
XYZ = ["x", "y", "z"]


def prepare(source, root, body):
    neuron = root / "data" / "neuron_downloads" / str(body)
    skeleton = neuron / "skeleton"
    skeleton.mkdir(parents=True)
    swc = source / "skeleton" / f"{body}.swc"
    nodes = pd.read_csv(swc, sep=r"\s+", comment="#", header=None,
                        names=["rowId", "swc_type", *XYZ, "radius", "link"])
    xyz = nodes[XYZ].to_numpy()
    nodes[["x_em", "y_em", "z_em"]] = xyz / [8, 8, 30]
    nodes[XYZ] = xyz / [16, 16, 15]
    nodes.to_parquet(skeleton / f"{body}_skeleton_em.parquet", index=False)
    shutil.copy2(source / "neuron_synapse_connections.parquet", neuron)
    return skeleton


def replay(original, root, body, *, bridge_from_v3_if_v11_unavailable=False):
    bridge_from_v3 = False
    snapshot = root / "original_scripts"
    snapshot.mkdir()
    for stage in STAGES:
        path = snapshot / stage
        shutil.copy2(original / stage, path)
        if stage.startswith(("01", "02")):
            previous = sys.argv
            try:
                sys.argv = [str(path), str(body)]
                runpy.run_path(str(path), init_globals={"ROOT": root})
            except RuntimeError as exc:
                if not (bridge_from_v3_if_v11_unavailable and stage.startswith("02")
                        and str(exc).startswith("No PRE/POST synapses attached to main skeleton component")):
                    raise
                bridge_from_v3 = True
                print("V11 unavailable; Step 03 will use the same skeleton geometry from V3", flush=True)
            finally:
                sys.argv = previous
            continue
        source = path.read_text()
        if bridge_from_v3 and stage.startswith("03"):
            source = source.replace("_v11_science_like.parquet", "_v3.parquet")
        old_root = "ROOT = Path('/home/cyf/JaneliaEM')"
        if source.count(old_root) != 1:
            raise ValueError(f"Unexpected root declaration in {stage}")
        namespace = {"__name__": "legacy_stage", "__file__": str(path), "WORK_ROOT": root}
        previous = sys.argv
        try:
            sys.argv = [str(path), str(body)]
            exec(compile(source.replace(old_root, "ROOT = WORK_ROOT"), str(path), "exec"), namespace)
        finally:
            sys.argv = previous
        if "process" in namespace:
            result = namespace["process"](body)
            print(stage, result, flush=True)
            if result["status"] != "success":
                raise RuntimeError(result)
    return {stage: hashlib.sha256((snapshot / stage).read_bytes()).hexdigest() for stage in STAGES}
