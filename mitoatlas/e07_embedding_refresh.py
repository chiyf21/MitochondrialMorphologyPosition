"""Rerun the fixed E07 panel with aligned historical and replacement Concat."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from fish2_downstream.common import dump, read, sha
from .handoff import inputs
from .phase2_models import SEED, e07, load_tables, site_data, write_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "source", "concat_export", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()

    frame, historical, handoff_hashes = inputs(args.handoff)
    export = args.concat_export
    manifest = read(export / "embedding_manifest.json")
    spec = manifest["representations"]["concat"]
    matrix_path = export / spec["path"]
    matrix_hash = sha(matrix_path)
    if matrix_hash != spec["sha256"]:
        raise ValueError("Replacement Concat checksum does not match export manifest")
    row_map = export / manifest["matrix_metadata"]
    identity = ["body_id", "record_id"]
    index = pd.read_csv(row_map, usecols=identity + ["embedding_row"], dtype={key:str for key in identity})
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if (matrix.shape != (len(index), spec["dimension"])
            or not index.embedding_row.eq(np.arange(len(index))).all()
            or index[identity].isna().any().any()
            or not np.isfinite(matrix).all()):
        raise ValueError("Invalid replacement Concat or embedding row map")
    aligned = frame[identity].merge(index, on=identity, how="left", validate="one_to_one", sort=False)
    if aligned.embedding_row.isna().any() or not aligned[identity].equals(frame[identity]):
        raise ValueError("Reviewed handoff lacks an exact replacement Concat match")

    sites, links, contacts, _, bodies, nearest = load_tables(args.source, frame)
    partner_counts = contacts.groupby("pre_site").bodyId_post.nunique()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "run.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        seed=SEED, numpy=version("numpy"), scikit_learn=version("scikit-learn"),
        handoff=str(args.handoff), handoff_hashes=handoff_hashes,
        phase2_input=str(args.source),
        phase2_hashes={name: sha(args.source / name) for name in
                       ("sites.parquet", "associations.parquet", "contacts.parquet")},
        replacement_export=str(export), replacement_map_sha256=sha(row_map),
        replacement_concat_sha256=matrix_hash, checkpoint_sha256=manifest["checkpoint_sha256"],
        aligned_mitochondria=len(aligned), reviewed_neurons=frame.body_id.nunique(),
    ))
    replacement = dict(historical, concat=pd.DataFrame(matrix[aligned.embedding_row.to_numpy(dtype=int)]))
    for name, reps in (("historical", historical), ("replacement", replacement)):
        out = args.output / name
        out.mkdir()
        models = site_data(sites, links, contacts, bodies, reps, nearest)
        e07(out / "E07", sites, models)
        anatomy = ["compartment", "roi", "soma_distance_um"]
        coverage = ["distance_um", "mito_count", "mito_volume_um3"]
        tasks = []
        for linkage, model in zip(("nearest", "pooled5um"), models):
            data = model.join(partner_counts.rename("partner_count"), on="site_index", validate="one_to_one")
            if data.partner_count.isna().any():
                raise ValueError("An eligible presynaptic site lacks an observed partner count")
            tasks.append((f"{linkage}_partner_count", data, ["partner_count"], anatomy, coverage,
                          "count", ["site_index"]))
        write_task(out / "partner_count", tasks, {"target": "distinct observed postsynaptic body IDs"})
        print(f"{name} complete: {len(models[0])} sites from {models[0].body_id.nunique()} neurons", flush=True)


if __name__ == "__main__":
    main()
