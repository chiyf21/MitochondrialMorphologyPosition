"""Compare shape between PC30 clusters on size-matched FISH2 instances."""
import argparse
from itertools import combinations
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fish2_downstream.common import dump, sha


TOLERANCE = .025


def match(rows, left, right):
    volumes = rows.log_volume.to_numpy()
    origins = rows.source_run_id.to_numpy()
    bodies = rows.body_id.to_numpy()
    candidates = []
    for origin in np.unique(origins[left]):
        a, b = left[origins[left] == origin], right[origins[right] == origin]
        if not len(b):
            continue
        delta, neighbor = cKDTree(volumes[b, None]).query(volumes[a, None], k=min(len(b), 24))
        if delta.ndim == 1:
            delta, neighbor = delta[:, None], neighbor[:, None]
        aa, jj = np.where(delta <= TOLERANCE)
        for i, j in zip(aa, jj):
            first, second = a[i], b[neighbor[i, j]]
            if bodies[first] != bodies[second]:
                candidates.append((delta[i, j], first, second))
    used_a, used_b, pairs = set(), set(), []
    for _, a, b in sorted(candidates):
        if a not in used_a and b not in used_b:
            pairs.append((a,b))
            used_a.add(a)
            used_b.add(b)
    return np.array(pairs, dtype=int).reshape(-1, 2)


def show_examples(picks, rows, output):
    fig, axes = plt.subplots(len(picks), 2, figsize=(6.5, 2.65*len(picks)), squeeze=False, layout="constrained")
    for panel, pair in zip(axes, picks):
        views = []
        for i in (pair["left_index"], pair["right_index"]):
            row = rows.iloc[i]
            em = np.load(row.em_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False) != 0
            z = mask.sum(axis=(1,2)).argmax()
            views.append((em[z], mask[z], row))
        low, high = np.quantile(np.concatenate([x[0].ravel() for x in views]), [.01,.99])
        for ax, (em, mask, row) in zip(panel, views):
            ax.imshow(em, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            if mask.any() and not mask.all():
                ax.contour(mask, [.5], colors=["#5ce1d2"], linewidths=.6)
            ax.set_title(f'C{row.k3} · {row.volume_um3:.3f} µm³ · {row.body_id}', fontsize=10)
            ax.axis("off")
    fig.savefig(output / "K3_volume_matched_examples.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_run", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("handcraft", type=Path)
    args = parser.parse_args()
    run = args.cluster_run
    assignments = run / "assignments.csv.gz"
    rows = pd.read_csv(assignments, dtype={"body_id": str})
    index = pd.read_csv(args.index, usecols=["record_id","source_run_id","em_path","mask_path"])
    shape_columns = ["PC1 Length","PC2 Length","PC3 Length","mito SA","SBI"]
    hand = pd.read_csv(args.handcraft, usecols=["record_id"]+shape_columns)
    rows = rows.merge(index, validate="one_to_one").merge(hand, validate="one_to_one", how="left")
    rows["log_volume"] = np.log10(rows.volume_um3)
    rows["elongation_PC1_PC2"] = rows["PC1 Length"] / rows["PC2 Length"]
    rows["flatness_PC3_PC2"] = rows["PC3 Length"] / rows["PC2 Length"]
    pcs = np.load(run / "pc30.npy", mmap_mode="r")
    if not rows.embedding_row.eq(np.arange(len(pcs))).all():
        raise ValueError("Volume review rows no longer align with PC30")
    output = run / "volume_matched"
    output.mkdir(exist_ok=False)
    summary, members, examples = [], [], []
    for k in (3,9):
        label = rows[f"k{k}"].to_numpy()
        centered_distance = np.empty(len(rows))
        for c in range(1,k+1):
            i = np.flatnonzero(label == c)
            centered_distance[i] = np.linalg.norm(pcs[i]-pcs[i].mean(0),axis=1)
        for a,b in combinations(range(1,k+1),2):
            pairs = match(rows,np.flatnonzero(label==a),np.flatnonzero(label==b))
            if not len(pairs):
                continue
            left,right = [rows.iloc[i].reset_index(drop=True) for i in pairs.T]
            record = dict(k=k, left_cluster=a, right_cluster=b, pairs=len(pairs),
                          neurons=pd.concat([left.body_id,right.body_id]).nunique(),
                          median_relative_volume_gap=float(np.median(abs(left.volume_um3/right.volume_um3-1))))
            for feature in ("elongation_PC1_PC2", "flatness_PC3_PC2", "mito SA", "SBI"):
                record[f"{feature}_left_median"] = left[feature].median()
                record[f"{feature}_right_median"] = right[feature].median()
                record[f"{feature}_left_minus_right_median"] = (left[feature]-right[feature]).median()
            summary.append(record)
            members.append(pd.DataFrame(dict(k=k,left_cluster=a,right_cluster=b,
                 left_record_id=left.record_id,right_record_id=right.record_id)))
            best = np.argmin(centered_distance[pairs].sum(1))
            u,v = pairs[best]
            examples.append(dict(k=k,left_cluster=a,right_cluster=b,left_index=u,right_index=v,
                                 left_record_id=rows.record_id[u],right_record_id=rows.record_id[v],
                                 left_volume_um3=rows.volume_um3[u],right_volume_um3=rows.volume_um3[v]))
    pd.DataFrame(summary).to_csv(output/"shape_summary.csv",index=False)
    pd.concat(members,ignore_index=True).to_csv(output/"pairs.csv.gz",index=False)
    pd.DataFrame(examples).to_csv(output/"examples.csv",index=False)
    show_examples([row for row in examples if row["k"]==3], rows, output)
    dump(output/"run.json",dict(k=[3,9],max_abs_log10_volume_gap=TOLERANCE, same_source_run=True,
        different_neurons=True, candidate_neighbors=24, assignments_sha256=sha(assignments), row_map_sha256=sha(args.index),
        handcraft_sha256=sha(args.handcraft),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
    print(pd.DataFrame(summary).query("k==3").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
