#!/usr/bin/env python3
"""Join morphology/EM features with skeleton features by stable record_id."""
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows

morph = read(root / "handcrafted_features.csv")
skel = read(root / "skeleton_length.csv")
by_id = {r["record_id"]: r for r in skel}
if len(by_id) != len(skel):
    raise SystemExit("duplicate record_id in skeleton_length.csv")
missing = [r["record_id"] for r in morph if r["record_id"] not in by_id]
extra = set(by_id) - {r["record_id"] for r in morph}
if missing or extra:
    raise SystemExit(f"row ID mismatch: missing_skeleton={len(missing)} extra_skeleton={len(extra)}")
fields = list(dict.fromkeys([*morph[0], *(k for r in skel for k in r if k not in morph[0])]))
out = root / "handcrafted_features_with_skeleton.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for row in morph:
        writer.writerow({**row, **by_id[row["record_id"]]})
print(f"joined {len(morph)} records; skeleton errors={sum(r.get('qc') != 'ok' for r in skel)}; wrote {out}")
