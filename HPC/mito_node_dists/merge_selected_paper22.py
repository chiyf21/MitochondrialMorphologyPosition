#!/usr/bin/env python3
"""Join selected 22D morphology values with the SWC-derived path-length column."""
import csv,sys
from pathlib import Path
root=Path(sys.argv[1])
skel_path=Path(sys.argv[2]) if len(sys.argv)>2 else root/"skeleton_length.csv"
def read(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
morph=read(root/"morphology_features.csv")
skel=read(skel_path)
by={r["record_id"]:r for r in skel}
if len(by)!=len(skel):raise SystemExit("duplicate skeleton record_id")
if {r["record_id"] for r in morph}!={r["record_id"] for r in skel}:raise SystemExit("morphology/skeleton record IDs differ")
fields=list(dict.fromkeys([*morph[0],*(k for r in skel for k in r if k not in morph[0])]))
out=root/"paper22_features_with_skeleton.csv"
with out.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in morph:
  s=by[r["record_id"]]
  w.writerow({**r,**s})
print(f"joined {len(morph)} rows; skeleton errors={sum(r.get('qc')!='ok' for r in skel)}; wrote {out}")
