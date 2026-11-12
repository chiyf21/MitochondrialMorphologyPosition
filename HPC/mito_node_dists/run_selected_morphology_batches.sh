#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:?dataset root required}
NEURONS=${2:?neuron root required}
OUTPUT=${3:?output directory required}
CHUNK=${4:-256}
WORKERS=${5:-16}
PYTHON=${PYTHON:-/home/cyf/.conda/envs/EM/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUTPUT/chunks"
TOTAL=$("$PYTHON" - "$DATASET/metadata/instances.csv" <<'PY'
import csv,sys
keep={"GC","CON","VMN","PC","MON2","MON1"}
with open(sys.argv[1],newline="",encoding="utf-8") as f:
 print(sum(r["cell_type"] in keep for r in csv.DictReader(f)))
PY
)
for ((start=0; start<TOTAL; start+=CHUNK)); do
  expected=$((TOTAL-start < CHUNK ? TOTAL-start : CHUNK))
  part=$(printf '%s/chunks/%06d.csv' "$OUTPUT" "$start")
  if [ -f "$part" ] && [ "$(($(wc -l < "$part")-1))" -eq "$expected" ]; then
    echo "skip chunk $start ($expected rows)"
    continue
  fi
  echo "run chunk $start ($expected rows)"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u "$HERE/compute_selected_morphology.py" \
      --dataset-root "$DATASET" --neuron-root "$NEURONS" \
      --output-csv "$part" --start-index "$start" --limit "$CHUNK" \
      --workers "$WORKERS"
done
"$PYTHON" - "$OUTPUT" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1]); parts=sorted((root/"chunks").glob("*.csv"))
rows=[]
for p in parts:
 with p.open(newline="",encoding="utf-8") as f: rows.extend(csv.DictReader(f))
if not rows: raise SystemExit("no completed chunks")
out=root/"morphology_features.csv"
with out.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} records from {len(parts)} chunks to {out}")
PY
