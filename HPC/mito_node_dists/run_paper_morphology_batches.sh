#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:?dataset root required}
OUTPUT=${2:?output root required}
CHUNK=${3:-64}
WORKERS=${4:-8}
PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUTPUT/chunks"
TOTAL=$("$PYTHON" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$DATASET/metadata/instances.csv")
for ((start=0; start<TOTAL; start+=CHUNK)); do
  expected=$((TOTAL-start < CHUNK ? TOTAL-start : CHUNK))
  part=$(printf '%s/chunks/%06d.csv' "$OUTPUT" "$start")
  if [ -f "$part" ] && [ "$(($(wc -l < "$part")-1))" -eq "$expected" ]; then
    echo "skip chunk $start ($expected rows)"
    continue
  fi
  echo "run chunk $start ($expected rows)"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$HERE/compute_paper_morphology.py" --dataset-root "$DATASET" \
      --output-csv "$part" --start-index "$start" --limit "$CHUNK" --workers "$WORKERS"
done
"$PYTHON" - "$OUTPUT" <<'PY'
import csv, sys
from pathlib import Path
root=Path(sys.argv[1])
parts=sorted((root/'chunks').glob('*.csv'))
rows=[r for p in parts for r in csv.DictReader(p.open(newline='', encoding='utf-8'))]
if not rows: raise SystemExit('no completed paper feature chunks')
with (root/'handcrafted_features.csv').open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'merged {len(rows)} rows from {len(parts)} chunks')
PY
echo "Paper-only mask feature table: $OUTPUT/handcrafted_features.csv"
