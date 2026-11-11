#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${1:?dataset root required}
OUTPUT_ROOT=${2:?output root required}
CHUNK_SIZE=${3:-64}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CALCULATOR="$SCRIPT_DIR/compute_skeleton_length_from_metadata.py"
NEURON_ROOT="$DATASET_ROOT/skeleton_metadata"
TOTAL=$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))) )' "$DATASET_ROOT/metadata/instances.csv")

for ((start=0; start<TOTAL; start+=CHUNK_SIZE)); do
  expected=$((TOTAL-start < CHUNK_SIZE ? TOTAL-start : CHUNK_SIZE))
  part=$(printf '%s/skeleton_chunks/%06d' "$OUTPUT_ROOT" "$start")
  if [ -f "$part/skeleton_length.csv" ] && [ "$(($(wc -l < "$part/skeleton_length.csv")-1))" -eq "$expected" ]; then
    echo "skip existing skeleton chunk start=$start rows=$expected"
    continue
  fi
  mkdir -p "$part"
  echo "start skeleton chunk start=$start rows=$expected"
  python "$CALCULATOR" --dataset-root "$DATASET_ROOT" --neuron-root "$NEURON_ROOT" \
    --start-index "$start" --limit "$CHUNK_SIZE" --output-csv "$part/skeleton_length.csv"
done

python - "$OUTPUT_ROOT" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
parts = sorted((root / "skeleton_chunks").glob("*"))
rows = [row for part in parts for row in csv.DictReader((part / "skeleton_length.csv").open(newline="", encoding="utf-8"))]
fields = list(dict.fromkeys(key for row in rows for key in row))
out = root / "skeleton_length.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
errors = sum(row.get("qc") != "ok" for row in rows)
print(f"merged skeleton rows={len(rows)} errors={errors}; wrote {out}")
PY
