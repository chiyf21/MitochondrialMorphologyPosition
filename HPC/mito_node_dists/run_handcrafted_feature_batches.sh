#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${1:?dataset root required}
OUTPUT_ROOT=${2:?output root required}
CHUNK_SIZE=${3:-64}
WORKERS=${4:-8}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXTRACTOR="$SCRIPT_DIR/compute_handcrafted_features.py"

TOTAL=$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))) )' "$DATASET_ROOT/metadata/instances.csv")
for ((start=0; start<TOTAL; start+=CHUNK_SIZE)); do
  expected=$((TOTAL-start < CHUNK_SIZE ? TOTAL-start : CHUNK_SIZE))
  part=$(printf '%s/chunks/%06d' "$OUTPUT_ROOT" "$start")
  if [ -f "$part/summary.json" ] && python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("instances_selected")==int(sys.argv[2]) else 1)' "$part/summary.json" "$expected"; then
    echo "skip existing chunk start=$start rows=$expected"
    continue
  fi
  echo "start chunk start=$start rows=$expected workers=$WORKERS"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$EXTRACTOR" --dataset-root "$DATASET_ROOT" --output-dir "$part" \
      --start-index "$start" --limit "$CHUNK_SIZE" --workers "$WORKERS"
done

python - "$OUTPUT_ROOT" "$WORKERS" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
parts = sorted((root / "chunks").glob("*"))
names = ("morphology_features.csv", "pixel_features.csv", "handcrafted_features.csv", "errors.csv")
for name in names:
    rows = [row for part in parts for row in csv.DictReader((part / name).open(newline="", encoding="utf-8"))]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["record_id", "source_run_id", "body_id", "cell_type", "global_instance", "voxel_count", "status"]
    with (root / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
summaries = [json.loads((part / "summary.json").read_text()) for part in parts]
summary = summaries[0].copy() if summaries else {}
summary.update(instances_selected=sum(s["instances_selected"] for s in summaries),
               errors=sum(s["errors"] for s in summaries), chunks=len(parts),
               source_run_ids=sorted({run for s in summaries for run in s["source_run_ids"]}),
               workers_per_chunk=int(sys.argv[2]))
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"merged chunks={len(parts)} instances={summary['instances_selected']} errors={summary['errors']}")
PY
