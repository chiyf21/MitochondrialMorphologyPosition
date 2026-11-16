#!/usr/bin/env python3
"""Parallel skeleton-geodesic feature calculation for the six selected cell types."""
import argparse,csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from compute_skeleton_length_from_metadata import process

_TYPES={"GC","CON","VMN","PC","MON2","MON1"}
_CACHE={}

def one(item):
 row,ds,neurons=item
 base={k:row.get(k,"") for k in ("record_id","source_run_id","body_id","cell_type","global_instance")}
 try:
  return process(row,Path(ds),Path(neurons),_CACHE)
 except Exception as e:
  return {**base,"qc":f"error: {type(e).__name__}: {e}"}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--dataset-root",type=Path,required=True)
 ap.add_argument("--neuron-root",type=Path,required=True)
 ap.add_argument("--output-csv",type=Path,required=True)
 ap.add_argument("--workers",type=int,default=16)
 ap.add_argument("--start-index",type=int,default=0)
 ap.add_argument("--limit",type=int)
 a=ap.parse_args()
 with (a.dataset_root/"metadata/instances.csv").open(newline="",encoding="utf-8") as f:
  rows=[r for r in csv.DictReader(f) if r["cell_type"] in _TYPES]
 rows=rows[a.start_index:]
 if a.limit is not None: rows=rows[:a.limit]
 if not rows: ap.error("No selected instances")
 a.output_csv.parent.mkdir(parents=True,exist_ok=True)
 fields=["record_id","source_run_id","body_id","cell_type","global_instance",
  "nearest_skeleton_node_id","nearest_node_x_global_vox","nearest_node_y_global_vox",
  "nearest_node_z_global_vox","nearest_surface_distance_nm","mapped_skeleton_node_count",
  "skeleton_endpoint_a_node_id","skeleton_endpoint_b_node_id","length_along_skeleton_nm",
  "length_along_skeleton_um","mapped_skeleton_geodesic_span_nm","skeleton_length_method",
  "swc_scale_applied","swc_header_scale","swc_coordinate_units","surface_mapping_method","surface_point_count","voxel_size_xyz_nm","qc","swc_path","neuron_manifest_path"]
 out=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,res in enumerate(pool.map(one,((r,str(a.dataset_root),str(a.neuron_root)) for r in rows),chunksize=1),1):
   out.append(res)
   if i==1 or i%100==0 or i==len(rows): print(f"processed {i}/{len(rows)}",flush=True)
 with a.output_csv.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 print(f"WROTE {a.output_csv}; rows={len(out)}; errors={sum(r.get('qc')!='ok' for r in out)}",flush=True)

if __name__=="__main__":main()
