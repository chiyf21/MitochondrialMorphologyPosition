"""Compare per-instance reads with one haloed spatial-block read."""
import json, os, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from cloudvolume import CloudVolume

INPUT=Path(os.environ["FLYEM_INPUT_JSON"])
DOWNLOADER=Path(os.environ["FLYEM_DOWNLOADER"])
sys.path.insert(0,str(DOWNLOADER.parent))
import download_flyem_mito_instances as dl

def xyz(vol, lo, hi):
 a=np.asarray(vol[int(lo[0]):int(hi[0]),int(lo[1]):int(hi[1]),int(lo[2]):int(hi[2])])
 while a.ndim>3: a=a[...,0]
 return a

def main():
 data=json.loads(INPUT.read_text()); neurons={int(n["bodyId"]):n for n in data["neurons"]}
 block=int(os.environ.get("FLYEM_BATCH_BLOCK","128"))
 counts=Counter((int(r["x"])//block,int(r["y"])//block,int(r["z"])//block) for r in data["mitochondria"])
 tile,n=counts.most_common(int(os.environ.get("FLYEM_TILE_RANK","0"))+1)[int(os.environ.get("FLYEM_TILE_RANK","0"))]
 indices=defaultdict(int); grouped=defaultdict(list)
 for r in data["mitochondria"]:
  b=int(r["bodyId"]); indices[b]+=1; key=(int(r["x"])//block,int(r["y"])//block,int(r["z"])//block)
  if key==tile: grouped[key].append((r,str(neurons[b].get("type") or "unknown"),indices[b]))
 jobs=grouped[tile]
 print("tile",tile,"instances",len(jobs),flush=True)
 scratch=Path(tempfile.mkdtemp(prefix="flyem-block-bench-",dir="/tmp"))
 dl.OUT=scratch/"baseline"
 for r,t,i in jobs:
  (dl.OUT/"results"/str(int(r["bodyId"]))/"instances"/f"instance_{i:06d}").mkdir(parents=True,exist_ok=True)
 baseline={}; base_s=None
 if os.environ.get("BENCH_REVERSE")!="1":
  t0=time.perf_counter()
  for r,t,i in jobs: baseline[(int(r["bodyId"]),i)]=dl.locate_and_save(r,t,i)
  base_s=time.perf_counter()-t0
  print("baseline_s",round(base_s,2),flush=True)

 seg=CloudVolume(dl.MITO_URL,mip=0,use_https=True,progress=False,fill_missing=True,cache=True)
 em=CloudVolume(dl.EM_URL,mip=0,use_https=True,progress=False,fill_missing=True,cache=True)
 bounds_lo=np.asarray(seg.bounds.minpt[:3],dtype=int); bounds_hi=np.asarray(seg.bounds.maxpt[:3],dtype=int)
 halo=np.array([64,64,32]); tile_lo=np.array(tile)*block; tile_hi=tile_lo+block
 lo=np.maximum(tile_lo-halo,bounds_lo); hi=np.minimum(tile_hi+halo,bounds_hi)
 t0=time.perf_counter(); labels=xyz(seg,lo,hi); seg_s=time.perf_counter()-t0
 block_em=None
 t0=time.perf_counter(); block_em=xyz(em,lo,hi); em_s=time.perf_counter()-t0
 dl.OUT=scratch/"batch"; results=[]; fallbacks=0
 for r,typ,i in jobs:
  body=int(r["bodyId"]); center=np.rint([r["x"],r["y"],r["z"]]).astype(int); lc=center-lo
  label=int(labels[tuple(lc)])
  fallback_center=(label==0)
  if label==0:
   fallback_center=True
   nz=np.argwhere(labels>0)
   if len(nz)==0: raise RuntimeError("empty segmentation tile")
   near=nz[np.argmin(((nz-lc)**2).sum(axis=1))]; label=int(labels[tuple(near)])
  full=labels==label
  occ=[np.flatnonzero(full.any(axis=tuple(j for j in range(3) if j!=axis))) for axis in range(3)]
  if fallback_center or any(len(x)==0 for x in occ):
   fallbacks+=1
   dl.OUT=scratch/"fallback"
   d=dl.OUT/"results"/str(body)/"instances"/f"instance_{i:06d}"; d.mkdir(parents=True,exist_ok=True)
   m=dl.locate_and_save(r,typ,i)
   dl.OUT=scratch/"batch"
   results.append((r,typ,i,m,True)); continue
  objlo=np.array([x.min() for x in occ]); objhi=np.array([x.max()+1 for x in occ])
  if np.any(objlo==0) or np.any(objhi==np.asarray(labels.shape)):
   fallbacks+=1
   dl.OUT=scratch/"fallback"
   d=dl.OUT/"results"/str(body)/"instances"/f"instance_{i:06d}"; d.mkdir(parents=True,exist_ok=True)
   m=dl.locate_and_save(r,typ,i)
   dl.OUT=scratch/"batch"
   results.append((r,typ,i,m,True)); continue
  gl=objlo+lo; gh=objhi+lo; cl=np.maximum(gl-4,np.maximum(bounds_lo,lo)); ch=np.minimum(gh+4,np.minimum(bounds_hi,hi))
  sl=tuple(slice(int(cl[k]-lo[k]),int(ch[k]-lo[k])) for k in range(3))
  mask=full[sl]; image=block_em[sl]
  if image.shape!=mask.shape: raise ValueError("EM/mask mismatch")
  out=dl.OUT/"results"/str(body)/"instances"/f"instance_{i:06d}"; out.mkdir(parents=True,exist_ok=True)
  np.save(out/"mask.npy",mask.transpose(2,1,0).astype(np.uint8,copy=False))
  np.save(out/"em.npy",image.transpose(2,1,0).astype(np.uint8,copy=False))
  md={"body_id":str(body),"cell_type":typ,"global_instance":i,"source_mito_label":label,"source_voxel_count":int(r.get("size") or 0),"voxel_count":int(mask.sum()),"center_xyz":center.tolist(),"bbox_xyz":[int(cl[0]),int(cl[1]),int(cl[2]),int(ch[0]),int(ch[1]),int(ch[2])],"array_axis_order":"ZYX","voxel_size_nm_xyz":[8,8,8],"mask_file":"mask.npy","em_file":"em.npy","status":"downloaded"}
  (out/"metadata.json").write_text(json.dumps(md))
  results.append((r,typ,i,md,False))
 batch_s=time.perf_counter()-t0
 if os.environ.get("BENCH_REVERSE")=="1":
  dl.OUT=scratch/"baseline"
  t0=time.perf_counter()
  for r,t,i in jobs: baseline[(int(r["bodyId"]),i)]=dl.locate_and_save(r,t,i)
  base_s=time.perf_counter()-t0
  print("baseline_s_after_batch",round(base_s,2),flush=True)
 mismatches=[]
 for r,typ,i,md,fb in results:
  ref=baseline[(int(r["bodyId"]),i)]
  out=scratch/("fallback" if fb else "batch")/"results"/str(int(r["bodyId"]))/"instances"/f"instance_{i:06d}"
  refdir=scratch/"baseline"/"results"/str(int(r["bodyId"]))/"instances"/f"instance_{i:06d}"
  ref_bbox=ref["bbox_xyz"]; got_bbox=md["bbox_xyz"]
  low=np.maximum(ref_bbox[:3],got_bbox[:3]); high=np.minimum(ref_bbox[3:],got_bbox[3:])
  def overlap(a,bbox):
   return a[low[2]-bbox[2]:high[2]-bbox[2],low[1]-bbox[1]:high[1]-bbox[1],low[0]-bbox[0]:high[0]-bbox[0]]
  if md["source_mito_label"]!=ref["source_mito_label"] or md["voxel_count"]!=ref["voxel_count"]: mismatches.append((int(r["bodyId"]),i,"metadata"))
  for name in ("mask.npy","em.npy"):
   if not np.array_equal(overlap(np.load(out/name),got_bbox),overlap(np.load(refdir/name),ref_bbox)): mismatches.append((int(r["bodyId"]),i,name))
 print(json.dumps({"tile_size_xyz":[block,block,block],"halo_xyz":halo.tolist(),"instances":len(jobs),"baseline_total_s":round(base_s,2),"batch_total_s":round(batch_s+seg_s+em_s,2),"segmentation_block_read_s":round(seg_s,2),"em_block_read_s":round(em_s,2),"local_extract_s":round(batch_s,2),"fallback_count":fallbacks,"mismatch_count":len(mismatches),"mismatches":mismatches[:5],"scratch":str(scratch)},indent=2),flush=True)
if __name__=="__main__": main()
