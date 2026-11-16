#!/usr/bin/env python3
"""Compute paper-style 22D mito morphology features for selected neuron types."""
import argparse, csv, hashlib, json, math, re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from mitochondria_feature_encoder import EncoderConfig, encode_one, plane_cross_section

SPACING_XYZ = np.array([8., 8., 30.])
SPACING_ZYX = SPACING_XYZ[::-1]
TILE_XYZ = np.array([128, 128, 32])
BLOCK_RE = re.compile(r"^x(-?[0-9]+)_y(-?[0-9]+)_z(-?[0-9]+)$")
TYPES = {"GC","CON","VMN","PC","MON2","MON1"}
FEATURES = [
 "size","mito SA","mito CA","convex hull compactness",
 "PC1 Length","PC1 inertia moment","PC1 symmetry","PC1 CA","PC1 Circum",
 "PC2 Length","PC2 inertia moment","PC2 symmetry","PC2 CA","PC2 Circum",
 "PC3 Length","PC3 inertia moment","PC3 symmetry","PC3 CA","PC3 Circum",
 "mito diameter","length_along_skeleton_nm","SBI"
]

def block_origin(path):
 m=BLOCK_RE.match(path.name)
 if not m: raise ValueError(f"invalid block name {path.name}")
 return np.array([int(v) for v in m.groups()],dtype=int)

@lru_cache(maxsize=256)
def load_block_mask(path_text):
 p=Path(path_text)
 if p.suffix==".npz":
  with np.load(p,allow_pickle=False) as z: a=z[z.files[0]]
 else: a=np.load(p,mmap_mode="r",allow_pickle=False)
 if a.shape != tuple(TILE_XYZ):
  raise ValueError(f"block array expected XYZ shape {tuple(TILE_XYZ)}, got {a.shape}: {p}")
 return np.asarray(a>0)

@lru_cache(maxsize=64)
def block_dirs(body_dir):
 return tuple(str(p) for p in (Path(body_dir)/"blocks").glob("x*_y*_z*"))

def load_context(neuron_root,body,center_xyz_vox,radius_nm):
 radius=np.ceil(radius_nm/SPACING_XYZ).astype(int)+1
 lo=np.floor(center_xyz_vox).astype(int)-radius
 hi=np.floor(center_xyz_vox).astype(int)+radius+1
 out=np.zeros(tuple((hi-lo)[::-1]),dtype=bool)
 for d in block_dirs(str(Path(neuron_root)/str(body))):
  dp=Path(d); start=block_origin(dp); stop=start+TILE_XYZ
  a0=np.maximum(lo,start); a1=np.minimum(hi,stop)
  if np.any(a1<=a0): continue
  mp=next((dp/n for n in ("neuron_mask.npy","neuron_mask.npz") if (dp/n).is_file()),None)
  if mp is None: continue
  arr=load_block_mask(str(mp))
  sx= slice(int(a0[0]-start[0]),int(a1[0]-start[0]))
  sy= slice(int(a0[1]-start[1]),int(a1[1]-start[1]))
  sz= slice(int(a0[2]-start[2]),int(a1[2]-start[2]))
  dx= slice(int(a0[0]-lo[0]),int(a1[0]-lo[0]))
  dy= slice(int(a0[1]-lo[1]),int(a1[1]-lo[1]))
  dz= slice(int(a0[2]-lo[2]),int(a1[2]-lo[2]))
  out[dz,dy,dx] |= arr[sx,sy,sz].transpose(2,1,0)
 return out,lo

@lru_cache(maxsize=128)
def load_swc(body_dir,body):
 path=Path(body_dir)/"skeleton"/f"{body}.swc"
 nodes=[]; parents=[]
 for line in path.read_text(errors="replace").splitlines():
  if not line or line.startswith("#"): continue
  f=line.split()
  if len(f)>=7: nodes.append([float(f[2]),float(f[3]),float(f[4]),float(f[5])]); parents.append(int(float(f[6])))
 xyzr=np.asarray(nodes,float); ids=np.arange(len(nodes))
 if not len(nodes): raise ValueError(f"no SWC nodes: {path}")
 tree=cKDTree(xyzr[:,:3])
 id_to_idx={int(float(line.split()[0])):i for i,line in enumerate([x for x in path.read_text(errors="replace").splitlines() if x and not x.startswith("#")])}
 # Reparse IDs safely, preserving SWC node identifiers.
 raw=[x.split() for x in path.read_text(errors="replace").splitlines() if x and not x.startswith("#")]
 node_ids=np.asarray([int(float(x[0])) for x in raw],dtype=np.int64)
 id_to_idx={int(v):i for i,v in enumerate(node_ids)}
 adj=[[] for _ in nodes]
 for i,parent in enumerate(parents):
  if parent < 0: continue
  if parent not in id_to_idx: continue
  j=id_to_idx[parent]
  adj[i].append(j); adj[j].append(i)
 return tree,xyzr,adj,node_ids

def tangent_at(body_dir,body,center_nm):
 tree,xyzr,adj,node_ids=load_swc(str(Path(body_dir)/str(body)),str(body))
 _,idx=tree.query(center_nm)
 idx=int(idx); peers=adj[idx]
 if not peers:
  _,inds=tree.query(center_nm,k=min(4,len(xyzr)))
  peers=[int(i) for i in np.atleast_1d(inds) if int(i)!=idx]
 if not peers: raise ValueError("nearest SWC node has no tangent neighbor")
 j=max(peers,key=lambda k:float(np.linalg.norm(xyzr[k,:3]-xyzr[idx,:3])))
 v=xyzr[j,:3]-xyzr[idx,:3]; norm=np.linalg.norm(v)
 if norm==0: raise ValueError("zero length SWC edge")
 return v/norm,float(xyzr[idx,3]),int(node_ids[idx])

def surface_area_faces(mask):
 m=np.asarray(mask,bool); dz,dy,dx=SPACING_ZYX
 a=(m[1:] & ~m[:-1]).sum()+(m[:-1] & ~m[1:]).sum()
 b=(m[:,1:] & ~m[:,:-1]).sum()+(m[:,:-1] & ~m[:,1:]).sum()
 c=(m[:,:,1:] & ~m[:,:,:-1]).sum()+(m[:,:,:-1] & ~m[:,:,1:]).sum()
 a+=(m[0].sum()+m[-1].sum()); b+=(m[:,0].sum()+m[:,-1].sum()); c+=(m[:,:,0].sum()+m[:,:,-1].sum())
 return float(a*dy*dx+b*dz*dx+c*dz*dy)

def compute_one(item):
 root,neuron_root,row,context_nm=item
 base={k:row.get(k,"") for k in ("record_id","source_run_id","body_id","cell_type","global_instance","voxel_count")}
 try:
  meta=json.loads((Path(root)/row["metadata_path"]).read_text())
  arr=np.load(Path(root)/row["mask_path"],mmap_mode="r",allow_pickle=False)
  if arr.ndim!=3: raise ValueError(f"mito mask not ZYX: {arr.shape}")
  x0,y0,z0,x1,y1,z1=map(int,meta["bbox_xyz"])
  if tuple(arr.shape[::-1]) not in ( (x1-x0,y1-y0,z1-z0),(x1-x0+1,y1-y0+1,z1-y0+1) ):
   # use explicit per-axis check below to preserve the half-open bbox convention
   if not np.all(np.abs(np.array(arr.shape[::-1])-[x1-x0,y1-y0,z1-z0])<=1):
    raise ValueError(f"bbox/mask mismatch: {meta['bbox_xyz']} {arr.shape}")
  coords=np.nonzero(arr)
  if not len(coords[0]): raise ValueError("empty mito mask")
  lo=np.array([int(c.min()) for c in coords]); hi=np.array([int(c.max())+1 for c in coords])
  mask=np.asarray(arr[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]>0)
  origin_xyz=np.array([x0+lo[2],y0+lo[1],z0+lo[0]],float)
  center_local_zyx=np.array([(mask.shape[0]-1)/2,(mask.shape[1]-1)/2,(mask.shape[2]-1)/2])
  # Use the mask centroid rather than the crop center.
  zz,yy,xx=np.nonzero(mask)
  center_local_zyx=np.array([zz.mean(),yy.mean(),xx.mean()])
  center_xyz_vox=origin_xyz+center_local_zyx[::-1]
  center_xyz_nm=center_xyz_vox*SPACING_XYZ
  seed=int.from_bytes(hashlib.sha256(row["record_id"].encode()).digest()[:8],"little")
  cfg=EncoderConfig(spacing_zyx_nm=tuple(SPACING_ZYX),min_voxels=1)
  morph=encode_one(mask,(slice(z0+lo[0],z0+hi[0]),slice(y0+lo[1],y0+hi[1]),slice(x0+lo[2],x0+hi[2])),None,cfg,np.random.default_rng(seed))
  morph["mito SA"]=surface_area_faces(mask)
  size=float(mask.sum()*np.prod(SPACING_XYZ))
  er=((3*size)/(4*np.pi))**(1/3)
  morph["size"]=size
  morph["SBI"]=float(np.log10((4*np.pi*er*er)/morph["mito SA"])) if morph["mito SA"]>0 else math.nan
  tangent,radius,node_id=tangent_at(neuron_root,row["body_id"],center_xyz_nm)
  nmask,nlo=load_context(neuron_root,row["body_id"],center_xyz_vox,context_nm)
  origin_idx=np.array([center_xyz_vox[2]-nlo[2],center_xyz_vox[1]-nlo[1],center_xyz_vox[0]-nlo[0]])
  half=max(400.,min(context_nm-16.,2*radius if radius>0 else 600.))
  skel_ca,skel_circ=plane_cross_section(nmask,origin_idx,tangent,SPACING_ZYX,half,8.)
  mito_ca,mito_circ=plane_cross_section(mask,center_local_zyx,tangent,SPACING_ZYX,half,8.)
  morph["mito CA"]=mito_ca
  values={k:morph.get(k,math.nan) for k in FEATURES if k!="length_along_skeleton_nm"}
  values["length_along_skeleton_nm"]=math.nan
  aux={"skel CA":skel_ca,"skel Circum":skel_circ,"nearest_skeleton_node_id":node_id,
       "skeleton_radius_nm":radius,"mito_CA_method":"local_SWC_tangent_plane",
       "skel_section_method":"block_neuron_mask_tangent_plane",
       "mito_SA_method":"anisotropic_exposed_voxel_faces",
       "voxel_size_xyz_nm":"8,8,30"}
  return {**base,**values,**aux,"status":"ok"}
 except Exception as e:
  return {**base,"status":f"error: {type(e).__name__}: {e}"}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--dataset-root",type=Path,required=True)
 ap.add_argument("--neuron-root",type=Path,required=True)
 ap.add_argument("--output-csv",type=Path,required=True)
 ap.add_argument("--workers",type=int,default=16)
 ap.add_argument("--context-nm",type=float,default=1000.)
 ap.add_argument("--start-index",type=int,default=0)
 ap.add_argument("--limit",type=int)
 a=ap.parse_args()
 rows=list(csv.DictReader((a.dataset_root/"metadata/instances.csv").open(newline="",encoding="utf-8")))
 rows=[r for r in rows if r["cell_type"] in TYPES]
 rows=rows[a.start_index:]
 if a.limit is not None: rows=rows[:a.limit]
 a.output_csv.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(["record_id","source_run_id","body_id","cell_type","global_instance","voxel_count",*FEATURES,"skel CA","skel Circum","nearest_skeleton_node_id","skeleton_radius_nm","mito_CA_method","skel_section_method","mito_SA_method","voxel_size_xyz_nm","status"]))
 with ProcessPoolExecutor(max_workers=a.workers) as pool, a.output_csv.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  items=((str(a.dataset_root),str(a.neuron_root),r,a.context_nm) for r in rows)
  for i,res in enumerate(pool.map(compute_one,items,chunksize=1),1):
   w.writerow(res)
   if i==1 or i%100==0 or i==len(rows): print(f"processed {i}/{len(rows)}",flush=True)
 print(f"WROTE {a.output_csv}; rows={len(rows)}")
if __name__=="__main__": main()
