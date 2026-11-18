"""Recompute the FISH2 handcraft panel using ClarkLab geometry kernels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from fish2_downstream.common import dump
from .handoff import inputs
from .handcraft_parity import ISO, SPACING, iso_binary, paired_ca, select_functions, tight, upstream

KEY = ["record_id","body_id","embedding_row"]
UPSTREAM_FILES = ["util_files/utils.py","util_files/voxel_utils.py","HPC/save_mito_synapse/mito_synapse_main.py"]
UPSTREAM_REV = "a1874cf"


def native_versions():
    # Reject accidental changes to the referenced ClarkLab sources.
    for path in UPSTREAM_FILES:
        expected = subprocess.check_output(["git","rev-parse",f"{UPSTREAM_REV}:{path}"],text=True).strip()
        current = subprocess.check_output(["git","hash-object",path],text=True).strip()
        if expected != current:
            raise ValueError(f"Upstream file changed: {path}")


@lru_cache(maxsize=1)
def shape_kernels():
    return upstream()


def unique_count(points):
    # A one-dimensional voxel key gives the same distinct triples as unique(axis=0).
    origin = points.min(axis=0)
    extent = points.max(axis=0)-origin+1
    return len(np.unique(np.ravel_multi_index((points-origin).T,tuple(extent))))


def sample_diameter(points):
    # A farthest pair is on the convex hull. This keeps the exact maximum while
    # avoiding a quadratic distance array when the author-style 1% sample is large.
    if len(points) >= 8:
        try:
            points = points[ConvexHull(points).vertices]
        except QhullError:
            pass
    return pdist(points).max()


def section_kernels():
    kernels = upstream()
    original = kernels["is_cropped"]

    def is_cropped(section, box_shape=None, theta=None, phi=None):
        if theta is None:
            return original(section,box_shape,theta,phi)
        bounds = np.array([[0,0,0],box_shape])
        xyz = kernels["calc_plane_coords"](theta,phi,bounds)
        support = ((xyz>=0)&(xyz<bounds[1,::-1])).all(axis=-1)
        support = kernels["get_center_object"](kernels["binary_fill_holes"](support))
        return np.any(kernels["get_border_mask"](support,"on") & (section==1))

    kernels["is_cropped"] = is_cropped
    return kernels


def geometry(row):
    fixed = {k:getattr(row,k) for k in KEY}
    try:
        mask = np.load(row.mask_path,mmap_mode="r",allow_pickle=False)>0
        count = np.count_nonzero(mask)
        if not count or count != row.voxel_count:
            raise ValueError("Mitochondrial mask voxel count disagrees with handoff")
        iso = np.pad(tight(iso_binary(tight(mask))),2)
        points = np.argwhere(iso)[:,::-1]
        native = shape_kernels()
        stats = native["get_foreground_stats"]
        _, sa = stats(iso,None,return_COM=False)
        sample_n = len(points)//100
        if sample_n<3:
            raise ValueError("Upstream one-percent PCA sample has fewer than three points")
        rng = random.Random(int(row.global_instance))
        pca = PCA().fit(points[rng.sample(range(len(points)),sample_n)])
        pc = pca.transform(points)
        result = fixed | dict(status="ok",size=count*np.prod(SPACING),iso_size=len(points)*ISO**3,
                              **{"mito SA":sa*ISO**2,"convex hull compactness":ConvexHull(points).area*ISO**2})
        issues = []
        pc_integer = pc.astype(int)
        n = unique_count(pc_integer)
        for k,direction in enumerate(pca.components_):
            flip = pc_integer.copy()
            flip[:,k] *= -1
            symmetry = 2 - unique_count(np.r_[flip,pc_integer])/n
            _,theta,phi = native["utils"].cart_2_spherical(*direction)
            try:
                section = native["find_cross_section"](theta,phi,iso)
                area,perimeter = stats(section,None,return_COM=False)
            except AssertionError:
                area,perimeter = np.nan,np.nan
                issues.append(f"PC{k+1}_empty_plane")
            values = {"Length":np.ptp(pc[:,k])*ISO,"inertia moment":np.square(np.delete(pc,k,axis=1)).sum()*ISO**2,
                      "symmetry":symmetry,"CA":area*ISO**2,"Circum":perimeter*ISO}
            result.update({f"PC{k+1} {key}":value for key,value in values.items()})
        distance_sample = points[rng.sample(range(len(points)),sample_n)]
        result["mito diameter"] = sample_diameter(distance_sample)*ISO
        result["SBI"] = np.log10((36*np.pi*result["size"]**2)**(1/3)/result["mito SA"])
        result["qc"] = "|".join(issues) if issues else "ok"
        return result
    except (AssertionError,ValueError,QhullError) as error:
        return fixed | dict(status="failed",qc=f"{type(error).__name__}: {error}")


@lru_cache(maxsize=24)
def skeleton_table(snapshot, body, correct_scale):
    nodes = pd.read_parquet(snapshot/"automatic"/str(body)/"skeleton_nodes.parquet")
    physical = nodes[["x_em","y_em","z_em"]].to_numpy()*SPACING[::-1]
    if correct_scale:
        physical *= [16,16,15]
    # SWC file radius is physical nm. The same 8 nm unit is used throughout the native math.
    xyz = physical / ISO
    pos = {v:i for i,v in enumerate(nodes.rowId)}
    parents = np.array([pos.get(v,-1) for v in nodes.link],dtype=int)
    vector = xyz[parents]-xyz
    vector[parents<0] = [0.,0.,1.]
    length = np.linalg.norm(vector,axis=1)
    if (length<=0).any():
        raise ValueError("Skeleton has zero-length SWC edges")
    theta = np.arccos(np.clip(vector[:,2]/length,-1,1))
    phi = np.arctan2(vector[:,1],vector[:,0])
    length[parents<0] = 0.
    return pd.DataFrame(np.c_[nodes.rowId,xyz,nodes.radius/ISO,nodes.link,length,theta,phi],
                        columns=["rowId","x","y","z","radius","link","distance","theta","phi"])


def section_length(row, roots, snapshot):
    native = section_kernels()
    volatile = {k:getattr(row,k) for k in KEY}
    mask = np.load(row.mask_path,mmap_mode="r",allow_pickle=False)>0
    result = {"neuron_ca_status":"missing_blocks","skeleton_length_status":"requires_mito_ca"}
    for root in roots:
        try:
            if not (root/str(row.body_id)/"blocks").is_dir():
                continue
            result = paired_ca(row,mask,root,native,section_max_dr=150)
        except (OSError,ValueError) as error:
            result = {"neuron_ca_status":f"context_error:{type(error).__name__}"}
        if result["neuron_ca_status"]=="ok":
            break
    result["skeleton_length_status"] = "requires_mito_ca"
    if result["neuron_ca_status"]!="ok":
        return volatile | result
    try:
        s = skeleton_table(snapshot,row.body_id,row.skeleton_xyz_rescaled)
        graph = s.to_numpy(float)
        helpers = select_functions("util_files/utils.py",("get_down_idxs","get_up_idx_dist","overlay_mito_on_df"),
                                   dict(np=np,spherical_2_cart=native["utils"].spherical_2_cart))
        origin = np.asarray(json.loads(row.bbox_xyz)[:3],dtype=float)
        points = np.argwhere(iso_binary(mask))[:,::-1].astype(float)
        points += origin*SPACING[::-1]/ISO
        length = row.pc1_length / ISO
        radius = np.sqrt(result["upstream_mito_ca"]/np.pi) / ISO
        edge = helpers["overlay_mito_on_df"](s,s,graph,points,0,length,radius)
        start,end = edge[0],edge[-1]
        if start[3]!="up" or end[3]!="down":
            raise ValueError("Unexpected upstream overlay endpoints")
        if start[1] == end[0]:
            status = "upstream_same_edge_pc1_fallback"
        else:
            path = helpers["get_down_idxs"](graph,start[1],graph[:,0]==end[1],count_start=True)
            if not len(path) or graph[path[-1],0]!=end[1]:
                raise ValueError("No connected upstream-to-downstream path")
            total = float(start[2]+graph[path[:-1],6].sum()+end[2])
            if np.isfinite(total) and total>=0:
                length,status = total,"upstream_edge_overhang"
            else:
                status = "upstream_negative_pc1_fallback"
        result["length_along_skeleton_nm"] = length*ISO
        result["skeleton_length_status"] = status
    except (AssertionError,IndexError,ValueError,TypeError) as error:
        result["skeleton_length_status"] = f"upstream_overlay_failed:{type(error).__name__}"
    return volatile | result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["geometry","sections","assemble"])
    for key in ("handoff","out"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--roots",type=Path,nargs="+",default=[])
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--workers",type=int,default=24)
    args = parser.parse_args()
    native_versions()
    frame,reps,hashes = inputs(args.handoff)
    args.out.mkdir(parents=True,exist_ok=True)
    geom = args.out/"geometry.csv"
    sec = args.out/"sections.csv"
    filename = {"geometry":geom,"sections":sec,"assemble":args.out/"handcrafted_features22.csv"}[args.phase]
    if filename.exists():
        raise FileExistsError(filename)
    if args.phase == "assemble":
        left = pd.read_csv(geom,dtype={"body_id":str})
        right = pd.read_csv(sec,dtype={"body_id":str})
        join = frame[KEY+["cell_type","compartment","handcrafted_feature_source"]].merge(
            left,on=KEY,validate="one_to_one").merge(right,on=KEY,validate="one_to_one")
        if len(join)!=len(frame):
            raise ValueError("Unexpected recomputation row count")
        join["mito CA"] = join.upstream_mito_ca
        feature = list(reps["handcrafted"])
        join["complete22"] = np.isfinite(join[feature]).all(axis=1)
        join.to_csv(filename,index=False)
        join.loc[join.complete22,KEY+feature].to_csv(args.out/"complete22_subset.csv",index=False)
        dump(args.out/"summary.json",dict(rows=len(join),complete22=int(join.complete22.sum()),
             geometry_status=join.status.value_counts().to_dict(),
             mito_ca_status=join.neuron_ca_status.value_counts().to_dict(),
             length_status=join.skeleton_length_status.value_counts().to_dict(),
             coverage={k:int(join[k].notna().sum()) for k in feature}))
        return
    if args.phase == "sections":
        if not args.roots or args.snapshot is None:
            parser.error("sections requires --roots and --snapshot")
        computed = pd.read_csv(geom,usecols=["record_id","PC1 Length"]).rename(columns={"PC1 Length":"pc1_length"})
        frame = frame.merge(computed,on="record_id",validate="one_to_one")
        work = partial(section_length,roots=args.roots,snapshot=args.snapshot)
    else:
        work = geometry
    if args.phase == "geometry":
        frame = frame.sort_values("voxel_count")
    jobs = (SimpleNamespace(**row) for row in frame.to_dict("records"))
    temporary = filename.with_suffix(".inprogress")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        buffer, count = [],0
        for result in pool.map(work,jobs,chunksize=4):
            buffer.append(result)
            count += 1
            if count % 200 == 0:
                print(f"{args.phase}: {count}/{len(frame)}",flush=True)
        pd.DataFrame(buffer).to_csv(temporary,index=False)
    temporary.rename(filename)
    dump(args.out/f"{args.phase}_run.json",dict(arguments=vars(args),handoff_hashes=hashes,
         upstream_revision=UPSTREAM_REV,commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
    print(f"{args.phase}: {count} rows written",flush=True)


if __name__=="__main__":
    main()
