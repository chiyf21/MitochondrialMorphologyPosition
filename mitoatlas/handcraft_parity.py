"""Sampled parity audit: run the upstream geometry kernels on FISH2 masks."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, center_of_mass, convolve, measurements, zoom
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import measure
from sklearn.decomposition import PCA

from fish2_downstream.common import dump
from .handoff import inputs
from HPC.mito_node_dists.mitochondria_feature_encoder import EncoderConfig, _pca_xyz, plane_cross_section
import sys
sys.path.insert(0, str(Path("HPC/mito_node_dists").resolve()))
from compute_selected_morphology import load_context

SPACING = np.array([30., 8., 8.])
ISO = 8.
SEED = 25


def select_functions(path, names, scope):
    source = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = [x for x in source.body if isinstance(x, ast.FunctionDef) and x.name in names]
    if {f.name for f in selected} != set(names):
        raise ValueError(f"Upstream functions missing in {path}")
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,"exec"),scope)
    return scope


def upstream():
    util = select_functions("util_files/utils.py",
        ("spherical_2_cart","cart_2_spherical","calc_orthonormal_basis"),{"np":np})
    scope = dict(np=np, utils=SimpleNamespace(**util), measure=measure, minimize=minimize,
                 measurements=measurements, binary_fill_holes=binary_fill_holes,
                 binary_erosion=binary_erosion,binary_dilation=binary_dilation,convolve=convolve)
    return select_functions("util_files/voxel_utils.py",("adjust_center","get_closest_true_idx",
        "get_center_object","calc_plane_coords","find_cross_section","get_border_mask",
        "get_foreground_stats","is_cropped","find_best_cross_section","get_CrossSection_info"),scope)


def tight(vol):
    coord = np.array(np.nonzero(vol))
    lo,hi = coord.min(axis=1),coord.max(axis=1)+1
    return vol[tuple(slice(a,b) for a,b in zip(lo,hi))]


def iso_binary(vol):
    # FISH2 differs physically from the upstream isotropic grid. Keep class
    # labels binary and use the native spacing for the two in-plane axes.
    return zoom(vol,SPACING/ISO,order=0,grid_mode=True,mode="grid-constant")


def paired_ca(row, vol, root, native):
    cfg_path = Path(root)/str(row.body_id)/"blocks"
    if not cfg_path.is_dir():
        return {"neuron_ca_status":"missing_blocks"}
    import json
    box = np.asarray(json.loads(row.bbox_xyz),dtype=int).reshape(2,3)
    center = box[0] + np.asarray(center_of_mass(vol))[::-1]
    neuron, lo = load_context(root,row.body_id,center,1500.)
    coords = np.array(np.nonzero(vol))
    mask_coords = coords + (box[0]-lo)[::-1,None]
    keep = ((mask_coords>=0)&(mask_coords<np.asarray(neuron.shape)[:,None])).all(axis=0)
    coords,mask_coords = coords[:,keep],mask_coords[:,keep]
    overlap = np.count_nonzero(neuron[tuple(mask_coords)])/np.count_nonzero(vol)
    if overlap < .90:
        return {"neuron_ca_status":"low_neuron_mask_overlap","neuron_overlap":overlap}
    m = np.zeros_like(neuron)
    m[tuple(mask_coords)] = True
    volumes = {"segmentation":iso_binary(neuron),"mito-objects":iso_binary(m)}
    def crop(bounds, kind):
        lower,upper = np.asarray(bounds,dtype=int)
        src = volumes[kind]
        if (lower<0).any() or (upper>src.shape).any():
            raise ValueError("native context exceeded crop")
        return src[tuple(slice(a,b) for a,b in zip(lower,upper))]
    native["get_subvol_any_size"] = crop
    anchor = np.rint((center-lo)*np.array([8.,8.,30.])/ISO).astype(int)
    try:
        skel,mito = native["get_CrossSection_info"](1,anchor,100,return_mito=True,max_dr=100)
    except (AssertionError,ValueError,IndexError) as e:
        return {"neuron_ca_status":f"native_section_failed:{type(e).__name__}","neuron_overlap":overlap}
    if skel is None or mito is None:
        return {"neuron_ca_status":"native_section_cropped","neuron_overlap":overlap}
    return {"neuron_ca_status":"ok","neuron_overlap":overlap,
            "upstream_mito_ca":np.count_nonzero(mito)*ISO**2,
            "upstream_neuron_ca":np.count_nonzero(skel)*ISO**2}


def audit(job, root):
    row,run_context = job
    native = upstream()
    vol = np.load(row.mask_path,mmap_mode="r",allow_pickle=False)>0
    mask = tight(vol)
    iso = np.pad(tight(iso_binary(mask)),2)
    native_stats = native["get_foreground_stats"]
    _,sa = native_stats(iso,None,return_COM=False)
    exposed = sum(np.count_nonzero(np.diff(np.pad(mask,1),axis=i))*a
                  for i,a in enumerate((64,240,240)))
    coords = np.argwhere(iso)[:,::-1]
    n = max(3,len(coords)//100)
    seed = int(row.global_instance)
    rng = random.Random(seed)
    axes = PCA().fit(coords[rng.sample(range(len(coords)),n)]).components_
    project = (coords-coords[rng.sample(range(len(coords)),n)].mean(axis=0)) @ axes.T
    angles = native["utils"].cart_2_spherical
    cfg = EncoderConfig()
    physical = np.argwhere(mask)[:,::-1]*SPACING[::-1]
    directions,_ = _pca_xyz(physical,np.random.default_rng(seed),cfg.pca_max_points)
    volume = int(mask.sum())*np.prod(SPACING)
    eq_radius = np.cbrt(3*volume/(4*np.pi))
    half = cfg.cs_half_extent_factor*max(eq_radius,2*SPACING.max())
    result = dict(record_id=row.record_id,source=row.handcrafted_feature_source,
                  body_id=row.body_id,upstream_size=len(coords)*ISO**3,
                  upstream_sa=sa*ISO**2,full_exposed_sa=exposed,
                  upstream_hull=ConvexHull(coords).area*ISO**2)
    result["upstream_diameter"] = pdist(coords[rng.sample(range(len(coords)),n)]).max()*ISO
    for i,(axis,direction) in enumerate(zip(axes,directions),1):
        _,theta,phi = angles(*axis)
        try:
            section = native["find_cross_section"](theta,phi,iso)
            area,circum = native_stats(section,None,return_COM=False)
        except AssertionError:
            area,circum = np.nan,np.nan
        result[f"upstream_PC{i}_CA"] = area*ISO**2
        result[f"upstream_PC{i}_Circum"] = circum*ISO
        result[f"upstream_PC{i}_Length"] = np.ptp(project[:,i-1])*ISO
        # Existing sampler multiplies its supplied normal by voxel spacing.
        # Divide first to isolate the anisotropic normal conversion.
        corrected = direction/SPACING[::-1]
        area,circum = plane_cross_section(mask,center_of_mass(mask),corrected,SPACING,half,cfg.cs_sample_step_nm)
        result[f"normal_fix_PC{i}_CA"] = area
        result[f"normal_fix_PC{i}_Circum"] = circum
    if run_context:
        result.update(paired_ca(row,vol,root,native))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff","neuron_root","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--per-source",type=int,default=30)
    p.add_argument("--contexts-per-source",type=int,default=6)
    args = p.parse_args()
    frame,reps,_ = inputs(args.handoff)
    rows = frame.groupby("handcrafted_feature_source",group_keys=False).sample(n=args.per_source,random_state=SEED)
    jobs = [(row,i<args.contexts_per_source) for _,group in rows.groupby("handcrafted_feature_source")
            for i,row in enumerate(group.itertuples(index=False))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        result = pd.DataFrame(pool.map(partial(audit,root=args.neuron_root),jobs))
    original = reps["handcrafted"].set_axis(frame.record_id).add_prefix("stored_")
    result = result.join(original,on="record_id",validate="one_to_one")
    args.out.mkdir(parents=True,exist_ok=False)
    result.to_csv(args.out/"paired_values.csv",index=False)
    dump(args.out/"run.json",dict(aspacing_zyx_nm=SPACING.tolist(),upstream_iso_nm=ISO,
         sample_seed=SEED,source_counts=result.source.value_counts().to_dict(),
         context_status=result.neuron_ca_status.value_counts().to_dict(),
         commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         comparisons="Upstream routines on NN 8 nm isotropic masks; neuron context uses available block masks, bounded native 100-voxel crop"))


if __name__ == "__main__":
    main()
