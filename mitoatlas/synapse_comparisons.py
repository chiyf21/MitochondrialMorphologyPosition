"""FISH2 E02 geometry-matched EM checks and E06 pre/post proximity contrasts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from fish2_downstream.common import dump, sha
from .handoff import inputs

SEED = 20261117


def arrays(row):
    em = np.load(row.em_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(row.mask_path, mmap_mode="r", allow_pickle=False) != 0
    if em.shape != mask.shape or em.ndim != 3 or not mask.any():
        raise ValueError(f"Incompatible EM and mask: {row.record_id}")
    return em, mask


def texture(row):
    em, mask = arrays(row)
    values = np.asarray(em[mask], dtype=float)
    lo, hi = np.quantile(values, [.1,.9])
    sample = em.ravel()[::max(1,em.size//20000)]
    crop = np.diff(np.quantile(sample,[.01,.99]))[0]
    result = dict(record_id=row.record_id,
                  crop_scaled_contrast=(hi-lo)/crop if crop > 0 else np.nan)
    gradient = []
    for axis in (1, 2):
        a, b = [slice(None)]*3, [slice(None)]*3
        a[axis], b[axis] = slice(1,None),slice(None,-1)
        a, b = tuple(a),tuple(b)
        both = mask[a] & mask[b]
        d = abs(em[a][both].astype(float)-em[b][both])
        gradient.append(d.mean() if len(d) else np.nan)
    g = np.asarray(gradient)
    result["in_plane_roughness"] = g.mean()/(hi-lo) if hi > lo else np.nan
    result["in_plane_anisotropy"] = abs(g[0]-g[1])/g.sum() if g.sum() > 0 else np.nan
    return result


def textures(frame, ids, out):
    selected = frame[frame.record_id.isin(ids)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        data = pd.DataFrame(pool.map(texture,selected.itertuples(index=False)))
    data.to_csv(out/"texture_features.csv",index=False)
    return data.set_index("record_id")


def summarize(table, cols, body, context):
    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in table.groupby(context) if context else [((),table)]:
        for col in cols:
            use = group[[body,col]].dropna()
            value = use.groupby(body)[col].mean().to_numpy()
            if not len(value):
                continue
            sampled = value[rng.integers(len(value),size=(1999,len(value)))].mean(1)
            lo,hi = np.quantile(sampled,[.025,.975])
            entries = key if isinstance(key,tuple) else (key,)
            rows.append(dict(zip(context,entries))|dict(feature=col,comparisons=len(use),
                bodies=len(value),estimate=value.mean(),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)


def gallery(examples, frame, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes

    path = out/"blind_gallery"
    path.mkdir()
    metadata = frame.set_index("record_id",drop=False)
    key = []
    shuffled = examples.sample(frac=1,random_state=SEED).reset_index(drop=True)
    for i, r in enumerate(shuffled.itertuples(index=False),1):
        row = metadata.loc[r.record_id]
        em,mask = arrays(row)
        zs = np.sum(mask,axis=(1,2)).argmax()
        ys = np.sum(mask,axis=(0,2)).argmax()
        xs = np.sum(mask,axis=(0,1)).argmax()
        fig, axes = plt.subplots(1,3,figsize=(8,3))
        values = em[mask]
        lo,hi = np.quantile(values,[.02,.98])
        for ax, image, outline, aspect, title in zip(
                axes,(em[zs],em[:,ys],em[:,:,xs]),(mask[zs],mask[:,ys],mask[:,:,xs]),
                (1,30/8,30/8),("XY","XZ","YZ")):
            ax.imshow(image,cmap="gray",vmin=lo,vmax=hi,aspect=aspect)
            if outline.any() and not outline.all() and min(outline.shape)>1:
                ax.contour(outline,levels=[.5],colors=["#54d6b7"],linewidths=.55)
            ax.set_title(title)
            ax.axis("off")
        fig.savefig(path/f"{i:03}.png",dpi=120,bbox_inches="tight")
        plt.close(fig)
        verts,faces,_,_ = marching_cubes(np.pad(mask,1),level=.5,spacing=(30.,8.,8.))
        with (path/f"{i:03}.obj").open("w") as file:
            file.writelines(f"v {x-8:.3f} {y-8:.3f} {z-30:.3f}\n" for z,y,x in verts)
            file.writelines(f"f {a+1} {b+1} {c+1}\n" for a,b,c in faces[:,::-1])
        key.append(r._asdict()|dict(blind_id=i,body_id=row.body_id,em_path=row.em_path,mask_path=row.mask_path))
    pd.DataFrame(key).to_csv(out/"gallery_key.csv",index=False)


def e02(frame, hand, source, out):
    pairs = pd.read_csv(source,dtype={"body_a":str,"body_b":str})
    features = hand.set_axis(frame.record_id)[["size","PC1 Length"]]
    for col in features:
        a = pairs.record_a.map(features[col])
        b = pairs.record_b.map(features[col])
        pairs[col] = np.maximum(a,b)/np.minimum(a,b)
    groups = ["cell_type","compartment"]
    pairs["geometry_pct"] = pairs.groupby(groups).geometry_distance.rank(pct=True)
    pairs = pairs[pairs.geometry_pct.le(.50) & pairs["size"].le(1.25) & pairs["PC1 Length"].le(1.25)]
    candidates = []
    for _, group in pairs.groupby("record_a"):
        ordered = group.sort_values(["concat_distance","record_b"])
        low,high = ordered.iloc[0],ordered.iloc[-1]
        if (low.record_b != high.record_b and low.concat_distance_percentile <= .5
                and high.concat_distance_percentile >= .75
                and abs(low.geometry_pct-high.geometry_pct) <= .20):
            candidates.append(dict(body_id=high.body_a,cell_type=high.cell_type,compartment=high.compartment,
                                   anchor=high.record_a,high=high.record_b,low=low.record_b,
                                   high_body=high.body_b,low_body=low.body_b,
                                   high_geometry=high.geometry_distance,low_geometry=low.geometry_distance,
                                   high_concat=high.concat_distance,low_concat=low.concat_distance))
    matched = pd.DataFrame(candidates)
    matched.to_csv(out/"matched_triples.csv",index=False)
    if matched.empty:
        return
    cols = ["anchor","high","low"]
    data = textures(frame,set(matched[cols].to_numpy().ravel()),out)
    for name in data:
        a = matched.anchor.map(data[name])
        matched[name] = abs(a-matched.high.map(data[name]))-abs(a-matched.low.map(data[name]))
    summarize(matched,list(data),"body_id",["compartment"]).to_csv(out/"effects.csv",index=False)
    matched.to_csv(out/"texture_comparisons.csv",index=False)
    pick = matched.assign(spread=matched.high_concat-matched.low_concat).sort_values(
        "spread",ascending=False).groupby("compartment").head(4).reset_index(drop=True)
    examples = pick[cols].rename_axis("example").stack().rename("record_id").reset_index()
    gallery(examples,frame,out)


def matched_groups(group, left, right):
    a = group[left].sort_values("record_id")
    b = group[right].sort_values("record_id")
    if a.empty or b.empty:
        return []
    xy = np.log(a[["size","PC1 Length"]].to_numpy())
    uv = np.log(b[["size","PC1 Length"]].to_numpy())
    diff = abs(xy[:,None,:]-uv[None,:,:])
    i,j = np.where((diff <= np.log(1.25)).all(axis=2))
    cost = diff[i,j].sum(axis=1)
    used_a,used_b,out = set(),set(),[]
    for k in np.argsort(cost,kind="stable"):
        x,y = i[k],j[k]
        if x not in used_a and y not in used_b:
            used_a.add(x)
            used_b.add(y)
            out.append((a.record_id.iloc[x],b.record_id.iloc[y],cost[k]))
    return out


def e06(frame, reps, source, out):
    labels = pd.read_csv(source,dtype={"body_id":str})
    labels = labels[labels.radius_um.eq(1)].copy()
    hand = reps["handcrafted"].set_axis(frame.record_id)
    labels = labels.join(hand[["size","PC1 Length"]],on="record_id")
    for role in ("pre","post"):
        dist = labels[f"nearest_{role}_um"]
        labels[f"{role}_near"] = dist.le(1).where(dist.notna())
        labels[f"{role}_far"] = dist.ge(2).where(dist.notna())
    labels.to_csv(out/"proximity_labels.csv",index=False)
    labels.groupby(["compartment","pre_near","post_near"],dropna=False).size().rename("instances").reset_index().to_csv(
        out/"coverage.csv",index=False)
    rows = []
    for (body,comp), group in labels.groupby(["body_id","compartment"]):
        pre,post = group.pre_near.eq(True),group.post_near.eq(True)
        pre_far,post_far = group.pre_far.eq(True),group.post_far.eq(True)
        for name,left,right in (("pre_near_vs_far",pre,pre_far),("post_near_vs_far",post,post_far),
                                ("pre_near_vs_post_near",pre & post_far,post & pre_far)):
            rows.extend(dict(body_id=body,compartment=comp,contrast=name,left=a,right=b,geometry_gap=d)
                        for a,b,d in matched_groups(group,left,right))
    matched = pd.DataFrame(rows)
    matched.to_csv(out/"matched_pairs.csv",index=False)
    if matched.empty:
        return
    tech = textures(frame,set(matched[["left","right"]].to_numpy().ravel()),out)
    values = pd.concat([hand,tech],axis=1)
    for name in values:
        matched[name] = matched.left.map(values[name])-matched.right.map(values[name])
    summarize(matched,list(values),"body_id",["compartment","contrast"]).to_csv(out/"effects.csv",index=False)
    emb = reps["concat"].to_numpy(dtype=float)
    emb = (emb-emb.mean(0))/np.maximum(emb.std(0),1e-12)
    lookup = frame.set_index("record_id").embedding_row
    rng = np.random.default_rng(SEED)
    separation = []
    for (comp,contrast), group in matched.groupby(["compartment","contrast"]):
        z = emb[group.left.map(lookup)]-emb[group.right.map(lookup)]
        z = pd.DataFrame(z).groupby(group.body_id.to_numpy()).mean().to_numpy()
        observed = np.linalg.norm(z.mean(0))/np.sqrt(z.shape[1])
        null = np.linalg.norm(rng.choice([-1,1],size=(1999,len(z)))@z/len(z),axis=1)/np.sqrt(z.shape[1])
        separation.append(dict(compartment=comp,contrast=contrast,bodies=len(z),
                               pairs=len(group),concat_shift=observed,p=(1+(null>=observed).sum())/2000))
    pd.DataFrame(separation).to_csv(out/"concat_shift.csv",index=False)
    matched.to_csv(out/"paired_measurements.csv",index=False)
    pick = matched.sort_values("geometry_gap").groupby(["compartment","contrast"]).head(2).reset_index(drop=True)
    examples = pick[["left","right"]].rename_axis("example").stack().rename("record_id").reset_index()
    gallery(examples,frame,out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=["E02","E06"])
    for name in ("handoff","source","out"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    frame,reps,hashes = inputs(args.handoff)
    args.out.mkdir(parents=True,exist_ok=False)
    if args.task == "E02":
        e02(frame,reps["handcrafted"],args.source,args.out)
    else:
        e06(frame,reps,args.source,args.out)
    dump(args.out/"run.json",dict(arguments=vars(args),input_sha256=hashes|{str(args.source):sha(args.source)},
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        bootstrap="paired contrasts summarized per anchor/target neuron, then resampled by neuron",
        textures="EM contrast and in-mask XY intensity gradients; no cristae labels"))


if __name__=="__main__":
    main()
