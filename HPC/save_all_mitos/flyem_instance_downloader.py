from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume


INPUT = Path(os.environ["FLYEM_INPUT_JSON"])
OUT = Path(os.environ["FLYEM_OUT"])
WORKERS = int(os.environ.get("FLYEM_WORKERS", "6"))
PAD = np.array([4, 4, 4], dtype=np.int64)
MITO_URL = (
    "precomputed://https://storage.googleapis.com/"
    "neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects"
)
EM_URL = (
    "precomputed://https://storage.googleapis.com/"
    "neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg"
)

_tls = threading.local()


def volumes():
    if not hasattr(_tls, "mito"):
        _tls.mito = CloudVolume(
            MITO_URL, mip=0, use_https=True, progress=False,
            fill_missing=True, cache=True,
        )
        _tls.em = CloudVolume(
            EM_URL, mip=0, use_https=True, progress=False,
            fill_missing=True, cache=True,
        )
    return _tls.mito, _tls.em


def as_float(value, default):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def locate_and_save(record, body_type, instance_index):
    started = time.perf_counter()
    timings = {"volume_setup": 0.0, "segmentation_read": 0.0, "mask_search": 0.0, "em_read": 0.0, "write": 0.0}
    body_id = int(record["bodyId"])
    center = np.rint([record["x"], record["y"], record["z"]]).astype(np.int64)
    stage = time.perf_counter()
    mito, em = volumes()
    timings["volume_setup"] = time.perf_counter() - stage
    bounds_lo = np.asarray(mito.bounds.minpt[:3], dtype=np.int64)
    bounds_hi = np.asarray(mito.bounds.maxpt[:3], dtype=np.int64)

    instance_dir = OUT / "results" / str(body_id) / "instances" / f"instance_{instance_index:06d}"
    instance_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = instance_dir / "metadata.json"
    if (instance_dir / "em.npy").exists() and (instance_dir / "mask.npy").exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    # neuPrint r0/r1/r2 are shape descriptors, not XYZ voxel radii.
    # Start with a modest crop and expand only when the label touches its edge.
    half = np.array([64, 64, 32], dtype=np.int64)
    found = None
    for _ in range(5):
        lo = np.maximum(center - half, bounds_lo)
        hi = np.minimum(center + half + 1, bounds_hi)
        stage = time.perf_counter()
        crop = np.asarray(mito[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        timings["segmentation_read"] += time.perf_counter() - stage
        while crop.ndim > 3:
            crop = crop[..., 0]
        stage = time.perf_counter()
        local_center = np.minimum(np.maximum(center - lo, 0), np.asarray(crop.shape) - 1)
        label = int(crop[tuple(local_center)])
        if label == 0:
            nonzero = np.argwhere(crop > 0)
            if len(nonzero) == 0:
                timings["mask_search"] += time.perf_counter() - stage
                half *= 2
                continue
            distances = ((nonzero - local_center) ** 2).sum(axis=1)
            nearest = nonzero[int(np.argmin(distances))]
            label = int(crop[tuple(nearest)])

        full_mask = crop == label
        occupied = [
            np.flatnonzero(full_mask.any(axis=tuple(j for j in range(3) if j != i)))
            for i in range(3)
        ]
        object_lo_local = np.asarray([axis.min() for axis in occupied])
        object_hi_local = np.asarray([axis.max() + 1 for axis in occupied])
        touches = np.any(object_lo_local == 0) or np.any(object_hi_local == np.asarray(crop.shape))
        if touches:
            timings["mask_search"] += time.perf_counter() - stage
            half *= 2
            continue
        object_lo = lo + object_lo_local
        object_hi = lo + object_hi_local
        crop_lo = np.maximum(object_lo - PAD, np.maximum(bounds_lo, lo))
        crop_hi = np.minimum(object_hi + PAD, np.minimum(bounds_hi, hi))
        slices = tuple(slice(int(crop_lo[i] - lo[i]), int(crop_hi[i] - lo[i])) for i in range(3))
        mask_xyz = full_mask[slices]
        timings["mask_search"] += time.perf_counter() - stage
        found = label, crop_lo, crop_hi, mask_xyz
        break
    if found is None:
        raise RuntimeError(f"could not locate mito near {center.tolist()}")

    label, crop_lo, crop_hi, mask_xyz = found
    stage = time.perf_counter()
    em_xyz = np.asarray(em[crop_lo[0]:crop_hi[0], crop_lo[1]:crop_hi[1], crop_lo[2]:crop_hi[2]])
    timings["em_read"] = time.perf_counter() - stage
    while em_xyz.ndim > 3:
        em_xyz = em_xyz[..., 0]
    if em_xyz.shape != mask_xyz.shape:
        raise ValueError(f"EM/mask shape mismatch: {em_xyz.shape} vs {mask_xyz.shape}")

    stage = time.perf_counter()
    np.save(instance_dir / "mask.npy", mask_xyz.transpose(2, 1, 0).astype(np.uint8, copy=False))
    np.save(instance_dir / "em.npy", em_xyz.transpose(2, 1, 0).astype(np.uint8, copy=False))
    metadata = {
        "body_id": str(body_id),
        "cell_type": body_type,
        "global_instance": instance_index,
        "source_mito_label": label,
        "source_voxel_count": int(as_float(record.get("size"), 0)),
        "voxel_count": int(mask_xyz.sum()),
        "center_xyz": center.tolist(),
        "bbox_xyz": [int(crop_lo[0]), int(crop_lo[1]), int(crop_lo[2]), int(crop_hi[0]), int(crop_hi[1]), int(crop_hi[2])],
        "array_axis_order": "ZYX",
        "voxel_size_nm_xyz": [8, 8, 8],
        "mask_file": "mask.npy",
        "em_file": "em.npy",
        "source_mito_layer": MITO_URL,
        "source_em_layer": EM_URL,
        "status": "downloaded",
    }
    if os.environ.get("FLYEM_PROFILE") == "1":
        timings["write"] = time.perf_counter() - stage
        timings["total"] = time.perf_counter() - started
        metadata["pipeline_timings_s"] = {k: round(v, 4) for k, v in timings.items()}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def save_spatial_batch(tile, jobs):
    """Read one haloed 128-cube once, then save each contained mito crop."""
    results = {}
    pending = []
    for record, body_type, index in jobs:
        body_id = int(record["bodyId"])
        out = OUT / "results" / str(body_id) / "instances" / f"instance_{index:06d}"
        if all((out / name).is_file() for name in ("em.npy", "mask.npy", "metadata.json")):
            results[(body_id, index)] = json.loads((out / "metadata.json").read_text())
        else:
            pending.append((record, body_type, index))

    if len(pending) < 4:
        for record, body_type, index in pending:
            body_id = int(record["bodyId"])
            results[(body_id, index)] = locate_and_save(record, body_type, index)
        return results

    mito, em = volumes()
    bounds_lo = np.asarray(mito.bounds.minpt[:3], dtype=np.int64)
    bounds_hi = np.asarray(mito.bounds.maxpt[:3], dtype=np.int64)
    halo = np.asarray([64, 64, 32], dtype=np.int64)
    tile_lo = np.asarray(tile, dtype=np.int64) * 128
    tile_hi = tile_lo + 128
    lo = np.maximum(tile_lo - halo, bounds_lo)
    hi = np.minimum(tile_hi + halo, bounds_hi)

    stage = time.perf_counter()
    labels = np.asarray(mito[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    while labels.ndim > 3:
        labels = labels[..., 0]
    seg_seconds = time.perf_counter() - stage

    valid = []
    fallback = []
    for record, body_type, index in pending:
        body_id = int(record["bodyId"])
        center = np.rint([record["x"], record["y"], record["z"]]).astype(np.int64)
        local_center = center - lo
        if np.any(local_center < 0) or np.any(local_center >= np.asarray(labels.shape)):
            fallback.append((record, body_type, index))
            continue

        label = int(labels[tuple(local_center)])
        if label == 0:
            fallback.append((record, body_type, index))
            continue

        full_mask = labels == label
        occupied = [
            np.flatnonzero(full_mask.any(axis=tuple(j for j in range(3) if j != axis)))
            for axis in range(3)
        ]
        if any(len(axis) == 0 for axis in occupied):
            fallback.append((record, body_type, index))
            continue

        object_lo_local = np.asarray([axis.min() for axis in occupied])
        object_hi_local = np.asarray([axis.max() + 1 for axis in occupied])
        if np.any(object_lo_local == 0) or np.any(object_hi_local == np.asarray(labels.shape)):
            fallback.append((record, body_type, index))
            continue

        object_lo = lo + object_lo_local
        object_hi = lo + object_hi_local
        crop_lo = np.maximum(object_lo - 4, np.maximum(bounds_lo, lo))
        crop_hi = np.minimum(object_hi + 4, np.minimum(bounds_hi, hi))
        slices = tuple(slice(int(crop_lo[k] - lo[k]), int(crop_hi[k] - lo[k])) for k in range(3))
        valid.append({
            "record": record, "body_type": body_type, "index": index,
            "body_id": body_id, "center": center, "label": label,
            "crop_lo": crop_lo, "crop_hi": crop_hi, "mask": full_mask[slices],
            "mask_search_s": 0.0,
        })

    for record, body_type, index in fallback:
        body_id = int(record["bodyId"])
        results[(body_id, index)] = locate_and_save(record, body_type, index)

    if valid:
        em_lo = np.min([item["crop_lo"] for item in valid], axis=0)
        em_hi = np.max([item["crop_hi"] for item in valid], axis=0)
        stage = time.perf_counter()
        em_block = np.asarray(em[em_lo[0]:em_hi[0], em_lo[1]:em_hi[1], em_lo[2]:em_hi[2]])
        while em_block.ndim > 3:
            em_block = em_block[..., 0]
        em_seconds = time.perf_counter() - stage

        for item in valid:
            cl, ch = item["crop_lo"], item["crop_hi"]
            slices = tuple(slice(int(cl[k] - em_lo[k]), int(ch[k] - em_lo[k])) for k in range(3))
            em_crop = em_block[slices]
            mask_crop = item["mask"]
            if em_crop.shape != mask_crop.shape:
                raise ValueError(f"EM/mask mismatch: {em_crop.shape} vs {mask_crop.shape}")

            body_id, index = item["body_id"], item["index"]
            instance_dir = OUT / "results" / str(body_id) / "instances" / f"instance_{index:06d}"
            instance_dir.mkdir(parents=True, exist_ok=True)
            np.save(instance_dir / "mask.npy", mask_crop.transpose(2, 1, 0).astype(np.uint8, copy=False))
            np.save(instance_dir / "em.npy", em_crop.transpose(2, 1, 0).astype(np.uint8, copy=False))
            metadata = {
                "body_id": str(body_id), "cell_type": item["body_type"],
                "global_instance": index, "source_mito_label": item["label"],
                "source_voxel_count": int(as_float(item["record"].get("size"), 0)),
                "voxel_count": int(mask_crop.sum()), "center_xyz": item["center"].tolist(),
                "bbox_xyz": [int(cl[0]), int(cl[1]), int(cl[2]), int(ch[0]), int(ch[1]), int(ch[2])],
                "array_axis_order": "ZYX", "voxel_size_nm_xyz": [8, 8, 8],
                "mask_file": "mask.npy", "em_file": "em.npy",
                "source_mito_layer": MITO_URL, "source_em_layer": EM_URL,
                "status": "downloaded",
            }
            if os.environ.get("FLYEM_PROFILE") == "1":
                metadata["pipeline_timings_s"] = {
                    "segmentation_read_shared_s": round(seg_seconds, 4),
                    "em_read_shared_s": round(em_seconds, 4),
                }
            (instance_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            results[(body_id, index)] = metadata

    return results


def main():
    data = json.loads(INPUT.read_text())
    neurons = {int(row["bodyId"]): row for row in data["neurons"]}
    records = data["mitochondria"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log_path = OUT / "logs" / "download.log"
    started = time.time()
    jobs = []
    for body_id, neuron in neurons.items():
        body_records = [r for r in records if int(r["bodyId"]) == body_id]
        body_dir = OUT / "results" / str(body_id)
        body_dir.mkdir(parents=True, exist_ok=True)
        (body_dir / "instances").mkdir(parents=True, exist_ok=True)
        (body_dir / "neuron_metadata.json").write_text(json.dumps(neuron, indent=2, default=str))
        for index, record in enumerate(body_records, start=1):
            jobs.append((record, str(neuron.get("type") or "unknown"), index))

    grouped = {}
    for job in jobs:
        r = job[0]
        tile = (int(r["x"]) // 128, int(r["y"]) // 128, int(r["z"]) // 128)
        grouped.setdefault(tile, []).append(job)
    tasks = []
    for tile in sorted(grouped):
        group = grouped[tile]
        if len(group) >= 4:
            tasks.append(("batch", tile, group))
        else:
            tasks.extend(("single", None, [job]) for job in group)

    with log_path.open("a", buffering=1) as log:
        def write(message):
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
            print(line, flush=True)
            log.write(line + "\n")

        write(f"starting bodies={len(neurons)} mitochondria={len(jobs)} tiles={len(tasks)} workers={WORKERS}")
        completed, failures = 0, []
        manifest_by_body = {body_id: [] for body_id in neurons}
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for kind, tile, group in tasks:
                if kind == "batch":
                    f = executor.submit(save_spatial_batch, tile, group)
                else:
                    r, t, i = group[0]
                    f = executor.submit(locate_and_save, r, t, i)
                futures[f] = (kind, tile, group)

            for future in as_completed(futures):
                kind, tile, group = futures[future]
                try:
                    if kind == "batch":
                        outputs = future.result()
                    else:
                        r, t, i = group[0]
                        outputs = {(int(r["bodyId"]), i): future.result()}
                except Exception as exc:
                    outputs = {}
                    write(f"batch failed tile={tile}: {type(exc).__name__}: {exc}; retrying per instance")
                    for r, t, i in group:
                        try:
                            outputs[(int(r["bodyId"]), i)] = locate_and_save(r, t, i)
                        except Exception as inner:
                            failures.append({"body_id": int(r["bodyId"]), "instance": i,
                                             "error": f"{type(inner).__name__}: {inner}"})
                for r, body_type, index in group:
                    body_id = int(r["bodyId"])
                    metadata = outputs.get((body_id, index))
                    if metadata is None:
                        continue
                    manifest_by_body[body_id].append(metadata)
                    completed += 1
                    write(f"{completed}/{len(jobs)} body={body_id} instance={index} voxels={metadata['voxel_count']}")

        for body_id, items in manifest_by_body.items():
            body_dir = OUT / "results" / str(body_id) / "instances"
            body_dir.mkdir(parents=True, exist_ok=True)
            manifest = {"body_id": str(body_id),
                "cell_type": str(neurons[body_id].get("type") or "unknown"),
                "instance_count": len(items), "voxel_size_nm_xyz": [8, 8, 8],
                "array_axis_order": "ZYX",
                "instances": sorted(items, key=lambda x: x["global_instance"])}
            (body_dir / "instances_manifest.json").write_text(json.dumps(manifest, indent=2))

        final = {"dataset":"hemibrain:v1.2.1","body_count":len(neurons),
            "mitochondria_count":len(jobs),"completed_count":completed,
            "failed_count":len(failures),"workers":WORKERS,
            "voxel_size_nm_xyz":[8,8,8],"format":"per-instance em.npy + mask.npy + metadata.json",
            "elapsed_seconds":round(time.time()-started,2),
            "status":"complete" if not failures else "partial"}
        if failures: (OUT / "logs" / "failures.json").write_text(json.dumps(failures, indent=2))
        (OUT / "manifest.json").write_text(json.dumps(final, indent=2))
        write(f"complete output={OUT}")


if __name__ == "__main__":
    main()
