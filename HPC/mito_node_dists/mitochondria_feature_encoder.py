from __future__ import annotations
"""Canonical single-file feature extraction implementation for neuron mitochondria."""
from pathlib import Path
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

EM_PY = Path("/home/cyf/.conda/envs/EM/bin/python")
CONNECTOMICS_PY = Path("/home/cyf/.conda/envs/connectomics/bin/python")

#!/usr/bin/env python3
"""Encode each mitochondrion in a 3D binary mask into the 22 morphological features
used by Sager et al. (Science 2025) on FlyEM Hemibrain mitochondria.

The published encoder lives in ClarkLabCode/MitochondrialMorphologyPosition
(HPC/save_all_mitos/mito_main.py and HPC/save_mito_synapse/mito_synapse_main.py).
Those scripts pull voxel clouds from neuPrint/DVID. This module keeps the same
feature definitions, but computes them from a local binary instance mask plus
the matching raw EM volume.

Skeleton-dependent quantities in the original 22-vector (skel CA, skel Circum,
length along skeleton) cannot be obtained from a mitochondria-only crop. They
are replaced by size, SBI and sphericity, which are the extra morphology terms
used in the paper's hemibrain classifier / microns morphometrics notebooks.
The 15 principal-component descriptors, surface area, compactness and diameter
are unchanged.
"""


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion, center_of_mass, generate_binary_structure, label, map_coordinates
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


# Paper morphology-classifier names, plus the three voxel-only substitutes.
FEATURE_NAMES: Tuple[str, ...] = (
    "size",
    "mito SA",
    "mito CA",
    "convex hull compactness",
    "PC1 Length",
    "PC1 inertia moment",
    "PC1 symmetry",
    "PC1 CA",
    "PC1 Circum",
    "PC2 Length",
    "PC2 inertia moment",
    "PC2 symmetry",
    "PC2 CA",
    "PC2 Circum",
    "PC3 Length",
    "PC3 inertia moment",
    "PC3 symmetry",
    "PC3 CA",
    "PC3 Circum",
    "mito diameter",
    "SBI",
    "sphericity",
)
assert len(FEATURE_NAMES) == 22

# Fish2 serial-section EM, as recorded in JaneliaEM/segment_mitochondria/README.md.
DEFAULT_SPACING_ZYX_NM = (30.0, 8.0, 8.0)
DEFAULT_MIN_VOXELS = 30


@dataclass
class EncoderConfig:
    spacing_zyx_nm: Tuple[float, float, float] = DEFAULT_SPACING_ZYX_NM
    min_voxels: int = DEFAULT_MIN_VOXELS
    pca_max_points: int = 8000
    diameter_max_points: int = 2000
    cs_half_extent_factor: float = 1.4
    cs_sample_step_nm: float = 8.0


def load_volume(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"{path} has shape {arr.shape}; expected a 3D volume")
    return arr


def label_mitochondria(mask: np.ndarray, min_voxels: int) -> Tuple[np.ndarray, int]:
    binary = mask > 0
    structure = generate_binary_structure(3, 3)  # 26-neighborhood
    labeled, n_raw = label(binary, structure=structure)
    if n_raw == 0:
        return labeled, 0
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[1:] = counts[1:] >= min_voxels
    relabel = np.cumsum(keep) * keep
    labeled = relabel[labeled].astype(np.int32, copy=False)
    return labeled, int(keep.sum())


def _subsample(coords: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    if len(coords) <= n_keep:
        return coords
    idx = rng.choice(len(coords), size=n_keep, replace=False)
    return coords[idx]


def _orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, helper)
    a /= np.linalg.norm(a) + 1e-12
    b = np.cross(n, a)
    b /= np.linalg.norm(b) + 1e-12
    return a, b


def voxel_surface_area(mask: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    """Anisotropic face-area surface estimate (foreground faces touching background)."""
    dz, dy, dx = spacing_zyx
    m = mask.astype(bool, copy=False)
    area = 0.0
    area += float(np.sum(m[1:, :, :] != m[:-1, :, :])) * dy * dx
    area += float(np.sum(m[:, 1:, :] != m[:, :-1, :])) * dz * dx
    area += float(np.sum(m[:, :, 1:] != m[:, :, :-1])) * dz * dy
    # Outer faces of the crop that still belong to the object.
    if m[0].any():
        area += float(np.sum(m[0])) * dy * dx
    if m[-1].any():
        area += float(np.sum(m[-1])) * dy * dx
    if m[:, 0].any():
        area += float(np.sum(m[:, 0])) * dz * dx
    if m[:, -1].any():
        area += float(np.sum(m[:, -1])) * dz * dx
    if m[:, :, 0].any():
        area += float(np.sum(m[:, :, 0])) * dz * dy
    if m[:, :, -1].any():
        area += float(np.sum(m[:, :, -1])) * dz * dy
    return 0.5 * area


def plane_cross_section(
    mask: np.ndarray,
    origin_idx_zyx: np.ndarray,
    normal_xyz: np.ndarray,
    spacing_zyx: Sequence[float],
    half_extent_nm: float,
    sample_step_nm: float,
) -> Tuple[float, float]:
    """Sample a plane through origin with physical normal, return area and circumference."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    # Convert paper PCA axes (x, y, z) into zyx voxel-index direction.
    normal_xyz = np.asarray(normal_xyz, dtype=float)
    n_phys = np.array([normal_xyz[2], normal_xyz[1], normal_xyz[0]]) * spacing
    if np.linalg.norm(n_phys) < 1e-12:
        return 0.0, 0.0
    a_phys, b_phys = _orthonormal_basis(n_phys)
    n_samp = int(max(8, np.ceil(2 * half_extent_nm / sample_step_nm))) | 1
    us = (np.arange(n_samp) - n_samp // 2) * sample_step_nm
    vs = us.copy()
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    # Physical offset in zyx nm, then back to voxel coordinates.
    delta_nm = uu[..., None] * a_phys + vv[..., None] * b_phys
    coords = origin_idx_zyx[None, None, :] + delta_nm / spacing
    sampled = map_coordinates(
        mask.astype(np.float32),
        [coords[..., 0], coords[..., 1], coords[..., 2]],
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5
    if not sampled.any():
        return 0.0, 0.0
    pixel_area = sample_step_nm * sample_step_nm
    area = float(sampled.sum()) * pixel_area
    border = sampled & ~binary_erosion(sampled, iterations=1, border_value=0)
    circum = float(border.sum()) * sample_step_nm
    return area, circum


def _pca_xyz(coords_xyz_nm: np.ndarray, rng: np.random.Generator, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    fit_coords = _subsample(coords_xyz_nm, max_points, rng)
    n_comp = min(3, max(1, fit_coords.shape[0] - 1), fit_coords.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(fit_coords)
    components = np.zeros((3, 3), dtype=float)
    components[:n_comp] = pca.components_
    for i in range(n_comp, 3):
        components[i, i] = 1.0
    transformed = (coords_xyz_nm - pca.mean_) @ components.T
    return components, transformed


def encode_one(
    mito_mask: np.ndarray,
    bbox_zyx: Tuple[slice, slice, slice],
    raw_crop: Optional[np.ndarray],
    cfg: EncoderConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    spacing = np.asarray(cfg.spacing_zyx_nm, dtype=float)
    voxel_volume = float(np.prod(spacing))
    zz, yy, xx = np.nonzero(mito_mask)
    n_vox = int(zz.size)
    size = n_vox * voxel_volume
    coords_zyx_nm = np.stack([zz * spacing[0], yy * spacing[1], xx * spacing[2]], axis=1)
    coords_xyz_nm = coords_zyx_nm[:, ::-1].copy()

    sa = voxel_surface_area(mito_mask, spacing)
    eq_radius = ((3.0 * size) / (4.0 * np.pi)) ** (1.0 / 3.0) if size > 0 else 0.0
    sphere_sa = 4.0 * np.pi * eq_radius * eq_radius
    sbi = float(np.log10(sa / sphere_sa)) if sa > 0 and sphere_sa > 0 else np.nan
    sphericity = float((np.pi ** (1.0 / 3.0) * (6.0 * size) ** (2.0 / 3.0)) / sa) if sa > 0 else np.nan

    compactness = np.nan
    if n_vox >= 4:
        try:
            hull = ConvexHull(coords_xyz_nm)
            compactness = float(hull.area)
        except (QhullError, ValueError):
            compactness = np.nan

    diameter = np.nan
    if n_vox >= 2:
        diam_pts = _subsample(coords_xyz_nm, cfg.diameter_max_points, rng)
        if len(diam_pts) >= 2:
            diameter = float(np.max(pdist(diam_pts)))

    pc_length = [np.nan, np.nan, np.nan]
    pc_inertia = [np.nan, np.nan, np.nan]
    pc_sym = [np.nan, np.nan, np.nan]
    pc_ca = [np.nan, np.nan, np.nan]
    pc_circum = [np.nan, np.nan, np.nan]
    mito_ca = np.nan

    com_idx = np.array(center_of_mass(mito_mask), dtype=float)
    half_extent = cfg.cs_half_extent_factor * max(eq_radius, float(np.max(spacing)) * 2.0)

    if n_vox >= 4:
        try:
            components, pc_coords = _pca_xyz(coords_xyz_nm, rng, cfg.pca_max_points)
            for i_pc in range(3):
                proj = pc_coords[:, i_pc]
                pc_length[i_pc] = float(proj.max() - proj.min())
                other = [j for j in range(3) if j != i_pc]
                pc_inertia[i_pc] = float(np.sum(pc_coords[:, other] ** 2))
                flipped = pc_coords.copy()
                flipped[:, i_pc] *= -1
                # Original code unique()'s integer voxel PCA coords. Bin at 8 nm so
                # anisotropic physical coordinates behave like Hemibrain voxels.
                bin_nm = float(min(spacing))
                uniq = len(np.unique(np.round(pc_coords / bin_nm).astype(int), axis=0))
                uniq_flip = len(
                    np.unique(np.round(np.vstack([pc_coords, flipped]) / bin_nm).astype(int), axis=0)
                )
                pc_sym[i_pc] = float(2.0 - (uniq_flip / max(uniq, 1)))
                ca, circum = plane_cross_section(
                    mito_mask,
                    com_idx,
                    components[i_pc],
                    spacing,
                    half_extent_nm=half_extent,
                    sample_step_nm=cfg.cs_sample_step_nm,
                )
                pc_ca[i_pc] = ca
                pc_circum[i_pc] = circum
            mito_ca = pc_ca[0]
        except ValueError:
            pass

    mean_intensity = np.nan
    if raw_crop is not None and raw_crop.size:
        mean_intensity = float(raw_crop[mito_mask].mean()) if n_vox else np.nan

    z0, y0, x0 = bbox_zyx[0].start, bbox_zyx[1].start, bbox_zyx[2].start
    centroid_z = z0 + float(zz.mean())
    centroid_y = y0 + float(yy.mean())
    centroid_x = x0 + float(xx.mean())

    feats = {
        "size": size,
        "mito SA": sa,
        "mito CA": mito_ca,
        "convex hull compactness": compactness,
        "PC1 Length": pc_length[0],
        "PC1 inertia moment": pc_inertia[0],
        "PC1 symmetry": pc_sym[0],
        "PC1 CA": pc_ca[0],
        "PC1 Circum": pc_circum[0],
        "PC2 Length": pc_length[1],
        "PC2 inertia moment": pc_inertia[1],
        "PC2 symmetry": pc_sym[1],
        "PC2 CA": pc_ca[1],
        "PC2 Circum": pc_circum[1],
        "PC3 Length": pc_length[2],
        "PC3 inertia moment": pc_inertia[2],
        "PC3 symmetry": pc_sym[2],
        "PC3 CA": pc_ca[2],
        "PC3 Circum": pc_circum[2],
        "mito diameter": diameter,
        "SBI": sbi,
        "sphericity": sphericity,
    }
    extra = {
        "n_voxels": n_vox,
        "centroid_z_voxel": centroid_z,
        "centroid_y_voxel": centroid_y,
        "centroid_x_voxel": centroid_x,
        "bbox_z0": z0,
        "bbox_z1": bbox_zyx[0].stop,
        "bbox_y0": y0,
        "bbox_y1": bbox_zyx[1].stop,
        "bbox_x0": x0,
        "bbox_x1": bbox_zyx[2].stop,
        "mean_em_intensity": mean_intensity,
        "touches_border": int(
            bbox_zyx[0].start == 0
            or bbox_zyx[1].start == 0
            or bbox_zyx[2].start == 0
            or False
        ),
    }
    feats.update(extra)
    return feats


def iter_instances(
    labeled: np.ndarray,
    raw: Optional[np.ndarray],
    pad: int = 2,
) -> Iterable[Tuple[int, np.ndarray, Tuple[slice, slice, slice], Optional[np.ndarray]]]:
    n_lab = int(labeled.max())
    # bounding boxes from one pass over unique labels would be nicer; region scan is OK.
    from skimage.measure import regionprops

    for prop in regionprops(labeled):
        z0, y0, x0, z1, y1, x1 = prop.bbox
        z0p, y0p, x0p = max(0, z0 - pad), max(0, y0 - pad), max(0, x0 - pad)
        z1p = min(labeled.shape[0], z1 + pad)
        y1p = min(labeled.shape[1], y1 + pad)
        x1p = min(labeled.shape[2], x1 + pad)
        sl = (slice(z0p, z1p), slice(y0p, y1p), slice(x0p, x1p))
        mito = labeled[sl] == prop.label
        raw_crop = None if raw is None else raw[sl]
        yield int(prop.label), mito, sl, raw_crop


def encode_volume(
    mask: np.ndarray,
    raw: Optional[np.ndarray],
    cfg: EncoderConfig,
    seed: int = 0,
) -> pd.DataFrame:
    labeled, n_kept = label_mitochondria(mask, cfg.min_voxels)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    print(f"labeling kept {n_kept} mitochondria (min_voxels={cfg.min_voxels})")
    for i, (mito_id, mito, sl, raw_crop) in enumerate(iter_instances(labeled, raw), start=1):
        # touches_border needs full volume shape; compute here.
        touches = int(
            sl[0].start == 0
            or sl[1].start == 0
            or sl[2].start == 0
            or sl[0].stop == labeled.shape[0]
            or sl[1].stop == labeled.shape[1]
            or sl[2].stop == labeled.shape[2]
        )
        rec = encode_one(mito, sl, raw_crop, cfg, rng)
        rec["mito_id"] = mito_id
        rec["touches_border"] = touches
        rows.append(rec)
        if i == 1 or i % 200 == 0 or i == n_kept:
            print(f"  encoded {i}/{n_kept}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["mito_id", *FEATURE_NAMES])
        return df
    ordered = ["mito_id", *FEATURE_NAMES] + [c for c in df.columns if c not in ("mito_id", *FEATURE_NAMES)]
    df = df[ordered].sort_values("mito_id").reset_index(drop=True)
    df.attrs["n_labeled"] = n_kept
    return df


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Encode mitochondria into 22 morphological features")
    p.add_argument("--mask", type=Path, default=here / "data" / "mask_fish2_updated_binary.ome.tif")
    p.add_argument("--raw", type=Path, default=here / "data" / "raw_em_crop_1280x1280x256_repaired.ome.tif")
    p.add_argument("--output-dir", type=Path, default=here / "results")
    p.add_argument("--min-voxels", type=int, default=DEFAULT_MIN_VOXELS)
    p.add_argument("--spacing-zyx-nm", type=float, nargs=3, default=list(DEFAULT_SPACING_ZYX_NM))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EncoderConfig(
        spacing_zyx_nm=tuple(args.spacing_zyx_nm),
        min_voxels=args.min_voxels,
    )
    mask = load_volume(args.mask)
    raw = load_volume(args.raw) if args.raw.exists() else None
    if raw is not None and raw.shape != mask.shape:
        raise ValueError(f"raw {raw.shape} does not match mask {mask.shape}")

    df = encode_volume(mask, raw, cfg, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mito_22feat.csv"
    npy_path = args.output_dir / "mito_22feat.npy"
    meta_path = args.output_dir / "mito_22feat_meta.json"
    df.to_csv(csv_path, index=False)
    feat_mat = df.reindex(columns=list(FEATURE_NAMES)).to_numpy(dtype=np.float64)
    np.save(npy_path, feat_mat)
    meta = {
        "n_mitochondria": int(len(df)),
        "feature_names": list(FEATURE_NAMES),
        "spacing_zyx_nm": list(cfg.spacing_zyx_nm),
        "min_voxels": cfg.min_voxels,
        "mask": str(args.mask),
        "raw": str(args.raw) if raw is not None else None,
        "mask_shape_zyx": list(mask.shape),
        "paper": (
            "Sager et al., Spatial and morphological organization of mitochondria "
            "in neurons across a connectome, Science 2025. Encoder reproduced from "
            "ClarkLabCode/MitochondrialMorphologyPosition."
        ),
        "csv": str(csv_path),
        "npy": str(npy_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"encoded {len(df)} mitochondria")
    print(f"wrote {csv_path}")
    print(f"wrote {npy_path}")
    print(f"wrote {meta_path}")



"""Seven compact, locally normalized EM-intensity features for one mitochondrion.

All arrays are ZYX and all distances/frequencies are physical (nm and cycles/nm).
"""

from pathlib import Path
import warnings

import numpy as np
from scipy import ndimage as ndi


FEATURE_NAMES = [
    'pixel_internal_contrast', 'pixel_internal_edge_density',
    'pixel_orientation_coherence', 'pixel_normal_pc1_angle_deg',
    'pixel_internal_depth_index', 'pixel_dominant_spatial_scale_nm',
    'pixel_spectral_entropy',
]


def _nan_result():
    return {name: float('nan') for name in FEATURE_NAMES}


def build_local_shell(mito_mask, voxel_size=(30., 8., 8.), shell_width_nm=50.):
    mask = np.asarray(mito_mask, dtype=bool)
    distance = ndi.distance_transform_edt(~mask, sampling=voxel_size)
    shell = (~mask) & (distance > 0) & (distance <= shell_width_nm)
    if not shell.any():
        shell = ~mask
    return shell


def normalize_local_intensity(raw_bbox, shell_mask, eps=1e-6):
    raw = np.asarray(raw_bbox, dtype=np.float32)
    values = raw[shell_mask]
    if values.size == 0:
        return np.full(raw.shape, np.nan, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return (raw - median) / (scale + eps)


def get_mito_interior_mask(mito_mask, voxel_size=(30., 8., 8.), boundary_exclusion_nm=15.):
    mask = np.asarray(mito_mask, dtype=bool)
    depth = ndi.distance_transform_edt(mask, sampling=voxel_size)
    exclusion = float(boundary_exclusion_nm)
    interior = mask & (depth >= exclusion)
    if interior.sum() < 3 and mask.any():
        positive = depth[mask]
        fallback = max(0.0, float(np.percentile(positive, 25)) if positive.size else 0.0)
        interior = mask & (depth >= fallback)
    return interior, depth


def compute_physical_gradient(image, voxel_size=(30., 8., 8.), gradient_sigma_nm=7.):
    sigma = tuple(float(gradient_sigma_nm) / float(v) for v in voxel_size)
    smoothed = ndi.gaussian_filter(np.asarray(image, dtype=np.float32), sigma=sigma)
    gz, gy, gx = np.gradient(smoothed, *map(float, voxel_size))
    gradient = np.sqrt(gz * gz + gy * gy + gx * gx)
    return np.stack((gz, gy, gx), axis=0), gradient


def compute_internal_contrast(image, interior):
    values = np.asarray(image)[interior]
    return float(np.percentile(values, 90) - np.percentile(values, 10)) if values.size else float('nan')


def compute_internal_edge_density(gradient, interior, shell, min_values=3):
    reference = np.asarray(gradient)[shell]
    values = np.asarray(gradient)[interior]
    if values.size < min_values:
        return float('nan')
    threshold = float(np.percentile(reference, 95)) if reference.size else float(np.percentile(values, 95))
    return float(np.mean(values > threshold))


def _mito_pc1(mask, voxel_size):
    coords = np.argwhere(mask).astype(float) * np.asarray(voxel_size, dtype=float)
    if len(coords) < 3:
        return None
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 1e-9 else None


def compute_orientation_features(gradient_field, gradient, interior, shell, mito_mask, voxel_size):
    shell_values = gradient[shell]
    threshold = float(np.percentile(shell_values, 95)) if shell_values.size else float(np.percentile(gradient[interior], 95))
    valid = interior & (gradient > threshold)
    count = int(valid.sum())
    if count < 3:
        return float('nan'), float('nan')
    vectors = gradient_field[:, valid].T
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    weights = gradient[valid].astype(float)
    tensor = (vectors * weights[:, None]).T @ vectors / (weights.sum() + 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    coherence = float((eigenvalues[0] - eigenvalues[1]) / (eigenvalues.sum() + 1e-12))
    pc1 = _mito_pc1(mito_mask, voxel_size)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(eigenvectors[:, 0], pc1)), 0., 1.)))) if pc1 is not None else float('nan')
    return coherence, angle


def compute_internal_depth_index(depth, gradient, interior, mito_mask):
    maximum = float(np.max(depth[mito_mask])) if mito_mask.any() else 0.
    weights = gradient[interior].astype(float)
    if maximum <= 0 or weights.sum() <= 1e-12:
        return float('nan')
    return float(np.sum((depth[interior] / maximum) * weights) / weights.sum())


def compute_spectral_features(image, interior, depth, voxel_size, spectrum_out=None):
    if int(interior.sum()) < 8:
        return float('nan'), float('nan')
    signal = np.zeros_like(image, dtype=np.float32)
    values = image[interior].astype(np.float32)
    max_depth = float(np.max(depth[interior]))
    soft = np.clip(depth / max_depth, 0., 1.) if max_depth > 0 else interior.astype(float)
    signal[interior] = (values - float(np.mean(values))) * soft[interior]
    power = np.abs(np.fft.fftn(signal)) ** 2
    freqs = [np.fft.fftfreq(n, d=float(spacing)) for n, spacing in zip(signal.shape, voxel_size)]
    zz, yy, xx = np.meshgrid(*freqs, indexing='ij')
    radius = np.sqrt(zz * zz + yy * yy + xx * xx)
    positive = radius > 0
    nyquist = min(0.5 / float(v) for v in voxel_size)
    extent_nm = max(float(n) * float(v) for n, v in zip(signal.shape, voxel_size))
    valid = positive & (radius < 0.9 * nyquist) & (radius > 1.5 / extent_nm)
    if not valid.any():
        return float('nan'), float('nan')
    edges = np.geomspace(float(radius[valid].min()), float(radius[valid].max()), 25)
    bin_index = np.digitize(radius, edges) - 1
    frequencies, powers = [], []
    for index in range(len(edges) - 1):
        selected = valid & (bin_index == index)
        if selected.any():
            frequencies.append(float(np.mean(radius[selected])))
            powers.append(float(np.mean(power[selected])))
    if not powers or not np.isfinite(powers).any() or np.sum(powers) <= 0:
        return float('nan'), float('nan')
    frequencies, powers = np.asarray(frequencies), np.asarray(powers)
    if spectrum_out is not None:
        out = Path(spectrum_out); out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.column_stack((frequencies, powers)))
    probabilities = powers / powers.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(len(probabilities))) if len(probabilities) > 1 else float('nan')
    peak_frequency = float(frequencies[int(np.argmax(powers))])
    return float(1. / peak_frequency) if peak_frequency > 0 else float('nan'), entropy


def extract_pixel_ultrastructure_features(raw_bbox, mito_mask, voxel_size=(30., 8., 8.), shell_width_nm=50., boundary_exclusion_nm=15., gradient_sigma_nm=7., debug=False, spectrum_out=None):
    """Return exactly seven pixel features; invalid cases become NaN, never batch-fatal."""
    result = _nan_result()
    try:
        raw, mask = np.asarray(raw_bbox), np.asarray(mito_mask, dtype=bool)
        if raw.shape != mask.shape or not mask.any():
            return result
        shell = build_local_shell(mask, voxel_size, shell_width_nm)
        normalized = normalize_local_intensity(raw, shell)
        interior, depth = get_mito_interior_mask(mask, voxel_size, boundary_exclusion_nm)
        if not interior.any():
            return result
        field, gradient = compute_physical_gradient(normalized, voxel_size, gradient_sigma_nm)
        result['pixel_internal_contrast'] = compute_internal_contrast(normalized, interior)
        result['pixel_internal_edge_density'] = compute_internal_edge_density(gradient, interior, shell)
        result['pixel_orientation_coherence'], result['pixel_normal_pc1_angle_deg'] = compute_orientation_features(field, gradient, interior, shell, mask, voxel_size)
        result['pixel_internal_depth_index'] = compute_internal_depth_index(depth, gradient, interior, mask)
        result['pixel_dominant_spatial_scale_nm'], result['pixel_spectral_entropy'] = compute_spectral_features(normalized, interior, depth, voxel_size, spectrum_out)
        return result
    except Exception as error:
        warnings.warn(f'pixel feature extraction skipped: {error}', RuntimeWarning, stacklevel=2)
        return result


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    z, y, x = np.indices((32, 64, 64))
    mask = ((z - 16) / 12) ** 2 + ((y - 32) / 25) ** 2 + ((x - 32) / 25) ** 2 < 1
    regular = mask * (2. * np.sin(y / 2.5))
    noisy = mask * rng.normal(size=mask.shape)
    a = extract_pixel_ultrastructure_features(regular, mask)
    b = extract_pixel_ultrastructure_features(noisy, mask)
    assert set(a) == set(FEATURE_NAMES) and np.isfinite(a['pixel_internal_contrast'])
    assert a['pixel_orientation_coherence'] > b['pixel_orientation_coherence']
    print('pixel_features smoke test passed')


#!/usr/bin/env python3
"""Compute skeleton-aware mito features from existing mito crop folders."""

import argparse, csv, json, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

BLOCK = re.compile(r"^x(-?\d+)_y(-?\d+)_z(-?\d+)$")
SPACING_XYZ = np.array([8.0, 8.0, 30.0])
TILE_XYZ = np.array([128, 128, 32])


def origin(path):
    m = BLOCK.match(path.name)
    if not m: raise ValueError(f"bad block name: {path.name}")
    return np.array([int(x) for x in m.groups()])


def load_mask(path):
    value = np.load(path)
    if isinstance(value, np.lib.npyio.NpzFile):
        try: return value[value.files[0]] > 0
        finally: value.close()
    return value > 0


def local_tangent(skel, point):
    xyz = skel[["x_em", "y_em", "z_em"]].to_numpy(float) * SPACING_XYZ
    distance, nearest = cKDTree(xyz).query(point * SPACING_XYZ)
    row = skel.iloc[int(nearest)]; ids = {int(r.rowId): i for i, r in skel.iterrows()}
    peers = ([ids[int(row.link)]] if int(row.link) in ids else [])
    peers += [i for i, r in skel.iterrows() if int(r.link) == int(row.rowId)]
    if not peers:
        peers = np.atleast_1d(cKDTree(xyz).query(xyz[int(nearest)], k=min(2, len(xyz)))[1])[1:].tolist()
    # At a branch, summing parent/child directions can cancel; use its longest edge.
    vector = max((xyz[i] - xyz[int(nearest)] for i in peers), key=np.linalg.norm)
    norm = np.linalg.norm(vector)
    if norm < 1e-6: raise ValueError("degenerate local skeleton tangent")
    return vector / norm, float(distance)


def load_context(raw_body, center, radius_nm):
    radius = np.ceil(radius_nm / SPACING_XYZ).astype(int)
    lo = np.floor(center).astype(int) - radius; hi = np.floor(center).astype(int) + radius + 1
    out = np.zeros(tuple((hi - lo)[::-1]), bool)  # ZYX
    for block in (raw_body / "blocks").glob("x*_y*_z*"):
        path = next((x for x in (block / "neuron_mask.npy", block / "neuron_mask.npz") if x.exists()), None)
        if path is None: continue
        start = origin(block); stop = start + TILE_XYZ; a0 = np.maximum(lo, start); a1 = np.minimum(hi, stop)
        if np.any(a1 <= a0): continue
        local = load_mask(path)
        src = tuple(slice(int(a0[i]-start[i]), int(a1[i]-start[i])) for i in range(3))
        dst = tuple(slice(int(a0[i]-lo[i]), int(a1[i]-lo[i])) for i in range(3))
        out[dst[2], dst[1], dst[0]] |= local[src].transpose(2, 1, 0)
    return out, lo


def cross_section(nmask, lo, center, tangent, radius_nm=600., step_nm=8.):
    seed = np.array([1., 0, 0]) if abs(tangent[0]) < .9 else np.array([0., 1., 0])
    u = np.cross(tangent, seed); u /= np.linalg.norm(u); v = np.cross(tangent, u)
    axis = np.arange(-radius_nm, radius_nm + step_nm, step_nm); yy, xx = np.meshgrid(axis, axis, indexing="ij")
    physical = center[None,None,:]*SPACING_XYZ + xx[...,None]*u + yy[...,None]*v
    grid = physical / SPACING_XYZ - lo
    plane = ndi.map_coordinates(nmask.astype(np.uint8), [grid[...,2], grid[...,1], grid[...,0]], order=0, mode="constant").astype(bool)
    labels, count = ndi.label(plane); c = len(axis)//2; chosen = labels[c,c]
    if not chosen and count:
        d = ndi.distance_transform_edt(labels == 0)
        chosen = min(range(1, count+1), key=lambda i: d[labels == i].min())
    mask = labels == chosen
    perimeter_edges = np.count_nonzero(mask[:,1:] != mask[:,:-1]) + np.count_nonzero(mask[1:,:] != mask[:-1,:])
    return float(mask.sum()*step_nm**2), float(perimeter_edges*step_nm), int(mask.sum())


def one_crop(folder, raw_root, skel_root, context_nm):
    meta = json.loads((folder / "metadata.json").read_text()); body = str(meta["body_id"])
    mask = np.load(folder / "mask_zyx.npy") > 0; crop_origin = np.asarray(meta["crop_origin_xyz"], float)
    center = crop_origin + np.argwhere(mask)[:, ::-1].mean(axis=0)
    tangent, distance = local_tangent(pd.read_csv(skel_root/body/"skeleton_nodes.csv"), center)
    nmask, nmask_origin = load_context(raw_root/body, center, context_nm)
    area, circumference, pixels = cross_section(nmask, nmask_origin, center, tangent, context_nm)
    pts = np.argwhere(mask)[:, ::-1] + crop_origin
    length = float(np.percentile((pts-center) @ (tangent*SPACING_XYZ), 99) - np.percentile((pts-center) @ (tangent*SPACING_XYZ), 1))
    return {"body_id":body, "mito_id":int(meta["mito_id"]), "n_voxels":int(meta["n_voxels"]), "skel CA":area, "skel Circum":circumference, "length along skeleton":length, "centroid_x":center[0], "centroid_y":center[1], "centroid_z":center[2], "nearest_skeleton_distance_nm":distance, "cross_section_pixels":pixels, "status":"ok"}


def safe_one_crop(folder, raw_root, skel_root, context_nm):
    try: return one_crop(folder, raw_root, skel_root, context_nm)
    except Exception as e:
        meta=json.loads((folder/"metadata.json").read_text())
        return {"body_id":meta["body_id"], "mito_id":meta["mito_id"], "status":f"error: {e}"}



def export_skeleton(source, target):
    code = """import csv,sys,pyarrow.parquet as pq
s,t=sys.argv[1:]
q=pq.read_table(s)
cols=[c for c in [\"rowId\",\"x_em\",\"y_em\",\"z_em\",\"radius\",\"link\",\"distance\"] if c in q.column_names]
if len(cols)!=7: raise SystemExit(\"skeleton lacks required EM-coordinate columns: \"+str(cols))
with open(t,\"w\",newline=\"\") as f:
 w=csv.writer(f); w.writerow(cols); w.writerows(zip(*(q[c].to_pylist() for c in cols)))
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(CONNECTOMICS_PY), "-c", code, str(source), str(target)], check=True)


def compute_skeleton_features(crops, body_id, raw_root, skeleton_root, jobs=1, context_nm=600.):
    folders=sorted((Path(crops)/str(body_id)).glob(f"{body_id}_mito*"))
    rows=[]
    if jobs == 1:
        for folder in folders:
            rows.append(safe_one_crop(folder, Path(raw_root), Path(skeleton_root), context_nm))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures=[pool.submit(safe_one_crop, folder, Path(raw_root), Path(skeleton_root), context_nm) for folder in folders]
            rows=[future.result() for future in as_completed(futures)]
    return pd.DataFrame(rows)
