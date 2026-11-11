# Handcrafted mitochondrial features

`compute_handcrafted_features.py` computes per-instance mask morphology and EM texture features for the canonical neuron mitochondria dataset. It imports `mitochondria_feature_encoder.py`, adapted from the existing JaneliaEM feature pipeline.

## Features

- 22 mask-derived morphology features: physical volume, surface area, cross-sectional area, convex-hull compactness, PCA-axis lengths/inertia/symmetry/cross-section/circumference, maximum diameter, SBI, and sphericity.
- 7 EM-derived ultrastructure features: internal contrast, edge density, orientation coherence, normal-to-PC1 angle, internal depth, dominant spatial scale, and spectral entropy.
- Metadata/QC columns include instance and neuron identifiers, physical centroid, mean EM intensity, and whether the mask touches its stored crop boundary.

The current voxel-size setting is `(30, 8, 8) nm` in ZYX order, consistent with the existing Fish2 serial-section pipeline. Confirm this against the source-volume metadata before interpreting physical measurements.

## Run

```bash
python HPC/mito_node_dists/compute_handcrafted_features.py \
  --dataset-root /path/to/neuron_mito_dataset \
  --output-dir /path/to/neuron_mito_dataset/handcraft_features/run_name
```

Outputs are `morphology_features.csv`, `pixel_features.csv`, `handcrafted_features.csv`, `errors.csv`, and `summary.json`. Keep these generated tables with the dataset, outside Git.

`length_along_skeleton_nm` is intentionally not inferred from a mitochondrion-only mask. It requires the corresponding neuron skeleton SWC and coordinate/voxel metadata; use `mito_mesh_skeleton_length.py` once complete skeleton inputs are available to the compute server.
