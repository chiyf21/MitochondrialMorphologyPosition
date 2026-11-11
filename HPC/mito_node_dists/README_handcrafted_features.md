# Handcrafted mitochondrial features

`compute_handcrafted_features.py` computes per-instance shape and EM features from the canonical dataset. It uses the existing JaneliaEM feature encoder. `run_handcrafted_feature_batches.sh` saves bounded chunks and can resume by skipping completed chunk summaries.

## Features

- 22 mask-derived morphology features: physical volume, surface area, cross-sectional area, convex-hull compactness, PCA-axis descriptors, maximum diameter, SBI, and sphericity.
- 7 EM-derived features: internal contrast, edge density, orientation coherence, normal-to-PC1 angle, internal depth, dominant spatial scale, and spectral entropy.
- Skeleton mapping features: nearest skeleton node/distance and geodesic span of mapped skeleton nodes (`length_along_skeleton_nm`).

Mask features use ZYX voxel spacing `(30, 8, 8) nm`, inherited from the Fish2 pipeline. Confirm this against the source-volume metadata before interpreting physical measurements.

## Run

```bash
bash HPC/mito_node_dists/run_handcrafted_feature_batches.sh \
  /path/to/neuron_mito_dataset /path/to/handcraft_features/all_instances_v2 64 8

bash HPC/mito_node_dists/run_skeleton_length_batches.sh \
  /path/to/neuron_mito_dataset /path/to/handcraft_features/all_instances_v2 64
```

The skeleton metadata directory is expected at `neuron_mito_dataset/skeleton_metadata/<body_id>/`. The supplied manifests declare global XYZ coordinates and the SWC header declares nanometers. In the supplied data, SWC positions align with the mitochondria bboxes after converting bbox voxels to nanometers; therefore the header's `scale=2` is recorded but not applied to those physical coordinates. `surface_mapping_method` records whether the exact marching-cubes mesh or a bounded sample of boundary voxels was used; oversized stored crops use the sampled method to avoid unbounded memory use.

Chunk CSVs and final tables are generated data and should remain outside Git. The morphology runner writes `morphology_features.csv`, `pixel_features.csv`, `handcrafted_features.csv`, `errors.csv`, and `summary.json`; the skeleton runner writes `skeleton_length.csv`. After both finish, create the combined table with:

```bash
python HPC/mito_node_dists/merge_handcrafted_with_skeleton.py /path/to/handcraft_features/all_instances_v2
```

## Paper morphology only

`compute_paper_morphology.py` computes the morphology descriptors from the paper without loading EM intensity arrays or calculating the seven optional texture features. It tight-crops each mask to its foreground bounds before morphology operations. `run_paper_morphology_batches.sh` runs resumable chunks and writes `handcrafted_features.csv`.

The original neuron segmentation volume is not present in the supplied dataset, so `skel CA` and `skel Circum` use the radius in the nearest SWC node as a circular-section estimate (`pi*r^2` and `2*pi*r`). The output records this in `skeleton_section_method`; these two columns are estimates, not direct voxel cross-sections. Append the already calculated `skeleton_length.csv` using `merge_handcrafted_with_skeleton.py`.

```bash
PYTHON=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python \
  bash HPC/mito_node_dists/run_paper_morphology_batches.sh \
  /path/to/neuron_mito_dataset /path/to/handcraft_features/paper_original_v1 64 8
```
