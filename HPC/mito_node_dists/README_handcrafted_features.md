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

## Selected 22-feature computation on neuron blocks

`compute_selected_morphology.py` and `run_selected_morphology_batches.sh` process the GC, CON, VMN, PC, MON2, and MON1 instance rows in the consolidated dataset, use the corresponding per-neuron block masks and SWC, and write resumable outputs outside Git. Neuron block arrays are stored XYZ (shape 128×128×32), while mitochondria crops are ZYX; the runner converts block masks while assembling local contexts. The metadata bbox remains global XYZ voxel coordinates, and SWC positions are global XYZ nanometers.

Run on 112:

```bash
bash HPC/mito_node_dists/run_selected_morphology_batches.sh \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset \
  /mnt/data21T_3/cyf/EM/neuron \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset/handcraft_features/paper22_all6_v1 \
  256 16
```

The table contains the paper's 22 input columns, plus QC/auxiliary neuron cross-section columns. Physical values use the 8×8×30 nm voxel spacing. Surface area uses an anisotropic exposed-face estimate; mito/skeleton cross-sections use a local SWC-tangent plane; the skeleton path-length column is filled by the separate SWC mapping stage. These are recorded in method columns and should be treated as a local-data adaptation of the paper pipeline.

Run the SWC path-length stage and merge after the morphology chunks finish:

```bash
bash HPC/mito_node_dists/run_selected_skeleton_batches.sh \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset \
  /mnt/data21T_3/cyf/EM/neuron \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset/handcraft_features/paper22_all6_v1/skeleton_tangent \
  256 16
python HPC/mito_node_dists/merge_selected_paper22.py \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset/handcraft_features/paper22_all6_v1 \
  /mnt/data21T_3/cyf/EM/neuron_mito_dataset/handcraft_features/paper22_all6_v1/skeleton_tangent/skeleton_length.csv
```

The local length_along_skeleton_nm is the mitochondrion surface extent projected onto the nearest SWC edge tangent. This is a local straight-tangent approximation to the paper's clean-skeleton edge-overhang routine. The geodesic span of all mapped SWC nodes is retained separately as mapped_skeleton_geodesic_span_nm for diagnostics and is not used as the 22nd feature because it can jump between spatially adjacent but distant skeleton branches.

## Dataset workflow

- FISH2: use the selected 22-feature computation above for the anisotropic
  masks (8 x 8 x 30 nm). Keep compartment labels as a separate annotation on
  skeleton node IDs so the 22-feature computation can also run on all instances.
- FlyEM hemibrain: use the paper's upstream workflow, including its native
  inputs and the analysis notebooks listed in the upstream README (e.g.
  `morphology embeddings - hemibrain.ipynb`). Check the input dataset and release
  before running the upstream notebook. Keep the FISH2 geometry adaptations
  scoped to FISH2.

## FISH2 manual compartment review

Build the review panel from the frozen paper22 feature CSV. This takes the
unique body IDs from the six selected cell types and keeps the original V3
Step 05 labels as review candidates.

```bash
python HPC/mito_node_dists/build_compartment_review.py FEATURES_CSV NEURON_ROOT OUTPUT_DIR --workers 16
python HPC/mito_node_dists/serve_compartment_review.py OUTPUT_DIR REVIEW_SOCKET
ssh -NT -L 127.0.0.1:18765:REVIEW_SOCKET USER@REVIEW_HOST
```

Run the Python commands on the data server with the same environment used
for the V3 replay and with NumPy, SciPy, pandas and PyArrow installed. Run the SSH
command on the desktop, then open http://127.0.0.1:18765/. Choose a new socket
path for a second live server. If restarting, remove a stale socket only
after checking the prior server is stopped.

The editor labels existing skeleton edges and inferred bridges. It supports
single-edge, unbranched-chain, path, same-label component and physical-radius
selection. PRE and POST endpoints can be toggled through the legend.

- `automatic/<body_id>/` holds the immutable candidate and node table.
- `drafts/<body_id>.json` holds changes saved for later review.
- `corrected/<body_id>/review.json` marks a completed review, including its
  remaining ambiguous count and paths to the corrected revision's two
  Parquet tables. Read these paths to obtain the accepted revision.
- `panel.json` stores the review panel and `batch_status/` records batch
  outcomes. The candidate builder skips already published candidates.

When V11 cannot attach any PRE or POST endpoints to its main component, the
batch uses the original V3 skeleton geometry to construct Step 03 bridges.
Step 04 still initializes the bridge labels from V3 as usual. Failed or
ambiguous V3 anatomical assignments remain available for manual review.

The five original algorithm files are tracked under `compartment_steps/`. The optional legacy directory can be supplied between `NEURON_ROOT` and `OUTPUT_DIR` to replay a different snapshot.
