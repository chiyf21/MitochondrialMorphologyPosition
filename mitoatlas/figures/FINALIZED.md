# FISH2 finalized figure analyses

This index records the frozen FISH2 figure analyses and the approved
synapse prediction experiment and the provisional clustering protocol. Run the commands from the repository root.
Keep data, checkpoints and generated figures in the experiment storage.

| Analysis | Figure or readout | Code reference |
| --- | --- | --- |
| R1 and R2 | Matched 21D and Concat confusion matrices. R1 has neuron and single-mitochondrion panels. R2 predicts compartment. | `fish2-fig2-classification-recovery-v1` |
| E01 | In-sample OLS recovery of 21 distinct handcrafted morphology measurements from Concat, with soma excluded. The figure sorts targets by R². | `fish2-fig2-classification-recovery-v1` |
| E02 | Shape-matched EM example pairs, including wider fields, mask outlines, surface point clouds and PNG scale bars. The selected examples illustrate texture variation at similar shape. | `fish2-fig2-e02-em-v1` |
| FISH2 PC30 clustering | Provisional Leiden six-cluster partition for annotation. | `6d0a9ad` (partition), `31f49b3` (UMAP) |
| Synapse pre/post | One pooled binary classifier design with separate fits for the two representations on identical neuron-disjoint folds. | `a8ae367` |

## R1, R2 and E01

See [the classification and rendering instructions](README.md). For E01
use `python -m mitoatlas.figures.feature_recovery_comparison --help`
with the frozen feature-recovery scores. Use the in-sample OLS readout
for the figure.

## E02

- `mitoatlas.figures.e02_texture_examples` selects example pairs from the
  morphology nearest-neighbor table. The larger selection uses
  `--large-min-voxels 50000 --large-neighbors 16`.
- `mitoatlas.figures.e02_export_figure_assets` exports surface point clouds,
  XY stacks and 21D cosine similarities.
- `mitoatlas.figures.e02_export_em_context` exports registered wide-field
  views from the original EM source. Run it with the frozen instance table,
  exported example instance list, FISH2 EM source and a new output path.
  Original and larger examples used widths of 320 and 384 pixels respectively.

## FISH2 PC30 clustering

The provisional partition uses the full 326-neuron FISH2 e043 VICReg
Concat export (`full326_vicreg25_25_1_e043_embeddings_20261119_01`),
with 29,752 mitochondria. Use the first 30 unwhitened PCs
(randomized PCA, seed 50000, variance explained 92.73%).
Build the exact Euclidean 90-nearest-neighbor graph with undirected
union symmetrization and local Gaussian weights. The bandwidth for each
point is its 90th neighbor distance. Run weighted Leiden with
`RBConfigurationVertexPartition`, resolution 0.5, two iterations and
seed 50000. The run uses igraph 0.11.9, leidenalg 0.10.2 and
scikit-learn 1.9.1.

Authoritative cluster labels are in
`pc30_coarse_clusters_v1/assignments.csv.gz` (column
`Leiden k=90 gamma=0.5`). Join on `body_id` and `record_id`; use
`embedding_row` to verify alignment with the e043 export. Raw integer
labels 0–5 correspond to C1–C6 in the figure. Counts are 6,536,
6,505, 5,919, 5,021, 3,485 and 2,286 in label order. The UMAP shows
these labels on the existing coordinates from
`pc30_neuron_stability_v1/assignments.csv.gz`.

- PC30 SHA-256 (file `pc30_neuron_stability_v1/pc30.npy`)
  `9b07f05476d58c8d82725b0cc8a64e5125c622d16134c5528beda4dd474e23d6`
- Cluster table SHA-256
  `50220a4b2a3532d3ea9afa3ea7ff6396b4f681635886d2737c4089539994cc16`

Run from the repository root with `FISH2_E043_EXPORT` pointing to the
export directory and `PC30_REPRO` set to a new output directory.

```bash
python -m mitoatlas.pc30_fine_clusters \
  "$FISH2_E043_EXPORT/pc30_neuron_stability_v1" "$PC30_REPRO" --coarse

python -m mitoatlas.plot_pc30_clusters \
  "$FISH2_E043_EXPORT/pc30_neuron_stability_v1/assignments.csv.gz" \
  "$PC30_REPRO/assignments.csv.gz" "Leiden k=90 gamma=0.5" \
  "$PC30_REPRO/FISH2_Leiden6_UMAP"
```

Four 80% instance-level resamples gave pairwise ARI median 0.675
(range 0.570–0.867) and a minimum per-cluster median best-match Jaccard
of 0.447. These results support provisional use for annotation.
The C1–C6 labels are cluster identifiers pending biological interpretation.

## Synapse pre/post

Use the full-326 FISH2 instance directory (with `synapse_proximity_v2`),
the matching E50 embedding export and the reviewed-panel metadata.
The reviewed metadata is used for supplementary strata. The primary
comparison pools all compartments and uses one representation at a time.

```bash
python -m mitoatlas.synapse_polarity_full \
  "$FISH2_FULL326" "$FULL326_E50_EXPORT" "$REVIEWED_INSTANCES_CSV" "$POLARITY_RUN"

python -m mitoatlas.synapse_polarity_compare \
  "$POLARITY_RUN/oof.csv" "$FISH2_FULL326/dataset_manifest.json" "$COMPARISON_RUN"
```

The label uses distances from the same neuron's annotated synapse endpoints
to the closest occupied mask voxel center. The near role is at most 1 µm
away and the other role at least 2 µm away. The primary analysis retains
eligible E50 Refine training neurons. The classifier's five folds are
separated by neuron. Classifier inputs are either Concat or the 21 distinct
handcrafted features. The primary comparison contains 17,851 instances
from 310 neurons, with AUROC 0.914 for Concat and 0.674 for 21D handcrafted.
These scores describe pre/post proximity prediction in the pooled cohort.

The same analysis on e043 Concat is recorded in
`synapse_polarity_comparison_v1` under the e043 export. It uses the
same 17,851 instances and neuron-disjoint folds. Pooled AUROC is 0.933
for Concat and 0.674 for 21D. Mean within-neuron AUROC across 285
neurons with both labels is 0.913 and 0.680, respectively.

For the e043 pre/post figure, render the paired per-neuron AUROC
scatter from the frozen out-of-fold table. Each point is one of the
285 neurons with both labels. The diagonal denotes equal AUROC.
The accompanying CSV contains both AUROCs and class counts per neuron.

```bash
python -m mitoatlas.figures.synapse_polarity_paired \
  "$FISH2_E043_EXPORT/synapse_polarity_comparison_v1/oof.csv" "$NEW_FIGURE_DIR"
```

E04–E10 and the full-cohort distance regression serve as exploratory
records for the synapse module. The finalized synapse figure comparison
uses pre/post classification.
