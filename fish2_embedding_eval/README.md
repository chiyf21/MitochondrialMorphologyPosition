# Fish2 embedding evaluation

This directory adds a read-only evaluation workflow for the completed E16 CLS/MLP fusion runs. It reuses the existing Fish2 Reviewed81 dataset and classifiers, and keeps all results outside Git.

## Inputs and commands

1. Choose the completed refinement checkpoint and preserve its run directory, including the resolved training configuration and step log. Reuse the canonical Reviewed81 descriptors for the historical Fusion50 endpoint, when applicable.
2. Export one canonical view per record with the native frozen model. Select an idle physical GPU and keep the same index in CUDA_VISIBLE_DEVICES and --physical-gpu. The Fish2 code path must contain the verified Reviewed81 input adapter and the native training checkout must match its launcher revision.

`ash
env CUDA_VISIBLE_DEVICES=2 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python fish2_embedding_eval/export_reviewed81.py \
  --checkpoint /ABS/run/checkpoints/refine_epoch005.pt \
  --fish2-code /ABS/fish2_downstream_worktree --physical-gpu 2 --output /ABS/new_smoke --smoke
`

Repeat with a fresh output directory and omit --smoke for all 5,568 instances. Register concat_12.npy and fused_12.npy separately with the shared Fish2 run.py register command, the export instance_map.csv, and the exact checkpoint. Use a unique descriptor path for each array. The historical canonical descriptors can be passed directly.

3. Supply a JSON configuration with the keys name, assets (Reviewed81 directory), fish2_entry (shared run.py) and runs. Each run has name, checkpoint, loss_profile (description of CLS scale in the loss), balance (boolean), concat (descriptor path), and project (descriptor path). Keep descriptors, paths, and results in the experiment directory, outside Git.

`ash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
  python fish2_embedding_eval/run.py --config /ABS/comparison_config.json --output /ABS/new_comparison
`

## Readouts and metrics

- **Concat1152** is the actual input to the fusion MLP, formed from Point CLS384 and Image CLS768 from Block 12. The on run uses the checkpoint's PairBalance buffers. The off run concatenates the unbalanced CLS readouts. The internal 768D concat after modality LayerNorm and the image projection is a different readout.
- **Project384** is the frozen fusion head output including final LayerNorm.
- Each UMAP uses the same 5,568 records, centered PCA-30, cosine distance, 30 neighbors, min_dist 0.1 and seed 20261116. Each panel is fit independently, so orientation and coordinate values are panel-local. For the compartment overlay, gray points are outside the accepted binary QC labels. Coloring never enters the fit.
- Each representation is evaluated using fixed-profile LR with train-fold scaling and Reviewed81 body-separated folds. Cell Type is pooled six-class classification on 5,568 instances and 81 bodies. Compartment is QC-filtered axon versus dendrite on 5,059 instances, trained within cell type. Grouping for compartment uses the true compartment and body ID (161 groups). Report balanced accuracy and macro-F1. The classifier split alone gives no guarantee of independence from encoder training.

## Loss accounting

The original 20-epoch Point CLS stage uses the mean of the layerwise NNA losses. The newer runs begin from the saved state at the start of refinement. For refinement, calculate the following sum on each of Blocks 9–12 and average over the four blocks:

- Fusion VICReg contributes 25 × invariance + 5 × variance + 0.2 × covariance.
- Filtered InfoNCE has weight 1 and temperature 0.2. The same-instance other view is positive. The 20 nearest other instances are filtered from eligible batch negatives at the standard batch size.
- Point and Image CLS each contribute 1.25 × variance + 0.05 × covariance. The historical run applies these to its baseline CLS outputs. The corrected runs use per-sample LayerNorm on Point Blocks 9–12 and Image Blocks 9–11 for these regularizers. Image Block 12 already uses the native final norm.
- The CSV stores raw and weighted components. Mean loss is over the recorded optimizer updates. If a resumed epoch contains duplicate batch log rows, the last row is used and the count is recorded. The pipeline checks that the weighted sum reproduces logged loss and that the selected endpoint agrees with the completion counter.

Compare total loss within a consistently defined objective. The separate common-term curve omits both CLS regularizers so their scale change can be seen explicitly. Final checkpoints can have different training budgets. Same-stage endpoint comparisons cover the actual completed runs and cannot isolate the effect of a change to training.

## Outputs

epoch_loss.csv, loss_curves.png, UMAP coordinates keyed by record ID, UMAP panels for both label sets, downstream_metrics.csv, six (for a three-model comparison) downstream run references, and complete.json. The shared Fish2 runner stores each fold, out-of-fold predictions, and confusion matrices. A new output directory is required on each invocation. The currently supported training loss profile is the E16 NNA + VICReg/InfoNCE/CLS objective described above.
