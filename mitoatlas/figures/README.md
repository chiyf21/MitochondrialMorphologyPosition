# FISH2 confusion figures

## Fit and draw

Run the two classification experiments on a frozen FISH2 handoff or the
aligned `inputs/` directory produced by `mitoatlas.pipeline`.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 python -m mitoatlas.confusion_benchmark \
    /path/to/reviewed_handoff /path/to/new_run
```

- R1 is cell type with one prediction per neuron. Separate axon and
  dendrite models predict single mitochondria, and mean log probabilities
  across both compartments give the final neuron prediction. The
  `R1_mito` readout scores those same out-of-fold predictions directly,
  with one prediction per mitochondrion.
- R2 is binary axon versus dendrite, with one prediction per
  mitochondrion. It fits separate classifiers for each known cell type.
- Both use the frozen by-neuron folds, balanced logistic regression and
  training-fold-only scaling by default. `--lc-scaling full_cohort`
  reproduces the historical scaling choice.
- For the fixed FISH2 baseline, the runner verifies that `mito CA` and
  `PC1 CA` are identical before dropping `mito CA`. This gives the 21D
  comparison without dropping a distinct feature from a future handoff.
- The figure colors show fractions within each true class, on the same
  0–1 scale. Cell counts and row fractions are in CSVs. The labels and supports
  must match across representations. Each task exports side-by-side and
  individual PNG and PDF files. Each matrix is exported without an attached
  colorbar. The shared vertical 0–1 scale is exported as
  `confusion_colorbar.pdf` and `confusion_colorbar.png`.
- The scale uses #f7fcf8 at 0, #54b8aa at 0.5, and #102a63 at 1.
  Axis titles and class labels use 24 pt and 22 pt respectively.

## Redraw without fitting

```bash
python -m mitoatlas.figures.confusion /path/to/new_run /path/to/figures \
    --handcraft-label 'Handcrafted 21D' --individual
```

The renderer reads `classification.csv` and the count-matrix CSVs.
When only an older fit is available, regenerate its summary with
`summarize_e03` from `mitoatlas.synapse_review` to recover the R1 single-mito
metrics from the stored `instance_oof.csv` files without fitting again.
The benchmark directory also stores fold predictions, metrics, input
hashes, source commit, and the list of handcraft features.
