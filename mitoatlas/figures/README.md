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
  across both compartments give the final neuron prediction.
- R2 is binary axon versus dendrite, with one prediction per
  mitochondrion. It fits separate classifiers for each known cell type.
- Both use the frozen by-neuron folds, balanced logistic regression and
  training-fold-only scaling by default. `--lc-scaling full_cohort`
  reproduces the historical scaling choice.
- For the fixed FISH2 baseline, the runner verifies that `mito CA` and
  `PC1 CA` are identical before dropping `mito CA`. This gives the 21D
  comparison without dropping a distinct feature from a future handoff.
- The figure colors show fractions within each true class, on the same
  0–1 scale. The printed cell values are counts. The labels and supports
  must match across representations. Each task exports side-by-side and
  single-matrix PNG and PDF files.

## Redraw without fitting

```bash
python -m mitoatlas.figures.confusion /path/to/new_run /path/to/figures \
    --handcraft-label 'Handcrafted 21D' --individual
```

The renderer reads `classification.csv` and the count-matrix CSVs.
The benchmark directory also stores fold predictions, metrics, input
hashes, source commit, and the list of handcraft features.
