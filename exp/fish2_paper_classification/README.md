# FISH2 R1 and R2 using the LC paper protocol

Run from the experiment repository with the QC handoff. This entry point runs both the author-style random forest and L2 logistic regression against the same sampled rows and folds.

```bash
python -m mitoatlas.fish2_paper_classification QC_HANDOFF OUTPUT
```

- R1 trains one classifier per compartment. The held-out mitochondria are combined per neuron by mean log10 probability with a five-mitochondrion minimum. Separate compartment results use the same minimum.
- R2 trains one classifier per cell type. Held-out probabilities are combined per body and true compartment. This is the primary R2 scoring level. Instance-level predictions are exported separately.
- Body IDs are shuffled reproducibly within each training scope. An invalid draw with missing training classes is redrawn. RF and LR use the same assignments.
- The benchmark defaults to full-cohort z-scoring, following the LC notebook. Use the `--scaling training_fold` variant to fit normalization on training rows only.
- FISH2 evaluates axon and dendrite across ten cell types. FISH2 Handcrafted drops the exactly duplicated cross-sectional area column.
- RF keeps 100 trees, minimum weighted leaf fraction 0.01, inverse class-frequency sample weights and seeded randomness. LR uses L2, C=1 and balanced class weighting.
- Pooling preserves the author notebook’s exact-zero convention. The comparison table counts groups for which every class receives a zero product.

Each classifier folder has per-instance and pooled predictions, confusion matrices and figures. `comparison.csv` lists both scoring levels. Use the grouped rows for the paper comparison.
