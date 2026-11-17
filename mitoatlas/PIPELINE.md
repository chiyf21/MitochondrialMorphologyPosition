# MitoAtlas R1–R5 and A1–A3

Run from the root of this Git checkout. Use the same entry point independently
for FISH2 and the designated FlyEM LC21 complete subset.

```bash
python -m mitoatlas.run --dataset FISH2 \
  --handcrafted /path/to/handcrafted.csv \
  --embedding point /path/to/point.csv \
  --embedding image /path/to/image.csv \
  --embedding fusion /path/to/fusion.csv \
  --texture /path/to/independent_measurements.csv \
  --classifier lr rf --out /path/to/new/run
```

- Handcrafted CSV: `body_id`, unique `record_id`, `cell_type`, the 22 feature
  columns described in [README](README.md), and `compartment` where available.
  FlyEM's `length along skeleton` is accepted as the corresponding column
  without a unit conversion. `compartment_qc`, when present, must equal
  `ok` for compartment analyses.
- Each learned CSV has `body_id`, `record_id`, and numeric `embedding_`
  columns. Every selected handcrafted row must have an exact ID match.
  Supply an audited common cohort if the representation coverages differ.
  For NPY files provide the row-to-ID mapping to produce a keyed CSV.
- The optional texture table has `body_id`, `record_id`, and independently
  measured numeric `texture_` columns. Record what the measurements mean
  outside the table. Do not use model output, its projection, or the same
  texture computation as both a predictor and endpoint.

Use `--tasks R1 R2 R3 R4 R5 A1 A2 A3` to select modules (all by default).
Use `--umap` to add UMAP coordinates to A1 if umap-learn is installed.
The output directory must be new. `run.json` records input hashes, code
revision, parameters, package versions and each task's status. Each task
writes its cohort and outputs to its own subdirectory.

| Task | First analysis and output |
| --- | --- |
| R1 | Pooled neuron type classification with five type-stratified, neuron-disjoint folds; instance and neuron scores and out-of-fold predictions. |
| R2 | Axon/dendrite classification fitted within each type with at least five neurons observed in both accepted compartments; aggregate groups use the annotated compartment. |
| R3 | Fold-local scaled Ridge predicts each hand-designed measure from each embedding. The handcrafted comparator excludes the target column; related measures can still encode it. |
| R4 | Original bilateral nearest-neighbor distribution distance on standardized features and the common paired axon/dendrite cohort. |
| R5 | Axon/dendrite type-shift cosine and paired neuron bootstrap on the same cohort as R4. |
| A1 | PCA spectrum, effective rank, candidate KMeans splits, stability across initializations, type/compartment composition, morphometric profiles and gallery IDs. These are descriptive summaries. |
| A2 | Nearest handcrafted matches within each type and accepted compartment, across different neurons. Exports the full directed candidate list up to the configured number of matches per anchor, distances and within-stratum learned distance percentiles. Inspect matched EM before interpreting the candidates. |
| A3 | Compares geometry-only Ridge against geometry plus each embedding using the same complete-case cohort and held-out neurons. Outputs out-of-fold residuals and changes in neuron-weighted R² and MAE. This tests incremental prediction relative to the chosen geometric baseline. |

These split definitions are shared across compared representations. The
existing frozen Reviewed81 benchmark remains available in `fish2_downstream`;
this entry point computes five folds from the input cohort. For A1–A3,
choose the independent measurement and matching controls before a paper run.
The first pipeline does not attempt skeleton, synapse or circuit analyses.


## Read results by experiment

Open `RUN/report.md` for the experiment index and follow each task link to its
summary and underlying files. For example, the R1 page shows instance and
neuron scores and links to confusion matrices; the A3 page shows per-target
changes from adding an embedding. Skipped tasks report their missing inputs.
Select a single experiment with `--tasks R4` and a new output directory.

To add or refresh the reports for an existing run without refitting models:

```bash
python -m mitoatlas.report --run-dir /path/to/run
```
