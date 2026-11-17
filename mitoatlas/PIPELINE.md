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
  --classifier lr --out /path/to/new/run
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
| R1 | Cell type classification within each compartment, then one per-neuron prediction from the combined axon and dendrite probabilities. |
| R2 | Axon/dendrite classification fitted within each type with training class coverage in each fold; aggregate groups use the annotated compartment. |
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

## Fixed FISH2 handoff (R1–R5)

Run the five aligned NPY tables directly with:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 \
  python -m mitoatlas.handoff /path/to/handoff_e50_12068 /path/to/new/run --workers 5
```

This reads the handoff's checksums and instance order. R1 and R3 use its
fixed cell-type folds, R2 uses the updated compartment folds, and R4/R5
use QC-passed axon/dendrite rows from neurons observed in both. R1/R2
pass Float64 features to the existing classifier so the probabilities pass
its normalization check. Use the saved reports to compare matched inputs.

For the FISH2 LC-style adaptation, run `--tasks R1 R2 --classification-protocol lc`
(the default). The common input uses accepted axon and dendrite instances.
The cable compartment is outside this protocol because class coverage is
insufficient on this panel. A handoff run keeps its fixed body folds. R1 fits
one LR per compartment and reports only the combined per-neuron result,
with at least five mitochondria per neuron across axon and dendrite. R2 fits one LR per cell
type and scores by body and true compartment. Both use the author's mean
log-probability aggregation and global z-scoring. LR balancing uses inverse
class frequency normalized to mean weight one. Global z-scoring includes
test-cohort feature statistics, as in the notebook; this is a replication
setting. Use `--classification-protocol legacy` for the previous pooled R1
with fold-local scaling.

To compare the LC classifiers, set `--classifier rf` for the handoff or
`--classifier lr rf` for the generic CSV entry. RF uses 100 trees, a 0.01
minimum weighted leaf fraction and inverse training-class frequency weights.
The saved group table flags ties caused by zero probabilities for every
class under the original log aggregation rule. Those ties are counted
under the original argmax convention and should be inspected.

Run `--tasks R2 --classifier rf --r2-include-cable` on a reviewed FISH2
handoff for the original three-compartment label set. The cohort includes only
cell types with all three labels in every five-fold training partition, and
keeps the handoff's fixed neuron folds. This cohort is distinct from the
all-type axon/dendrite comparison.
