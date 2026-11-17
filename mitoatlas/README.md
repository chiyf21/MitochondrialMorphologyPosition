# MitoAtlas morphology analysis

Run from the experiment repository root with Python, NumPy, pandas and SciPy.
This entry point runs R4 (distribution differences) and R5
(cross-compartment coordination) within a single dataset. Run it separately on
FISH2 and FlyEM using the same analysis options.

```bash
# Point these environment variables at the frozen FISH2 and FlyEM data directories.
python -m mitoatlas.morphology --dataset FISH2 \
  --handcrafted "$FISH2_REVIEWED81/fish2_handcraft_81neurons_22_with_compartment.csv" \
  --embedding "$FISH2_REVIEWED81/fish2_model_embedding_81neurons_384.csv" \
  --embedding-name fusion384 --out /path/to/new/fish2_run --smoke

python -m mitoatlas.morphology --dataset FlyEM \
  --handcrafted "$FLYEM_COMPLETE_SUBSET/author_features22.csv" \
  --out /path/to/new/flyem_run --smoke
```

Remove `--smoke` for the full eligible cohort. FISH2 uses Reviewed81; FlyEM
uses the LC21 `complete_data_subset`. Any learned CSV can be added with
`--embedding` if it has `body_id`, `record_id`, and `embedding_` feature
columns. The selected rows must all have exact matches. For NPY input, export an
ID-keyed CSV using the saved row map first. Inputs are read-only.

Both analyses use neurons with accepted instances in both axon and dendrite;
FISH2 uses `compartment_qc=ok`. A type needs at least two eligible neurons
(configurable via `--min-bodies`). Smoke selects up to three types and three
neurons per type, then up to twelve rows per neuron and compartment. The
identical selected instance list is used for all representations in a run.
Each feature is z-scored across this descriptive cohort, excluding constant
columns. This fit is for descriptive analysis only.

- **R4** uses the original LC notebook's pooled bidirectional nearest-neighbor
  Euclidean distance. It compares type pairs within each compartment and
  compartments within each type. It is descriptive and can depend on group
  sample size. Raw distance magnitudes depend on representation dimension.
- **R5** computes the cosine between type-centroid difference vectors in
  axon and dendrite. Centroids weight each instance equally; percentile
  intervals use a stratified neuron bootstrap, keeping axon and dendrite
  paired. Intervals are per contrast and are not multiplicity-adjusted tests.
  Undefined cosines are blank and the number of valid draws is recorded.

The output directory must be new. It records the selected cohort, both result
tables, source hashes, feature columns, parameters, software and Git revision
in `run.json`. The shared two-compartment cohort is a harmonized adaptation
of the three-compartment LC analysis.
