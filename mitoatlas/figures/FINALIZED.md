# FISH2 finalized figure analyses

This index records the frozen FISH2 figure analyses and the approved
synapse prediction experiment. Run the commands from the repository root.
Keep data, checkpoints and generated figures in the experiment storage.

| Analysis | Figure or readout | Code reference |
| --- | --- | --- |
| R1 and R2 | Matched 21D and Concat confusion matrices. R1 has neuron and single-mitochondrion panels. R2 predicts compartment. | `fish2-fig2-classification-recovery-v1` |
| E01 | In-sample OLS recovery of 21 distinct handcrafted morphology measurements from Concat, with soma excluded. The figure sorts targets by R². | `fish2-fig2-classification-recovery-v1` |
| E02 | Shape-matched EM example pairs, including wider fields, mask outlines, surface point clouds and PNG scale bars. The selected examples illustrate texture variation at similar shape. | `fish2-fig2-e02-em-v1` |
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

E04–E10 and the full-cohort distance regression serve as exploratory
records for the synapse module. The finalized synapse figure comparison
uses pre/post classification.
