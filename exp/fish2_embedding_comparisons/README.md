# FISH2 embedding comparisons

The studies below use Reviewed81. Record the cohort version, model checkpoint
and readout, input descriptors, and output folder with each run. Training
weights, embedding exports and generated figures stay outside Git.

| Experiment | Comparison | Source commits |
| --- | --- | --- |
| Final E16 checkpoint evaluation | Native concat and projected CLS embeddings, loss accounting, UMAP and fixed downstream probes | `d4a635f`, `039e5e3`, `078f13d` |
| Equal-width Concat768 | Image MLP versus PairBalance variants at the equal-width internal concat stage. Projected output is a separate readout. | `79e0874` |
| Compartment UMAP overlays | Handcrafted features, image CLS, and fused outputs from point-only and image-only refinement on matched instances | `04227dc`, `52da289`, `61a8c5e` |

The [checkpoint comparison runner](../../fish2_embedding_eval/README.md) and
[compartment UMAP script](../../HPC/mito_node_dists/plot_fish2_reviewed_umap.py)
are shared entry points. The [fixed evaluator](../../fish2_downstream/README.md)
documents the classification protocol and registered embeddings. The
checkpoint exporter requires the matching model code and data adapter.
UMAP panels are fitted separately; positions and distances are panel-local.
