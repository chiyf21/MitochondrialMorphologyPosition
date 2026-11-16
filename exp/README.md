# Archived experiments

This directory preserves exploratory code and records the comparison it was used for. Active workflows live elsewhere in the repository. Input datasets, generated dashboards, and results are stored outside Git.

## FISH2 compartment diagnostics

The files in [fish2_compartment_diagnostics](fish2_compartment_diagnostics/) are snapshots from the exploratory review branch at `d50e3b2`. They preserve the diagnostics as originally used, including their historical paths and input assumptions.

| Script | Experiment | Source commit |
| --- | --- | --- |
| `fish2_navis_pilot.py` | NAVis pilot on FISH2 skeletons | `c646408` |
| `compare_pc_compartments.py` | Original FISH2 versus NAVis on PC neurons | `de734f9` |
| `compare_step02_panel.py` | Step 02 versus NAVis across six neurons | `5f6c283` |
| `show_step05_gallery.py` | Step 05 gallery and expanded PC gallery | `ae3752f`, `1e5d200` |
| `show_pc_compartment_steps.py` | Five-step replay and axon candidates for PC 100076106 | `18a0a1e` |
| `show_pc_synapse_cloud.py` | Presynaptic and postsynaptic locations with the V11 split for PC 100076106 | `fe01147` |

For new analyses, use the [five-step pipeline](../HPC/mito_node_dists/compartment_pipeline.py) and the [review server](../HPC/mito_node_dists/serve_compartment_review.py). The active review tools are maintained with those scripts.
