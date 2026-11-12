# LC21 FlyEM paper morphometrics

Run from the repository root with NumPy, SciPy, scikit-image and scikit-learn.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python HPC/mito_node_dists/compute_lc21_paper_geometry.py --manifest /data/lc21/manifest.json --output /results/lc21/paper22_geometry.csv --workers 16

The input manifest is the ordered LC21 instance list. Each record has a stable
record_id and an absolute relative_dir field pointing to a directory with
mask.npy (ZYX) and metadata.json (global XYZ bbox, source center, source voxel
count, and spacing). Only 8 nm isotropic inputs are accepted. The source
voxel count is used for size and SBI. The actual mask count is also exported.

The 22 feature columns follow the authors' LC notebook. This stage fills the
20 columns that can be computed from masks and source metadata. The mito CA and
length along skeleton columns stay blank. These require local neuronal
segmentation and the cleaned skeleton. Use status and record_id for QC and
joining; blank cells must not be interpreted as zero.

The voxel surface estimator and PCA-plane sampler are extracted from the
authors' helpers. All geometry is computed in the authors' voxel units.
The PCA fit and diameter use separate 1% voxel samples from a reproducible
per-instance random stream. Convex-hull area is evaluated on boundary voxels.
Symmetry counts use exact packed integer row IDs. Farthest-pair distances use
the sampled points' convex hull, with a bounded-memory fallback.

Outputs are generated data and stay outside Git. While running, the ordered
table is written to a .partial.csv file. A final CSV may still contain rows
whose status reports an error and should be reviewed before analysis.

## Context features

Fetch raw neuPrint skeletons on a host with an existing credential. The
credential remains on that host. Transfer the SWC files to the compute host.

    python HPC/mito_node_dists/fetch_lc21_skeletons.py --manifest /data/lc21/manifest.json --output /results/lc21/raw_swc
    python HPC/mito_node_dists/compute_lc21_context.py --manifest /data/lc21/manifest.json --geometry /results/lc21/paper22_geometry.csv --skeletons /results/lc21/raw_swc --skeleton-kind raw_neuprint --output /results/lc21/paper22_context_raw.csv --workers 24

The context runner also requires CloudVolume. It uses the public FlyEM neuron
and grouped mitochondria layers. Join context and geometry by record_id.
The raw SWC version uses original neuPrint nodes, raw radii, and parent-edge
orientations as inputs to the paper's cross-section and overhang logic. This is
provisional. The paper instead repositions and resamples nodes, estimates radii
from segmentation, and prunes branches before the feature computation.
The default maximum section radius is 300 voxels to bound memory. For an
author-cleaned CSV input directory, use --skeleton-kind author_cleaned and
--max-radius 700 to reproduce the paper's section-search ceiling. Context values
are in voxel units (mito CA in square voxels); missing and fallback cases are
recorded in the section_status and length_status columns.

### Complete feature table

Run `python HPC/mito_node_dists/complete_lc21_paper_features.py /results/lc21`
with both input CSVs in the same output directory. The resulting
`paper22_complete_pc1.csv` keeps all 22 numeric features from `size` through
`SBI`, plus instance metadata and provenance. Missing or nonpositive context
values are filled with `PC1 CA` and `PC1 Length` respectively. Positive context
values, including the context runner's existing PC1 fallbacks, are preserved.

### Published compartment annotations

Run `extract_lc21_dryad_annotations.py` with the feature output directory as
its argument. The script reads a Dryad token from `~/.config/dryad/token`
and extracts cohort files using HTTP byte ranges. Then run
`label_lc21_author_compartments.py` with the same directory. It writes
`author_dryad/paper22_author_compartments.csv`, preferring exact within-body
XYZ matches in the author's mitochondria tables and using the nearest
cleaned skeleton node for remaining instances when available. Join by
`body_id` and `record_id`.

### Annotated morphometric tables

Run `python HPC/mito_node_dists/export_flyem_annotated_features.py /results/lc21`
after generating the completed features and author compartment table.
`flyem_mito_handcrafted_annotated.csv` contains the full cohort, annotations
and reconstructed 22-dimensional vectors. The companion file
`author_dryad/flyem_mito_author_features_exact.csv` contains exact within-body
XYZ matches to the published mitochondria tables. It copies 21 author
features and derives SBI with the authors' LC morphology notebook formula.
Both tables join on `body_id` and `record_id`.
