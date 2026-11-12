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
