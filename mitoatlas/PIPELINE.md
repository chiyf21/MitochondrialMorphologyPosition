# MitoAtlas FISH2 标准下游流程

新 Concat embedding 统一从 `python -m mitoatlas.pipeline` 进入。流水线按
`record_id` 与冻结面板对齐，复用神经元划分、区室标签和空间缓存。
旧版逐实验入口见后半部分，可用于复现历史运行。

## 单次运行的输入

- `panel` 是包含 `manifest.json` 的冻结、已复核 FISH2 handoff。
- `concat.npy` 是形状为 `[实例数, embedding 维数]` 的有限浮点矩阵。
- `embedding_rows.csv` 每行一个 `record_id`。可加 `embedding_row` 写明 NPY 的
  0-based 行号。没有行号列时，CSV 行顺序就是 NPY 行顺序。每个面板实例必须恰好出现一次。
- `output` 是新目录。程序校验输入并记录 Git 提交与输入哈希。结果总索引是 `output/README.md`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
panel=/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/exports/primary10x15_native768_stage_review_20261117_01/handoff_e50_12068
python -m mitoatlas.pipeline "$panel" /path/to/concat.npy /path/to/embedding_rows.csv /path/to/new_run
```

只跑第一组实验时，在命令末尾加 `--modules morphology`。模块可组合。
R1/R2 默认采用 LC 概率聚合和训练折内标准化。加
`--lc-scaling full_cohort` 可复现原 notebook 的全队列标准化，
`--classifier rf` 可切换随机森林。R3、E04/E05 和第二轮模型仍按照各自协议拟合。

### 各模块读取的现成产物

所有模块共用已经导出的 Concat、实例行映射、复核面板内的手工特征和神经元划分。

| 模块 | 另需提供或可复用的产物 |
| --- | --- |
| morphology | E01 可复用同面板的 `mask_solidity.csv`。没有该缓存时，E01 从面板引用的现有实例 mask 计算凸包填充率。 |
| em | E02 读取面板记录指向的现有实例 EM 和 mask，用于纹理统计和盲审图。配对名单随本次 embedding 重新生成。 |
| synapse | E04/E05/E06 读取已有的 E05 `windows.csv` 及其 `run.json`。E06 的纹理和盲审图还会读取已保存的实例 EM 和 mask。 |
| connectivity | E07–E10 读取 `phase2_data` 已生成的连接缓存，包括位点、接触、线粒体关联、骨架区段和解剖表。 |

常规重跑从这些已有产物开始。执行内容是下游拟合、统计、配对和出图。
若需要更新面板或空间缓存，下文列出了独立的数据准备入口。模型权重、
原始数据、embedding 和生成的实验结果保存在工作目录之外，由输入路径引用。

## 实验清单与输出

| 模块 | 实验 | 主输出 |
| --- | --- | --- |
| morphology | E01（R3）形态恢复，E03（R1/R2）类型与区室，R4 分布距离，R5 区室协调，A1 探索性聚类 | `01_morphology/`。E01 输出折外 R² 和置信区间图。E03 输出神经元类型及单线粒体区室的混淆矩阵。 |
| em | E02 外部形态匹配后的 EM 纹理差异，包含 A2 近邻匹配 | `02_em/E02/`。每次换 embedding 都重新选择配对。 |
| synapse | E04 最近同区室突触距离，E05 1/2/5 µm 局部计数与密度，E06 邻近和远离配对 | `03_synapse/`。位置和距离标签直接复用，模型与配对统计重新计算。 |
| connectivity | E07 位点 fan-out，E08 目标类型与区段组成，E09 神经元连接指纹，E10 两侧匹配可行性 | `04_connectivity/`。逐项报告可拟合范围与数据门槛，达到门槛的任务才输出预测。 |

同一实验只在清单出现一次。R1/R2 收到 E03，R3 收到 E01，A2 收到 E02。
E07 从第二轮接触数据运行。A3 已搁置。E10 当前入口只做配对可行性。

## 更换其他输入

- `--handcrafted FILE` 接完整的、按 `record_id` 对齐的 22 列手工 CSV。
  默认沿用冻结 handoff 的 FISH2 基线。单独重算手工特征的入口是
  `mitoatlas.handcraft_upstream`。先检查完整性再替换。
- `--train-bodies FILE` 接该模型已知的训练神经元 CSV，含 `body_id` 列。
  E04/E05 在有足够五折支持时额外评估排除这些神经元的子集。新模型默认不套用旧 E50 的训练标签。
- 固定 150 神经元面板默认从 handoff 父目录使用已保存的 E05 窗口、E01 凸包填充率和第二轮连接缓存。
  可用 `--windows`、`--solidity` 和 `--connectivity` 显式指定。
  新面板需要按后面的旧入口从对应神经元骨架与连接表重建缓存。
  E05/E04/E06 与 E07–E10 会检查缓存所属面板，缺缓存时记录 BLOCKED。
- 只换 embedding 时不用重新算骨架、突触距离或手工特征。每个正式运行使用新的输出目录，
  旧试验可按运行索引回查。

## 历史运行及低层构建命令


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

Run the fixed FISH2 handoff directly with the Concat embedding and handcrafted reference:

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

The NPY handoff also supports `--tasks A1 A2`. A1 fits PCA and KMeans
on the full aligned cohort. A2 uses QC-passed axon and dendrite instances.
The optional EM shortlist takes the closest quarter in handcrafted geometry
within type and compartment, limits volume and PC1 length ratios to 1.25,
and keeps pairs in the top 10% of distance for a given learned representation
within that stratum. It writes up to two unique pairs per type/compartment
stratum for each representation. The complete directed candidate table is
retained. Selection thresholds are descriptive and suitable for review only.

## R3: geometry retention outside the soma

R3 excludes instances annotated as `soma`. Other annotated compartments and
ambiguous instances remain in the cohort. The fixed handoff keeps the original
cell-type body folds and runs the Concat embedding against the 22
continuous hand-designed targets:

```bash
python -m mitoatlas.handoff /path/to/handoff_e50_12068 /path/to/new/run \
  --tasks R3
```

This runs Ridge (alpha=10) with input scaling fitted within each training fold.
Targets stay in their original units. Reported R² and MAE use out-of-fold
predictions and equal total weight per neuron. The hand-designed table supplies the targets for this run.

## E04 FISH2 distance pilot

`python -m mitoatlas.synapse_distance HANDOFF REVIEW_SNAPSHOT ORIGINAL_REVIEW_ROOT
SYNAPSE_SNAPSHOT OUTPUT --bodies BODY_ID ...` generates labels and projection QC.
It reads nm coordinates from the reviewed skeleton and synapse attachment
table, deduplicates endpoints by role and physical XYZ within each neuron,
and reprojects them onto the same original edges. Artificial bridge edges
are excluded. Unreachable sites, failed projections, and masks with no
sampled skeleton intersection leave distance fields blank. The occupancy
distance is the minimum over samples inside the mask, with at most 4 nm
sample spacing. Use this pilot to assess coverage before fitting any models.

To run E04 for the complete FISH2 panel, provide all reviewed body IDs to
`mitoatlas.synapse_distance --bodies … --workers 12`, then use
`python -m mitoatlas.synapse_regression HANDOFF DISTANCE_OUTPUT NEW_OUTPUT`.
Ridge uses fixed neuron folds, train-only z-scoring, type and basic geometry
controls, and separately reports axon/dendrite, pre/post and raw/log distance.
MAE gains use paired bootstrap intervals across neurons. Missing distances
are excluded per endpoint and distance definition and counted in coverage.csv.

## FISH2 E01–E07 review

Reuse the frozen FISH2 handoff, the full E04 distance run, the A2 candidate
table and the 150 per-body raw connection snapshots. Set BLAS and OpenMP
threads to one before running the process-parallel labels. The report reads
the prior no-soma R3 and LC-style R1/R2 LR results from the export directory.

```bash
python -m mitoatlas.synapse_load "$handoff" "$e04/distances" "$out/E05" --workers 16
python -m mitoatlas.synapse_fanout "$handoff" "$e04/distances" "$connections" "$out/E07"
python -m mitoatlas.synapse_comparisons E02 "$handoff" "$a2/candidate_pairs.csv" "$out/E02"
python -m mitoatlas.synapse_comparisons E06 "$handoff" "$out/E05/windows.csv" "$out/E06"
python -m mitoatlas.synapse_review "$export" "$out"
python -m mitoatlas.synapse_report "$out"
```

The original E04 path and targets can cross reviewed compartment boundaries.
The E04 sensitivity and E05/E06 labels use only original edges annotated as
the instance's compartment. E05 zeros are counts of recorded sites. E07 uses
coordinate-defined pre sites from the raw table and reports observed
fan-out, with distinct post endpoints and distinct partner bodies as separate
targets. Its nearest-mito linkage uses a centroid projection within 5 µm.

## Phase II FISH2 connectivity

Use a new output directory and set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS
to one. Reuse the Phase I full E04 distance directory and E07 site labels.
The roster is the 326-neuron FISH2 neuron_index.csv, used only to map body
IDs to available types. Only reviewed panel neurons contribute embeddings.

```bash
python -m mitoatlas.phase2_data "$handoff" "$e04/distances" "$phase1/E07/site_labels.csv" \
    "$connections" "$roster" "$out/inputs"
python -m mitoatlas.phase2_models "$handoff" "$out/inputs" "$out"
python -m mitoatlas.phase2_report "$out" "$handoff"
```

Phase II recovers archived contact and endpoint identifiers without spatial
clustering. Fixed body-level folds are used for all estimable tasks. E08
gates label classes on per-fold training support, and E10 reports the counts
of feasible matched null pairs. The observed partner profiles depend on
what was captured in the 150 per-body connection archives.
