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

只需细胞类型和区室的 Concat／手工 21D 混淆矩阵时，可使用 [分类与出图入口](figures/README.md)。

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

## 数据准备和历史对照

- [低层命令和历史复现](REFERENCE.md)记录空间缓存构建、LC 风格参数与逐实验入口。
- [同一数据集的 R4/R5 CSV 入口](README.md)用于既有 Reviewed81 或 FlyEM 数据。
- [早期探索实验](../exp/README.md)保留区室诊断和旧模型比较。
