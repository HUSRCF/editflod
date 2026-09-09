# 蛋白质一步结构编辑：项目实践方案 v0.1

**暂定题目：Reference-Preserving Protein Structure Editing via Mutation-Conditioned Transport**  
**方案日期：2026-09-08**  
**状态：研究与工程设计初稿。未完成模型部署、训练或基准测试；文中的参数是建议起点，不是已验证结果。**

## 0. 执行摘要

输入父本蛋白结构 X0、父本序列 S0 和指定突变序列 S1，输出突变体结构 X1。研究重点不是重新折叠，而是利用已有构象预测必要更新，同时减少无关结构漂移。

首版限定等长单链、天然短链或完整独立结构域、单点替换。先预测 N/CA/C/O 主链，再用独立侧链重建模块补全目标序列。暂不处理插入删除、大尺度折叠转换、复合物重排、功能自由能或动力学。

主路线分两阶段：

1. 使用已有序列条件结构生成器，验证共享噪声下的源—目标条件差分，并比较单噪声级和双噪声级的单次结构更新。
2. 只有编辑算子在实验配对上显示价值，才蒸馏或监督训练真正单次前向的编辑学生。

三项必须同时评价：突变响应准确性、无关漂移、端到端成本。父本复制不是成功编辑；整体坐标系一致也不是结构精度证据。

## 1. 研究问题与可支持的贡献

### 1.1 核心假设

H1：在同一个带噪父本结构上比较源序列与突变序列，可抵消部分共同重建偏差，改善保留—响应折中。

H2：在统一的父本局部坐标系内表达编辑，并约束链几何，比不加约束的坐标残差更稳定。

H3：对于一个父本对应多个突变候选的工作负载，复用父本表征和压缩编辑算子可以降低总成本。

这些是假设，不应预先写成实验结论。ChordEdit 的图像最优控制解释不能不经推导直接当作蛋白质上的证明。

### 1.2 与已有工作的关系

PreMut 已研究野生型结构加单点突变到突变体结构；不能声称首个该任务模型。ChordEdit 提供条件差分、短时间窗组合与结构保留的启发；FoldFlow-2 提供序列条件主链生成器和可查询结构输出；ESMFold 提供从序列重新预测的效率与精度对照。[1–5]

预期贡献应由实验决定：参照保留的编辑方法、严格突变响应评测、单父本多候选的高通量实现。不能仅凭残差连接或 NFE 数量宣称新原理。

## 2. 首版任务契约

| 项目 | v0.1 范围 |
|---|---|
| 输入 | 父本 mmCIF/PDB、链与残基映射、S0、S1 |
| 长度 | 先 64–128 aa，再扩展到 256 aa；不任意截断长链 |
| 突变 | 首先 1 个替换；2–4 个替换另列扩展集 |
| 输出 | N/CA/C/O 坐标、残基映射、位移、几何检查、耗时元数据 |
| 目标状态 | 原折叠状态附近的突变相关结构 |
| 主要条件 | 优先独立可折叠单链；必需配体/伙伴缺失的样本不混入主集 |
| 不做 | 插入删除、任意状态切换、MD 平衡系综、功能/自由能保证 |

无突变输入必须返回原输入主链；实现时从原坐标更新，不用理想模板重新生成原结构，否则零更新也可能产生漂移。

## 3. 数据协议

### 3.1 来源与用途

以 PreMut 的 MutData2022/2023 数据入口作为配对来源和复现起点，再依据质量与环境一致性重新审计。仓库给出了 Zenodo 数据入口，但本方案未下载并验证所有数据文件。[2]

数据分三类：

- **实验配对**：主要训练监督、开发和最终评价。不能用教师预测替代实验终点。
- **同序列结构重复**：估计不依赖突变的结构背景差异；不是要求模型把所有同序列构象都视为相同。
- **人工编辑与教师预测**：只作为训练辅助或蒸馏标签，记录来源，不能作为实验成功证据。

### 3.2 清洗规则

核对链、序列和原子映射；明确 author residue number、insertion code 与内部 0-based index。缺失残基保留 mask，不因缺失坐标重新编号。处理 alternate conformer 和不完整主链。

主集优先选择相同或可比的配体、伙伴、寡聚状态与实验条件；将 apo/holo、明显不同构象状态等配对独立记录。不要将所有观测差异直接解释成突变的因果效应。

保存父本质量、突变体质量和实验方法。不要对真实实验结构使用不存在的 pLDDT 字段；B-factor、occupancy 与预测模型置信度分别管理。

### 3.3 划分

先形成同源家族/近邻编辑连通组，再划分训练、开发与锁定测试。可将 30% 序列一致性和双向覆盖条件作为保守起点，但应记录实际聚类程序与阈值。一个父本的所有编辑、反向编辑和裁剪必须同组。

官方 PreMut split 用于复现；自建严格 split 用于新实验，不能把两者混为同一评测。记录预训练底座的数据暴露范围；重新划分学生数据不等于消除了预训练泄漏。

### 3.4 必要字段

`pair_id, parent_id, family_id, split, source_file, target_file, source_chain, target_chain, source_sequence, target_sequence, residue_map, mutation_indices, atom_mask, environment_metadata, experiment_metadata, source_checksum, target_checksum, label_source, teacher_config_hash`

建议产物：`pairs.parquet`、`split_manifest.json`、`audit_report.json`、逐样本排除原因表。

开发规模起点：8–16 对用于过拟合排错，32–64 对用于机制开发。最终测试规模以清洗后数据和家族数为准，不预先承诺不存在的样本数量。

## 4. 底座接入与工程核查

### 4.1 首选机制底座

首选 FoldFlow-2 的基础权重作为候选，而非多样性强化微调版本。它采用序列条件残基框架建模，论文和配置使用 ESM2-650M；官方 release 0.2.0 列出 `ff2_base.pth`。[3,4]

**重要：发布说明使用了 unconditional model 的措辞；代码支持条件路径，不等于该权重已被验证为准确的突变效应教师。必须通过本项目的条件响应检查。**

本次核查代码中：

- `foldflow/models/ff2flow/flow_model.py` 的默认 eval 分支会屏蔽序列。
- 必须显式调用 `conditional_generation()`。
- `train(False)` 会重置条件生成状态；之后再调用 `.eval()` 可能再次关闭条件路径。
- `forward()` 中会调用序列编码器；缓存改造必须明确区分父本序列缓存、目标序列缓存和结构状态缓存。
- 模型返回残基框架、平移/旋转向量场及 psi 等信息，可封装为统一 adapter。[4]

统一接口规范，而非已有可运行 API：

`encode_sequence(sequence) -> SequenceCache`

`noise_source(source_frames, time, seed) -> NoisyState`

`predict_endpoint(noisy_state, time, sequence_cache) -> PredictedFrames`

`edit(source, source_sequence, target_sequence, config) -> EditResult`

### 4.2 首批单元测试

无突变恒等性；共享噪声一致性；改变目标序列时条件特征确实改变；批处理与逐样本结果一致；全局旋转平移等变性；Å/内部缩放一致；链与 mask 保持正确；完整调用无隐藏 recycling；空缺主链不进入损失。

几何指数/对数和分支相减先使用 FP32。网络本体混合精度的结果应与 FP32 小样本比较，防止微弱突变差分被数值误差淹没。

## 5. 单次更新编辑原型

### 5.1 统一到父本局部坐标

每个残基具有父本框架 `T0_i=(R0_i,t0_i)`。对同一带噪父本，分别用源、目标序列预测终点框架 `(Rhat_i^c,that_i^c)`。

定义父本局部表示：

`xi_i^c = [ R0_i^T (that_i^c-t0_i), Log(R0_i^T Rhat_i^c)^vee ]`

再取：

`d_i(t) = xi_i^target(t) - xi_i^source(t)`。

这是局部坐标中的终点编辑差分，是本项目拟议的工程基线，不是已证明等价于 ChordEdit 的物理/最优控制场。旋转差过大或接近对数映射分支边界时须标记不可靠，不能强行视为近邻编辑。

直接对来自不同状态的旋转矩阵或向量场相减不可作为无说明的实现。更忠实的原生向量场版本应单独实现时间尺度转换及切空间映射，并作为消融。

### 5.2 单噪声与双噪声

先测试 `u_i=d_i(t)`。再测试 `u_i=w1*d_i(t1)+w2*d_i(t2)`，其中两个状态从同一父本、同一基础噪声按官方日程构造。

平移与旋转分别设置更新尺度。初版不加入迭代 refinement；添加时单独报告。

更新：

`t1_i = t0_i + R0_i * eta_trans * u_trans_i`

`R1_i = R0_i * Exp([eta_rot * u_rot_i]_x)`

将这一刚体变化作用于父本残基实际 N/CA/C/O 坐标，保留零更新时的精确原结构。SO(3) 合法不等于整条肽链几何合法，需另做链连接和碰撞检查。

### 5.3 实验矩阵

C0：复制父本主链。

C1：同底座只用目标条件做普通部分去噪；至少测 1/4/8 次结构更新。

C2：单噪声级源—目标终点差分。

C3：共享噪声的双噪声级组合。

C4：在同一噪声级重复两次源—目标差分，总计四个条件查询，与 C3 匹配预算，用于区分时间组合收益和单纯多查询收益。

C5：C3 但源目标使用独立噪声，用于测试共享噪声的作用。

所有时间/更新尺度仅在开发集选择。不要为每个测试目标根据其真值搜索最优步长或噪声。

### 5.4 条件信号与教师准入

改变突变身份、打乱 edit 标签、保持序列不变分别测试。如果模型对不同编辑没有区别，先查条件 mask、缓存键、精度和索引，再判断底座能力。

教师需在独立开发实验配对上取得有用的保留—响应折中，不能只凭低位移或高置信度通过准入。准入脚本还会用相同共享噪声重复查询，并以固定容差检查响应摘要的可重复性；非有限、无响应或不稳定的配对都不能进入伪标签池。不通过时，不大规模生成伪标签。可选择有限配对监督适配，但若适配后仍无优势，则报告该编辑算子不成立。

## 6. 真正单次前向学生

教师通过准入后，构造：

`H0 = ParentEncoder(X0,S0)`

`edit_i = 1[S0_i!=S1_i] * Embedding(S0_i,S1_i)`

`delta_frames = EditNetwork(H0,edit)`

`X1 = ApplyLocalFrameUpdate(X0,delta_frames)`。

父本编码可缓存。编辑网络建议起点为 4–6 层结构图/几何注意力模块，隐藏维度 128–256，结合序列边和空间邻接；允许影响传播到突变位置之外。参数量实际构建后统计。

使用父本局部方向、相对框架、距离等几何特征，输出局部位移和旋转。无突变时显式返回原主链。不要为了恒等性而在每个推理请求里隐藏额外源分支网络评估。

实现上，学生输出头采用零初始化，使训练初始点严格等于复制父本；训练目标对平移（Å）和旋转（弧度）使用独立尺度，尺度随 checkpoint 保存并在独立评估时复用。可选的增量范数正则只作为稳定性消融，不能替代实验结构监督。

先建立两种条件输入：

- 质量模式：缓存父本 ESM，目标序列 ESM 重新计算；这是较完整的序列条件对照。
- 高通量模式：父本表征缓存，加离散编辑编码器，不重算目标完整 PLM；这是需单独验证的近似，不称作完全相同的推理。

结构主结果先用固定种子或确定性学生。多样本版本单独评价，不宣称单次输出表达了正确的物理系综。

## 7. 训练与损失

### 7.1 顺序

先做配对监督小样本过拟合，再做纯实验配对学生基线；教师通过准入后才加入伪标签蒸馏。比较相同学生架构、相同数据与可比较 GPU 预算的直接监督和蒸馏版本。

候选起点：冻结 PLM；编辑学生学习率 1e-4；AdamW；weight decay 0.01；micro-batch 1；梯度累积 16；先长度 128。已有底座解冻部分可从较小学习率 1e-5 开始。均为待调试参数。

当前实现提供 `delta_norm_weight` 作为可选的预测增量范数正则。它只能抑制小数据训练中的异常大更新；若开启，必须与未开启版本在相同划分和预算下比较，不能单独作为精度改进证据。

### 7.2 最小损失

`L = L_target + lambda_geo*L_geometry + lambda_KD*L_distill`

`L_target`：实验终点的局部框架/坐标误差与距离误差，区域分别归一化；避免大量远端近零变化淹没编辑区。

实现中的 batch 目标损失先按每个配对的有效残基数归一化，再对配对取平均，避免长链样本在混合 batch 中隐式获得更大权重。平移和旋转通道另用独立单位尺度归一化。

`L_geometry`：链连接、局部角度、碰撞等；阈值与实现版本固定。

`L_distill`：与合格教师输出的父本局部编辑表示或终点几何的差。只用于辅助标签样本；标签来源、权重和置信标准明确记录。

当前训练 CLI 已支持读取 `TeacherCache` 的父本局部 delta，并通过
`--distill-weight` 将其作为可选辅助项；默认仍只使用实验突变体目标，且
cache 必须先通过 teacher admission 和 manifest fingerprint 校验。

不要另加一个与终点误差代数相同的“响应损失”后宣称新监督：

`[(Dhat-D0)-(Dtrue-D0)] = Dhat-Dtrue`。

弱保留正则是可选消融，不作为必需项。若采用，只在训练参考支持稳定的部分使用；不把所有远端或未突变残基一律固定，不要求物理上多对一的编辑严格循环可逆。

实验监督与教师监督分开记录，不能用重复教师标签人为扩大真实样本数量。

## 8. 评测方案

### 8.1 必要基线

| 基线 | 回答的问题 |
|---|---|
| 复制父本主链 | 是否真的预测了主链变化 |
| 复制主链 + 统一侧链重建 | 是否只是利用原结构做侧链更新 |
| PreMut | 是否优于已有相同输入任务的方法 |
| ESMFold 默认与零额外 recycling | 相较从序列重新预测的效率与精度 |
| 同底座部分去噪、单差分、双差分 | 改进究竟来自哪个机制 |
| 同架构直接监督学生 | 蒸馏与预训练编辑场是否必要 |

PreMut 原始结果与带 refinement 结果分别报告。ESMFold 的零额外 recycling 仍然执行一次普通 folding trunk，不能解释为没有运行结构预测。[5]

侧链可以使用 AttnPacker 等已有方法作为候选统一后处理，但须检查实际接口是否固定主链、是否还有 refinement，并将全部成本计入。[6]

### 8.2 区域与指标

区域由父本结构预先定义，例如到突变 Cα 的距离：≤10 Å 为局部，10–15 Å 为过渡，>15 Å 为远端。这是建议起点，阈值冻结于开发阶段。

主指标包括：局部主链误差、局部 lDDT/接触误差、共有原子的距离变化误差、远端预测误差、链断裂/碰撞及覆盖率。整体 TM-score 和整体 RMSD 为补充。

预测漂移不等于错误。远端评测既报告与父本的变化量，也报告与实验终点的误差。真实远端响应样本单列。

对齐采用所有方法相同的残基对应和刚体叠合协议，另报不依赖刚体对齐的距离指标。不得用原始未叠合 RMSD 构造对 ESMFold 的人为优势。

实现同时报告 `parent_to_prediction_frame_rmsd` 与
`remote_scaffold_frame_drift`，二者不做额外 Kabsch 叠合，只用于审计参照
坐标系中的实际漂移；对齐后的 RMSD 仍用于内部形状比较。

按家族而非单个残基/近邻突变做统计汇总和 bootstrap。报告均值、中位数、失败率和组级置信区间；各方法比较同一组可评估目标，失败不能静默删除。

每次机制网格结果还应保存运行元数据，包括 manifest 指纹、endpoint
factory、划分、噪声/步长/单位尺度、batch size 和运行环境版本，避免不同
实验配置被误作同一结果。

### 8.3 性能协议

分别报告：

- 冷父本：父本预处理、编码、目标序列编码、编辑、侧链与文件输出。
- 热父本：父本缓存后单个新突变的完整成本。
- 一个父本对应 N=1/32/128 个候选的总成本和吞吐。

统一 GPU、精度、长度、batch、warmup 和同步方式。记录 p50/p95、峰值显存和 OOM；模型加载时间另列。

报告结构更新次数、条件分支数、网络调用次数、序列编码次数、采样数和后处理成本。两条件×两噪声级即使合并为一次 batch 调用，也不能隐去四个条件评估的计算量。

比较主链任务时统一评主链；全原子端到端另列，不能把主链编辑器与含侧链和 refinement 的整套系统只按输出时间简单比较。

## 9. 阶段产物与推进门槛

| 阶段 | 产物 | 推进依据 |
|---|---|---|
| P0 协议与环境 | split/manifest、baseline wrapper、条件生成测试、运行元数据 | 无泄漏、映射和条件路径正确 |
| P1 机制实验 | 32–64 对开发配对上的 C0–C5 结果、局部误差—远端漂移图 | 差分相较匹配预算对照显示可重复收益 |
| P2 单次前向学生 | 直接监督学生、蒸馏学生、等变性与几何报告 | 学生保留必要变化，且没有靠复制或隐藏后处理取胜 |
| P3 锁定评测 | 独立家族测试、1/32/128 候选成本、失败分析 | 精度与成本折中支持清楚的贡献 |
| P4 扩展 | 2–4 点联合编辑、低质量父本鲁棒性、侧链一体化 | 主任务已经成立后再扩展 |

建议内部目标可以设为：局部误差相对强参照基线有约 10% 改善，或在预先定义的非劣界内显著提速；远端实验误差和几何失败率不恶化。这里的数字是立项阈值，不是预测结果；误差容忍度必须结合开发集背景噪声确定。

如果差分方法不优于普通部分去噪，不为保留方法名而无限堆叠模块。如果主链复制始终最强，应检查样本是否缺乏可辨主链响应，以及项目是否应收窄为侧链/局部原子编辑。

## 10. 建议工程目录

```
protein_edit/
  configs/
  data/manifest/
  adapters/foldflow2_adapter.py  # 当前原型：ospedit/adapters.py + foldflow_batch.py
  models/source_chart_editor.py
  models/cached_context_student.py
  geometry/frames.py
  geometry/update.py
  geometry/violations.py
  baselines/copy.py
  baselines/premut.py
  baselines/esmfold.py
  training/losses.py
  evaluation/regions.py
  evaluation/metrics.py
  evaluation/runtime.py
  tests/
  scripts/audit_pairs.py
  scripts/run_mechanism_grid.py
  scripts/build_teacher_cache.py  # 已实现：准入后生成可审计 endpoint 标签缓存
  scripts/train_student.py
  scripts/evaluate_frozen.py
```

这是拟实现的目录，不是声称已经存在的代码。

关键测试至少包括：`test_identity_edit`、`test_sequence_condition_enabled`、`test_shared_noise`、`test_rigid_equivariance`、`test_residue_mapping`、`test_no_leakage`、`test_batch_consistency`、`test_runtime_accounting`。

## 11. 算力与复现

按单张 24GB RTX 4090 的保守预算规划，但尚未在该设备实测。教师和 ESMFold 按需顺序加载，先小长度、batch 1；学生训练使用冻结序列表征或父本缓存。缓存参数更新前后的边界必须一致，不能使用过期编码。

先实测 50–100 个完整训练/推理调用，再按实际吞吐估算 GPU 小时。分别记录序列编码、教师标签生成、学生训练和评价成本，不给出未经运行的加速倍数。

固定代码 commit、checkpoint checksum、依赖环境、dataset manifest 和随机种子。FoldFlow 仓库声明 CC BY-NC 4.0；复用代码、权重和衍生发布前应分别核对对应授权。[4]

本次核查到的 FoldFlow 主分支 commit：`9d2c260813da3c9a2bc944973953f63ea6d71203`。部署时用实际验证可运行的版本冻结，不盲目跟随主分支。

## 12. 论文与展示产物

图1：输入父本、指定突变、差分场与单次结构更新；明确不从随机噪声重新折叠。

图2：局部突变体误差—远端实验误差/漂移—运行成本的折中；必须包含复制父本。

图3：一个父本、多个候选的总成本曲线，区分冷启动和缓存后成本。

图4：条件差分、共享噪声、双时间窗、几何表示、缓存学生的消融。

图5：真实局部响应、真实远端响应和失败案例，不只挑对齐漂亮的例子。

若实验支持，可使用以下结论表述：

> Given a reference protein structure, our method predicts mutation-conditioned structural updates with reduced spurious scaffold drift and lower inference cost, while preserving or improving mutant-structure accuracy.

只有完整的单次前向学生通过验证后才使用 one-pass；单次更新但多条件查询时用 single-update。不要声称普适优于 ESMFold 的从头折叠、首次突变体结构预测、正确热力学系综或更高生物活性。

## 参考资料与核查入口

[1] Lu et al. ChordEdit: One-Step Low-Energy Transport for Image Editing. arXiv:2602.19083v2. 原论文讨论共享噪声条件差分、时间窗控制场和可选 refinement。

[2] Mahmud, Morehead, Cheng. Accurate prediction of protein tertiary structural changes induced by single-site mutations with equivariant graph neural networks. bioRxiv:10.1101/2023.10.03.560758. 官方仓库 `jianlin-cheng/PreMut`；数据 DOI `10.5281/zenodo.8401256`。

[3] Huguet et al. Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Backbone Generation. arXiv:2405.20313v2 / NeurIPS 2024。

[4] 官方仓库 `DreamFold/FoldFlow`：README、`foldflow/models/ff2flow/flow_model.py`、`runner/config/model/ff2.yaml`、release 0.2.0。

[5] 官方仓库 `facebookresearch/esm`：`esm/esmfold/v1/trunk.py`，显式 no_recycles 会加一以包含普通前向。

[6] McPartlon and Xu. An end-to-end deep learning method for protein side-chain packing and inverse folding. PNAS, 2023. DOI `10.1073/pnas.2216438120`。

[7] Li et al. Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights. arXiv:2606.14042。作为等预算噪声级/时窗消融的补充参考，不将其结论直接当作蛋白质实验事实。
