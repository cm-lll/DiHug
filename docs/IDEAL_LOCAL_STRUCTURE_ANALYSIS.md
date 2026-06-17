# Ideal 机制与局部结构（三角形/聚类）问题分析

## 设计前提：局部训练 + 全图结构指标

- **训练是局部的**：每步只对当前 **query 边**（一块子集）做预测，其余边保持当前噪声状态；不在一帧内「整图」预测。
- **采样是多轮拼出全图**：通过多轮生成（每轮更新一块 query），逐步把全图生成出来。
- **Ideal 的定位**：在「只允许改当前 query、其余为当前噪声图」的约束下，**局部最优**的目标——即通过**只调整这一块 query**，使得 **当前步意义下的全图**（噪声图 + query 上的赋值）在**规定的全局结构指标**（三角形数量、聚类、度分布等）上尽量接近真实图 A。  
  因此 ideal 图 = 在当前噪声图基础上，仅改 query 边后、在全局结构损失下最优的「理想可达全图」；训练目标 CE(pred, ideal) 是让**局部预测**逼近这一**局部优化得到的最优**，从而多轮采样后全图在结构上趋近真实图。

**注意**：这里说的「全图」指「当前步的完整图」（当前噪声 + query 部分），即结构损失是在这一「全图」上算的，而不是「一次性训练整张图」；优化变量只有 query 边，是局部优化。

---

## 问题现象

- 已添加 ideal 相关配置，但**三角形数量、聚类系数等稍高阶局部结构**无明显改善。
- 模型在**度分布**上表现好，**局部结构**较差。

## 根因分析

### 1. Ideal 分支未开启（最主要原因）

**结论：当前配置下 ideal 机制实际上没有参与训练。**

- 在 `diffusion_model_sparse.py` 中，计算 `ideal_full` 并用其作为 `structure_target` 的**整段逻辑**都在 `if structure_only_global:` 分支内（约 709–1210 行）。
- 配置里 `structure_only_global: false`（见 `configs/model/discrete.yaml` 第 41 行），因此：
  - 不会进入全图分支；
  - 不会计算 `ideal_full`；
  - 训练目标始终是 **CE(pred, true)**（直接对真实边标签），没有「以当前 t 下结构最优的 ideal 为目标」。
- 因此模型只在对齐「真实边 0/1」，度分布可以通过边数/存在性预测间接对齐，但**没有显式以三角形、聚类等结构为目标的监督**，局部结构难以改善。

**修复：** 将 `structure_only_global` 设为 **`true`**，才能启用 ideal 机制。

---

### 2. 大图时三角形/聚类项被跳过

- `compute_structure_loss_scalar` 内部通过 `_compute_structure_stats` 得到 `A_true_node` / `A_pred_node`，再在 `_structure_losses_graph_metrics` 中计算 **Clustering** 和 **Triangles**。
- 在 `train_metrics.py` 第 410 行附近：

  ```python
  if n_node <= getattr(self, "structure_loss_max_nodes", 3000) and n_node > 0:
      # 才计算 A_true_node, A_pred_node
  ```

- 当 **节点数 > structure_loss_max_nodes（默认 3000）** 时，`A_true_node` / `A_pred_node` 为 `None`，`triangles_loss` 和 `clustering_loss` 直接为 0（见 532–556 行）。
- 因此：
  - 若训练/采样的图节点数 > 3000，则 **ideal 优化时根本没有三角形和聚类项**，只有 degree_mmd 和 edge_types_tv；
  - 即使开启了 ideal，局部结构（三角形、聚类）也收不到梯度，自然不会改善。

**修复建议：**

- 若必须支持大图：可考虑在结构损失里对节点做**子图采样**（例如每步随机取 3000 节点诱导子图）再算 A_node / 三角形 / 聚类，使局部结构项在 ideal 中仍能参与（需改代码）。
- 若图不大：适当提高 `structure_loss_max_nodes`，确保你的图节点数不超过该值，从而三角形/聚类项被计算。

---

### 3. Ideal 优化中局部结构梯度相对较弱

- 当 ideal 分支真正开启且 `structure_loss_type: 'graph_metrics'` 时，ideal 的构造是：对 query 上的 soft 做若干步梯度下降，最小化  
  `L = degree_mmd + clustering + triangles + edge_types_tv`。
- 可能的问题：
  - **度分布**是全局直方图 L1，梯度覆盖所有边，且归一化后量级稳定；
  - **三角形/聚类**对单条边的梯度是局部的（只影响与该边相关的三角形），且 `structure_triangles_normalize: true` 时做了相对误差归一化并 `clamp(max=10)`，单边梯度可能偏小；
  - **fix_k=true** 时，ideal 只在「固定 query 边数 k」下重分配哪 k 条边为 1，若 `structure_ideal_steps` 或 `structure_ideal_lr` 偏小，ideal 可能仍很接近 noisy，对局部结构的引导不够强。

**修复建议：**

- 在启用 ideal 且希望强化局部结构时，可适当：
  - 提高 **`triangles_loss_weight`**、**`clustering_loss_weight`**（相对 `degree_mmd_loss_weight`）；
  - 提高 **`structure_ideal_steps`**（如 8–10）或略提高 **`structure_ideal_lr`**，使 ideal 更充分向「高三角形/高聚类」方向移动。

---

### 4. 训练目标只对 query 做 CE(pred, ideal)，不直接约束全图结构

- 当前训练损失是：在 query 边上 **CE(pred, ideal)**，没有直接对「生成全图」的三角形数/聚类系数做 loss。
- 因此局部结构的改善完全依赖：
  1. ideal 本身是否在优化时被三角形/聚类项推动（见上：需开启 ideal + 大图时不被 max_nodes 截断）；
  2. 模型学到的 pred 是否逼近该 ideal。
- 若 1）中三角形/聚类在 ideal 里被跳过或权重过小，则 ideal 更偏向度/边类型，局部结构就不会有明显提升。

---

## 建议操作清单

| 项目 | 说明 |
|------|------|
| **必须** | 将 **`structure_only_global`** 设为 **`true`**，否则 ideal 完全不生效。 |
| **必须** | 若图节点数 > 3000，要么增大 **`structure_loss_max_nodes`**（注意显存），要么实现子图采样再算 A_node/三角形/聚类。 |
| **建议** | 启用 ideal 后，适当增大 **`triangles_loss_weight`**、**`clustering_loss_weight`**，并尝试增加 **`structure_ideal_steps`** 或 **`structure_ideal_lr`**。 |
| **原图三角形多、生成很少时** | 开启 **`triangle_aux_loss_weight`**（如 0.5~1.0）：在 CE(pred, ideal) 之外再加一项「预测全图 vs 真实图」的三角形+聚类损失，让模型直接收到「多预测能形成三角形的边」的梯度（见下节）。 |
| **可选** | 用 **`log_predict_behavior: true`** 观察 `n_ideal_exist`、`n_pred_exist` 等，确认 ideal 与预测行为是否符合预期。 |

---

## 三角形辅助损失（triangle_aux_loss_weight）

当 **ideal 已开**但生成图三角形仍远少于原图（如原图 300+、生成 40+）时，可启用 **三角形辅助损失**：

- **含义**：在原有损失 CE(pred, ideal) 之外，对 **预测全图**（noisy + query 处用模型预测）与 **真实图** 再算一次「三角形 + 聚类」结构损失，并乘以 **`triangle_aux_loss_weight`** 加到总损失里。
- **作用**：模型除学习「逼近 ideal」外，还会直接收到「预测图三角形/聚类不足」的梯度，更倾向于预测能形成三角形/聚类的边。
- **配置**：在 `configs/model/discrete.yaml` 中设置 `triangle_aux_loss_weight: 0.5` 或 `1.0`（0 表示关闭）。建议 ACM 等子图先试 0.5，若仍不足再试 1.0。

---

## 配置修改示例（仅作参考）

在 `configs/model/discrete.yaml`（或对应 experiment 的 model 覆盖）中：

```yaml
# 启用 ideal 全图分支（必须）
structure_only_global: true

# 若图节点数 > 3000，按需提高或实现子图采样
structure_loss_max_nodes: 3000   # 或更大，注意 OOM

# 强化局部结构在 ideal 优化中的权重
triangles_loss_weight: 3.0       # 原 2.0
clustering_loss_weight: 1.5      # 原 1.0
structure_ideal_steps: 8        # 原 5
# structure_ideal_lr: 0.15      # 可选：略提高
```

修改后需重新训练；采样时无需改配置，ideal 仅影响训练目标。
