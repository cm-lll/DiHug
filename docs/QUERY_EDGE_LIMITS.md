# Query 边数上限：配置与代码位置

需要调整「query 边数上限」时，可对照下表改配置或代码。

## 1. 配置项汇总

| 配置项 | 作用域 | 默认 | 说明 |
|--------|--------|------|------|
| **general.val_max_query_edges_per_fam** | 仅验证 | `null` | 异质图验证时**每个关系族**每图最多采样的 query 边数（用于 NLL）。`null` = 不限制，用全量可能边。 |
| **general.val_apply_max_query_edges_per_batch** | 仅验证 | `false` | 是否在验证时用 `model.max_query_edges_per_batch` 对**总 query 边数**截断。`false` = 验证不截断。 |
| **model.max_query_edges_per_batch** | 训练 +（可选）验证 | `0`（不限制） | 训练时：单步 query 边数超过此值则随机子采样到该值，避免 OOM。为 0 表示不限制。若 `val_apply_max_query_edges_per_batch=true` 则验证也会用此值截断总 query。 |

## 2. 代码位置

- **训练**：`sparse_diffusion/diffusion_model_sparse.py`  
  - 约 **82–83 行**：从 `cfg.model` 读取 `max_query_edges_per_batch`。  
  - 约 **587–591 行**：若 `max_query_edges_per_batch > 0` 且当前 query 边数超过该值，则随机子采样到该上限。

- **验证（异质图按族）**：`sparse_diffusion/diffusion_model_sparse.py`  
  - 约 **1542–1544 行**：`val_max_query_edges_per_fam`：每族 `num_query_edges_fam = min(num_fam_possible_edges, val_max)`，仅当 `val_max` 非空且 > 0 时生效。  
  - 约 **1637–1645 行**：在合并各族 query 后，若 `val_apply_max_query_edges_per_batch=True` 且 `max_query_edges_per_batch > 0`，则对总 query 边做随机子采样到该上限。

## 3. 单图 DBLP 推荐

- 训练：`model.max_query_edges_per_batch: 0` 表示不限制；若显存不足再设如 25000。  
- 验证：不加上限——`val_max_query_edges_per_fam` 不设（或 `null`），`val_apply_max_query_edges_per_batch: false`（默认），验证用全量 query。

## 4. 验证与测试一致（val_sampling_style）

参考 [SparseDiff](https://github.com/qym7/SparseDiff) 的测试流程：**验证 = 多步去噪、每步分批预测 + 解析后验，得到生成图后再算指标**。

- **general.val_sampling_style**（默认 `false`）：设为 `true` 且为异质图时，`validation_step` 会调用与测试相同的采样流程（`sample_batch_fixed_nodes`：固定节点、多步边去噪、每步内多块 query + 解析后验），得到一张生成图，再计算生成图与真实图之间的**结构损失标量**（relation_matrix / metapath2 / metapath3 / subtype_degree 加权和），作为本步的 val loss 并写入 `val/epoch_NLL`（供 checkpoint 监控）。
- **代码位置**：`sparse_diffusion/diffusion_model_sparse.py` 的 `validation_step` 开头，当 `val_sampling_style and self.heterogeneous` 时进入该分支。
- **说明**：每 val batch 跑一整条采样链，耗时与一次 test 采样相当；单图 DBLP 一般仅 1 个 val batch，故每 epoch 验证约等于跑一次完整采样并回报结构损失。
