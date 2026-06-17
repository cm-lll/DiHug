# 为什么采样会 OOM：2% 只限制「预测边」，不限制「计算图」

## 你的直觉（训练阶段）

- **2%**：每次只 **query** 2% 的可能边（`edge_fraction`），算 loss 只在这 2% 上。
- **全局指标**：关系矩阵 / metapath / 度分布等可以「已有全局统计 + 本步 query 的局部增量」来更新，不需要每步全图重算。

这两点在 **训练** 里都是对的：训练时我们确实只对 2% 的边做预测和梯度，结构损失也可以按增量方式设计。

## OOM 出在哪里：采样的 model.forward

OOM 发生在 **采样** 的 `sample_p_zs_given_zt` → `forward` → `heterogeneous_transconv_layer.message`，也就是 **GNN 在计算图上的前向传播**，和「是否用 2% 做 loss」、是否用增量全局指标 **无关**。

## 2% 在训练 vs 采样中的不同含义

### 训练时

1. 只 **采样 2% 的边** 作为 query（再配合 `max_query_edges_per_batch` 截断）。
2. 构建 **计算图**：`comp = get_computational_graph(query_edges, clean_edge_index=当前噪声图的边, ...)`。
3. 这里 **clean** = 当前 batch 的 **噪声图边**（来自真实图加噪），对单图 DBLP 来说边数大致是「真实边数」量级，相对可控。
4. 模型只在 **comp** 上做一次 forward，在 **query 边** 上输出预测并算 loss；结构指标可以只基于这 2% 的预测做增量更新。

所以训练时：**参与计算的边** ≈ 当前噪声图边 + 2% query，且 query 还有上限，显存相对可控。

### 采样时

1. **每一步** 的当前状态是 `sparse_noisy_data["edge_index_t"]`，也就是 **当前扩散步的整张图**（所有在「当前时刻」存在的边）。
2. 虽然也是按 2% 一批边来 **预测**（内层循环 `len_loop = ceil(1/edge_fraction)`，每轮只预测本轮的 2% 边），但构建计算图时：
   ```text
   comp = get_computational_graph(
       triu_query_edge_index=本轮的 2% query,
       clean_edge_index=sparse_noisy_data["edge_index_t"],  # 当前步的「全图边」
       ...
   )
   ```
   即 **comp = 当前步全图边(clean) + 本轮的 query 边**，合并去重后，**comp 的边数主要由「当前步全图」决定**。
3. 因此 **GNN 的输入** 是「当前扩散状态下的整图」，不是「只有 2% 的边」。2% 只决定了 **每一步里我们给模型多少条边做预测并采样**，**不** 决定 **模型看到多少条边做消息传递**。
4. 随着扩散进行，中间某几步的边数会很多（例如 11k 节点、几十万边），再乘上 batch_size（例如 2）、乘上 GNN 的隐藏维和层数，`message` 里类似 `Y * (edge_attr_mul.unsqueeze(-1) + 1) + ...` 的中间张量就会很大，触发 OOM。

所以：**采样时 2% 只限制了「被预测/被采样的边」**，**没有限制「参与 message passing 的边」**；参与 message passing 的一直是 **当前步的整张图**。

## 小结

| 项目           | 训练                         | 采样（导致 OOM 的部分）           |
|----------------|------------------------------|------------------------------------|
| 2% 限制的是    | 参与 loss 的 query 边        | 每轮被预测的 query 边              |
| 模型 forward 看到的图 | comp = 噪声图边 + query（边数相对可控） | comp = **当前步全图边** + query（边数可很大） |
| 全局/结构指标  | 可用「已有全局 + 本步 query 局部」增量 | 不参与 OOM（OOM 在 GNN forward）   |

因此会出现「训练时只有 2%、结构也是增量更新，却仍然在采样阶段 OOM」：**显存主要耗在采样时每一步在「整张当前图」上做 GNN forward**，而不是耗在 2% 或全局指标计算上。

---

## 之前针对 OOM 的代码调整

1. **大图时测试采样 batch 改为 1**（`diffusion_model_sparse.py` 的 `on_test_epoch_end`）  
   - 若 `cond_edge_gen_sample_nodes=True` 且训练图节点数 > 5000（如 DBLP ~11k），将 `to_generate` 限制为 1，即每次只对一张图调用 `sample_batch`，降低单次扩散的显存峰值。

2. **每批采样后清空 CUDA 缓存**（同上）  
   - 每执行完一次 `sample_batch` 并 `samples.append(sampled_batch)` 后，若 `self.device.type == "cuda"` 则执行 `torch.cuda.empty_cache()`，减轻显存碎片。

3. **采样每步打印显式边数 / 族可能边数**（`sample_p_zs_given_zt` 内）  
   - 每扩散步开始时打印：`t_norm`、`当前显式边数`、`族/可能边数(本步query总)`，便于观察是否出现显式边数异常增大。

---

## 采样日志一行含义与计算方式

每步会打印类似：

```text
[采样] t_norm=0.9900 当前显式边数=137979 族/可能边数(本步query总)=493140
```

### 各项含义与计算

| 字段 | 含义 | 计算方式 |
|------|------|----------|
| **t_norm** | 当前扩散步的归一化时间 | `t_float[0].item()`，即本步的 \(t/T\)（0~1）。t_norm=1.0 表示起点（纯噪声），越小表示越接近去噪结束。 |
| **当前显式边数** | 本步**开始时**图中已有的边数 | `edge_index.shape[1]`，即 `sparse_sampled_data` 在本步入口的边数。上一步各块 query 按「预测为存在则保留」merge 后得到，会随模型预测增减。 |
| **族/可能边数(本步query总)** | 本步**所有内层循环**要预测的 query 边总数 | 异质图：各关系族「本步 query 边」之和，即 `all_query_edge_index.shape[1]`（约等于 各族 `ceil(edge_fraction × 族内可能边数)` 之和 × 块数相关）。同质图：`num_edges.sum().item()`。即本步会在这 493140 条「候选边」上分批做预测并 merge。 |

### 为何「当前显式边数」会远大于初始边数？

- 初始（t_norm=1.0）显式边数 = 噪声图边数（如 36645），与真实图边数一致。
- 每步内有多块 query，每块做「删掉当前图中落在本块 query 的边 → 把本块 query 里预测为存在的边加回」。
- 若模型偏向预测「存在」，每块加的多、删的少，**当前显式边数**会快速上升（如 36645 → 137979），多步后易 OOM。详见 `docs/EDGE_EXPLOSION_ANALYSIS.md`。
