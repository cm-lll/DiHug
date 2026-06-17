# MP 边组成与采样内层循环实现说明

## 「全部可能边」与训练/采样一致

- **概念**：训练和采样都遵循「全部可能边」：可能边 = 按图/关系族定义的所有 (u,v) 或 (src_type,dst_type) 组合。
- **训练**：每次只对全部可能边中的**一块**做预测。每步随机取一块（比例 k = edge_fraction，如 2%），comp = 加噪显式边 + 该块 query，loss 仅在该块上计算。
- **采样**：同一扩散步内做 1/k 次（如 50 次），每轮一块，块之间不重叠；若开启内层更新，每轮显式边用「上轮显式边 + 本轮 query」预测后更新。同质图分支对 num_edges 全量再切块（50 轮覆盖全部可能边）；异质图分支每族取 k*num_fam_possible_edges 条再切块（与训练同口径）。

## 训练：MP 边的组成

与你的理解一致，代码实现是：

1. **显式噪声边**：来自扩散加噪得到的当前时刻图 `data.edge_index` / `edge_attr`（加噪过程带来）。
2. **抽样 query 边**：按 `edge_fraction`（如 2%）从可能边中抽样，作为「本轮要预测」的边；可以是隐式边（与已有噪声边可重合）。
3. **合并**：`get_computational_graph(triu_query_edge_index, clean_edge_index=噪声图边, clean_edge_attr=...)` → `comp_edge_index = hstack([clean, query])` 再 coalesce，得到 MP 用的边集。

**补充真实边（已按结构模式自动关闭）**：  
训练时有一项「在 query 里补充一定比例真实边」，用来保证边 CE 可算、避免随机 query 几乎无正边。当 **`structure_only` 或 `structure_only_global` 为 true** 时，代码已自动不再补充真实边（两处判断均增加 `and not structure_only and not structure_only_global`）。若未用结构模式仍想关闭，可设 `model.train_query_supplement_real_edges: false`。

---

## 采样：1/k 次循环与「噪声图每轮更新」的正确语义

- **「全部」指所有可能边**：不是「某次随机抽出的 query 边」打乱再切块，而是**所有可能边**（例如 n 个节点的图有 n(n-1)/2 或按关系族划分的可能边）被分成 1/k 块，每轮取其中一块作为 query，k 轮后覆盖全部可能边。
- **循环次数**：`len_loop = ceil(1.0 / self.edge_fraction)`，例如 `edge_fraction=0.02` 时为 50 次；50 块不重叠，合起来 = 全部可能边。
- **每轮逻辑（举例）**：始终只维护「显式边」；初始显式边 = 加噪后的显式边（如 0–19），其余为隐式（空边）。第 1 轮：用「上轮显式边 ∪ 本轮 query（如 10–29）」做 comp、在 query 上预测；本块 query 位置按预测更新、其余显式边保留 → 得到新一轮的显式边。下一轮再用「上轮更新后的显式边 ∪ 本轮 query」重复，直到 50 块覆盖全部可能边。
- **实现要点**：每轮做的都是「上轮更新过的显式边 + 本轮 query 边」做 MP 和预测，再用预测结果更新显式边（本块 query 由预测决定显式/隐式，非 query 位置保留上轮显式边）。代码在开启 `autoregressive` 或 `sampling_update_graph_per_inner_loop` 时，每轮末把当前显式边中属于本块 query 的去掉、并上本块 query 里预测为显式的边，得到下一轮的显式边 `new_edge_index` / `new_edge_attr`，下一轮用其作为 `sparse_noisy_data["edge_index_t"]`。
- **异质图**：全部可能边按关系族构造后，`randperm` 打乱再按 `num_query_edges_per_loop` 切块，50 块覆盖全部；同质图用 condensed 索引按 `num_edges_per_loop * i` 切块，同样覆盖全部可能边。

---

## 建议配置

1. **训练**：不关注边类别时关闭「补充真实边」  
   - 已实现：当 `structure_only` 或 `structure_only_global` 为 true 时，自动不补充真实边；无需改 config。
2. **采样**：若要「每轮用上轮更新后的显式边 + 本轮 query 做预测并更新显式边」  
   - 将 `model.autoregressive` 设为 `true`；或  
   - 将 `model.sampling_update_graph_per_inner_loop` 设为 `true`（内层循环每轮按预测更新显式边，与 `autoregressive` 解耦）。
