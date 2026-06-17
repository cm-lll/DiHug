# 额外维度说明（Extra Features）

`extra_features` 在 `compute_input_dims` 时对 **节点 X**、**边 E**、**图级 y** 追加的维度如下。配置来自 `configs/model/discrete.yaml`（如 `num_eigenvectors=8`, `num_eigenvalues=5`, `num_degree=10`, `edge_features: 'all'` 等）。

---

## 一、节点额外维度（加到 input_dims.X）

| 来源 | 同质图 | 异质图 | 说明 |
|------|--------|--------|------|
| **AdjacencyFeatures (kcyclesx)** | 3 | 0 | 同质：每点 k3/k4/k5 环计数（归一化）；异质：关掉且不占维 |
| **PositionalEncoding** | 30 | 30 | `positional_encoding: True` 时：sin/cos 位置编码，默认 D=30 |
| **EigenFeatures / HeterogeneousGraphFeatures** | 8+1=9 | 8 | 同质：特征向量 (num_eigenvectors+1)，其中 +1 为 not_lcc；异质：关系类型度特征，num_eigenvectors 维 |

**默认（positional_encoding=False, eigenfeatures=True）**  
- 同质：X 额外 **3 + 9 = 12** 维  
- 异质：X 额外 **0 + 8 = 8** 维  

---

## 二、边额外维度（加到 input_dims.E）

| 来源 | 同质图 | 异质图 | 说明 |
|------|--------|--------|------|
| **AdjacencyFeatures (edge_feats)** | 见下 | 0 | 异质：关掉 path/localngbs，不占维 |

同质图下由 `edge_features` 配置决定：

- `'all'`：path_features (max_degree=10) + local_neighbors (1) → **11 维**
- `'dist'`：仅 path_features → **10 维**
- `'localngbs'`：仅 local_neighbors (Adamic–Adar) → **1 维**
- `null`/其他：**0 维**

**默认（edge_features: 'all'）**  
- 同质：E 额外 **11** 维  
- 异质：E 额外 **0** 维  

---

## 三、图级额外维度（加到 input_dims.y）

| 来源 | 同质图 | 异质图 | 说明 |
|------|--------|--------|------|
| **AdjacencyFeatures (y_feat)** | 4 + 1 + (num_degree+2) + dx + de | 1 + (num_degree+2) + dx + de | 见下 |
| **n (节点比例)** | 1 | 1 | 有效节点数 / max_n_nodes |
| **EigenFeatures eval_feat** | 1 + num_eigenvalues | 1 + num_eigenvalues | 连通分量数 + 特征值相关统计 |

**AdjacencyFeatures 的 y 组成（dist_feat=True 时）**  
- kcyclesy_part：同质 4 维（k3,k4,k5,k6 图级）；异质 0 维  
- degree_dist：num_degree+2 维（度分布，默认 12）  
- node_dist：dx 维（节点类型/子类型分布）  
- edge_dist：de 维（边类型分布）  

异质图时，若启用 `hetero_global_features`，会用 **HeterogeneousGlobalFeatures** 替代上述 AdjacencyFeatures 的 y：节点类型分布 + 各 (src_type, dst_type) 对的关系族密度等，维度由 `type_names` 与 `family_ranges` 决定。

---

## 四、汇总（默认配置）

- **同质图**  
  - X：+12（3 cycle + 9 eigen）  
  - E：+11（path 10 + localngbs 1）  
  - y：由 kcyclesy(4) + n(1) + degree_dist(12) + node_dist(dx) + edge_dist(de) + eval(6) 等组成  

- **异质图**  
  - X：+8（仅 eigen 关系类型度特征，无 cycle、无占位零维）  
  - E：+0  
  - y：由 hetero_global（或 degree/node/edge_dist）+ n(1) + eval(6) 等组成  

各维具体含义见 `sparse_diffusion/diffusion/extra_features.py`（AdjacencyFeatures、PositionalEncoding、EigenFeatures、HeterogeneousGraphFeatures、HeterogeneousGlobalFeatures）。
