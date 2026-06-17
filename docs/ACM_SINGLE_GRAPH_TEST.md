# 用 ACM 子图测试单图性能

你当前的单图流程（dblp_single：一张图 train/val/test）和 ACM 子图流程（多张图、train/val/test 各一堆图）**用的是同一套异质图 + edge-only 模型**。要“拿 ACM 子图测试单图性能”，有两种做法。

---

## 方式一：直接跑现有 ACM 实验（推荐）

ACM 子图本身是**多图**：每个 batch 里 1～2 张图，每张图独立做扩散/query，和“单图”在模型和训练逻辑上一致，只是数据从“一张大图”变成“多张小图”。

**步骤：**

1. **确认数据**  
   配置里 ACM 的 `datadir` 是 `data/ACM_subgraphs`（见 `configs/dataset/acm_subgraphs.yaml`）。  
   保证该目录下已有处理好的 ACM 子图（含 train/val/test 的 processed 等），和之前跑 ACM 时用的结构一致。

2. **训练 + 验证**  
   ```bash
   ./run_acm_train.sh
   ```
   或指定单卡：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python sparse_diffusion/main.py +experiment=acm_train
   ```

3. **实验配置**  
   `configs/experiment/acm_train.yaml` 里已经：
   - `override /dataset: acm_subgraphs`
   - `batch_size: 2`（每步 2 张 ACM 子图）
   - `edge_only_model: true`、`use_edge_subtype_ce: true`、结构损失等，和单图 DBLP 那套一致  

   若你想更接近“每步一张图”（和 DBLP 单图一样），可在该 yaml 里把 `train.batch_size` 改为 `1`。

这样得到的就是**在 ACM 子图上的单图式性能**（同一模型、同一训练/验证方式，只是数据换成 ACM）。

---

## 方式二：只取一张 ACM 图做 train/val/test（已实现）

已支持：在配置里打开单图模式后，train/val/test 都只用**同一张** ACM 子图。

**1. 用现成实验配置（推荐）**

```bash
python sparse_diffusion/main.py +experiment=acm_single_train
```

该实验会：
- 使用 `dataset: acm_subgraphs`，并打开 `dataset.single_graph: true`
- 从 **train** split 取 **第 0 张**图，作为 train/val/test 的唯一样本
- `train.batch_size: 1`，每步就是这一张图

**2. 自定义“用哪一张图”**

在命令行覆盖或改 `configs/dataset/acm_subgraphs.yaml`：

- `dataset.single_graph: true` — 开启单图
- `dataset.single_graph_split: 'train'` — 从哪个 split 取图：`train` / `val` / `test`
- `dataset.single_graph_index: 0` — 该 split 中第几张图（从 0 开始）

例如用 val 的第 2 张图：

```bash
python sparse_diffusion/main.py +experiment=acm_single_train dataset.single_graph_split=val dataset.single_graph_index=2
```

**3. 单图模式下的日志**

启动时会打印类似：

`[ACM] single_graph=True: using train[0] for train/val/test (1 graph, 522 nodes, 882 edges)`

---

## 小结

- **多图 ACM**：`./run_acm_train.sh` 或 `+experiment=acm_train`，可选 `train.batch_size=1`。
- **单图 ACM（一张图 train/val/test）**：`+experiment=acm_single_train`，或保持 `acm_subgraphs` 并设 `dataset.single_graph=true`、`single_graph_split`、`single_graph_index`。
