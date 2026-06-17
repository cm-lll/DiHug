# 子图提取脚本说明

## extract_subgraphs_ego.py

基于 **BFS k-hop ego 扩展** 提取连通子图，不直接切割，保证：
- **连通性**：每个子图都是连通的（从种子节点 BFS 扩展得到）
- **节点数控制**：每个子图节点数在 `[min_nodes, max_nodes]` 范围内（默认 ≤500）

### 使用方法

```bash
python dataprocess/scripts/extract_subgraphs_ego.py \
    <nodes.jsonl> <edges.jsonl> <输出目录> \
    --dataset dblp|imdb|pubmed|aminer \
    --min-nodes 50 --max-nodes 500 \
    --n-subgraphs 100 --seed 42
```

### 支持的数据集

| 数据集 | 节点类型 | 边类型 | 说明 |
|--------|----------|--------|------|
| **DBLP** | Author, Paper, Conference, Term | author_paper, paper_conference, paper_term, author_author | 学术网络 |
| **IMDB** | Movie, Actor, Director | movie_actor, movie_director | 电影网络 |
| **PubMed** | Paper | paper_paper | 医学文献（Planetoid，仅引用图） |
| **Aminer** | Paper, Author, Venue | author_paper, paper_venue, paper_paper | 学术引用 |

### 输入格式

- **nodes.jsonl**：每行一个 JSON，需包含 `id`, `type`, `subtype`（或 `label_id`）
- **edges.jsonl**：每行一个 JSON，需包含 `src`, `dst`, `type`

### 输出格式

与 ACM_subgraphs 一致，可直接被 `ACMSubgraphsDataset` 加载：
```
<out_dir>/
  subgraph_000/
    nodes.pt    # 节点张量
    edges.pt    # 边张量
    meta.json   # 元信息
  subgraph_001/
    ...
```

### 示例

```bash
# DBLP（使用 DBLP_four_area_processed 全图）
python dataprocess/scripts/extract_subgraphs_ego.py \
    data/DBLP_four_area_processed/nodes.jsonl \
    data/DBLP_four_area_processed/edges.jsonl \
    data/DBLP_subgraphs_ego \
    --dataset dblp --max-nodes 500 --n-subgraphs 100

# 若已有 IMDB/PubMed/Aminer 的 nodes.jsonl + edges.jsonl
python dataprocess/scripts/extract_subgraphs_ego.py \
    data/IMDB/nodes.jsonl data/IMDB/edges.jsonl \
    data/IMDB_subgraphs --dataset imdb
```

### 逻辑说明

1. **种子选择**：从配置的 `seed_types`（如 Paper、Author）中挑选度数≥2 的节点
2. **BFS 扩展**：从每个种子出发 BFS，直到节点数达到 `max_nodes`
3. **子图导出**：保留扩展到的节点及其中所有边，转为 ACM 格式

这样得到的子图天然连通，无需后处理。
