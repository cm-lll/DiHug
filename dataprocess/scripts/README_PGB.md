# PGB (PubMed Graph Benchmark) 异质图流程

## 数据来源

Zenodo: https://zenodo.org/record/6406776  
含 Paper、Author、引用等，约 20GB（10 个分片）。

## 流程

```bash
# 1. 下载（默认只下 part1，约 2GB）
python dataprocess/scripts/download_pgb.py --output-dir data/raw/PGB

# 2. 预处理 -> nodes.jsonl, edges.jsonl
python dataprocess/scripts/preprocess_pubmed_pgb.py \
    --input-dir data/raw/PGB/extracted \
    --output-dir data/PubMed_PGB_processed

# 若数据量大，可限制篇数做快速测试
python dataprocess/scripts/preprocess_pubmed_pgb.py --limit 50000

# 3. 子图提取
python dataprocess/scripts/extract_subgraphs_ego.py \
    data/PubMed_PGB_processed/nodes.jsonl \
    data/PubMed_PGB_processed/edges.jsonl \
    data/PubMed_PGB_subgraphs \
    --dataset pubmed_pgb --max-nodes 500 --n-subgraphs 100

# 4. 训练
cd sparse_diffusion && python main.py +experiment=pubmed_pgb_single_train
```

## 子类别

| 节点 | 子类别 |
|------|--------|
| Paper | Review, Clinical, Article, CaseReport, Study, Other（来自 publication_type） |
| Author | High(4+篇), Medium(2-3篇), Low(1篇) |
