# Semantic richness (label-agnostic)

## Default (structured vs permuted control)

Compares only **`structured.json`** and **`randomized.json`** (same topology; permuted breaks label–structure alignment). Per-axis max normalization and the radar plot use **these two versions only**.

**SchemaRichness**：对粗粒度「节点类型数 + 边类型数」取对数，刻画类型表规模；**仅置换 subtype** 时通常不变。**HierarchyDepth**：是否出现节点/边的 `subtype` 字段，取 1/2/3；structured 与置换对照若均有节点与边 subtype，则二者相同。二者仍写入 `metrics_raw.json`；当所有已加载版本上这两项**完全一致**时，不再写入归一化表与雷达图，仅保留 MI、ConnSigDiv、边可预测性、三元稀疏性四条**结构敏感**轴。使用 `--versions raw,structured,randomized` 且 raw 与后两者在深度/模式上不同时，会恢复六轴雷达。

```bash
python evaluate_semantic_richness_label_agnostic.py --dir . --out out
```

Requires: `structured.json`, `randomized.json` under `--dir`.

## Optional three-way (include raw)

```bash
python evaluate_semantic_richness_label_agnostic.py --dir . --out out \
  --versions raw,structured,randomized
```

Requires additionally: `raw.json`.
