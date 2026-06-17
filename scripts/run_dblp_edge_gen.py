#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DBLP 固定节点、边去噪生成：使用训练好的模型，在 DBLP 节点上从噪声边初始化，去噪生成边。
用法:
  python scripts/run_dblp_edge_gen.py <checkpoint路径> [--out generated_edges.jsonl]
  或: CKPT=path/to/last.ckpt python scripts/run_dblp_edge_gen.py
"""
import argparse
import json
import os
import os.path as osp
import sys

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="DBLP 固定节点边去噪生成")
    parser.add_argument("ckpt", nargs="?", default=os.environ.get("CKPT"), help="Checkpoint 路径")
    parser.add_argument("--out", default="generated_dblp_edges.jsonl", help="输出边列表路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    if not args.ckpt or not osp.isfile(args.ckpt):
        print("用法: python scripts/run_dblp_edge_gen.py <checkpoint路径> [--out generated_edges.jsonl]")
        print("  或: CKPT=path/to/last.ckpt python scripts/run_dblp_edge_gen.py")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(42)

    # 使用 dblp_train 配置（configs 相对于项目根）
    proj_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    with hydra.initialize(config_path=osp.join(proj_root, "configs"), version_base="1.3"):
        cfg = hydra.compose(config_name="config", overrides=["+experiment=dblp_train"])
    OmegaConf.resolve(cfg)

    from sparse_diffusion.datasets.dblp_single_dataset import DBLPSingleDataModule, DBLPSingleInfos
    from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion
    from sparse_diffusion.diffusion.extra_features import DummyExtraFeatures
    from sparse_diffusion import utils

    datamodule = DBLPSingleDataModule(cfg)
    dataset_infos = DBLPSingleInfos(datamodule)
    domain_features = DummyExtraFeatures()
    train_metrics = __import__("sparse_diffusion.metrics.train_metrics", fromlist=["TrainAbstractMetricsDiscrete"]).TrainAbstractMetricsDiscrete()
    val_sampling_metrics = __import__("sparse_diffusion.metrics.sampling_metrics", fromlist=["SamplingMetrics"]).SamplingMetrics(cfg, dataset_infos)
    test_sampling_metrics = __import__("sparse_diffusion.metrics.sampling_metrics", fromlist=["SamplingMetrics"]).SamplingMetrics(cfg, dataset_infos)

    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        extra_features=DummyExtraFeatures(),
        domain_features=domain_features,
        val_sampling_metrics=val_sampling_metrics,
        test_sampling_metrics=test_sampling_metrics,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model = model.cuda()

    # 加载 DBLP 训练图
    train_data = datamodule.train_dataset[0]
    fixed_data = train_data.clone()
    fixed_data.x = dataset_infos.to_one_hot(fixed_data).x
    fixed_data = fixed_data.cuda()

    print(f"固定节点数: {fixed_data.x.shape[0]}, 去噪生成边...")
    generated = model.sample_batch_fixed_nodes(
        fixed_data,
        keep_chain=0,
        number_chain_steps=10,
        save_final=1,
    )
    edge_index = generated.edge_index
    edge_attr = generated.edge_attr
    mask = edge_attr != 0
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]

    # 加载原始节点 id 映射（用于写出可读的边）
    nodes_path = osp.join(datamodule.inner.raw_dir, "nodes.jsonl")
    id2idx = {}
    idx2id = {}
    if osp.isfile(nodes_path):
        with open(nodes_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                n = json.loads(line)
                nid = str(n["id"])
                id2idx[nid] = i
                idx2id[i] = nid

    edge_family_offsets = getattr(dataset_infos, "edge_family_offsets", {}) or {}
    fam_sorted = sorted(edge_family_offsets.items(), key=lambda x: x[1])

    def attr_to_fam(attr_val):
        a = int(attr_val)
        if a <= 0:
            return "no_edge"
        for i, (fam, off) in enumerate(fam_sorted):
            next_off = fam_sorted[i + 1][1] if i + 1 < len(fam_sorted) else 999
            if off <= a < next_off:
                return fam
        return "unknown"

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for k in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, k].item())
            dst_idx = int(edge_index[1, k].item())
            etype = attr_to_fam(edge_attr[k].item())
            src_id = idx2id.get(src_idx, str(src_idx))
            dst_id = idx2id.get(dst_idx, str(dst_idx))
            f.write(json.dumps({"src": src_id, "dst": dst_id, "type": etype}) + "\n")

    print(f"生成边数: {edge_index.shape[1]} (过滤 no-edge 后), 已保存到 {out_path}")


if __name__ == "__main__":
    main()
