# -*- coding: utf-8 -*-
"""
单图 DBLP 数据集：从 data/DBLP_four_area_processed/nodes.jsonl 与 edges.jsonl 加载一张异质图。
train/val/test 均返回同一张图（单图训练），用于「边子类别 CE + 结构损失」联合训练。
"""
import json
import os
import os.path as osp
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from hydra.utils import get_original_cwd
from torch_geometric.data import Data, InMemoryDataset

from sparse_diffusion.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from sparse_diffusion.utils import load_pt
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform, Statistics, load_pickle, resolve_data_path, save_pickle
from sparse_diffusion.metrics.metrics_utils import atom_type_counts, edge_counts, node_counts


# DBLP 节点类型顺序（与 preprocess 一致）
NODE_TYPE_NAMES = ["Author", "Paper", "Conference", "Term"]
# 子类别（领域）：与 AREA_LABEL_MAP 一致
SUBTYPE_NAMES = ["Database", "DataMining", "AI", "InformationRetrieval"]
# 边关系族及其端点类型（用于 fam_endpoints）
FAM_ENDPOINTS = {
    "author_paper": {"src_type": "Author", "dst_type": "Paper"},
    "paper_conference": {"src_type": "Paper", "dst_type": "Conference"},
    "paper_term": {"src_type": "Paper", "dst_type": "Term"},
    "author_author": {"src_type": "Author", "dst_type": "Author"},
    "paper_paper": {"src_type": "Paper", "dst_type": "Paper"},
    "conference_conference": {"src_type": "Conference", "dst_type": "Conference"},
    "term_term": {"src_type": "Term", "dst_type": "Term"},
    "generic": {"src_type": "Author", "dst_type": "Paper"},  # fallback
}


def _load_nodes(nodes_path: str) -> Tuple[List[Dict], Dict[str, int], Dict[str, str]]:
    """加载 nodes.jsonl，返回 (nodes_list, id2idx, id2type)。节点按类型顺序排列后按 id 排序。"""
    nodes = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes.append(json.loads(line))
    # 稳定顺序：先按 type（Author, Paper, Conference, Term），再按 id
    type_order = {t: i for i, t in enumerate(NODE_TYPE_NAMES)}
    nodes.sort(key=lambda n: (type_order.get(n["type"], 99), str(n["id"])))
    id2idx = {str(n["id"]): i for i, n in enumerate(nodes)}
    id2type = {str(n["id"]): n["type"] for n in nodes}
    return nodes, id2idx, id2type


def _load_edges(edges_path: str, id2idx: Dict[str, int]) -> List[Tuple[int, int, str]]:
    """加载 edges.jsonl，返回 [(src_idx, dst_idx, edge_type), ...]。"""
    edges = []
    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = str(row["src"])
            dst = str(row["dst"])
            if src not in id2idx or dst not in id2idx:
                continue
            etype = row.get("type", "generic")
            if etype not in FAM_ENDPOINTS:
                etype = "generic"
            edges.append((id2idx[src], id2idx[dst], etype))
    return edges


def _build_vocab_from_data(edges: List[Tuple[int, int, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """从边类型构建 edge_family2id 与 edge_family_offsets。每个 family 一个子类型（id=offset）。"""
    families = sorted(set(e[2] for e in edges))
    families = [f for f in families if f in FAM_ENDPOINTS]
    if not families:
        families = ["author_paper", "paper_conference", "paper_term"]
    edge_family2id = {f: i for i, f in enumerate(families)}
    edge_family_offsets = {}
    cur = 1
    for f in families:
        edge_family_offsets[f] = cur
        cur += 1
    edge_label2id = {f: edge_family_offsets[f] for f in families}
    return edge_label2id, edge_family2id, edge_family_offsets


class DBLPSingleDataset(InMemoryDataset):
    """单图 DBLP：从 nodes.jsonl + edges.jsonl 构建一张 PyG Data，train/val/test 均为同一图。"""

    def __init__(self, split: str, root: str, transform=None, pre_transform=None, heterogeneous: bool = True):
        assert split in {"train", "val", "test"}
        self.split = split
        self.heterogeneous = heterogeneous
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = load_pt(osp.join(self.processed_dir, self.processed_file_names[0]))

        pdir = self.processed_dir
        fnames = self.processed_file_names
        self.statistics = Statistics(
            num_nodes=load_pickle(osp.join(pdir, fnames[1])),
            node_types=np.load(osp.join(pdir, fnames[2]), allow_pickle=False),
            bond_types=np.load(osp.join(pdir, fnames[3]), allow_pickle=False),
            node_subtype_by_type=load_pickle(osp.join(pdir, fnames[5])) if osp.isfile(osp.join(pdir, fnames[5])) else None,
            edge_subtype_by_family=load_pickle(osp.join(pdir, fnames[6])) if osp.isfile(osp.join(pdir, fnames[6])) else None,
            node_type_distribution=load_pickle(osp.join(pdir, fnames[7])) if osp.isfile(osp.join(pdir, fnames[7])) else None,
            edge_family_distribution=load_pickle(osp.join(pdir, fnames[8])) if osp.isfile(osp.join(pdir, fnames[8])) else None,
        )

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return ["nodes.jsonl", "edges.jsonl"]

    @property
    def processed_file_names(self):
        files = [
            f"{self.split}.pt",
            f"{self.split}_n.pickle",
            f"{self.split}_node_types.npy",
            f"{self.split}_bond_types.npy",
            "vocab.json",
            f"{self.split}_node_subtype_by_type.pickle",
            f"{self.split}_edge_subtype_by_family.pickle",
            f"{self.split}_node_type_distribution.pickle",
            f"{self.split}_edge_family_distribution.pickle",
            f"{self.split}_edge_family_avg_counts.pickle",
        ]
        return files

    def download(self):
        pass

    def process(self):
        raw_dir = self.raw_dir
        nodes_path = osp.join(raw_dir, "nodes.jsonl")
        edges_path = osp.join(raw_dir, "edges.jsonl")
        if not osp.isfile(nodes_path) or not osp.isfile(edges_path):
            raise FileNotFoundError(f"DBLP raw files not found: {nodes_path}, {edges_path}")

        nodes, id2idx, id2type = _load_nodes(nodes_path)
        edges = _load_edges(edges_path, id2idx)
        n_nodes = len(nodes)

        # 节点：全局子类别 id。每类 4 个子类型，type_offsets: Author=0, Paper=4, Conference=8, Term=12
        type_sizes = [4] * len(NODE_TYPE_NAMES)
        type_offsets = {}
        cur = 0
        for t, sz in zip(NODE_TYPE_NAMES, type_sizes):
            type_offsets[t] = cur
            cur += sz
        node_type2id = {t: i for i, t in enumerate(NODE_TYPE_NAMES)}
        subtype2id = {s: i for i, s in enumerate(SUBTYPE_NAMES)}

        node_state = torch.empty(n_nodes, dtype=torch.long)
        node_type_id = torch.empty(n_nodes, dtype=torch.long)
        node_subtype_local = torch.empty(n_nodes, dtype=torch.long)
        for i, n in enumerate(nodes):
            t = n["type"]
            sub = n.get("subtype", "DataMining")
            local_sub = subtype2id.get(sub, 0)
            node_type_id[i] = node_type2id.get(t, 0)
            node_subtype_local[i] = local_sub
            node_state[i] = type_offsets.get(t, 0) + local_sub

        # 边：从边类型构建 vocab
        edge_label2id, edge_family2id, edge_family_offsets = _build_vocab_from_data(edges)
        num_edge_types = 1 + len(edge_label2id)

        edge_index_list = []
        edge_attr_list = []
        edge_family_list = []
        for src, dst, etype in edges:
            gid = edge_label2id.get(etype, 1)
            fid = edge_family2id.get(etype, 0)
            edge_index_list.append([src, dst])
            edge_attr_list.append(gid)
            edge_family_list.append(fid)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
        edge_family = torch.tensor(edge_family_list, dtype=torch.long)

        data = Data(
            x=node_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type_id,
            node_subtype=node_subtype_local,
            edge_family=edge_family,
            y=torch.zeros((1, 0), dtype=torch.float),
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        data_list = [data]

        # 统计
        num_nodes = node_counts(data_list)
        node_types_arr = atom_type_counts(data_list, num_classes=cur)
        bond_types_arr = edge_counts(data_list, num_bond_types=num_edge_types)

        # 异质图分布
        node_type_distribution = {}
        for t in NODE_TYPE_NAMES:
            cnt = (node_type_id == node_type2id[t]).sum().item()
            node_type_distribution[t] = cnt / n_nodes if n_nodes else 1.0 / len(NODE_TYPE_NAMES)

        type_pair_fam = defaultdict(lambda: defaultdict(int))
        for src, dst, etype in edges:
            t_src = nodes[src]["type"]
            t_dst = nodes[dst]["type"]
            type_pair_fam[(t_src, t_dst)][etype] += 1
        edge_family_distribution = {}
        for (st, dt), fam_counts in type_pair_fam.items():
            total = sum(fam_counts.values())
            edge_family_distribution[(st, dt)] = {f: c / total for f, c in fam_counts.items()}

        node_subtype_by_type = {}
        for t in NODE_TYPE_NAMES:
            mask = node_type_id == node_type2id[t]
            if mask.any():
                subs = node_state[mask] - type_offsets[t]
                cnts = torch.bincount(subs.clamp(min=0, max=3), minlength=4).float()
                cnts = cnts / cnts.sum().clamp(min=1e-8)
                node_subtype_by_type[t] = cnts
            else:
                node_subtype_by_type[t] = torch.ones(4) / 4.0

        edge_subtype_by_family = {}
        for fam in edge_family2id:
            fid = edge_family2id[fam]
            m = (edge_family == fid) & (edge_attr != 0)
            if m.any():
                vals = edge_attr[m] - edge_family_offsets[fam]
                cnts = torch.bincount(vals.clamp(min=0), minlength=1).float()
                cnts = cnts / cnts.sum().clamp(min=1e-8)
                edge_subtype_by_family[fam] = cnts
            else:
                edge_subtype_by_family[fam] = torch.ones(1)

        edge_family_avg_edge_counts = {}
        for fam in edge_family2id:
            m = (edge_family == edge_family2id[fam]) & (edge_attr != 0)
            edge_family_avg_edge_counts[fam] = float(m.sum().item())

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types_arr)
        np.save(self.processed_paths[3], bond_types_arr)

        vocab = {
            "node_type_names": NODE_TYPE_NAMES,
            "node_type2id": node_type2id,
            "type_offsets": type_offsets,
            "edge_label2id": edge_label2id,
            "edge_family2id": edge_family2id,
            "edge_family_offsets": edge_family_offsets,
            "fam_endpoints": FAM_ENDPOINTS,
            "heterogeneous": self.heterogeneous,
        }
        with open(osp.join(self.processed_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        save_pickle(node_subtype_by_type, osp.join(self.processed_dir, self.processed_file_names[5]))
        save_pickle(edge_subtype_by_family, osp.join(self.processed_dir, self.processed_file_names[6]))
        save_pickle(node_type_distribution, osp.join(self.processed_dir, self.processed_file_names[7]))
        save_pickle(edge_family_distribution, osp.join(self.processed_dir, self.processed_file_names[8]))
        save_pickle(edge_family_avg_edge_counts, osp.join(self.processed_dir, self.processed_file_names[9]))


class DBLPSingleDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        root_path = resolve_data_path(self.datadir, get_original_cwd())
        pre_transform = RemoveYTransform()
        heterogeneous = getattr(cfg.dataset, "heterogeneous", True)
        datasets = {
            "train": DBLPSingleDataset(split="train", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
            "val": DBLPSingleDataset(split="val", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
            "test": DBLPSingleDataset(split="test", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
        }
        self.statistics = {"train": datasets["train"].statistics, "val": datasets["val"].statistics, "test": datasets["test"].statistics}
        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class DBLPSingleInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = False
        self.use_charge = False
        # Ensure diffusion model enters heterogeneous-specific paths.
        self.heterogeneous = bool(getattr(datamodule.cfg.dataset, "heterogeneous", True))
        self.dataset_name = datamodule.dataset_name
        self.node_types = datamodule.inner.statistics.node_types
        self.bond_types = datamodule.inner.statistics.bond_types
        # Required by SamplingMetrics.compute_all_metrics (val/test reference statistics).
        self.statistics = datamodule.statistics
        super().complete_infos(datamodule.statistics, len(datamodule.inner.statistics.node_types))

        from sparse_diffusion.utils import PlaceHolder
        self.output_dims = PlaceHolder(X=datamodule.inner.statistics.node_types.shape[0], E=datamodule.inner.statistics.bond_types.shape[0], y=0, charge=0)

        vocab_path = osp.join(datamodule.inner.processed_dir, "vocab.json")
        if osp.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.heterogeneous = bool(vocab.get("heterogeneous", self.heterogeneous))
            self.node_type_names = vocab.get("node_type_names", [])
            self.type_offsets = {k: int(v) for k, v in vocab.get("type_offsets", {}).items()}
            if not self.type_offsets and self.node_type_names:
                # 按 4 子类/类型 推导
                self.type_offsets = {}
                o = 0
                for t in self.node_type_names:
                    self.type_offsets[t] = o
                    o += 4
            self.edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
            self.edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
            self.fam_endpoints = vocab.get("fam_endpoints", FAM_ENDPOINTS)
            self.edge_label2id = {k: int(v) for k, v in vocab.get("edge_label2id", {}).items()}
        else:
            self.node_type_names = NODE_TYPE_NAMES
            self.type_offsets = {t: i * 4 for i, t in enumerate(NODE_TYPE_NAMES)}
            self.edge_family2id = {}
            self.edge_family_offsets = {}
            self.fam_endpoints = FAM_ENDPOINTS
            self.edge_label2id = {}

        self.edge_family_marginals = {}
        self.edge_family_avg_edge_counts = load_pickle(osp.join(datamodule.inner.processed_dir, "train_edge_family_avg_counts.pickle")) if osp.isfile(osp.join(datamodule.inner.processed_dir, "train_edge_family_avg_counts.pickle")) else {}
        self.node_type_distribution = getattr(datamodule.inner.statistics, "node_type_distribution", None) or {}
        self.edge_family_distribution = getattr(datamodule.inner.statistics, "edge_family_distribution", None) or {}
        self.node_subtype_by_type = getattr(datamodule.inner.statistics, "node_subtype_by_type", None) or {}
        self.edge_subtype_by_family = getattr(datamodule.inner.statistics, "edge_subtype_by_family", None) or {}
        self.vocab_path = datamodule.inner.processed_dir

        # Build per-family edge marginals required by HeterogeneousMarginalUniformTransition.
        # Format per family: [p_no_edge, p_subtype_1, p_subtype_2, ...].
        if self.heterogeneous and self.edge_family_offsets:
            train_num_nodes = datamodule.statistics["train"].num_nodes
            total_graphs = float(sum(train_num_nodes.values())) if train_num_nodes else 0.0
            mean_num_nodes = (
                sum(float(n) * float(c) for n, c in train_num_nodes.items()) / total_graphs
                if total_graphs > 0
                else float(datamodule.inner.data.x.shape[0])
            )

            fam_sorted = sorted(self.edge_family_offsets.items(), key=lambda x: x[1])
            for i, (fam_name, offset) in enumerate(fam_sorted):
                if i + 1 < len(fam_sorted):
                    next_offset = fam_sorted[i + 1][1]
                else:
                    next_offset = int(self.output_dims.E)
                num_subtypes = max(int(next_offset - offset), 1)

                src_type = self.fam_endpoints.get(fam_name, {}).get("src_type")
                dst_type = self.fam_endpoints.get(fam_name, {}).get("dst_type")
                p_src = float(self.node_type_distribution.get(src_type, 0.0))
                p_dst = float(self.node_type_distribution.get(dst_type, 0.0))
                n_src = mean_num_nodes * p_src
                n_dst = mean_num_nodes * p_dst
                if src_type == dst_type:
                    w_fam = max(n_src * max(n_src - 1.0, 0.0), 0.0)
                else:
                    w_fam = max(n_src * n_dst, 0.0)
                m_fam = float(self.edge_family_avg_edge_counts.get(fam_name, 0.0))
                u1 = (m_fam / w_fam) if w_fam > 0 else 0.0
                u1 = max(0.0, min(1.0, u1))
                u0 = 1.0 - u1

                subtype_dist = self.edge_subtype_by_family.get(fam_name)
                if subtype_dist is None:
                    subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / float(num_subtypes)
                else:
                    if not isinstance(subtype_dist, torch.Tensor):
                        subtype_dist = torch.tensor(subtype_dist, dtype=torch.float)
                    if subtype_dist.numel() != num_subtypes:
                        subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / float(num_subtypes)
                    elif subtype_dist.sum() > 0:
                        subtype_dist = subtype_dist / subtype_dist.sum()
                    else:
                        subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / float(num_subtypes)

                fam_marginals = torch.zeros(num_subtypes + 1, dtype=torch.float)
                fam_marginals[0] = u0
                fam_marginals[1:] = u1 * subtype_dist
                self.edge_family_marginals[fam_name] = fam_marginals
