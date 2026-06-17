import time
import os
import os.path as osp
import math
import pickle
import json
import traceback

import torch
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.conv_transformer_model import GraphTransformerConv
from models.pyhgt_denoiser import PyHGTDenoiser
from diffusion.noise_schedule import (
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
)
from diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition

from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from sparse_diffusion.metrics.sampling_metrics import format_structure_metrics_table

from analysis.visualization import Visualizer
from sparse_diffusion import utils
from sparse_diffusion.diffusion import diffusion_utils
from sparse_diffusion.diffusion.sample_edges_utils import (
    get_computational_graph,
    mask_query_graph_from_comp_graph,
    sample_non_existing_edge_attr,
    condensed_to_matrix_index_batch,
    matrix_to_condensed_index_batch,
)
from sparse_diffusion.diffusion.sample_edges import (
    sample_query_edges,
    sample_non_existing_edges_batched,
    sample_non_existing_edges_batched_heterogeneous,
    sampled_condensed_indices_uniformly,
)
from sparse_diffusion.models.sign_pos_encoder import SignNetNodeEncoder


class DiscreteDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    val_iterations = None

    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        extra_features,
        domain_features,
        val_sampling_metrics,
        test_sampling_metrics,
    ):
        super().__init__()
        self.automatic_optimization = (
            int(getattr(cfg.model, "train_query_repeats", 1)) <= 1
            and not bool(getattr(cfg.model, "train_all_blocks_per_noise", False))
        )

        self.in_dims = dataset_infos.input_dims
        self.out_dims = dataset_infos.output_dims
        self.use_charge = cfg.model.use_charge and self.out_dims.charge > 1
        self.node_dist = dataset_infos.nodes_dist
        self.extra_features = extra_features
        self.domain_features = domain_features
        # 当配置关闭额外特征时，不调用 extra/domain 计算、不分配空张量、不拼接
        ef = getattr(cfg.model, "extra_features", None)
        self._no_extra_features = ef is None or (isinstance(ef, str) and ef.lower() == "null")
        self.sign_net = cfg.model.sign_net
        if not self.sign_net:
            cfg.model.sn_hidden_dim = 0

        # sparse settings
        self.edge_fraction = cfg.model.edge_fraction
        self.autoregressive = cfg.model.autoregressive
        self.use_block_query = bool(getattr(cfg.model, "block_query", False))
        self.block_partition_mode = str(getattr(cfg.model, "block_partition_mode", "connected_bfs"))
        self.block_partition_rho = getattr(cfg.model, "block_partition_rho", None)
        self.train_partition_source = str(getattr(cfg.model, "train_partition_source", "noisy"))
        self._true_block_cache = {}
        self._inter_block_pool_cache = {}

        self.cfg = cfg
        self.test_variance = cfg.general.test_variance
        self.dataset_info = dataset_infos
        self.visualization_tools = Visualizer(dataset_infos)
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        self.train_loss = TrainLossDiscrete(
            cfg.model.lambda_train,
            self.edge_fraction,
            self.dataset_info,
            exist_pos_weight=getattr(cfg.model, "exist_pos_weight", None),
            exist_loss_type=getattr(cfg.model, "exist_loss_type", "bce"),
            exist_focal_gamma=getattr(cfg.model, "exist_focal_gamma", 2.0),
            exist_focal_alpha=getattr(cfg.model, "exist_focal_alpha", 0.75),
            edge_neg_weight=getattr(cfg.model, "edge_neg_weight", 1.0),
            relation_matrix_loss_weight=getattr(cfg.model, "relation_matrix_loss_weight", 1.0),
            relation_matrix_loss_normalize=getattr(cfg.model, "relation_matrix_loss_normalize", True),
            metapath2_loss_weight=getattr(cfg.model, "metapath2_loss_weight", 1.0),
            metapath2_loss_normalize=getattr(cfg.model, "metapath2_loss_normalize", True),
            metapath3_loss_weight=getattr(cfg.model, "metapath3_loss_weight", 1.0),
            metapath3_loss_normalize=getattr(cfg.model, "metapath3_loss_normalize", True),
            subtype_degree_loss_weight=getattr(cfg.model, "subtype_degree_loss_weight", 1.0),
            subtype_degree_loss_normalize=getattr(cfg.model, "subtype_degree_loss_normalize", True),
            subtype_degree_max=getattr(cfg.model, "subtype_degree_max", 100),
            structure_loss_max_edges=getattr(cfg.model, "structure_loss_max_edges", 0),
            subtype_degree_use_full_graph_true=getattr(cfg.model, "subtype_degree_use_full_graph_true", False),
            edge_only_model=getattr(cfg.model, "edge_only_model", False),
            structure_loss_type=getattr(cfg.model, "structure_loss_type", "legacy"),
            degree_mmd_loss_weight=getattr(cfg.model, "degree_mmd_loss_weight", 1.0),
            clustering_loss_weight=getattr(cfg.model, "clustering_loss_weight", 1.0),
            triangles_loss_weight=getattr(cfg.model, "triangles_loss_weight", 1.0),
            wedge_closure_loss_weight=getattr(cfg.model, "wedge_closure_loss_weight", 0.0),
            edge_types_tv_loss_weight=getattr(cfg.model, "edge_types_tv_loss_weight", 1.0),
            structure_loss_max_nodes=getattr(cfg.model, "structure_loss_max_nodes", 3000),
            structure_triangles_normalize=getattr(cfg.model, "structure_triangles_normalize", True),
        )
        # 验证阶段用与训练相同的基础度量：relation_matrix_L1、metapath2/3_L1、subtype_degree_L1 的累加器（每 epoch 重置）
        self._val_relation_matrix_L1_sum = 0.0
        self._val_relation_matrix_L1_count = 0
        self._val_metapath2_L1_sum = 0.0
        self._val_metapath2_L1_count = 0
        self._val_metapath3_L1_sum = 0.0
        self._val_metapath3_L1_count = 0
        self._val_subtype_degree_L1_sum = 0.0
        self._val_subtype_degree_L1_count = 0
        self.train_metrics = train_metrics
        self.val_sampling_metrics = val_sampling_metrics
        self.test_sampling_metrics = test_sampling_metrics

        # TODO: transform to torchmetrics.MetricCollection
        self.val_nll = NLL()
        self.val_exist_nll = NLL()
        self.val_subtype_nll = NLL()
        self.val_exist_kl = NLL()
        self.val_subtype_kl = NLL()
        self.val_exist_logp = NLL()
        self.val_subtype_logp = NLL()
        # Legacy diffusion-VLB metrics (kept for compatibility; not used by the new val logging path)
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.best_nll = 1e8
        self.best_epoch = 0

        # TODO: transform to torchmetrics.MetricCollection
        self.test_nll = NLL()
        self.test_exist_nll = NLL()
        self.test_subtype_nll = NLL()
        self.test_exist_kl = NLL()
        self.test_subtype_kl = NLL()
        self.test_exist_logp = NLL()
        self.test_subtype_logp = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        if self.use_charge:
            self.val_charge_kl = SumExceptBatchKL()
            self.val_charge_logp = SumExceptBatchMetric()
            self.test_charge_kl = SumExceptBatchKL()
            self.test_charge_logp = SumExceptBatchMetric()

        # 获取异质图相关参数
        heterogeneous = getattr(self.dataset_info, "heterogeneous", False)
        num_node_types = 0
        num_node_subtypes = 0
        num_relation_types = 0
        type_offsets = None
        edge_family_offsets = None
        
        if heterogeneous:
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            num_node_types = len(node_type_names) if node_type_names else 0
            num_node_subtypes = self.out_dims.X  # 所有子类别的总数
            num_relation_types = num_node_types * num_node_types if num_node_types > 0 else 0
            type_offsets = getattr(self.dataset_info, "type_offsets", None)
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", None)
            
        denoiser_name = str(getattr(cfg.model, "denoiser", "graph_transformer")).lower()
        denoiser_cls = PyHGTDenoiser if denoiser_name in {"hgt", "pyhgt", "pyhgt_denoiser"} else GraphTransformerConv
        if denoiser_cls is PyHGTDenoiser:
            print("[DiHuG] Using pyHGT denoiser")
        model_kwargs = dict(
            n_layers=cfg.model.n_layers,
            input_dims=self.in_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.out_dims,
            sn_hidden_dim=cfg.model.sn_hidden_dim,
            output_y=cfg.model.output_y,
            dropout=cfg.model.dropout,
            heterogeneous=heterogeneous,
            num_node_types=num_node_types,
            num_node_subtypes=num_node_subtypes,
            num_relation_types=num_relation_types,
            type_embed_dim=getattr(cfg.model, "type_embed_dim", 64),
            subtype_embed_dim=getattr(cfg.model, "subtype_embed_dim", 64),
            relation_embed_dim=getattr(cfg.model, "relation_embed_dim", 64),
            edge_family_offsets=edge_family_offsets,
            type_offsets=type_offsets,
            use_type_modulation=getattr(cfg.model, "use_type_modulation", True),
            edge_only_model=getattr(cfg.model, "edge_only_model", False),
        )
        if denoiser_cls is PyHGTDenoiser:
            model_kwargs.update(
                subtype_dim=getattr(cfg.model, "subtype_dim", 32),
                use_edge_phi_fusion=getattr(cfg.model, "use_edge_phi_fusion", True),
                use_dual_softmax=getattr(cfg.model, "use_dual_softmax", True),
                use_time_film=getattr(cfg.model, "use_time_film", False),
                use_edge_state_update=getattr(cfg.model, "use_edge_state_update", False),
                edge_state_update_mode=getattr(cfg.model, "edge_state_update_mode", "all"),
            )
        self.model = denoiser_cls(**model_kwargs)

        # whether to use sign net
        if self.sign_net and cfg.model.extra_features == "all":
            self.sign_net = SignNetNodeEncoder(
                dataset_infos, cfg.model.sn_hidden_dim, cfg.model.num_eigenvectors
            )

        # whether to use scale layers
        self.scaling_layer = cfg.model.scaling_layer
        (
            self.node_scaling_layer,
            self.edge_scaling_layer,
            self.graph_scaling_layer,
        ) = self.get_scaling_layers()

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps,
            skip=self.cfg.general.skip,
        )

        # Marginal transition（兼容 node_types/edge_types 为 numpy 的情况，如 DBLP single）
        node_types = self.dataset_info.node_types
        if not isinstance(node_types, torch.Tensor):
            node_types = torch.as_tensor(node_types, dtype=torch.float, device=self.device)
        else:
            node_types = node_types.float().to(self.device)
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types
        if not isinstance(edge_types, torch.Tensor):
            edge_types = torch.as_tensor(edge_types, dtype=torch.float, device=self.device)
        else:
            edge_types = edge_types.float().to(self.device)
        e_marginals = edge_types / torch.sum(edge_types)

        if not self.use_charge:
            charge_marginals = node_types.new_zeros(0)
        else:
            charge_marginals = (
                self.dataset_info.charge_types * node_types[:, None]
            ).sum(dim=0)

        # 检查是否为异质图模式
        self.heterogeneous = getattr(self.dataset_info, "heterogeneous", False)
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_marginals") and len(self.dataset_info.edge_family_marginals) > 0:
            self.transition_model = HeterogeneousMarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.out_dims.y,
                charge_marginals=charge_marginals,
                edge_family_marginals=getattr(self.dataset_info, "edge_family_marginals", None),
                edge_family_offsets=getattr(self.dataset_info, "edge_family_offsets", None),
            )
        else:
            # 同质图模式：使用原始转移矩阵
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.out_dims.y,
                charge_marginals=charge_marginals,
            )

        self.limit_dist = utils.PlaceHolder(
            X=x_marginals,
            E=e_marginals,
            y=torch.ones(self.out_dims.y) / self.out_dims.y,
            charge=charge_marginals,
        )

        self.save_hyperparameters(ignore=["train_metrics", "sampling_metrics", "val_sampling_metrics", "test_sampling_metrics"])
        # 非验证 epoch 也提供 val/epoch_NLL（用上一轮 val 的值或 inf），避免 ModelCheckpoint 因缺少 monitor key 报警
        self._last_val_epoch_nll = float("inf")
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        # 每 N 步才跑一次 train_metrics（relation_matrix/metapath2/subtype_degree 等较耗时），设为 >1 可加速
        self.train_metrics_every_n_steps = int(getattr(cfg.train, "train_metrics_every_n_steps", 1))

    def _rho_for_partition(self) -> float:
        rho = self.block_partition_rho
        if rho is None:
            rho = self.edge_fraction
        return float(min(max(float(rho), 1e-8), 1.0))



    def _log_hetero_metis_block_summary(
        self,
        blocks,
        batch_nodes,
        local_edge_index,
        local_edge_family,
        node_t,
        type_offsets,
        id2edge_family,
        graph_idx,
        rel_balance_power,
    ):
        if not blocks:
            return
        sorted_types = sorted((int(v), str(k)) for k, v in type_offsets.items())
        fam_names = {int(k): str(v) for k, v in id2edge_family.items()}
        sizes = [len(b) for b in blocks]
        print(
            f"[BLOCK] hetero_metis graph={graph_idx} power={float(rel_balance_power):.3f} "
            f"blocks={len(blocks)} size_min={min(sizes)} size_max={max(sizes)} "
            f"size_mean={sum(sizes)/len(sizes):.1f}"
        )
        top_k = int(getattr(self.cfg.model, "hetero_metis_log_top_blocks", 5) or 0)
        if top_k <= 0:
            return
        edge_src = local_edge_index[0].detach().cpu() if local_edge_index.numel() else torch.empty(0, dtype=torch.long)
        edge_dst = local_edge_index[1].detach().cpu() if local_edge_index.numel() else torch.empty(0, dtype=torch.long)
        edge_fam = local_edge_family.detach().cpu() if local_edge_family.numel() else torch.empty(0, dtype=torch.long)
        local_n = int(batch_nodes.numel())
        deg_cpu = torch.zeros(local_n, dtype=torch.float32)
        for u, v in zip(edge_src.tolist(), edge_dst.tolist()):
            if 0 <= int(u) < local_n and 0 <= int(v) < local_n and int(u) != int(v):
                deg_cpu[int(u)] += 1.0
                deg_cpu[int(v)] += 1.0
        block_loads = []
        for block in blocks:
            if block:
                block_loads.append(float(deg_cpu[torch.tensor(block, dtype=torch.long)].sum().item()))
        if block_loads:
            print(
                f"[BLOCK] degree_load min={min(block_loads):.1f} max={max(block_loads):.1f} "
                f"mean={sum(block_loads)/len(block_loads):.1f}"
            )
        node_types_cpu = node_t[batch_nodes].detach().cpu()
        for block_id, block in enumerate(blocks[:top_k]):
            block_cpu = torch.tensor(block, dtype=torch.long)
            if block_cpu.numel() == 0:
                continue
            block_set = set(int(x) for x in block_cpu.tolist())
            type_parts = []
            for idx, (offset, type_name) in enumerate(sorted_types):
                next_offset = sorted_types[idx + 1][0] if idx + 1 < len(sorted_types) else self.out_dims.X
                cnt = int(((node_types_cpu[block_cpu] >= offset) & (node_types_cpu[block_cpu] < next_offset)).sum().item())
                if cnt > 0:
                    type_parts.append(f"{type_name}:{cnt}")
            fam_counts = {}
            intra_edges = 0
            for e_idx, (u, v) in enumerate(zip(edge_src.tolist(), edge_dst.tolist())):
                if u in block_set and v in block_set:
                    intra_edges += 1
                    fam_id = int(edge_fam[e_idx].item()) if e_idx < edge_fam.numel() else 0
                    fam_counts[fam_id] = fam_counts.get(fam_id, 0) + 1
            top_fams = sorted(fam_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            fam_parts = []
            for fam_id, cnt in top_fams:
                ratio = float(cnt) / max(float(intra_edges), 1.0)
                fam_parts.append(f"{fam_names.get(fam_id, fam_id)}:{cnt}({ratio:.2f})")
            print(
                f"[BLOCK] id={block_id} nodes={len(block)} intra_edges={intra_edges} "
                f"degree_load={float(deg_cpu[block_cpu].sum().item()):.1f} "
                f"types={{{', '.join(type_parts)}}} top_families={{{', '.join(fam_parts)}}}"
            )


    def _cache_hetero_block_templates(self, blocks, batch_nodes, node_t, type_offsets):
        """Store only block type-count templates; no true block members are used for sampling."""
        if not blocks or not type_offsets:
            return
        sorted_types = sorted(((str(k), int(v)) for k, v in type_offsets.items()), key=lambda x: x[1])
        node_types_cpu = node_t[batch_nodes].detach().cpu()
        templates = []
        for block in blocks:
            block_cpu = torch.tensor(block, dtype=torch.long)
            if block_cpu.numel() == 0:
                continue
            counts = []
            for idx, (_, offset) in enumerate(sorted_types):
                next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
                cnt = int(((node_types_cpu[block_cpu] >= offset) & (node_types_cpu[block_cpu] < next_offset)).sum().item())
                counts.append(cnt)
            if sum(counts) > 0:
                templates.append(tuple(counts))
        if templates:
            self._hetero_block_templates = templates
            self._hetero_block_template_type_order = [name for name, _ in sorted_types]
            if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_hetero_block_templates", False):
                print(f"[BLOCK] cached type templates: {len(templates)} blocks, type_order={self._hetero_block_template_type_order}")
                self._logged_hetero_block_templates = True

    def _cache_hetero_block_edge_templates(self, blocks, local_edge_index, local_edge_family):
        """Cache true block/block-pair edge quotas by relation family for template sampling init."""
        if not blocks:
            return
        block_of = {}
        for block_id, block in enumerate(blocks):
            for n in block:
                block_of[int(n)] = int(block_id)
        intra = {}
        inter = {}
        if local_edge_index is not None and local_edge_index.numel() > 0:
            srcs = local_edge_index[0].detach().cpu().tolist()
            dsts = local_edge_index[1].detach().cpu().tolist()
            fams = local_edge_family.detach().cpu().tolist() if local_edge_family is not None and local_edge_family.numel() else [0] * len(srcs)
            for u, v, fam in zip(srcs, dsts, fams):
                bu = block_of.get(int(u))
                bv = block_of.get(int(v))
                if bu is None or bv is None:
                    continue
                fam = int(fam)
                if bu == bv:
                    key = (int(bu), fam)
                    intra[key] = int(intra.get(key, 0)) + 1
                else:
                    a, b = sorted((int(bu), int(bv)))
                    key = (a, b, fam)
                    inter[key] = int(inter.get(key, 0)) + 1
        self._hetero_block_edge_templates = {
            "num_blocks": int(len(blocks)),
            "intra": intra,
            "inter": inter,
        }
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_hetero_block_edge_templates", False):
            total_intra = sum(intra.values())
            total_inter = sum(inter.values())
            print(
                f"[BLOCK] cached edge templates: blocks={len(blocks)} "
                f"intra_edges={total_intra} inter_edges={total_inter}"
            )
            self._logged_hetero_block_edge_templates = True

    def _sample_edges_from_candidates(self, candidates, quota):
        quota = int(quota)
        if candidates is None or candidates.numel() == 0 or quota <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        take = min(quota, int(candidates.shape[1]))
        if take <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        perm = torch.randperm(int(candidates.shape[1]), device=self.device)[:take]
        return candidates[:, perm]

    def _candidate_edges_for_block_family(self, block_nodes, src_nodes, dst_nodes, same_type):
        if block_nodes is None or block_nodes.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        total_nodes = int(max(int(block_nodes.max().item()) + 1, int(src_nodes.max().item()) + 1 if src_nodes.numel() else 0, int(dst_nodes.max().item()) + 1 if dst_nodes.numel() else 0))
        in_block = torch.zeros(max(total_nodes, 1), dtype=torch.bool, device=self.device)
        in_block[block_nodes.long()] = True
        src_block = src_nodes[in_block[src_nodes]] if src_nodes.numel() else src_nodes
        dst_block = dst_nodes[in_block[dst_nodes]] if dst_nodes.numel() else dst_nodes
        if src_block.numel() == 0 or dst_block.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        if same_type:
            pairs = torch.triu_indices(src_block.numel(), src_block.numel(), offset=1, device=self.device)
            if pairs.shape[1] == 0:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            return torch.stack([src_block[pairs[0]], src_block[pairs[1]]], dim=0)
        flat = torch.arange(int(src_block.numel() * dst_block.numel()), device=self.device)
        return torch.stack([src_block[flat // dst_block.numel()], dst_block[flat % dst_block.numel()]], dim=0)

    def _candidate_edges_for_block_pair_family(self, block_a, block_b, src_nodes, dst_nodes, same_type):
        if block_a is None or block_b is None or block_a.numel() == 0 or block_b.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        max_node = 0
        for t in (block_a, block_b, src_nodes, dst_nodes):
            if t is not None and t.numel() > 0:
                max_node = max(max_node, int(t.max().item()) + 1)
        in_a = torch.zeros(max(max_node, 1), dtype=torch.bool, device=self.device)
        in_b = torch.zeros(max(max_node, 1), dtype=torch.bool, device=self.device)
        in_a[block_a.long()] = True
        in_b[block_b.long()] = True
        if same_type:
            a = src_nodes[in_a[src_nodes]] if src_nodes.numel() else src_nodes
            b = src_nodes[in_b[src_nodes]] if src_nodes.numel() else src_nodes
            if a.numel() == 0 or b.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            flat = torch.arange(int(a.numel() * b.numel()), device=self.device)
            u = a[flat // b.numel()]
            v = b[flat % b.numel()]
            return torch.stack([torch.minimum(u, v), torch.maximum(u, v)], dim=0)
        parts = []
        src_a = src_nodes[in_a[src_nodes]] if src_nodes.numel() else src_nodes
        dst_b = dst_nodes[in_b[dst_nodes]] if dst_nodes.numel() else dst_nodes
        if src_a.numel() > 0 and dst_b.numel() > 0:
            flat = torch.arange(int(src_a.numel() * dst_b.numel()), device=self.device)
            parts.append(torch.stack([src_a[flat // dst_b.numel()], dst_b[flat % dst_b.numel()]], dim=0))
        src_b = src_nodes[in_b[src_nodes]] if src_nodes.numel() else src_nodes
        dst_a = dst_nodes[in_a[dst_nodes]] if dst_nodes.numel() else dst_nodes
        if src_b.numel() > 0 and dst_a.numel() > 0:
            flat = torch.arange(int(src_b.numel() * dst_a.numel()), device=self.device)
            parts.append(torch.stack([src_b[flat // dst_a.numel()], dst_a[flat % dst_a.numel()]], dim=0))
        return torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.long, device=self.device)


    def _sample_family_labels_from_marginal(self, fam_name, num_edges, edge_family_offsets):
        num_edges = int(num_edges)
        if num_edges <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        start = max(1, int(edge_family_offsets.get(str(fam_name), 1)))
        if str(fam_name) not in marginals:
            return torch.full((num_edges,), start, dtype=torch.long, device=self.device)
        m = marginals[str(fam_name)]
        if not torch.is_tensor(m):
            m = torch.tensor(m, dtype=torch.float32, device=self.device)
        else:
            m = m.to(self.device, dtype=torch.float32)
        pos = m.reshape(-1)[1:]
        if pos.numel() <= 0 or float(pos.sum().detach().cpu()) <= 0.0:
            return torch.full((num_edges,), start, dtype=torch.long, device=self.device)
        probs = pos / pos.sum().clamp_min(1e-12)
        local = torch.multinomial(probs, num_samples=num_edges, replacement=True)
        return (local.long() + start).clamp(0, self.out_dims.E - 1)

    def _apply_block_marginal_initial_edges(self, sparse_sampled_data):
        """Initialize z_T with block candidate spaces but global family limit densities.

        This avoids using true block edge-count templates: each block/block-pair family
        candidate pool receives round(P(edge|family) * pool_size) random edges.
        """
        if not bool(getattr(self.cfg.model, "sampling_block_marginal_init", False)):
            return sparse_sampled_data
        pseudo_blocks = getattr(sparse_sampled_data, "pseudo_blocks", None)
        if not pseudo_blocks:
            return sparse_sampled_data
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        if not (id2edge_family and fam_endpoints and type_offsets and edge_family_offsets and marginals):
            return sparse_sampled_data

        node_t = sparse_sampled_data.anchor_node_subtype.long().to(self.device)
        batch = sparse_sampled_data.batch.long().to(self.device)
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[str(type_name)] = int(next_offset - int(offset))

        selected_edges = []
        selected_labels = []
        selected_intra = 0
        selected_inter = 0
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        for graph_idx in range(bs):
            graph_blocks = [
                b.long().to(self.device)
                for b in pseudo_blocks
                if b is not None and b.numel() > 0 and int(batch[b[0]].item()) == graph_idx
            ]
            if not graph_blocks:
                graph_blocks = [b.long().to(self.device) for b in pseudo_blocks if b is not None and b.numel() > 0]
            if not graph_blocks:
                continue
            graph_mask = batch == int(graph_idx)
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                density = self._family_density_from_marginal(fam_name, marginals)
                if density is None or density <= 0.0:
                    continue
                src_type = str(fam_endpoints[fam_name]["src_type"])
                dst_type = str(fam_endpoints[fam_name]["dst_type"])
                if src_type not in type_offsets or dst_type not in type_offsets:
                    continue
                src_offset = int(type_offsets[src_type])
                dst_offset = int(type_offsets[dst_type])
                src_size = int(type_sizes.get(src_type, 0))
                dst_size = int(type_sizes.get(dst_type, 0))
                src_nodes = torch.where((node_t >= src_offset) & (node_t < src_offset + src_size) & graph_mask)[0]
                dst_nodes = torch.where((node_t >= dst_offset) & (node_t < dst_offset + dst_size) & graph_mask)[0]
                if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                    continue
                same_type = src_type == dst_type

                for block_nodes in graph_blocks:
                    cand = self._candidate_edges_for_block_family(block_nodes, src_nodes, dst_nodes, same_type)
                    quota = int(round(float(density) * int(cand.shape[1])))
                    chosen = self._sample_edges_from_candidates(cand, quota)
                    if chosen.numel() > 0:
                        selected_edges.append(chosen)
                        selected_labels.append(self._sample_family_labels_from_marginal(fam_name, chosen.shape[1], edge_family_offsets))
                        selected_intra += int(chosen.shape[1])

                for i in range(len(graph_blocks)):
                    for j in range(i + 1, len(graph_blocks)):
                        cand = self._candidate_edges_for_block_pair_family(
                            graph_blocks[i], graph_blocks[j], src_nodes, dst_nodes, same_type
                        )
                        quota = int(round(float(density) * int(cand.shape[1])))
                        chosen = self._sample_edges_from_candidates(cand, quota)
                        if chosen.numel() > 0:
                            selected_edges.append(chosen)
                            selected_labels.append(self._sample_family_labels_from_marginal(fam_name, chosen.shape[1], edge_family_offsets))
                            selected_inter += int(chosen.shape[1])

        if not selected_edges:
            return sparse_sampled_data
        edge_index = torch.cat(selected_edges, dim=1).long()
        edge_labels = torch.cat(selected_labels, dim=0).long().clamp(0, self.out_dims.E - 1)
        if edge_index.shape[1] > 1:
            total_nodes = int(max(1, sparse_sampled_data.node.shape[0]))
            key = edge_index[0].long() * total_nodes + edge_index[1].long()
            order = torch.argsort(key)
            edge_index = edge_index[:, order]
            edge_labels = edge_labels[order]
            key_sorted = key[order]
            keep = torch.ones(edge_labels.shape[0], dtype=torch.bool, device=self.device)
            keep[1:] = key_sorted[1:] != key_sorted[:-1]
            edge_index = edge_index[:, keep]
            edge_labels = edge_labels[keep]
        sparse_sampled_data.edge_index = edge_index
        sparse_sampled_data.edge_attr = F.one_hot(edge_labels, num_classes=self.out_dims.E).float()
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_block_marginal_init", False):
            print(
                f"[采样] block-marginal init edges={int(edge_index.shape[1])} "
                f"intra={selected_intra} inter={selected_inter} families={len(id2edge_family)}"
            )
            self._logged_sampling_block_marginal_init = True
        return sparse_sampled_data

    def _apply_block_family_budget_projection(self, sparse_sampled_data):
        """Prune sampled edges to local block/family budgets from global marginals.

        This is a sampling-time stabilizer, not an oracle: budgets are computed as
        P(edge | family) times each pseudo block or block-pair candidate pool size.
        It only removes over-budget edges and never adds missing edges.
        """
        if not bool(getattr(self.cfg.model, "sampling_block_family_budget_projection", False)):
            return sparse_sampled_data
        source = str(getattr(self.cfg.model, "sampling_block_family_budget_source", "marginal")).lower()
        if source != "marginal":
            return sparse_sampled_data
        pseudo_blocks = getattr(sparse_sampled_data, "pseudo_blocks", None)
        if not pseudo_blocks:
            return sparse_sampled_data
        edge_index = getattr(sparse_sampled_data, "edge_index", None)
        edge_attr = getattr(sparse_sampled_data, "edge_attr", None)
        if edge_index is None or edge_attr is None or edge_index.numel() == 0:
            return sparse_sampled_data

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        if not (id2edge_family and fam_endpoints and type_offsets and marginals):
            return sparse_sampled_data

        device = self.device
        budget_multiplier = float(getattr(self.cfg.model, "sampling_block_family_budget_multiplier", 1.0) or 1.0)
        budget_multiplier = max(0.0, budget_multiplier)
        intra_budget_multiplier = float(
            getattr(self.cfg.model, "sampling_block_family_budget_intra_multiplier", budget_multiplier) or 0.0
        )
        inter_budget_multiplier = float(
            getattr(self.cfg.model, "sampling_block_family_budget_inter_multiplier", budget_multiplier) or 0.0
        )
        intra_budget_multiplier = max(0.0, intra_budget_multiplier)
        inter_budget_multiplier = max(0.0, inter_budget_multiplier)
        keep_policy = str(getattr(self.cfg.model, "sampling_block_family_budget_keep_policy", "random")).lower()
        edge_index = edge_index.long().to(device)
        if edge_attr.dim() > 1:
            edge_labels = edge_attr.to(device).argmax(dim=-1).long()
        else:
            edge_labels = edge_attr.long().to(device).reshape(-1)
        if edge_labels.numel() != edge_index.shape[1]:
            return sparse_sampled_data

        node_t = sparse_sampled_data.anchor_node_subtype.long().to(device)
        batch = sparse_sampled_data.batch.long().to(device)
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[str(type_name)] = int(next_offset - int(offset))

        label_to_fam = self._label_to_family_lookup().to(device)
        valid_label = (edge_labels > 0) & (edge_labels < int(label_to_fam.numel()))
        edge_fams = torch.full_like(edge_labels, -1)
        if valid_label.any():
            edge_fams[valid_label] = label_to_fam[edge_labels[valid_label]].long()

        num_nodes = int(node_t.shape[0])
        block_of = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        for block_idx, block_nodes in enumerate(pseudo_blocks):
            if block_nodes is None or block_nodes.numel() == 0:
                continue
            block_of[block_nodes.long().to(device)] = int(block_idx)

        budgets = {}
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        for graph_idx in range(bs):
            graph_blocks = []
            for global_idx, block_nodes in enumerate(pseudo_blocks):
                if block_nodes is None or block_nodes.numel() == 0:
                    continue
                block_nodes = block_nodes.long().to(device)
                if int(batch[block_nodes[0]].item()) == int(graph_idx):
                    graph_blocks.append((int(global_idx), block_nodes))
            if not graph_blocks:
                continue
            graph_mask = batch == int(graph_idx)
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                density = self._family_density_from_marginal(fam_name, marginals)
                if density is None:
                    continue
                src_type = str(fam_endpoints[fam_name]["src_type"])
                dst_type = str(fam_endpoints[fam_name]["dst_type"])
                if src_type not in type_offsets or dst_type not in type_offsets:
                    continue
                src_offset = int(type_offsets[src_type])
                dst_offset = int(type_offsets[dst_type])
                src_size = int(type_sizes.get(src_type, 0))
                dst_size = int(type_sizes.get(dst_type, 0))
                src_nodes = torch.where((node_t >= src_offset) & (node_t < src_offset + src_size) & graph_mask)[0]
                dst_nodes = torch.where((node_t >= dst_offset) & (node_t < dst_offset + dst_size) & graph_mask)[0]
                if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                    continue
                same_type = src_type == dst_type

                for block_id, block_nodes in graph_blocks:
                    cand = self._candidate_edges_for_block_family(block_nodes, src_nodes, dst_nodes, same_type)
                    budgets[(int(graph_idx), "intra", int(block_id), int(block_id), int(fam_id))] = int(
                        round(float(density) * intra_budget_multiplier * int(cand.shape[1]))
                    )

                for local_i in range(len(graph_blocks)):
                    bi, block_i = graph_blocks[local_i]
                    for local_j in range(local_i + 1, len(graph_blocks)):
                        bj, block_j = graph_blocks[local_j]
                        cand = self._candidate_edges_for_block_pair_family(block_i, block_j, src_nodes, dst_nodes, same_type)
                        a, b = (int(bi), int(bj)) if int(bi) <= int(bj) else (int(bj), int(bi))
                        budgets[(int(graph_idx), "inter", a, b, int(fam_id))] = int(
                            round(float(density) * inter_budget_multiplier * int(cand.shape[1]))
                        )

        if not budgets:
            return sparse_sampled_data

        groups = {}
        u = edge_index[0].long()
        v = edge_index[1].long()
        bu = block_of[u]
        bv = block_of[v]
        graph_ids = batch[u].long()
        for idx in range(int(edge_index.shape[1])):
            fam = int(edge_fams[idx].item())
            bi = int(bu[idx].item())
            bj = int(bv[idx].item())
            if fam < 0 or bi < 0 or bj < 0:
                continue
            graph_idx = int(graph_ids[idx].item())
            if bi == bj:
                key = (graph_idx, "intra", bi, bj, fam)
            else:
                a, b = (bi, bj) if bi <= bj else (bj, bi)
                key = (graph_idx, "inter", a, b, fam)
            if key in budgets:
                groups.setdefault(key, []).append(idx)

        keep = torch.ones((int(edge_index.shape[1]),), dtype=torch.bool, device=device)
        removed = 0
        deg_for_keep = None
        if keep_policy == "degree_spread":
            deg_for_keep = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
            deg_for_keep.scatter_add_(0, u, torch.ones_like(u, dtype=torch.float32))
            deg_for_keep.scatter_add_(0, v, torch.ones_like(v, dtype=torch.float32))
        for key, idxs in groups.items():
            budget = max(0, int(budgets.get(key, len(idxs))))
            n = len(idxs)
            if n <= budget:
                continue
            local = torch.tensor(idxs, dtype=torch.long, device=device)
            if keep_policy == "degree_spread" and deg_for_keep is not None:
                # Keep edges that spread mass to lower-degree endpoints. A tiny
                # random jitter avoids deterministic ties within equal-degree pools.
                score = deg_for_keep[u[local]] + deg_for_keep[v[local]]
                score = score + 1e-4 * torch.rand_like(score)
                order = torch.argsort(score, descending=False)
                drop = local[order[budget:]]
            else:
                perm = torch.randperm(n, device=device)
                drop = local[perm[budget:]]
            keep[drop] = False
            removed += int(drop.numel())

        if removed <= 0:
            return sparse_sampled_data

        before = int(edge_index.shape[1])
        edge_index = edge_index[:, keep]
        edge_labels = edge_labels[keep].clamp(0, self.out_dims.E - 1)
        sparse_sampled_data.edge_index = edge_index
        sparse_sampled_data.edge_attr = F.one_hot(edge_labels, num_classes=self.out_dims.E).float()
        after = int(edge_index.shape[1])
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_block_family_budget_projection", False):
            print(
                f"[采样-BUDGET] block-family marginal projection "
                f"before={before} after={after} removed={removed} groups={len(groups)} "
                f"mult={budget_multiplier:.3f} intra={intra_budget_multiplier:.3f} "
                f"inter={inter_budget_multiplier:.3f} keep={keep_policy}"
            )
            self._logged_sampling_block_family_budget_projection = True
        return sparse_sampled_data

    def _apply_block_template_initial_edges(self, sparse_sampled_data):
        if bool(getattr(self.cfg.model, "sampling_block_marginal_init", False)):
            return sparse_sampled_data
        if not bool(getattr(self.cfg.model, "sampling_block_template_init", False)):
            return sparse_sampled_data
        templates = getattr(self, "_hetero_block_edge_templates", None)
        pseudo_blocks = getattr(sparse_sampled_data, "pseudo_blocks", None)
        if not templates or not pseudo_blocks:
            return sparse_sampled_data
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        if not (id2edge_family and fam_endpoints and type_offsets and edge_family_offsets):
            return sparse_sampled_data

        node_t = sparse_sampled_data.anchor_node_subtype.long().to(self.device)
        batch = sparse_sampled_data.batch.long().to(self.device)
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[str(type_name)] = int(next_offset - int(offset))

        selected_edges = []
        selected_labels = []
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        for graph_idx in range(bs):
            graph_blocks = [b.long().to(self.device) for b in pseudo_blocks if b is not None and b.numel() > 0 and int(batch[b[0]].item()) == graph_idx]
            if not graph_blocks:
                graph_blocks = [b.long().to(self.device) for b in pseudo_blocks if b is not None and b.numel() > 0]
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                src_type = str(fam_endpoints[fam_name]["src_type"])
                dst_type = str(fam_endpoints[fam_name]["dst_type"])
                if src_type not in type_offsets or dst_type not in type_offsets:
                    continue
                src_offset = int(type_offsets[src_type])
                dst_offset = int(type_offsets[dst_type])
                src_size = int(type_sizes.get(src_type, 0))
                dst_size = int(type_sizes.get(dst_type, 0))
                graph_mask = batch == int(graph_idx)
                src_nodes = torch.where((node_t >= src_offset) & (node_t < src_offset + src_size) & graph_mask)[0]
                dst_nodes = torch.where((node_t >= dst_offset) & (node_t < dst_offset + dst_size) & graph_mask)[0]
                if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                    continue
                same_type = src_type == dst_type
                label = max(1, int(edge_family_offsets.get(fam_name, 1)))
                for block_id, block_nodes in enumerate(graph_blocks):
                    quota = int(templates.get("intra", {}).get((int(block_id), int(fam_id)), 0))
                    cand = self._candidate_edges_for_block_family(block_nodes, src_nodes, dst_nodes, same_type)
                    chosen = self._sample_edges_from_candidates(cand, quota)
                    if chosen.numel() > 0:
                        selected_edges.append(chosen)
                        selected_labels.append(torch.full((chosen.shape[1],), label, dtype=torch.long, device=self.device))
                for i in range(len(graph_blocks)):
                    for j in range(i + 1, len(graph_blocks)):
                        quota = int(templates.get("inter", {}).get((int(i), int(j), int(fam_id)), 0))
                        cand = self._candidate_edges_for_block_pair_family(graph_blocks[i], graph_blocks[j], src_nodes, dst_nodes, same_type)
                        chosen = self._sample_edges_from_candidates(cand, quota)
                        if chosen.numel() > 0:
                            selected_edges.append(chosen)
                            selected_labels.append(torch.full((chosen.shape[1],), label, dtype=torch.long, device=self.device))
        if not selected_edges:
            return sparse_sampled_data
        edge_index = torch.cat(selected_edges, dim=1).long()
        edge_labels = torch.cat(selected_labels, dim=0).long().clamp(0, self.out_dims.E - 1)
        if edge_index.shape[1] > 1:
            key = edge_index[0].long() * int(max(1, sparse_sampled_data.node.shape[0])) + edge_index[1].long()
            order = torch.argsort(key)
            edge_index = edge_index[:, order]
            edge_labels = edge_labels[order]
            keep = torch.ones(edge_labels.shape[0], dtype=torch.bool, device=self.device)
            key_sorted = key[order]
            keep[1:] = key_sorted[1:] != key_sorted[:-1]
            edge_index = edge_index[:, keep]
            edge_labels = edge_labels[keep]
        sparse_sampled_data.edge_index = edge_index
        sparse_sampled_data.edge_attr = F.one_hot(edge_labels, num_classes=self.out_dims.E).float()
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_block_template_init", False):
            print(f"[采样] block-template init edges={int(edge_index.shape[1])} families={len(id2edge_family)}")
            self._logged_sampling_block_template_init = True
        return sparse_sampled_data

    def _ensure_hetero_block_templates_from_data(self, data=None):
        """Build block type templates on demand, e.g. for test_only checkpoints."""
        if getattr(self, "_hetero_block_templates", None):
            return
        source = data
        if source is None and hasattr(self.dataset_info, "datamodule"):
            dm = self.dataset_info.datamodule
            if hasattr(dm, "train_dataset") and len(dm.train_dataset) > 0:
                source = dm.train_dataset[0]
            elif hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
                source = dm.test_dataset[0]
        if source is None:
            return
        try:
            source = source.clone()
        except Exception:
            pass
        try:
            source = self.dataset_info.to_one_hot(source)
        except Exception:
            pass
        source = source.to(self.device)
        if getattr(source, "batch", None) is None:
            source.batch = torch.zeros(source.x.shape[0], dtype=torch.long, device=self.device)
            source.ptr = torch.tensor([0, source.x.shape[0]], dtype=torch.long, device=self.device)
        elif getattr(source, "ptr", None) is None:
            counts = torch.bincount(source.batch.long(), minlength=int(source.batch.max().item()) + 1)
            source.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), counts.cumsum(0)])
        # This computes and caches true blocks/templates as a side effect. Query result is ignored.
        _ = self._heterogeneous_global_block_query_edges(source, None)

    def _build_type_template_pseudo_blocks(self, anchor_node_subtype, batch, type_offsets):
        templates = getattr(self, "_hetero_block_templates", None)
        if not templates or not type_offsets:
            return None
        sorted_types = sorted(((str(k), int(v)) for k, v in type_offsets.items()), key=lambda x: x[1])
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        all_blocks = []
        for graph_idx in range(bs):
            graph_blocks = [[] for _ in templates]
            for type_idx, (_, offset) in enumerate(sorted_types):
                next_offset = sorted_types[type_idx + 1][1] if type_idx + 1 < len(sorted_types) else self.out_dims.X
                nodes = torch.where((batch == graph_idx) & (anchor_node_subtype >= offset) & (anchor_node_subtype < next_offset))[0]
                if nodes.numel() == 0:
                    continue
                nodes = nodes[torch.randperm(nodes.numel(), device=nodes.device)]
                cursor = 0
                remaining_blocks = []
                for block_id, tmpl in enumerate(templates):
                    want = int(tmpl[type_idx]) if type_idx < len(tmpl) else 0
                    take = min(want, max(0, int(nodes.numel()) - cursor))
                    if take > 0:
                        graph_blocks[block_id].append(nodes[cursor:cursor + take])
                        cursor += take
                    remaining_blocks.append(block_id)
                # If node counts differ from the template graph, distribute leftovers without overlap.
                rr = 0
                while cursor < nodes.numel() and remaining_blocks:
                    block_id = remaining_blocks[rr % len(remaining_blocks)]
                    graph_blocks[block_id].append(nodes[cursor:cursor + 1])
                    cursor += 1
                    rr += 1
            for parts in graph_blocks:
                if parts:
                    all_blocks.append(torch.cat(parts, dim=0))
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_pseudo_blocks", False):
            sizes = [int(b.numel()) for b in all_blocks]
            if sizes:
                print(f"[采样] type-template pseudo blocks={len(sizes)} size_min={min(sizes)} size_max={max(sizes)} size_mean={sum(sizes)/len(sizes):.1f}")
            self._logged_sampling_pseudo_blocks = True
        return all_blocks or None

    def _take_inter_block_edges_balanced(
        self,
        deficit,
        block_id,
        pseudo_blocks,
        fam_id,
        src_all,
        dst_all,
        src_block,
        dst_block,
        same_type,
        graph_idx,
        total_nodes,
        inter_state,
    ):
        """Take deterministic, non-overlapping inter-block candidates for one family.

        For a current block b, split its missing quota across b-other block-pair pools
        proportionally to each pool's remaining legal candidate count, e.g. b1-b2 and
        b1-b3 contribute according to |b1*b2| : |b1*b3|. Pools are keyed by canonical
        block-pair + family, and cursors prevent duplicate consumption when the other
        block is visited later in the same diffusion step.
        """
        deficit = int(deficit)
        if deficit <= 0 or not pseudo_blocks:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        pools = []
        pool_cache = None
        sig_cache = None
        if inter_state is not None:
            pool_cache = inter_state.setdefault("_pool_cache", getattr(self, "_inter_block_pool_cache", {}))
            sig_cache = inter_state.setdefault("_block_sigs", {})

        def _block_sig(idx, nodes):
            if sig_cache is not None and int(idx) in sig_cache:
                return sig_cache[int(idx)]
            sig = tuple(sorted(int(x) for x in nodes.detach().cpu().tolist()))
            if sig_cache is not None:
                sig_cache[int(idx)] = sig
            return sig

        current_nodes = pseudo_blocks[int(block_id)].long().to(self.device)
        current_sig = _block_sig(int(block_id), current_nodes)
        src_all_set = None
        dst_all_set = None
        for other_id, other_nodes in enumerate(pseudo_blocks):
            if int(other_id) == int(block_id) or other_nodes is None or other_nodes.numel() == 0:
                continue
            other_nodes = other_nodes.long().to(self.device)
            other_sig = _block_sig(int(other_id), other_nodes)
            pair_key = (int(graph_idx), min(int(block_id), int(other_id)), max(int(block_id), int(other_id)), int(fam_id))
            sig_a, sig_b = (current_sig, other_sig) if int(block_id) <= int(other_id) else (other_sig, current_sig)
            cache_key = (pair_key, int(total_nodes), bool(same_type), sig_a, sig_b)
            if pool_cache is not None and cache_key in pool_cache:
                cand = pool_cache[cache_key]
                if cand.device != self.device:
                    cand = cand.to(self.device)
                    pool_cache[cache_key] = cand
                if inter_state is not None:
                    pool_sizes = inter_state.setdefault("_pool_sizes", {})
                    pool_sizes[pair_key] = max(int(pool_sizes.get(pair_key, 0)), int(cand.shape[1]))
                pools.append((pair_key, cand))
                continue
            if same_type:
                if src_all_set is None:
                    src_all_set = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
                    src_all_set[src_all.long()] = True
                other = other_nodes[src_all_set[other_nodes]]
                if src_block.numel() == 0 or other.numel() == 0:
                    continue
                flat = torch.arange(int(src_block.numel() * other.numel()), device=self.device)
                a = src_block[flat // other.numel()]
                b = other[flat % other.numel()]
                cand = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=0)
                cand = cand[:, cand[0] != cand[1]]
            else:
                if src_all_set is None:
                    src_all_set = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
                    src_all_set[src_all.long()] = True
                    dst_all_set = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
                    dst_all_set[dst_all.long()] = True
                parts = []
                other_dst = other_nodes[dst_all_set[other_nodes]]
                if src_block.numel() > 0 and other_dst.numel() > 0:
                    flat = torch.arange(int(src_block.numel() * other_dst.numel()), device=self.device)
                    parts.append(torch.stack([src_block[flat // other_dst.numel()], other_dst[flat % other_dst.numel()]], dim=0))
                other_src = other_nodes[src_all_set[other_nodes]]
                if other_src.numel() > 0 and dst_block.numel() > 0:
                    flat = torch.arange(int(other_src.numel() * dst_block.numel()), device=self.device)
                    parts.append(torch.stack([other_src[flat // dst_block.numel()], dst_block[flat % dst_block.numel()]], dim=0))
                cand = torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.long, device=self.device)
            if cand.numel() == 0:
                continue
            key_sort = cand[0].long() * int(total_nodes) + cand[1].long()
            order = torch.argsort(key_sort)
            cand = cand[:, order]
            if cand.shape[1] > 1:
                key_sorted = key_sort[order]
                keep = torch.ones_like(key_sorted, dtype=torch.bool)
                keep[1:] = key_sorted[1:] != key_sorted[:-1]
                cand = cand[:, keep]
            if pool_cache is not None:
                pool_cache[cache_key] = cand
            if inter_state is not None:
                pool_sizes = inter_state.setdefault("_pool_sizes", {})
                pool_sizes[pair_key] = max(int(pool_sizes.get(pair_key, 0)), int(cand.shape[1]))
            pools.append((pair_key, cand))
        if not pools:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        pool_infos = []
        total_remaining = 0
        for pair_key, cand in pools:
            cursor = int(inter_state.get(pair_key, 0))
            remaining = max(0, int(cand.shape[1]) - cursor)
            if remaining > 0:
                pool_infos.append({
                    "pair_key": pair_key,
                    "cand": cand,
                    "cursor": cursor,
                    "remaining": remaining,
                    "quota": 0,
                    "frac": 0.0,
                })
                total_remaining += remaining
        if total_remaining <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        target = min(int(deficit), int(total_remaining))
        assigned = 0
        for info in pool_infos:
            exact = float(target) * float(info["remaining"]) / float(total_remaining)
            quota = min(int(math.floor(exact)), int(info["remaining"]))
            info["quota"] = quota
            info["frac"] = exact - float(quota)
            assigned += quota

        # Distribute the rounding remainder to the pools with the largest fractional share.
        leftover = target - assigned
        for info in sorted(pool_infos, key=lambda x: x["frac"], reverse=True):
            if leftover <= 0:
                break
            room = int(info["remaining"]) - int(info["quota"])
            if room <= 0:
                continue
            add = min(room, leftover)
            info["quota"] += add
            leftover -= add

        # If caps still left quota unassigned, fill sequentially from remaining pools.
        if leftover > 0:
            for info in pool_infos:
                if leftover <= 0:
                    break
                room = int(info["remaining"]) - int(info["quota"])
                if room <= 0:
                    continue
                add = min(room, leftover)
                info["quota"] += add
                leftover -= add

        selected = []
        for info in pool_infos:
            quota = int(info["quota"])
            if quota <= 0:
                continue
            start = int(info["cursor"])
            end = start + quota
            selected.append(info["cand"][:, start:end])
            inter_state[info["pair_key"]] = end
        if not selected:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.cat(selected, dim=1)

    def _edge_family_label_ranges(self):
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        ranges = {}
        for fam_name, offset in edge_family_offsets.items():
            offset = int(offset)
            next_offset = int(self.out_dims.E)
            for _, other_offset in edge_family_offsets.items():
                other_offset = int(other_offset)
                if offset < other_offset < next_offset:
                    next_offset = other_offset
            ranges[str(fam_name)] = (max(1, offset), max(1, next_offset))
        return ranges

    def _mask_edge_logits_by_query_family(self, logits, query_edge_family):
        if logits is None or logits.numel() == 0 or query_edge_family is None:
            return logits
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        fam_ranges = self._edge_family_label_ranges()
        if not id2edge_family or not fam_ranges:
            return logits
        qfam = query_edge_family.long().to(logits.device).reshape(-1)
        if qfam.numel() != logits.shape[0]:
            return logits
        unknown = qfam < 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, 0] = True
        if unknown.any():
            mask[unknown, :] = True
        for fam_id, fam_name in id2edge_family.items():
            if fam_name not in fam_ranges:
                continue
            lo, hi = fam_ranges[fam_name]
            rows = qfam == int(fam_id)
            if rows.any() and lo < hi:
                mask[rows, lo:hi] = True
        return logits.masked_fill(~mask, -1e10)

    def _align_query_family_to_edges(self, original_edge_index, original_family, aligned_edge_index):
        if original_family is None or original_edge_index is None or aligned_edge_index is None:
            return original_family
        if original_edge_index.numel() == 0 or aligned_edge_index.numel() == 0:
            return original_family
        if int(original_edge_index.shape[1]) == int(aligned_edge_index.shape[1]) and torch.equal(original_edge_index, aligned_edge_index):
            return original_family.long().to(aligned_edge_index.device)
        # Coalesce can reorder query edges and can drop duplicates already present in clean graph.
        # Rebuild the family vector by endpoint so sampling masks stay aligned with logits.
        edge_to_family = {}
        ou = original_edge_index[0].detach().cpu().tolist()
        ov = original_edge_index[1].detach().cpu().tolist()
        of = original_family.detach().cpu().tolist()
        for u, v, fam in zip(ou, ov, of):
            edge_to_family[(int(u), int(v))] = int(fam)
        au = aligned_edge_index[0].detach().cpu().tolist()
        av = aligned_edge_index[1].detach().cpu().tolist()
        vals = []
        missing = 0
        for u, v in zip(au, av):
            fam = edge_to_family.get((int(u), int(v)))
            if fam is None:
                fam = edge_to_family.get((int(v), int(u)))
            if fam is None:
                missing += 1
                fam = -1
            vals.append(fam)
        out = torch.tensor(vals, dtype=torch.long, device=aligned_edge_index.device)
        if missing > 0 and getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_query_family_align_missing", False):
            print(f"[采样] warning: missing family for {missing}/{len(vals)} coalesced query edges; fallback to unmasked sampling for them")
            self._logged_query_family_align_missing = True
        return out

    def _select_original_query_outputs(self, original_edge_index, original_family, comp_query_edge_index, comp_query_logits):
        """Select logits for the original canonical query edges from a bidirectional MP query set."""
        if original_edge_index is None or comp_query_edge_index is None or comp_query_logits is None:
            return original_edge_index, original_family, comp_query_logits
        if original_edge_index.numel() == 0 or comp_query_edge_index.numel() == 0 or comp_query_logits.numel() == 0:
            return original_edge_index, original_family, comp_query_logits
        if (
            int(original_edge_index.shape[1]) == int(comp_query_edge_index.shape[1])
            and torch.equal(original_edge_index, comp_query_edge_index)
        ):
            fam = original_family.long().to(comp_query_edge_index.device) if original_family is not None else None
            return original_edge_index, fam, comp_query_logits

        comp_map = {}
        cu = comp_query_edge_index[0].detach().cpu().tolist()
        cv = comp_query_edge_index[1].detach().cpu().tolist()
        for idx, (u, v) in enumerate(zip(cu, cv)):
            comp_map.setdefault((int(u), int(v)), int(idx))

        keep_edges = []
        keep_logits = []
        keep_family = []
        ou = original_edge_index[0].detach().cpu().tolist()
        ov = original_edge_index[1].detach().cpu().tolist()
        of = original_family.detach().cpu().tolist() if original_family is not None else [None] * len(ou)
        missing = 0
        for u, v, fam in zip(ou, ov, of):
            idx = comp_map.get((int(u), int(v)))
            if idx is None:
                missing += 1
                continue
            keep_edges.append([int(u), int(v)])
            keep_logits.append(comp_query_logits[idx:idx + 1])
            if fam is not None:
                keep_family.append(int(fam))
        if not keep_logits:
            empty_edges = torch.empty((2, 0), dtype=torch.long, device=comp_query_edge_index.device)
            empty_logits = comp_query_logits[:0]
            empty_family = torch.empty((0,), dtype=torch.long, device=comp_query_edge_index.device) if original_family is not None else None
            return empty_edges, empty_family, empty_logits
        selected_edge_index = torch.tensor(keep_edges, dtype=torch.long, device=comp_query_edge_index.device).t().contiguous()
        selected_logits = torch.cat(keep_logits, dim=0)
        selected_family = None
        if original_family is not None:
            selected_family = torch.tensor(keep_family, dtype=torch.long, device=comp_query_edge_index.device)
        if missing > 0 and getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_query_output_missing", False):
            print(f"[采样] warning: missing logits for {missing}/{len(ou)} original query edges after MP coalesce; skipped them")
            self._logged_query_output_missing = True
        return selected_edge_index, selected_family, selected_logits

    def _log_sampling_family_exist_rates(self, exist_prob, sampled_labels, query_edge_family):
        if query_edge_family is None or getattr(self, "_logged_sampling_family_exist_rates", False):
            return
        if exist_prob is None or sampled_labels is None or exist_prob.numel() == 0:
            return
        qfam = query_edge_family.long().to(exist_prob.device).reshape(-1)
        if qfam.numel() != exist_prob.shape[0]:
            return
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        if not id2edge_family:
            return
        print("[采样-FAM] family-wise existence rates for first sampled query round")
        for fam_id, fam_name in sorted(id2edge_family.items(), key=lambda kv: kv[1]):
            mask = qfam == int(fam_id)
            if not mask.any():
                continue
            mean_p = float(exist_prob[mask].mean().detach().cpu())
            sampled_rate = float((sampled_labels[mask] > 0).float().mean().detach().cpu())
            real_density = None
            ratio = None
            if fam_name in marginals:
                m = marginals[fam_name]
                try:
                    if not torch.is_tensor(m):
                        m = torch.tensor(m)
                    if m.numel() > 0:
                        real_density = max(0.0, min(1.0, 1.0 - float(m.reshape(-1)[0].detach().cpu().item())))
                        ratio = sampled_rate / max(real_density, 1e-12)
                except Exception:
                    real_density = None
                    ratio = None
            if real_density is None:
                print(
                    f"[采样-FAM] {fam_name}: query={int(mask.sum().item())} "
                    f"mean_p={mean_p:.6f} sampled={sampled_rate:.6f} real_density=NA ratio=NA"
                )
            else:
                print(
                    f"[采样-FAM] {fam_name}: query={int(mask.sum().item())} "
                    f"mean_p={mean_p:.6f} sampled={sampled_rate:.6f} "
                    f"real_density={real_density:.6f} ratio={ratio:.2f}"
                )
        self._logged_sampling_family_exist_rates = True

    def _label_to_family_lookup(self):
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        lookup = torch.full((int(self.out_dims.E),), -1, dtype=torch.long)
        if not edge_family_offsets or not edge_family2id:
            return lookup
        fam_offsets_sorted = sorted((int(v), str(k)) for k, v in edge_family_offsets.items())
        for off_idx, (offset, fam_name) in enumerate(fam_offsets_sorted):
            next_offset = fam_offsets_sorted[off_idx + 1][0] if off_idx + 1 < len(fam_offsets_sorted) else int(self.out_dims.E)
            fam_id = int(edge_family2id.get(fam_name, -1))
            if fam_id >= 0 and offset < next_offset:
                lookup[max(0, offset):max(0, next_offset)] = fam_id
        return lookup

    def _get_reference_graph_for_sampling_stats(self):
        source = None
        if hasattr(self.dataset_info, "datamodule"):
            dm = self.dataset_info.datamodule
            for attr in ("train_dataset", "test_dataset", "val_dataset"):
                ds = getattr(dm, attr, None)
                if ds is not None and len(ds) > 0:
                    source = ds[0]
                    break
        if source is None:
            return None
        try:
            source = source.clone()
        except Exception:
            pass
        try:
            source = self.dataset_info.to_one_hot(source)
        except Exception:
            pass
        return source

    def _ensure_sampling_degree_pair_reference(self):
        if getattr(self, "_sampling_degree_pair_reference", None) is not None:
            return self._sampling_degree_pair_reference
        source = self._get_reference_graph_for_sampling_stats()
        if source is None or not hasattr(source, "edge_index") or not hasattr(source, "edge_attr"):
            self._sampling_degree_pair_reference = {}
            return self._sampling_degree_pair_reference
        edge_index = source.edge_index.detach().cpu().long()
        edge_attr = source.edge_attr.detach().cpu()
        if edge_attr.dim() > 1:
            labels = edge_attr.argmax(dim=-1).long()
        else:
            labels = edge_attr.long().reshape(-1)
        n_nodes = int(source.x.shape[0] if hasattr(source, "x") else source.node.shape[0])
        label_to_fam = self._label_to_family_lookup().cpu()
        if edge_index.numel() == 0 or labels.numel() == 0 or n_nodes <= 0:
            self._sampling_degree_pair_reference = {}
            return self._sampling_degree_pair_reference

        # Use unique undirected positive edges for degree bins and family pair statistics.
        seen = set()
        edges = []
        fams = []
        for col in range(int(edge_index.shape[1])):
            u = int(edge_index[0, col].item())
            v = int(edge_index[1, col].item())
            if u == v:
                continue
            label = int(labels[col].item())
            if label <= 0 or label >= int(label_to_fam.numel()):
                continue
            fam = int(label_to_fam[label].item())
            if fam < 0:
                continue
            a, b = (u, v) if u < v else (v, u)
            key = (a, b, fam)
            if key in seen:
                continue
            seen.add(key)
            edges.append((a, b))
            fams.append(fam)

        degree = torch.zeros(n_nodes, dtype=torch.float32)
        for u, v in edges:
            degree[u] += 1.0
            degree[v] += 1.0
        num_bins = max(2, int(getattr(self.cfg.model, "sampling_degree_pair_bins", 5) or 5))
        if n_nodes > 1:
            qs = torch.linspace(0, 1, steps=num_bins + 1)[1:-1]
            boundaries = torch.quantile(degree, qs).unique(sorted=True)
        else:
            boundaries = torch.empty((0,), dtype=torch.float32)
        degree_bins = torch.bucketize(degree, boundaries).long()

        pair_counts = {}
        family_totals = {}
        for (u, v), fam in zip(edges, fams):
            bu = int(degree_bins[u].item())
            bv = int(degree_bins[v].item())
            a, b = (bu, bv) if bu <= bv else (bv, bu)
            fam = int(fam)
            d = pair_counts.setdefault(fam, {})
            d[(a, b)] = int(d.get((a, b), 0)) + 1
            family_totals[fam] = int(family_totals.get(fam, 0)) + 1
        ref = {
            "degree": degree,
            "degree_bins": degree_bins,
            "boundaries": boundaries,
            "pair_counts": pair_counts,
            "family_totals": family_totals,
        }
        self._sampling_degree_pair_reference = ref
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_degree_pair_reference", False):
            print(
                f"[采样-DEGPAIR] cached reference degree-pair bins={num_bins} "
                f"families={len(pair_counts)} edges={len(edges)} boundaries={boundaries.tolist()}"
            )
            self._logged_sampling_degree_pair_reference = True
        return ref

    def _family_density_from_marginal(self, fam_name, marginals):
        if fam_name not in marginals:
            return None
        m = marginals[fam_name]
        try:
            if not torch.is_tensor(m):
                m = torch.tensor(m)
            if m.numel() > 0:
                return max(0.0, min(1.0, 1.0 - float(m.reshape(-1)[0].detach().cpu().item())))
        except Exception:
            return None
        return None

    def _sample_topk_density_by_family(self, exist_prob, has_valid_pos, query_edge_family):
        has_edge = torch.zeros_like(exist_prob, dtype=torch.bool)
        qfam = query_edge_family.long().to(exist_prob.device).reshape(-1)
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        used_mask = torch.zeros_like(has_edge)
        if qfam.numel() == exist_prob.shape[0] and id2edge_family:
            for fam_id, fam_name in id2edge_family.items():
                mask = (qfam == int(fam_id)) & has_valid_pos
                n = int(mask.sum().item())
                if n <= 0:
                    continue
                density = self._family_density_from_marginal(fam_name, marginals)
                if density is None:
                    local_idx = mask.nonzero(as_tuple=True)[0]
                    has_edge[local_idx] = torch.rand_like(exist_prob[local_idx]) < exist_prob[local_idx]
                else:
                    k = int(round(float(density) * n))
                    k = max(0, min(k, n))
                    if k > 0:
                        local_idx = mask.nonzero(as_tuple=True)[0]
                        top_local = torch.topk(exist_prob[local_idx], k=k, largest=True).indices
                        has_edge[local_idx[top_local]] = True
                used_mask |= mask
        fallback = (~used_mask) & has_valid_pos
        if fallback.any():
            has_edge[fallback] = torch.rand_like(exist_prob[fallback]) < exist_prob[fallback]
        return has_edge

    def _sample_topk_structure_by_family(self, exist_prob, has_valid_pos, query_edge_family):
        """Structure-aware topk sampling per family WITHOUT true graph statistics.

        Uses the model's own predicted exist_prob as a structure signal:
        - Per-family quota = edge_fraction * mean_p_exist * n_query  (adaptive, no true marginals)
        - Within each family, topk edges by exist_prob.
        """
        has_edge = torch.zeros_like(exist_prob, dtype=torch.bool)
        qfam = query_edge_family.long().to(exist_prob.device).reshape(-1)
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        used_mask = torch.zeros_like(has_edge)
        base_frac = float(self.edge_fraction)
        if qfam.numel() == exist_prob.shape[0] and id2edge_family:
            for fam_id, fam_name in id2edge_family.items():
                mask = (qfam == int(fam_id)) & has_valid_pos
                n = int(mask.sum().item())
                if n <= 0:
                    continue
                mean_conf = float(exist_prob[mask].mean().item())
                k = int(round(base_frac * max(mean_conf, 1e-6) * n))
                k = max(1, min(k, n))
                local_idx = mask.nonzero(as_tuple=True)[0]
                top_local = torch.topk(exist_prob[local_idx], k=k, largest=True).indices
                has_edge[local_idx[top_local]] = True
                used_mask |= mask
        fallback = (~used_mask) & has_valid_pos
        if fallback.any():
            has_edge[fallback] = torch.rand_like(exist_prob[fallback]) < exist_prob[fallback]
        return has_edge

    def _sample_topk_degree_pair_by_family(self, exist_prob, has_valid_pos, query_edge_family, query_edge_index):
        if query_edge_index is None or query_edge_index.numel() == 0:
            return self._sample_topk_density_by_family(exist_prob, has_valid_pos, query_edge_family)
        ref = self._ensure_sampling_degree_pair_reference()
        degree_bins = ref.get("degree_bins") if ref else None
        pair_counts = ref.get("pair_counts", {}) if ref else {}
        family_totals = ref.get("family_totals", {}) if ref else {}
        if degree_bins is None or degree_bins.numel() == 0 or not pair_counts:
            return self._sample_topk_density_by_family(exist_prob, has_valid_pos, query_edge_family)

        has_edge = torch.zeros_like(exist_prob, dtype=torch.bool)
        qfam = query_edge_family.long().to(exist_prob.device).reshape(-1)
        qedge = query_edge_index.long().to(exist_prob.device)
        if qfam.numel() != exist_prob.shape[0] or qedge.shape[1] != exist_prob.shape[0]:
            return self._sample_topk_density_by_family(exist_prob, has_valid_pos, query_edge_family)

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        dbins = degree_bins.to(exist_prob.device)
        max_node = int(dbins.numel())
        valid_endpoint = (qedge[0] >= 0) & (qedge[0] < max_node) & (qedge[1] >= 0) & (qedge[1] < max_node)
        bin_u = torch.zeros_like(qfam)
        bin_v = torch.zeros_like(qfam)
        if valid_endpoint.any():
            bin_u[valid_endpoint] = dbins[qedge[0, valid_endpoint]]
            bin_v[valid_endpoint] = dbins[qedge[1, valid_endpoint]]
        pair_a = torch.minimum(bin_u, bin_v)
        pair_b = torch.maximum(bin_u, bin_v)

        used_mask = torch.zeros_like(has_edge)
        log_parts = []
        min_bin_edges = max(0, int(getattr(self.cfg.model, "sampling_degree_pair_min_bin_edges", 1) or 1))
        if id2edge_family:
            for fam_id, fam_name in id2edge_family.items():
                fam_id = int(fam_id)
                mask = (qfam == fam_id) & has_valid_pos & valid_endpoint
                n = int(mask.sum().item())
                if n <= 0:
                    continue
                density = self._family_density_from_marginal(fam_name, marginals)
                if density is None or fam_id not in pair_counts or family_totals.get(fam_id, 0) <= 0:
                    local_idx = mask.nonzero(as_tuple=True)[0]
                    has_edge[local_idx] = torch.rand_like(exist_prob[local_idx]) < exist_prob[local_idx]
                    used_mask |= mask
                    continue
                target = int(round(float(density) * n))
                target = max(0, min(target, n))
                strength = float(getattr(self.cfg.model, "sampling_degree_pair_strength", 1.0) or 1.0)
                strength = max(0.0, min(1.0, strength))
                pair_target = int(round(float(target) * strength))
                pair_target = max(0, min(pair_target, target))
                local_selected = torch.zeros_like(has_edge)
                assigned = 0
                pair_infos = []
                fam_total = float(max(1, int(family_totals.get(fam_id, 0))))
                for pair_key, cnt in pair_counts.get(fam_id, {}).items():
                    a, b = int(pair_key[0]), int(pair_key[1])
                    pmask = mask & (pair_a == a) & (pair_b == b)
                    pn = int(pmask.sum().item())
                    if pn <= 0:
                        continue
                    exact = float(pair_target) * float(cnt) / fam_total
                    quota = int(math.floor(exact))
                    if pair_target > 0 and cnt >= min_bin_edges and exact > 0.0:
                        quota = max(1, quota)
                    quota = max(0, min(quota, pn))
                    pair_infos.append({"mask": pmask, "quota": quota, "frac": exact - math.floor(exact), "n": pn})
                    assigned += quota
                # If forced minimums over-assigned tiny buckets, trim from lowest fractional quotas.
                if assigned > pair_target:
                    excess = assigned - pair_target
                    for info in sorted(pair_infos, key=lambda x: x["frac"]):
                        if excess <= 0:
                            break
                        drop = min(int(info["quota"]), excess)
                        info["quota"] -= drop
                        excess -= drop
                    assigned = pair_target
                for info in pair_infos:
                    quota = int(info["quota"])
                    if quota <= 0:
                        continue
                    local_idx = info["mask"].nonzero(as_tuple=True)[0]
                    take = min(quota, int(local_idx.numel()))
                    if take > 0:
                        top_local = torch.topk(exist_prob[local_idx], k=take, largest=True).indices
                        local_selected[local_idx[top_local]] = True
                selected_count = int(local_selected.sum().item())
                if selected_count < target:
                    remaining = mask & (~local_selected)
                    rn = int(remaining.sum().item())
                    add = min(target - selected_count, rn)
                    if add > 0:
                        ridx = remaining.nonzero(as_tuple=True)[0]
                        top_local = torch.topk(exist_prob[ridx], k=add, largest=True).indices
                        local_selected[ridx[top_local]] = True
                has_edge |= local_selected
                used_mask |= mask
                if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_degree_pair_rates", False):
                    log_parts.append(f"{fam_name}:target={target},pair_target={pair_target},selected={int(local_selected.sum().item())},pairs={len(pair_infos)}")
        fallback = (~used_mask) & has_valid_pos
        if fallback.any():
            has_edge[fallback] = torch.rand_like(exist_prob[fallback]) < exist_prob[fallback]
        if getattr(self, "local_rank", 0) == 0 and log_parts and not getattr(self, "_logged_sampling_degree_pair_rates", False):
            print("[采样-DEGPAIR] " + "; ".join(log_parts[:10]))
            self._logged_sampling_degree_pair_rates = True
        return has_edge

    def _sample_edge_labels_hierarchical(self, logits, query_edge_family=None, query_edge_index=None):
        """Sample edge labels with the same existence/subtype factorization used in training."""
        if logits is None or logits.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        logits = self._mask_edge_logits_by_query_family(logits, query_edge_family)
        no_edge_logit = logits[:, 0]
        pos_logits = logits[:, 1:]
        has_valid_pos = torch.isfinite(pos_logits).any(dim=-1)
        pos_logsum = torch.logsumexp(pos_logits, dim=-1)
        exist_logits = pos_logsum - no_edge_logit
        temp = float(getattr(self.cfg.model, "sampling_exist_temperature", 1.0) or 1.0)
        temp = max(temp, 1e-6)
        bias = float(getattr(self.cfg.model, "sampling_exist_logit_bias", 0.0) or 0.0)
        exist_logits = exist_logits / temp - bias
        exist_prob = torch.sigmoid(exist_logits).clamp(min=0.0, max=1.0)
        selection = str(getattr(self.cfg.model, "sampling_edge_selection", "bernoulli") or "bernoulli").lower()
        has_edge = torch.zeros_like(exist_prob, dtype=torch.bool)
        if selection == "topk_density" and query_edge_family is not None:
            has_edge = self._sample_topk_density_by_family(exist_prob, has_valid_pos, query_edge_family)
        elif selection == "topk_degree_pair" and query_edge_family is not None:
            has_edge = self._sample_topk_degree_pair_by_family(
                exist_prob, has_valid_pos, query_edge_family, query_edge_index
            )
        elif selection == "topk_structure" and query_edge_family is not None:
            has_edge = self._sample_topk_structure_by_family(exist_prob, has_valid_pos, query_edge_family)
        else:
            has_edge = (torch.rand_like(exist_prob) < exist_prob) & has_valid_pos
        out = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        idx = has_edge.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            sub_logits = pos_logits[idx]
            sub = torch.softmax(sub_logits, dim=-1).multinomial(1).flatten()
            out[idx] = sub + 1
        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_exist_rate", False):
            print(
                f"[采样] hierarchical edge sampling: query={int(logits.shape[0])} "
                f"mode={selection} bias={bias:.3f} temp={temp:.3f} "
                f"mean_p_exist={float(exist_prob.mean().detach().cpu()):.6f} "
                f"sampled_pos_rate={float((out > 0).float().mean().detach().cpu()):.6f}"
            )
            self._logged_sampling_exist_rate = True
        if getattr(self, "local_rank", 0) == 0:
            self._log_sampling_family_exist_rates(exist_prob, out, query_edge_family)
        return out

    def _log_block_query_coverage(self, inter_state, prefix="[BLOCK-COVER]"):
        if not inter_state or getattr(self, "_logged_block_query_coverage", False):
            return
        stats = inter_state.get("_coverage_stats", {})
        if not stats:
            return
        pool_sizes = inter_state.get("_pool_sizes", {})
        remaining_by_fam = {}
        pool_total_by_fam = {}
        for pair_key, size in pool_sizes.items():
            if not isinstance(pair_key, tuple) or len(pair_key) != 4:
                continue
            fam_id = int(pair_key[3])
            size = int(size)
            cursor = int(inter_state.get(pair_key, 0))
            pool_total_by_fam[fam_id] = pool_total_by_fam.get(fam_id, 0) + size
            remaining_by_fam[fam_id] = remaining_by_fam.get(fam_id, 0) + max(0, size - cursor)

        pool_cache = inter_state.get("_pool_cache", {}) if isinstance(inter_state, dict) else {}
        print(f"{prefix} edge_fraction={float(self.edge_fraction):.4f} families={len(stats)} pool_cache_entries={len(pool_cache)}")
        for (graph_idx, fam_id), rec in sorted(stats.items(), key=lambda kv: (kv[0][0], str(kv[1].get("fam_name", kv[0][1])))):
            actual_possible = max(1, int(rec.get("actual_possible", 0)))
            formula_possible = int(rec.get("formula_possible", 0))
            target = int(rec.get("target", 0))
            intra = int(rec.get("intra", 0))
            inter = int(rec.get("inter", 0))
            selected = int(rec.get("selected", intra + inter))
            remain = int(remaining_by_fam.get(int(fam_id), 0))
            pool_total = int(pool_total_by_fam.get(int(fam_id), 0))
            coverage = float(selected) / float(actual_possible)
            target_ratio = float(target) / float(actual_possible)
            print(
                f"{prefix} graph={int(graph_idx)} family={rec.get('fam_name', fam_id)} "
                f"blocks={int(rec.get('blocks', 0))} selected={selected} "
                f"intra={intra} inter={inter} remain_inter={remain}/{pool_total} "
                f"target={target} actual_possible={actual_possible} formula_possible={formula_possible} "
                f"coverage={coverage:.4f} target_ratio={target_ratio:.4f}"
            )
        self._logged_block_query_coverage = True

    def _sampling_type_template_query_edges(self, anchor_node_subtype, batch, pseudo_blocks, block_ids=None, inter_state=None, all_blocks=None):
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
        id2edge_family = {v: k for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
        type_offsets = getattr(self.dataset_info, "type_offsets", {})
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
        edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        if not (id2edge_family and fam_endpoints and type_offsets and edge_family_offsets and pseudo_blocks):
            return None, None, None
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[type_name] = int(next_offset - offset)
        all_e, all_b, all_f = [], [], []
        total_nodes = int(anchor_node_subtype.shape[0])
        if block_ids is None:
            block_ids = list(range(len(pseudo_blocks)))
        if inter_state is None:
            inter_state = {}
        inter_blocks = all_blocks if all_blocks is not None else pseudo_blocks
        for block_pos, block_nodes in enumerate(pseudo_blocks):
            block_id = int(block_ids[block_pos]) if block_pos < len(block_ids) else int(block_pos)
            if block_nodes.numel() == 0:
                continue
            graph_idx = int(batch[block_nodes[0]].item()) if batch.numel() else 0
            block_mask = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
            block_mask[block_nodes.long()] = True
            batch_mask = batch == graph_idx
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints or edge_family_avg_edge_counts.get(fam_name, 0) == 0:
                    continue
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                if src_type not in type_offsets or dst_type not in type_offsets:
                    continue
                src_offset = int(type_offsets[src_type]); dst_offset = int(type_offsets[dst_type])
                src_size = int(type_sizes.get(src_type, 0)); dst_size = int(type_sizes.get(dst_type, 0))
                src_all = torch.where((anchor_node_subtype >= src_offset) & (anchor_node_subtype < src_offset + src_size) & batch_mask)[0]
                dst_all = torch.where((anchor_node_subtype >= dst_offset) & (anchor_node_subtype < dst_offset + dst_size) & batch_mask)[0]
                if src_all.numel() == 0 or dst_all.numel() == 0:
                    continue
                same_type = src_type == dst_type
                actual_possible = int(src_all.numel() * (src_all.numel() - 1) // 2) if same_type else int(src_all.numel() * dst_all.numel())
                num_possible = actual_possible
                if num_possible <= 0:
                    continue
                num_query = min(num_possible, max(1, int(math.ceil(float(self.edge_fraction) * num_possible))))
                src_block = src_all[block_mask[src_all]]
                dst_block = dst_all[block_mask[dst_all]]
                if src_block.numel() == 0 or dst_block.numel() == 0:
                    continue
                if same_type:
                    pairs = torch.triu_indices(src_block.numel(), src_block.numel(), offset=1, device=self.device)
                    if pairs.shape[1] == 0:
                        continue
                    selected = torch.stack([src_block[pairs[0]], src_block[pairs[1]]], dim=0)
                else:
                    flat_count = int(src_block.numel() * dst_block.numel())
                    flat = torch.arange(flat_count, device=self.device)
                    selected = torch.stack([src_block[flat // dst_block.numel()], dst_block[flat % dst_block.numel()]], dim=0)
                coverage_rec = None
                if inter_state is not None:
                    coverage_stats = inter_state.setdefault("_coverage_stats", {})
                    coverage_key = (int(graph_idx), int(fam_id))
                    coverage_rec = coverage_stats.setdefault(coverage_key, {
                        "fam_name": str(fam_name),
                        "formula_possible": int(num_possible),
                        "actual_possible": int(actual_possible),
                        "target": 0,
                        "intra": 0,
                        "inter": 0,
                        "selected": 0,
                        "blocks": 0,
                    })
                    coverage_rec["formula_possible"] = max(int(coverage_rec.get("formula_possible", 0)), int(num_possible))
                    coverage_rec["actual_possible"] = max(int(coverage_rec.get("actual_possible", 0)), int(actual_possible))
                    coverage_rec["target"] += int(num_query)
                    coverage_rec["intra"] += int(selected.shape[1])
                    coverage_rec["blocks"] += 1

                if bool(getattr(self.cfg.model, "block_query_inter_fill", False)):
                    deficit = int(num_query) - int(selected.shape[1])
                    if deficit > 0:
                        inter = self._take_inter_block_edges_balanced(
                            deficit=deficit,
                            block_id=block_id,
                            pseudo_blocks=inter_blocks,
                            fam_id=fam_id,
                            src_all=src_all,
                            dst_all=dst_all,
                            src_block=src_block,
                            dst_block=dst_block,
                            same_type=same_type,
                            graph_idx=graph_idx,
                            total_nodes=total_nodes,
                            inter_state=inter_state,
                        )
                        if inter.shape[1] > 0:
                            if coverage_rec is not None:
                                coverage_rec["inter"] += int(inter.shape[1])
                            selected = torch.cat([selected, inter], dim=1)
                if coverage_rec is not None:
                    coverage_rec["selected"] += int(selected.shape[1])
                if selected.shape[1] == 0:
                    continue
                all_e.append(selected)
                all_b.append(torch.full((selected.shape[1],), graph_idx, dtype=torch.long, device=self.device))
                all_f.append(torch.full((selected.shape[1],), int(fam_id), dtype=torch.long, device=self.device))
        if not all_e:
            return None, None, None
        return torch.cat(all_e, dim=1), torch.cat(all_b, dim=0), torch.cat(all_f, dim=0)

    def _heterogeneous_global_block_query_edges(self, data, sparse_noisy_data, forced_block_id=None):
        """Sample queries from one shared hetero hard block across all families.

        This is the block-query mode used by ``block_partition_mode=hetero_metis``:
        partition the true heterogeneous graph once per graph, select one block,
        enumerate relation-legal candidates inside that same block, then optionally
        fill from block-to-other candidates up to the original edge_fraction budget.
        """
        from sparse_diffusion.graph_partition.connected_blocks import hetero_metis_blocks_from_graph

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
        id2edge_family = {v: k for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
        type_offsets = getattr(self.dataset_info, "type_offsets", {})
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
        if not (id2edge_family and fam_endpoints and type_offsets and edge_family_offsets):
            return None, None

        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[type_name] = int(next_offset - offset)

        node_t = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x
        edge_attr_discrete = data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1 else data.edge_attr
        bs = int(data.batch.max() + 1)
        rho = self._rho_for_partition()
        rel_balance_power = float(getattr(self.cfg.model, "hetero_metis_relation_balance_power", 0.5))
        refine_degree_balance = bool(getattr(self.cfg.model, "hetero_metis_refine_degree_balance", False))
        refine_max_iter = int(getattr(self.cfg.model, "hetero_metis_refine_max_iter", 200) or 0)
        all_query_edge_index = []
        all_query_edge_batch = []

        fam_offsets_sorted = sorted((int(v), str(k)) for k, v in edge_family_offsets.items())

        for graph_idx in range(bs):
            batch_nodes = torch.where(data.batch == graph_idx)[0]
            if batch_nodes.numel() == 0:
                continue
            graph_edge_mask = data.batch[data.edge_index[0]] == graph_idx
            graph_edges = data.edge_index[:, graph_edge_mask]
            graph_edge_attr = edge_attr_discrete[graph_edge_mask]
            local_lookup = {int(n.item()): j for j, n in enumerate(batch_nodes)}
            local_edges = []
            local_family = []
            for col in range(graph_edges.shape[1]):
                s = int(graph_edges[0, col].item())
                d = int(graph_edges[1, col].item())
                if s not in local_lookup or d not in local_lookup:
                    continue
                label = int(graph_edge_attr[col].item())
                fam_id = 0
                for off_idx, (offset, fam_name) in enumerate(fam_offsets_sorted):
                    next_offset = fam_offsets_sorted[off_idx + 1][0] if off_idx + 1 < len(fam_offsets_sorted) else self.out_dims.E
                    if offset <= label < next_offset:
                        fam_id = int(edge_family2id.get(fam_name, 0))
                        break
                local_edges.append((local_lookup[s], local_lookup[d]))
                local_family.append(fam_id)
            if local_edges:
                local_edge_index = torch.tensor(local_edges, dtype=torch.long, device=self.device).t().contiguous()
                local_edge_family = torch.tensor(local_family, dtype=torch.long, device=self.device)
            else:
                local_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                local_edge_family = torch.empty((0,), dtype=torch.long, device=self.device)

            cache_key = (
                "hetero_metis",
                int(graph_idx),
                int(batch_nodes.numel()),
                int(local_edge_index.shape[1]),
                round(float(rho), 8),
                round(float(rel_balance_power), 8),
                bool(refine_degree_balance),
                int(refine_max_iter),
            )
            blocks = self._true_block_cache.get(cache_key)
            if blocks is not None and getattr(self, "_hetero_block_edge_templates", None) is None:
                self._cache_hetero_block_edge_templates(blocks, local_edge_index, local_edge_family)
            if blocks is None:
                blocks = hetero_metis_blocks_from_graph(
                    local_edge_index,
                    local_edge_family,
                    int(batch_nodes.numel()),
                    rho,
                    relation_balance_power=rel_balance_power,
                    node_type_local=node_t[batch_nodes],
                    refine_degree_balance=refine_degree_balance,
                    refine_max_iter=refine_max_iter,
                )
                if not blocks:
                    blocks = [list(range(int(batch_nodes.numel())))]
                self._true_block_cache[cache_key] = blocks
                self._cache_hetero_block_templates(blocks, batch_nodes, node_t, type_offsets)
                self._cache_hetero_block_edge_templates(blocks, local_edge_index, local_edge_family)
                if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_hetero_metis_blocks", False):
                    self._log_hetero_metis_block_summary(
                        blocks,
                        batch_nodes,
                        local_edge_index,
                        local_edge_family,
                        node_t,
                        type_offsets,
                        id2edge_family,
                        graph_idx,
                        rel_balance_power,
                    )
                    self._logged_hetero_metis_blocks = True

            if forced_block_id is None:
                block_id = int(torch.randint(len(blocks), (1,), device=self.device).item())
            else:
                block_id = int(forced_block_id) % max(1, len(blocks))
            block = blocks[block_id]
            all_block_nodes = [
                batch_nodes[torch.tensor(b, dtype=torch.long, device=self.device)]
                for b in blocks
            ]
            block_nodes = all_block_nodes[block_id]
            train_inter_state = {}
            block_mask = torch.zeros(data.x.shape[0], dtype=torch.bool, device=self.device)
            block_mask[block_nodes] = True
            full_block_query = bool(getattr(self.cfg.model, "block_query_full_block", False))
            do_inter_fill = bool(getattr(self.cfg.model, "block_query_inter_fill", False))
            max_block_edges = int(getattr(self.cfg.model, "block_query_max_edges_per_family", 0) or 0)

            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                if src_type not in type_offsets or dst_type not in type_offsets:
                    continue
                fam_offset = int(edge_family_offsets.get(fam_name, 0))
                next_fam_offset = self.out_dims.E
                for _, other_offset in edge_family_offsets.items():
                    other_offset = int(other_offset)
                    if fam_offset < other_offset < next_fam_offset:
                        next_fam_offset = other_offset
                fam_edge_mask = (edge_attr_discrete >= fam_offset) & (edge_attr_discrete < next_fam_offset)
                if not fam_edge_mask.any():
                    continue

                src_offset = int(type_offsets[src_type])
                dst_offset = int(type_offsets[dst_type])
                src_size = int(type_sizes.get(src_type, 0))
                dst_size = int(type_sizes.get(dst_type, 0))
                batch_mask = data.batch == graph_idx
                src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size) & batch_mask
                dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size) & batch_mask
                batch_src_nodes = torch.where(src_mask)[0]
                batch_dst_nodes = torch.where(dst_mask)[0]
                if batch_src_nodes.numel() == 0 or batch_dst_nodes.numel() == 0:
                    continue
                same_type = src_type == dst_type
                if same_type:
                    num_possible = int(batch_src_nodes.numel() * (batch_src_nodes.numel() - 1) // 2)
                else:
                    num_possible = int(batch_src_nodes.numel() * batch_dst_nodes.numel())
                if num_possible <= 0:
                    continue
                num_query = min(num_possible, max(1, int(math.ceil(float(self.edge_fraction) * num_possible))))

                block_src_nodes = batch_src_nodes[block_mask[batch_src_nodes]]
                block_dst_nodes = batch_dst_nodes[block_mask[batch_dst_nodes]]
                if block_src_nodes.numel() == 0 or block_dst_nodes.numel() == 0:
                    continue

                if same_type:
                    local_pairs = torch.triu_indices(block_src_nodes.numel(), block_src_nodes.numel(), offset=1, device=self.device)
                    if local_pairs.shape[1] == 0:
                        continue
                    candidate_edges = torch.stack([block_src_nodes[local_pairs[0]], block_src_nodes[local_pairs[1]]], dim=0)
                else:
                    flat_count = int(block_src_nodes.numel() * block_dst_nodes.numel())
                    if flat_count <= 0:
                        continue
                    flat = torch.arange(flat_count, device=self.device)
                    candidate_edges = torch.stack(
                        [block_src_nodes[flat // block_dst_nodes.numel()], block_dst_nodes[flat % block_dst_nodes.numel()]],
                        dim=0,
                    )
                if candidate_edges.shape[1] == 0:
                    continue
                if full_block_query:
                    selected = candidate_edges
                else:
                    take = min(num_query, int(candidate_edges.shape[1]))
                    selected = candidate_edges[:, torch.randperm(candidate_edges.shape[1], device=self.device)[:take]]

                if full_block_query and do_inter_fill:
                    deficit = int(num_query) - int(selected.shape[1])
                    if deficit > 0:
                        inter_edges = self._take_inter_block_edges_balanced(
                            deficit=deficit,
                            block_id=block_id,
                            pseudo_blocks=all_block_nodes,
                            fam_id=fam_id,
                            src_all=batch_src_nodes,
                            dst_all=batch_dst_nodes,
                            src_block=block_src_nodes,
                            dst_block=block_dst_nodes,
                            same_type=same_type,
                            graph_idx=graph_idx,
                            total_nodes=int(data.x.shape[0]),
                            inter_state=train_inter_state,
                        )
                        if inter_edges.shape[1] > 0:
                            selected = torch.cat([selected, inter_edges], dim=1)


                if max_block_edges > 0 and selected.shape[1] > max_block_edges:
                    selected = selected[:, torch.randperm(selected.shape[1], device=self.device)[:max_block_edges]]
                all_query_edge_index.append(selected)
                all_query_edge_batch.append(torch.full((selected.shape[1],), graph_idx, dtype=torch.long, device=self.device))

        if not all_query_edge_index:
            return None, None
        return torch.cat(all_query_edge_index, dim=1), torch.cat(all_query_edge_batch, dim=0)

    def _heterogeneous_block_query_edges(self, data, sparse_noisy_data, forced_block_id=None):
        if self.block_partition_mode == "hetero_metis":
            return self._heterogeneous_global_block_query_edges(data, sparse_noisy_data, forced_block_id=forced_block_id)

        """Sample one relation-aware local block per graph/family.

        The original SparseDiff-block condensed index is homogeneous and upper-triangular.
        DiHuG keeps the same training idea but scopes candidates to each relation family:
        (src_type, family, dst_type). This prevents query edges from crossing invalid
        heterogeneous endpoint spaces.
        """
        from sparse_diffusion.graph_partition.connected_blocks import partition_blocks_from_graph

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
        id2edge_family = {v: k for k, v in edge_family2id.items()}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
        type_offsets = getattr(self.dataset_info, "type_offsets", {})
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
        node_type_names = getattr(self.dataset_info, "node_type_names", [])
        if not (id2edge_family and fam_endpoints and type_offsets and edge_family_offsets):
            return None, None

        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_sizes = {}
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = sorted_types[idx + 1][1] if idx + 1 < len(sorted_types) else self.out_dims.X
            type_sizes[type_name] = int(next_offset - offset)

        node_t = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x
        edge_attr_discrete = data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1 else data.edge_attr
        bs = int(data.batch.max() + 1)
        rho = self._rho_for_partition()
        all_query_edge_index = []
        all_query_edge_batch = []

        for fam_id, fam_name in id2edge_family.items():
            if fam_name not in fam_endpoints:
                continue
            src_type = fam_endpoints[fam_name]["src_type"]
            dst_type = fam_endpoints[fam_name]["dst_type"]
            if src_type not in type_offsets or dst_type not in type_offsets:
                continue

            fam_offset = int(edge_family_offsets.get(fam_name, 0))
            next_fam_offset = self.out_dims.E
            for _, other_offset in edge_family_offsets.items():
                other_offset = int(other_offset)
                if fam_offset < other_offset < next_fam_offset:
                    next_fam_offset = other_offset
            fam_edge_mask = (edge_attr_discrete >= fam_offset) & (edge_attr_discrete < next_fam_offset)
            fam_edge_index_all = data.edge_index[:, fam_edge_mask]
            if fam_edge_index_all.numel() == 0:
                continue

            src_offset = int(type_offsets[src_type])
            dst_offset = int(type_offsets[dst_type])
            src_size = int(type_sizes.get(src_type, 0))
            dst_size = int(type_sizes.get(dst_type, 0))
            src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)
            dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)

            for graph_idx in range(bs):
                batch_mask = data.batch == graph_idx
                batch_src_nodes = torch.where(src_mask & batch_mask)[0]
                batch_dst_nodes = torch.where(dst_mask & batch_mask)[0]
                if batch_src_nodes.numel() == 0 or batch_dst_nodes.numel() == 0:
                    continue

                same_type = src_type == dst_type
                num_possible = int(batch_src_nodes.numel() * batch_dst_nodes.numel())
                if same_type:
                    num_possible -= int(batch_src_nodes.numel())
                if num_possible <= 0:
                    continue
                num_query = min(num_possible, max(1, int(math.ceil(float(self.edge_fraction) * num_possible))))

                fam_batch_mask = data.batch[fam_edge_index_all[0]] == graph_idx
                fam_edge_index = fam_edge_index_all[:, fam_batch_mask]
                union_nodes = torch.unique(torch.cat([batch_src_nodes, batch_dst_nodes], dim=0))
                if union_nodes.numel() == 0:
                    continue
                local_lookup = {int(n.item()): i for i, n in enumerate(union_nodes)}
                local_edges = []
                for col in range(fam_edge_index.shape[1]):
                    s = int(fam_edge_index[0, col].item())
                    d = int(fam_edge_index[1, col].item())
                    if s in local_lookup and d in local_lookup:
                        local_edges.append((local_lookup[s], local_lookup[d]))
                if local_edges:
                    local_edge_index = torch.tensor(local_edges, dtype=torch.long, device=self.device).t().contiguous()
                else:
                    local_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

                cache_key = (
                    "true",
                    int(graph_idx),
                    fam_name,
                    int(union_nodes.numel()),
                    int(fam_edge_index.shape[1]),
                    self.block_partition_mode,
                    round(float(rho), 8),
                )
                blocks = self._true_block_cache.get(cache_key)
                if blocks is None:
                    blocks = partition_blocks_from_graph(
                        local_edge_index,
                        int(union_nodes.numel()),
                        rho,
                        mode=self.block_partition_mode,
                    )
                    if not blocks:
                        blocks = [list(range(int(union_nodes.numel())))]
                    self._true_block_cache[cache_key] = blocks
                block = blocks[int(torch.randint(len(blocks), (1,), device=self.device).item())]
                block_nodes = union_nodes[torch.tensor(block, dtype=torch.long, device=self.device)]
                block_src_nodes = block_nodes[torch.isin(block_nodes, batch_src_nodes)]
                block_dst_nodes = block_nodes[torch.isin(block_nodes, batch_dst_nodes)]
                if block_src_nodes.numel() == 0 or block_dst_nodes.numel() == 0:
                    block_src_nodes = batch_src_nodes
                    block_dst_nodes = batch_dst_nodes

                if same_type:
                    local_pairs = torch.triu_indices(block_src_nodes.numel(), block_src_nodes.numel(), offset=1, device=self.device)
                    if local_pairs.shape[1] == 0:
                        continue
                    candidate_edges = torch.stack(
                        [block_src_nodes[local_pairs[0]], block_src_nodes[local_pairs[1]]],
                        dim=0,
                    )
                else:
                    flat_count = int(block_src_nodes.numel() * block_dst_nodes.numel())
                    if flat_count <= 0:
                        continue
                    flat = torch.arange(flat_count, device=self.device)
                    candidate_edges = torch.stack(
                        [
                            block_src_nodes[flat // block_dst_nodes.numel()],
                            block_dst_nodes[flat % block_dst_nodes.numel()],
                        ],
                        dim=0,
                    )
                if candidate_edges.shape[1] == 0:
                    continue
                full_block_query = bool(getattr(self.cfg.model, "block_query_full_block", False))
                if full_block_query:
                    selected = candidate_edges
                else:
                    take = min(num_query, int(candidate_edges.shape[1]))
                    perm = torch.randperm(candidate_edges.shape[1], device=self.device)[:take]
                    selected = candidate_edges[:, perm]

                if full_block_query and bool(getattr(self.cfg.model, "block_query_inter_fill", False)):
                    deficit = int(num_query) - int(selected.shape[1])
                    if deficit > 0:
                        if same_type:
                            outside_nodes = batch_src_nodes[~torch.isin(batch_src_nodes, block_src_nodes)]
                            if outside_nodes.numel() > 0 and block_src_nodes.numel() > 0:
                                flat_count_inter = int(block_src_nodes.numel() * outside_nodes.numel())
                                flat_inter = torch.arange(flat_count_inter, device=self.device)
                                a = block_src_nodes[flat_inter // outside_nodes.numel()]
                                b_other = outside_nodes[flat_inter % outside_nodes.numel()]
                                inter_edges = torch.stack([torch.minimum(a, b_other), torch.maximum(a, b_other)], dim=0)
                                valid_inter = inter_edges[0] != inter_edges[1]
                                inter_edges = inter_edges[:, valid_inter]
                                if inter_edges.shape[1] > 1:
                                    key = inter_edges[0] * int(data.x.shape[0]) + inter_edges[1]
                                    order = torch.argsort(key)
                                    key_sorted = key[order]
                                    is_first = torch.ones_like(key_sorted, dtype=torch.bool)
                                    is_first[1:] = key_sorted[1:] != key_sorted[:-1]
                                    inter_edges = inter_edges[:, order[is_first]]
                            else:
                                inter_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        else:
                            inter_parts = []
                            dst_out = batch_dst_nodes[~torch.isin(batch_dst_nodes, block_dst_nodes)]
                            if block_src_nodes.numel() > 0 and dst_out.numel() > 0:
                                flat_count_inter = int(block_src_nodes.numel() * dst_out.numel())
                                flat_inter = torch.arange(flat_count_inter, device=self.device)
                                inter_parts.append(torch.stack([
                                    block_src_nodes[flat_inter // dst_out.numel()],
                                    dst_out[flat_inter % dst_out.numel()],
                                ], dim=0))
                            src_out = batch_src_nodes[~torch.isin(batch_src_nodes, block_src_nodes)]
                            if src_out.numel() > 0 and block_dst_nodes.numel() > 0:
                                flat_count_inter = int(src_out.numel() * block_dst_nodes.numel())
                                flat_inter = torch.arange(flat_count_inter, device=self.device)
                                inter_parts.append(torch.stack([
                                    src_out[flat_inter // block_dst_nodes.numel()],
                                    block_dst_nodes[flat_inter % block_dst_nodes.numel()],
                                ], dim=0))
                            inter_edges = torch.cat(inter_parts, dim=1) if inter_parts else torch.empty((2, 0), dtype=torch.long, device=self.device)
                        if inter_edges.shape[1] > 0:
                            take_inter = min(deficit, int(inter_edges.shape[1]))
                            perm_inter = torch.randperm(inter_edges.shape[1], device=self.device)[:take_inter]
                            selected = torch.cat([selected, inter_edges[:, perm_inter]], dim=1)

                max_block_edges = int(getattr(self.cfg.model, "block_query_max_edges_per_family", 0) or 0)
                if max_block_edges > 0 and selected.shape[1] > max_block_edges:
                    perm = torch.randperm(selected.shape[1], device=self.device)[:max_block_edges]
                    selected = selected[:, perm]
                all_query_edge_index.append(selected)
                all_query_edge_batch.append(
                    torch.full((selected.shape[1],), graph_idx, dtype=torch.long, device=self.device)
                )

        if not all_query_edge_index:
            return None, None
        return torch.cat(all_query_edge_index, dim=1), torch.cat(all_query_edge_batch, dim=0)

    def _append_all_positive_query_edges(self, data, query_edge_index, query_edge_batch):
        """Append all true positive edges to the training query set.

        The appended edges are still inserted into the computational graph as
        no-edge query placeholders by get_computational_graph; their true labels
        are only used by the loss target. This strengthens positive supervision
        without feeding ground-truth edge labels to the denoiser input.
        """
        if not bool(getattr(self.cfg.model, "query_include_all_positive_edges", False)):
            return query_edge_index, query_edge_batch
        if data.edge_index.numel() == 0 or data.edge_attr.numel() == 0:
            return query_edge_index, query_edge_batch

        edge_attr = data.edge_attr
        edge_label = edge_attr.argmax(dim=-1) if edge_attr.dim() > 1 else edge_attr.long()
        pos_mask = edge_label.reshape(-1) > 0
        if not pos_mask.any():
            return query_edge_index, query_edge_batch

        pos_edge_index = data.edge_index[:, pos_mask].long()
        pos_edge_batch = data.batch[pos_edge_index[0]].long()
        if query_edge_index is None or query_edge_index.numel() == 0:
            merged_edge_index = pos_edge_index
            merged_edge_batch = pos_edge_batch
        else:
            merged_edge_index = torch.cat([query_edge_index.long(), pos_edge_index], dim=1)
            merged_edge_batch = torch.cat([query_edge_batch.long(), pos_edge_batch], dim=0)

        # Deduplicate by directed endpoint and graph id. Query targets are recovered
        # from the clean graph later, so duplicates only waste memory.
        num_total_nodes = int(data.x.shape[0])
        key = (
            merged_edge_batch * (num_total_nodes * num_total_nodes)
            + merged_edge_index[0] * num_total_nodes
            + merged_edge_index[1]
        )
        order = torch.argsort(key)
        key_sorted = key[order]
        is_first = torch.ones_like(key_sorted, dtype=torch.bool)
        if key_sorted.numel() > 1:
            is_first[1:] = key_sorted[1:] != key_sorted[:-1]
        keep = order[is_first].sort().values
        merged_edge_index = merged_edge_index[:, keep]
        merged_edge_batch = merged_edge_batch[keep]

        if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_query_pos_injection", False):
            added = int(pos_edge_index.shape[1])
            total = int(merged_edge_index.shape[1])
            print(f"[QUERY] include_all_positive_edges=true: appended_pos_edges={added}, query_edges_after_unique={total}")
            self._logged_query_pos_injection = True
        return merged_edge_index, merged_edge_batch

    def _compute_query_edge_count_loss(self, pred, true_data):
        """Calibrate query-level expected edge count: mean P(edge) vs true edge ratio."""
        pred_edge = pred.edge_attr
        target = true_data.edge_attr
        if pred_edge.numel() == 0 or target.numel() == 0:
            return pred_edge.sum() * 0.0
        if target.dim() > 1:
            target = target.argmax(dim=-1)
        pred_edge_flat = pred_edge.reshape(-1, pred_edge.shape[-1])
        prob = torch.softmax(pred_edge_flat, dim=-1)
        pred_exist_mean = (1.0 - prob[:, 0]).mean()
        true_exist_mean = (target.reshape(-1).long() > 0).to(dtype=pred_edge.dtype, device=pred_edge.device).mean()
        return torch.abs(pred_exist_mean - true_exist_mean)

    def _compute_closure_positive_loss(self, pred, true_data, full_data, num_nodes):
        """Extra BCE on positive query edges weighted by true common-neighbor count."""
        pred_edge = pred.edge_attr
        target = true_data.edge_attr
        edge_index = pred.edge_index
        if pred_edge.numel() == 0 or target.numel() == 0 or edge_index.numel() == 0:
            return pred_edge.sum() * 0.0
        if target.dim() > 1:
            target = target.argmax(dim=-1)
        labels = target.reshape(-1).long()
        pos_mask = labels > 0
        if not pos_mask.any():
            return pred_edge.sum() * 0.0

        full_edge_index = getattr(full_data, "edge_index", None)
        full_edge_attr = getattr(full_data, "edge_attr", None)
        if full_edge_index is None or full_edge_index.numel() == 0:
            return pred_edge.sum() * 0.0
        src_full = full_edge_index[0].long().to(pred_edge.device)
        dst_full = full_edge_index[1].long().to(pred_edge.device)
        if full_edge_attr is not None and full_edge_attr.numel() > 0:
            if full_edge_attr.dim() > 1:
                full_labels = full_edge_attr.argmax(dim=-1).to(pred_edge.device)
            else:
                full_labels = full_edge_attr.long().to(pred_edge.device)
            full_pos = full_labels.reshape(-1) > 0
            src_full = src_full[full_pos]
            dst_full = dst_full[full_pos]
        if src_full.numel() == 0:
            return pred_edge.sum() * 0.0

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=pred_edge.device)
        valid = (src_full >= 0) & (src_full < num_nodes) & (dst_full >= 0) & (dst_full < num_nodes) & (src_full != dst_full)
        src_full = src_full[valid]
        dst_full = dst_full[valid]
        if src_full.numel() == 0:
            return pred_edge.sum() * 0.0
        adj[src_full, dst_full] = True
        adj[dst_full, src_full] = True

        q_src = edge_index[0].long()[pos_mask]
        q_dst = edge_index[1].long()[pos_mask]
        valid_q = (q_src >= 0) & (q_src < num_nodes) & (q_dst >= 0) & (q_dst < num_nodes)
        if not valid_q.any():
            return pred_edge.sum() * 0.0
        q_src = q_src[valid_q]
        q_dst = q_dst[valid_q]
        common = (adj[q_src] & adj[q_dst]).sum(dim=-1).to(dtype=pred_edge.dtype)
        if bool(getattr(self.cfg.model, "closure_pos_score_log1p", True)):
            common = torch.log1p(common)

        top_q = float(getattr(self.cfg.model, "closure_pos_top_quantile", 0.0) or 0.0)
        if top_q > 0.0:
            positive = common > 0
            if not positive.any():
                return pred_edge.sum() * 0.0
            top_q = max(0.0, min(1.0, top_q))
            threshold = torch.quantile(common[positive].detach().float(), top_q).to(common.device, dtype=common.dtype)
            keep = positive & (common >= threshold)
            if not keep.any():
                return pred_edge.sum() * 0.0
            common = common[keep]
        else:
            keep = torch.ones_like(common, dtype=torch.bool)

        if bool(getattr(self.cfg.model, "closure_pos_score_normalize", True)):
            active = common > 0
            if active.any():
                common = common / common[active].mean().clamp(min=1e-8)
        score_cap = float(getattr(self.cfg.model, "closure_pos_score_cap", 0.0) or 0.0)
        if score_cap > 0.0:
            common = common.clamp(max=score_cap)
        if common.sum() <= 0:
            return pred_edge.sum() * 0.0

        pred_flat = pred_edge.reshape(-1, pred_edge.shape[-1])
        exist_logits = torch.logsumexp(pred_flat[:, 1:], dim=-1) - pred_flat[:, 0]
        pos_logits = exist_logits[pos_mask][valid_q][keep]
        bce_pos = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction="none",
        )
        return (bce_pos * common.detach()).mean()

    def _compute_query_degree_loss(self, pred, true_data, num_nodes, node_type=None):
        """Degree-aware loss on current query edges.

        ``distribution`` preserves the previous normalized node-degree distribution
        objective. ``raw`` compares per-node expected query degree to true query
        degree directly. ``mixed`` combines both, which is useful when edge-count
        calibration is correct but generated edges are spread across the wrong nodes.
        """
        edge_index = pred.edge_index
        pred_edge = pred.edge_attr
        target = true_data.edge_attr
        if edge_index.numel() == 0 or pred_edge.numel() == 0 or target.numel() == 0:
            return pred_edge.sum() * 0.0
        if target.dim() > 1:
            target = target.argmax(dim=-1)
        target_exist = (target.reshape(-1).long() > 0).to(dtype=pred_edge.dtype, device=pred_edge.device)
        pred_prob = F.softmax(pred_edge.reshape(-1, pred_edge.shape[-1]), dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]

        pred_deg = pred_edge.new_zeros(num_nodes)
        true_deg = pred_edge.new_zeros(num_nodes)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        pred_deg.scatter_add_(0, src, pred_exist)
        pred_deg.scatter_add_(0, dst, pred_exist)
        true_deg.scatter_add_(0, src, target_exist)
        true_deg.scatter_add_(0, dst, target_exist)

        if true_deg.sum() <= 0:
            return pred_edge.sum() * 0.0

        loss_type = str(getattr(self.cfg.model, "degree_loss_type", "l1")).lower()
        mode = str(getattr(self.cfg.model, "degree_loss_mode", "distribution")).lower()
        normalized_degree = bool(getattr(self.cfg.model, "degree_loss_normalize", True))

        def pair_loss(a, b, weight=None, reduction="mean"):
            if loss_type == "mse":
                diff = (a - b).pow(2)
            else:
                diff = (a - b).abs()
            if weight is not None:
                diff = diff * weight
            if reduction == "sum":
                return diff.sum()
            return diff.mean()

        losses = []
        if mode in {"distribution", "mixed"}:
            pred_dist = pred_deg
            true_dist = true_deg
            if normalized_degree:
                pred_dist = pred_dist / pred_dist.sum().clamp(min=1e-8)
                true_dist = true_dist / true_dist.sum().clamp(min=1e-8)
            reduction = "sum" if normalized_degree else "mean"
            losses.append(pair_loss(pred_dist, true_dist, reduction=reduction))

        if mode in {"raw", "mixed"}:
            raw_weight = float(getattr(self.cfg.model, "degree_loss_raw_weight", 1.0) or 0.0)
            if raw_weight > 0:
                hub_power = float(getattr(self.cfg.model, "degree_loss_hub_power", 0.0) or 0.0)
                node_weight = None
                if hub_power > 0:
                    node_weight = (true_deg + 1.0).pow(hub_power)
                    node_weight = node_weight / node_weight.mean().clamp(min=1e-8)

                if bool(getattr(self.cfg.model, "degree_loss_type_balanced", True)) and node_type is not None:
                    nt = node_type.to(pred_edge.device).long().reshape(-1)
                    if nt.numel() == num_nodes:
                        raw_terms = []
                        for t in torch.unique(nt):
                            mask = nt == t
                            if not mask.any():
                                continue
                            scale = true_deg[mask].mean().clamp(min=1.0)
                            w = node_weight[mask] if node_weight is not None else None
                            raw_terms.append(pair_loss(pred_deg[mask] / scale, true_deg[mask] / scale, weight=w))
                        if raw_terms:
                            losses.append(raw_weight * torch.stack(raw_terms).mean())
                    else:
                        scale = true_deg.mean().clamp(min=1.0)
                        losses.append(raw_weight * pair_loss(pred_deg / scale, true_deg / scale, weight=node_weight))
                else:
                    scale = true_deg.mean().clamp(min=1.0)
                    losses.append(raw_weight * pair_loss(pred_deg / scale, true_deg / scale, weight=node_weight))

        if losses:
            return torch.stack(losses).sum()
        return pred_edge.sum() * 0.0

    def _training_t_override(self, data, batch_idx):
        schedule = str(getattr(self.cfg.model, "train_t_schedule", "random") or "random").lower()
        if schedule in {"random", "none"}:
            return None
        if schedule != "cycle":
            if hasattr(self, 'local_rank') and self.local_rank == 0 and not hasattr(self, "_warned_train_t_schedule"):
                print(f"[TRAIN] unknown train_t_schedule={schedule}; falling back to random")
                self._warned_train_t_schedule = True
            return None

        bs = int(getattr(data, "num_graphs", 1) or 1)
        try:
            batch_idx_int = int(batch_idx)
        except Exception:
            batch_idx_int = 0
        num_batches = 1
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            nb = getattr(trainer, "num_training_batches", 1)
            try:
                num_batches = max(1, int(nb))
            except Exception:
                num_batches = 1
        epoch_idx = int(getattr(self, "current_epoch", 0))
        base_step = epoch_idx * num_batches + batch_idx_int
        repeats = max(1, int(getattr(self.cfg.model, "train_t_cycle_repeats", 1) or 1))
        cycle_step = base_step // repeats
        t_vals = ((torch.arange(bs, device=self.device, dtype=torch.long) + cycle_step) % int(self.T)) + 1
        return t_vals.view(bs, 1).float()

    def training_step(self, data, i):
        if bool(getattr(self.cfg.model, "train_all_blocks_per_noise", False)):
            if data.edge_index.numel() == 0:
                if hasattr(self, 'local_rank') and self.local_rank == 0:
                    print("Found a batch with no edges. Skipping.")
                return None

            if not self.use_block_query:
                return self._training_step_once(data, i)

            opt = self.optimizers()
            data_one_hot = self.dataset_info.to_one_hot(data)
            t_override = self._training_t_override(data_one_hot, i)
            sparse_noisy_data = self.apply_sparse_noise(data_one_hot, t_override=t_override)

            block_count = int(getattr(self.cfg.model, "train_all_blocks_count", 0) or 0)
            if block_count <= 0:
                ef = max(float(getattr(self, "edge_fraction", 1.0) or 1.0), 1e-8)
                block_count = max(1, int(math.ceil(1.0 / ef)))

            block_order = torch.arange(block_count, device=self.device)
            if bool(getattr(self.cfg.model, "train_all_blocks_shuffle", True)) and block_count > 1:
                block_order = block_order[torch.randperm(block_count, device=self.device)]

            losses_detached = []
            for block_id_t in block_order:
                opt.zero_grad()
                out = self._training_step_once(
                    data_one_hot,
                    i,
                    data_is_one_hot=True,
                    sparse_noisy_data_override=dict(sparse_noisy_data),
                    forced_block_id=int(block_id_t.item()),
                )
                if out is None:
                    continue
                loss_piece = out["loss"]
                self.manual_backward(loss_piece)
                clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
                if clip_val > 0:
                    self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
                opt.step()
                losses_detached.append(loss_piece.detach())

            if not losses_detached:
                return None
            return {"loss": torch.stack(losses_detached).mean()}

        repeats = max(1, int(getattr(self.cfg.model, "train_query_repeats", 1)))
        if repeats == 1:
            return self._training_step_once(data, i)

        if data.edge_index.numel() == 0:
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Found a batch with no edges. Skipping.")
            return None

        opt = self.optimizers()
        opt.zero_grad()
        data_one_hot = self.dataset_info.to_one_hot(data)
        t_override = self._training_t_override(data_one_hot, i)
        sparse_noisy_data = self.apply_sparse_noise(data_one_hot, t_override=t_override)
        losses_detached = []

        for _ in range(repeats):
            out = self._training_step_once(
                data_one_hot,
                i,
                data_is_one_hot=True,
                sparse_noisy_data_override=dict(sparse_noisy_data),
            )
            if out is None:
                continue
            loss_piece = out["loss"] / float(repeats)
            self.manual_backward(loss_piece)
            losses_detached.append(loss_piece.detach())

        if not losses_detached:
            return None
        clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
        if clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
        opt.step()
        return {"loss": torch.stack(losses_detached).sum()}

    def _training_step_once(self, data, i, data_is_one_hot=False, sparse_noisy_data_override=None, forced_block_id=None, t_override=None):
        # The above code is using the Python debugger module `pdb` to set a breakpoint at a specific
        # line of code. When the code is executed, it will pause at that line and allow you to
        # interactively debug the program.
        if data.edge_index.numel() == 0:
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Found a batch with no edges. Skipping.")
            return
        step_idx = int(i)
        # Map discrete classes to one hot encoding
        if not data_is_one_hot:
            data = self.dataset_info.to_one_hot(data)

        if sparse_noisy_data_override is None:
            if t_override is None:
                t_override = self._training_t_override(data, i)
            sparse_noisy_data = self.apply_sparse_noise(data, t_override=t_override)
        else:
            sparse_noisy_data = sparse_noisy_data_override
        # ----- 随机选边用途 2/2：构造 query 边 -----
        # 遵循「全部可能边」概念：每次只对全部可能边中的一块做预测。query = 该块（比例 k=edge_fraction）。
        # 异质图：按族在族内可能边上采样 k*num_fam_possible_edges；同质图：全局 k*num_edges。
        # comp = 加噪后的显式边 + 本块 query 边；loss 仅在本块 query 上计算。
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
            triu_query_edge_index, query_edge_batch = (None, None)
            if self.use_block_query:
                triu_query_edge_index, query_edge_batch = self._heterogeneous_block_query_edges(data, sparse_noisy_data, forced_block_id=forced_block_id)
            if triu_query_edge_index is not None and query_edge_batch is not None:
                pass
            else:
                if self.use_block_query and getattr(self, "local_rank", 0) == 0:
                    print("[DiHuG] block query produced no edges; falling back to uniform heterogeneous query")
            block_query_only = (
                self.use_block_query
                and triu_query_edge_index is not None
                and query_edge_batch is not None
                and not bool(getattr(self.cfg.model, "block_query_include_uniform", True))
            )
            # 异质图模式：按关系族分别进行均匀采样（只处理有边的关系族）
            # 参考原项目实现，但需要区分关系族
            # 获取关系族信息
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            id2edge_family = {v: k for k, v in edge_family2id.items()}
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            
            # 获取节点类型信息
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            node_type_names = getattr(self.dataset_info, "node_type_names", [])

            # 如果 type_offsets 不存在，尝试从 meta.json 推断（不加载 vocab.json）
            if not type_offsets and node_type_names:
                import os.path as osp
                import json
                vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                if not vocab_path or not osp.exists(vocab_path):
                    if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                        vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                
                if osp.exists(vocab_path) and hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                    subgraph_dirs = [d for d in os.listdir(self.dataset_info.datamodule.inner.root) 
                                    if osp.isdir(osp.join(self.dataset_info.datamodule.inner.root, d)) and d.startswith("subgraph_")]
                    if len(subgraph_dirs) > 0:
                        meta0 = json.load(open(osp.join(self.dataset_info.datamodule.inner.root, subgraph_dirs[0], "meta.json"), "r", encoding="utf-8"))
                        schema_by_type = meta0.get("schema_by_type", {})
                        type_sizes = [len(schema_by_type.get(t, [])) for t in node_type_names]
                        type_offsets = {}
                        cur = 0
                        for t, size in zip(node_type_names, type_sizes):
                            type_offsets[t] = cur
                            cur += size
            
            # 为每个关系族分别生成查询边
            # 每族取「全部可能边中的一块」：|Eq_fam| = k * num_fam_possible_edges（族内可能边数），k 由 edge_fraction 控制
            all_query_edge_index_list = []
            all_query_edge_batch_list = []
            if block_query_only:
                all_query_edge_index_list.append(triu_query_edge_index)
                all_query_edge_batch_list.append(query_edge_batch)
            else:
                use_subtype_block_sampling = bool(
                    getattr(self.cfg.model, "query_subtype_block_sampling", False)
                )
                subtype_intra_ratio = float(
                    getattr(self.cfg.model, "query_subtype_intra_ratio", 0.8)
                )
                subtype_intra_ratio = max(0.0, min(1.0, subtype_intra_ratio))
            
                bs = int(data.batch.max() + 1)
                num_nodes_per_graph = data.ptr.diff()  # (bs,)
                node_t = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x  # (N,) - 全局子类别ID
            
                # 统计每个批次中每个关系族的真实边数（从原始数据中统计）
                edge_attr_discrete = data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1 else data.edge_attr  # (E,)
                edge_family_offsets = self.dataset_info.edge_family_offsets
                type_sizes_global = {}
                if type_offsets:
                    sorted_types_global = sorted(type_offsets.items(), key=lambda x: x[1])
                    for type_i, (type_name, type_offset) in enumerate(sorted_types_global):
                        if type_i + 1 < len(sorted_types_global):
                            type_sizes_global[type_name] = sorted_types_global[type_i + 1][1] - type_offset
                        else:
                            type_sizes_global[type_name] = self.out_dims.X - type_offset

                balanced_family_sampling = bool(
                    getattr(self.cfg.model, "query_family_balanced_sampling", True)
                )
                family_query_budget = {}
                if balanced_family_sampling and type_offsets:
                    family_possible_per_batch = {}
                    family_active_per_batch = {}
                    total_possible_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                    active_family_count = torch.zeros(bs, dtype=torch.long, device=self.device)
                    for _, fam_name_plan in id2edge_family.items():
                        if fam_name_plan not in fam_endpoints:
                            continue
                        src_type_plan = fam_endpoints[fam_name_plan]["src_type"]
                        dst_type_plan = fam_endpoints[fam_name_plan]["dst_type"]
                        if src_type_plan not in type_offsets or dst_type_plan not in type_offsets:
                            continue

                        fam_offset_plan = edge_family_offsets.get(fam_name_plan, 0)
                        next_offset_plan = self.out_dims.E
                        for _, other_offset_plan in edge_family_offsets.items():
                            if other_offset_plan > fam_offset_plan and other_offset_plan < next_offset_plan:
                                next_offset_plan = other_offset_plan
                        fam_edge_mask_plan = (edge_attr_discrete >= fam_offset_plan) & (edge_attr_discrete < next_offset_plan)
                        fam_edge_index_plan = data.edge_index[:, fam_edge_mask_plan]
                        if fam_edge_index_plan.shape[1] > 0:
                            fam_edge_batch_plan = data.batch[fam_edge_index_plan[0]]
                            unique_b, counts_b = torch.unique(fam_edge_batch_plan, sorted=True, return_counts=True)
                            has_edges = torch.zeros(bs, dtype=torch.bool, device=self.device)
                            has_edges[unique_b] = counts_b > 0
                        else:
                            has_edges = torch.zeros(bs, dtype=torch.bool, device=self.device)

                        src_offset_plan = type_offsets[src_type_plan]
                        dst_offset_plan = type_offsets[dst_type_plan]
                        src_size_plan = type_sizes_global.get(src_type_plan, 0)
                        dst_size_plan = type_sizes_global.get(dst_type_plan, 0)
                        src_mask_plan = (node_t >= src_offset_plan) & (node_t < src_offset_plan + src_size_plan)
                        dst_mask_plan = (node_t >= dst_offset_plan) & (node_t < dst_offset_plan + dst_size_plan)
                        possible = torch.zeros(bs, dtype=torch.long, device=self.device)
                        for b_plan in range(bs):
                            batch_mask_plan = data.batch == b_plan
                            n_src_plan = int((src_mask_plan & batch_mask_plan).sum().item())
                            n_dst_plan = int((dst_mask_plan & batch_mask_plan).sum().item())
                            n_possible_plan = n_src_plan * n_dst_plan
                            if src_type_plan == dst_type_plan:
                                n_possible_plan -= n_src_plan
                            possible[b_plan] = max(0, n_possible_plan)
                        active = has_edges & (possible > 0)
                        family_possible_per_batch[fam_name_plan] = possible
                        family_active_per_batch[fam_name_plan] = active
                        total_possible_per_batch += torch.where(active, possible, torch.zeros_like(possible))
                        active_family_count += active.long()

                    total_budget_per_batch = torch.ceil(
                        float(self.edge_fraction) * total_possible_per_batch.float()
                    ).long()
                    per_family_budget = torch.where(
                        active_family_count > 0,
                        torch.ceil(total_budget_per_batch.float() / active_family_count.clamp_min(1).float()).long(),
                        torch.zeros_like(total_budget_per_batch),
                    )
                    for fam_name_plan, possible in family_possible_per_batch.items():
                        active = family_active_per_batch[fam_name_plan]
                        budget = torch.minimum(per_family_budget, possible)
                        family_query_budget[fam_name_plan] = torch.where(active, budget, torch.zeros_like(budget))
            
                for fam_id, fam_name in id2edge_family.items():
                    if fam_name not in fam_endpoints:
                        continue
                
                    src_type = fam_endpoints[fam_name]["src_type"]
                    dst_type = fam_endpoints[fam_name]["dst_type"]
                
                    # 计算该关系族的offset范围
                    offset = edge_family_offsets.get(fam_name, 0)
                    next_offset = self.out_dims.E
                    for _, other_offset in edge_family_offsets.items():
                        if other_offset > offset and other_offset < next_offset:
                            next_offset = other_offset
                
                    # 统计每个批次中该关系族的真实边数 m_fam（从原始数据中统计）
                    fam_edge_mask = (edge_attr_discrete >= offset) & (edge_attr_discrete < next_offset)  # (E,)
                    fam_edge_index = data.edge_index[:, fam_edge_mask]  # (2, E_fam)
                    if fam_edge_index.shape[1] > 0:
                        fam_edge_batch = data.batch[fam_edge_index[0]]  # (E_fam,)
                        unique_fam_batch, counts_fam = torch.unique(fam_edge_batch, sorted=True, return_counts=True)
                        num_fam_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                        num_fam_edges_per_batch[unique_fam_batch] = counts_fam.long()
                    else:
                        num_fam_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                
                    # 只在有边的关系族内采样 query 边，无边的族跳过（避免 0 边族参与采样与 2^24 等问题）
                    if num_fam_edges_per_batch.sum() == 0:
                        continue
                
                    # 计算每个批次中该关系族的 src_type 和 dst_type 节点数
                    if type_offsets and src_type in type_offsets and dst_type in type_offsets:
                        src_offset = type_offsets[src_type]
                        dst_offset = type_offsets[dst_type]
                    
                        # 计算每个节点类型的 size
                        type_sizes = {}
                        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                        for i, (t, off) in enumerate(sorted_types):
                            if i + 1 < len(sorted_types):
                                type_sizes[t] = sorted_types[i + 1][1] - off
                            else:
                                type_sizes[t] = self.out_dims.X - off
                    
                        src_size = type_sizes.get(src_type, 0)
                        dst_size = type_sizes.get(dst_type, 0)
                    
                        # 计算每个批次中 src_type 和 dst_type 的节点数
                        src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)  # (N,)
                        dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)  # (N,)
                    
                        # 为每个批次生成该关系族的查询边
                        for b in range(bs):
                            batch_mask = (data.batch == b)
                            batch_src_nodes = torch.where(src_mask & batch_mask)[0]  # 全局节点索引
                            batch_dst_nodes = torch.where(dst_mask & batch_mask)[0]  # 全局节点索引
                        
                            if len(batch_src_nodes) == 0 or len(batch_dst_nodes) == 0:
                                continue
                        
                            # Uniform query over valid pairs. By default, split each graph's
                            # query budget evenly across active relation families so large endpoint
                            # spaces do not dominate smaller type pairs.
                            num_src = len(batch_src_nodes)
                            num_dst = len(batch_dst_nodes)
                            num_fam_possible_edges = num_src * num_dst
                            if src_type == dst_type:
                                num_fam_possible_edges = num_src * num_dst - num_src
                            if balanced_family_sampling and fam_name in family_query_budget:
                                num_query_edges_fam = int(family_query_budget[fam_name][b].item())
                            else:
                                k = float(self.edge_fraction)
                                num_query_edges_fam = min(
                                    num_fam_possible_edges,
                                    max(1, int(math.ceil(k * num_fam_possible_edges)))
                                )

                            if num_query_edges_fam <= 0:
                                continue
                        
                            # 使用 condensed_index 方式采样（与采样时保持一致）
                            if src_type == dst_type:
                                if use_subtype_block_sampling:
                                    # 子类别分块采样（同类型关系族）：
                                    # 优先抽取 src/dst 子类别相同的节点对，并保留一定跨子类别边。
                                    local_subtype = node_t[batch_src_nodes].long()
                                    all_pairs_local = torch.triu_indices(
                                        num_src, num_src, offset=1, device=self.device
                                    )
                                    if all_pairs_local.shape[1] > 0:
                                        src_local = all_pairs_local[0]
                                        dst_local = all_pairs_local[1]
                                        same_mask = (
                                            local_subtype[src_local] == local_subtype[dst_local]
                                        )
                                        intra_pairs = all_pairs_local[:, same_mask]
                                        inter_pairs = all_pairs_local[:, ~same_mask]
                                        desired_intra = int(
                                            round(float(num_query_edges_fam) * subtype_intra_ratio)
                                        )
                                        desired_intra = max(
                                            0, min(int(num_query_edges_fam), desired_intra)
                                        )

                                        n_intra = min(desired_intra, int(intra_pairs.shape[1]))
                                        remaining = int(num_query_edges_fam) - n_intra
                                        n_inter = min(remaining, int(inter_pairs.shape[1]))
                                        remaining = remaining - n_inter
                                        if remaining > 0:
                                            n_intra = min(
                                                n_intra + remaining, int(intra_pairs.shape[1])
                                            )

                                        selected_parts = []
                                        if n_intra > 0:
                                            intra_perm = torch.randperm(
                                                intra_pairs.shape[1], device=self.device
                                            )[:n_intra]
                                            selected_parts.append(intra_pairs[:, intra_perm])
                                        if n_inter > 0:
                                            inter_perm = torch.randperm(
                                                inter_pairs.shape[1], device=self.device
                                            )[:n_inter]
                                            selected_parts.append(inter_pairs[:, inter_perm])
                                        if selected_parts:
                                            selected_local = torch.cat(selected_parts, dim=1)
                                            if selected_local.shape[1] > 1:
                                                shuf = torch.randperm(
                                                    selected_local.shape[1], device=self.device
                                                )
                                                selected_local = selected_local[:, shuf]
                                            fam_query_edge_index = torch.stack(
                                                [
                                                    batch_src_nodes[selected_local[0]],
                                                    batch_src_nodes[selected_local[1]],
                                                ],
                                                dim=0,
                                            )
                                            if fam_query_edge_index.shape[1] > 0:
                                                fam_query_edge_batch = torch.full(
                                                    (fam_query_edge_index.shape[1],),
                                                    b,
                                                    dtype=torch.long,
                                                    device=self.device,
                                                )
                                                all_query_edge_index_list.append(
                                                    fam_query_edge_index
                                                )
                                                all_query_edge_batch_list.append(
                                                    fam_query_edge_batch
                                                )
                                                continue
                                # 同类型：使用上三角矩阵的 condensed_index（排除自环）
                                num_fam_nodes = num_src
                                max_condensed_value_fam = num_fam_nodes * (num_fam_nodes - 1) // 2
                            
                                if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                    num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                    max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                    sampled_condensed_fam, _ = sampled_condensed_indices_uniformly(
                                        max_condensed_value=max_condensed_value_fam_tensor,
                                        num_edges_to_sample=num_query_edges_fam_tensor,
                                        return_mask=False
                                    )
                                
                                    # 将 condensed_index 转换为 matrix_index
                                    fam_query_edge_index_local = condensed_to_matrix_index_batch(
                                        condensed_index=sampled_condensed_fam,
                                        num_nodes=torch.tensor([num_fam_nodes], device=self.device, dtype=torch.long),
                                        edge_batch=torch.zeros(len(sampled_condensed_fam), device=self.device, dtype=torch.long),
                                        ptr=torch.tensor([0, num_fam_nodes], device=self.device, dtype=torch.long),
                                    ).long()
                                
                                    # 边界检查：确保索引在有效范围内
                                    # 注意：fam_query_edge_index_local 是相对于 num_fam_nodes 的局部索引
                                    # 需要确保它不超过 batch_src_nodes 的长度
                                    valid_mask = (fam_query_edge_index_local[0] >= 0) & (fam_query_edge_index_local[0] < num_fam_nodes) & \
                                                (fam_query_edge_index_local[1] >= 0) & (fam_query_edge_index_local[1] < num_fam_nodes) & \
                                                (fam_query_edge_index_local[0] < len(batch_src_nodes)) & \
                                                (fam_query_edge_index_local[1] < len(batch_src_nodes))
                                    if not valid_mask.all():
                                        # 过滤无效索引
                                        fam_query_edge_index_local = fam_query_edge_index_local[:, valid_mask]
                                        if fam_query_edge_index_local.shape[1] == 0:
                                            continue
                                
                                    # 将局部索引转换回全局节点索引
                                    # 再次检查索引范围
                                    if (fam_query_edge_index_local[0].max() >= len(batch_src_nodes)) or \
                                       (fam_query_edge_index_local[1].max() >= len(batch_src_nodes)):
                                        # 如果索引超出范围，跳过
                                        continue
                                
                                    fam_query_edge_index = torch.stack([
                                        batch_src_nodes[fam_query_edge_index_local[0]],
                                        batch_src_nodes[fam_query_edge_index_local[1]]
                                    ], dim=0)
                                else:
                                    continue
                            else:
                                # 不同类型：直接使用 src*dst 的矩阵进行抽样
                                max_condensed_value_fam = num_src * num_dst
                            
                                if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                    num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                    max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                    sampled_flat_indices, _ = sampled_condensed_indices_uniformly(
                                        max_condensed_value=max_condensed_value_fam_tensor,
                                        num_edges_to_sample=num_query_edges_fam_tensor,
                                        return_mask=False
                                    )
                                
                                    # 将展平的索引转换为 (src_idx, dst_idx) 的矩阵坐标
                                    src_indices_local = sampled_flat_indices // num_dst
                                    dst_indices_local = sampled_flat_indices % num_dst
                                
                                    # 将局部索引转换回全局节点索引
                                    fam_query_edge_index = torch.stack([
                                        batch_src_nodes[src_indices_local],
                                        batch_dst_nodes[dst_indices_local]
                                    ], dim=0)
                                else:
                                    continue
                        
                            if fam_query_edge_index.shape[1] > 0:
                                fam_query_edge_batch = torch.full(
                                    (fam_query_edge_index.shape[1],),
                                    b,
                                    dtype=torch.long,
                                    device=self.device,
                                )
                                all_query_edge_index_list.append(fam_query_edge_index)
                                all_query_edge_batch_list.append(fam_query_edge_batch)
            
                    # 合并所有关系族的查询边（不做正边补充）
            if len(all_query_edge_index_list) > 0:
                uniform_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)  # (2, E_query)
                uniform_query_edge_batch = torch.cat(all_query_edge_batch_list)  # (E_query,)
                if self.use_block_query and triu_query_edge_index is not None and query_edge_batch is not None:
                    triu_query_edge_index = torch.cat(
                        [triu_query_edge_index, uniform_query_edge_index], dim=1
                    )
                    query_edge_batch = torch.cat([query_edge_batch, uniform_query_edge_batch], dim=0)
                else:
                    triu_query_edge_index = uniform_query_edge_index
                    query_edge_batch = uniform_query_edge_batch
            else:
                raise RuntimeError(
                    "Heterogeneous training produced no query edges for any relation family. "
                    "Check type_offsets/fam_endpoints and family query-edge construction."
                )
        else:
            # Homogeneous fallback: original SparseDiff uniform random query edges.
            triu_query_edge_index, query_edge_batch = sample_query_edges(
                    num_nodes_per_graph=data.ptr.diff(), edge_proportion=self.edge_fraction
            )

        # 严格一致性检查：query_edge_batch 必须与 query 边数一一对应
        if query_edge_batch.dim() != 1:
            raise RuntimeError(
                    f"query_edge_batch must be 1D, got shape={tuple(query_edge_batch.shape)}"
            )
        if query_edge_batch.shape[0] != triu_query_edge_index.shape[1]:
            raise RuntimeError(
                    "query edge/batch size mismatch before building computational graph: "
                    f"E_query={triu_query_edge_index.shape[1]}, "
                    f"E_batch={query_edge_batch.shape[0]}"
            )

        # Optional query block training (single-card friendly):
        # split query edges into A/B blocks, then randomly train one block per step.
        block_train_mode = str(
            getattr(self.cfg.model, "query_block_train_mode", "none")
        ).lower()
        block_overlap_ratio = float(
            getattr(self.cfg.model, "query_block_overlap_ratio", 0.0)
        )
        block_overlap_ratio = max(0.0, min(0.5, block_overlap_ratio))
        if block_train_mode in {"random_ab", "sequential_ab"} and triu_query_edge_index.shape[1] >= 2:
            num_q_edges = int(triu_query_edge_index.shape[1])
            perm = torch.randperm(num_q_edges, device=self.device)
            cut = max(1, num_q_edges // 2)
            idx_a = perm[:cut]
            idx_b = perm[cut:]
            if idx_b.numel() == 0:
                    idx_b = idx_a

            # Keep a small shared bridge set in both blocks to reduce block boundary tearing.
            shared_k = int(round(float(num_q_edges) * block_overlap_ratio))
            if shared_k > 0:
                    shared = perm[: min(shared_k, num_q_edges)]
                    idx_a = torch.unique(torch.cat([idx_a, shared], dim=0), sorted=False)
                    idx_b = torch.unique(torch.cat([idx_b, shared], dim=0), sorted=False)

            if block_train_mode == "random_ab":
                    chosen = idx_a if torch.rand(1, device=self.device).item() < 0.5 else idx_b
            else:
                    # Sequential A/B training (single-forward approximation):
                    # even global_step uses A, odd global_step uses B.
                    step_parity = int(getattr(self, "global_step", 0)) % 2
                    chosen = idx_a if step_parity == 0 else idx_b
            triu_query_edge_index = triu_query_edge_index[:, chosen]
            query_edge_batch = query_edge_batch[chosen]

        triu_query_edge_index, query_edge_batch = self._append_all_positive_query_edges(
            data, triu_query_edge_index, query_edge_batch
        )

        # 计算图使用加噪后的边（edge_index_t / edge_attr_t），故所有查询边位置（含并上的真实 (src,dst)）
        # 上模型看到的都是加噪后的子类别；真实标签由 mask_query_graph_from_comp_graph 从 data 取。
        query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
            triu_query_edge_index=triu_query_edge_index,
            clean_edge_index=sparse_noisy_data["edge_index_t"],
            clean_edge_attr=sparse_noisy_data["edge_attr_t"],
            heterogeneous=self.heterogeneous,
            for_message_passing=True,  # 训练时用于消息传递，需要双向信息流通
            total_num_nodes=data.x.shape[0],
        )

        # pass sparse comp_graph to dense comp_graph for ease calculation
        sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
        sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sparse_pred = self.forward(sparse_noisy_data)

        # 异质图：限制节点预测只能在其所属类型的子类别范围内（训练时也要限制）
        if self.heterogeneous and hasattr(self.dataset_info, "type_offsets"):
            type_offsets = self.dataset_info.type_offsets
            if type_offsets and sparse_pred.node.numel() > 0:
                # 关系族约束用于限制边类别空间，必须基于“固定真实节点类型”而非噪声节点，
                # 否则会把真实正边所属类别错误屏蔽，导致 subtype CE 异常增大。
                current_node_subtype = data.x
                if current_node_subtype.dim() > 1:
                    current_node_subtype = current_node_subtype.argmax(dim=-1)  # (N,)
                else:
                    current_node_subtype = current_node_subtype.long()  # (N,)

                num_nodes = current_node_subtype.shape[0]
                num_subtypes = self.out_dims.X
                node_type_mask = torch.zeros((num_nodes, num_subtypes), device=self.device)

                # 计算每个类型的size
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                type_sizes = {}
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        type_sizes[t_name] = sorted_types[i + 1][1] - off
                    else:
                        type_sizes[t_name] = num_subtypes - off

                # 为每个节点生成mask
                for i, (t_name, offset) in enumerate(sorted_types):
                    type_size = type_sizes.get(t_name, 0)
                    if type_size <= 0:
                        continue
                    # 找到属于该类型的节点
                    if i == len(sorted_types) - 1:
                        # 最后一个类型
                        type_mask = current_node_subtype >= offset
                    else:
                        next_offset = sorted_types[i + 1][1]
                        type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)

                    if type_mask.any():
                        # 允许该类型范围内的所有子类别
                        node_type_mask[type_mask, offset:offset + type_size] = 1.0

                # 应用mask：将不属于当前节点类别的子类别的logits设为-inf
                # 这样softmax后这些类别的概率为0，不会影响损失计算
                node_type_mask_inv = 1.0 - node_type_mask  # (N, dx)
                sparse_pred.node = sparse_pred.node - node_type_mask_inv * 1e10  # 将不允许的类别设为-inf
        # Compute the loss on the query edges only
        sparse_pred.edge_attr = sparse_pred.edge_attr[query_mask]
        sparse_pred.edge_index = comp_edge_index[:, query_mask]

        # 异质图：限制边预测只能在其所属关系族的子类型范围内（训练时也要限制，与节点预测一致）
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(getattr(self.dataset_info, "edge_family_offsets", {})) > 0:
            edge_family_offsets = self.dataset_info.edge_family_offsets
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})

            if edge_family_offsets and fam_endpoints and type_offsets and sparse_pred.edge_attr.numel() > 0:
                # 获取查询边的节点类型（从 comp_edge_index 和节点子类别推断）
                query_edge_index = sparse_pred.edge_index  # (2, E_query)
                num_query_edges = query_edge_index.shape[1]

                # 获取当前节点的子类别ID（从噪声图中）
                current_node_subtype = sparse_noisy_data["node_t"]
                if current_node_subtype.dim() > 1:
                    current_node_subtype = current_node_subtype.argmax(dim=-1)  # (N,)
                else:
                    current_node_subtype = current_node_subtype.long()  # (N,)

                # 推断每个节点的类型
                node_type_ids = torch.zeros_like(current_node_subtype) - 1  # -1 表示未知
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        next_offset = sorted_types[i + 1][1]
                        type_mask = (current_node_subtype >= off) & (current_node_subtype < next_offset)
                    else:
                        type_mask = current_node_subtype >= off
                    node_type_ids[type_mask] = i

                # 为每条查询边生成关系族 mask
                edge_family_mask = torch.zeros((num_query_edges, self.out_dims.E), device=self.device)

                for fam_name, endpoints in fam_endpoints.items():
                    if fam_name not in edge_family_offsets:
                        continue

                    # 获取端点类型（fam_endpoints 的结构是 {fam_name: {"src_type": ..., "dst_type": ...}}）
                    src_type = endpoints.get("src_type", None)
                    dst_type = endpoints.get("dst_type", None)
                    if src_type is None or dst_type is None:
                        continue

                    offset = edge_family_offsets[fam_name]
                    next_offset = self.out_dims.E
                    for other_fam_name, other_offset in edge_family_offsets.items():
                        if other_offset > offset and other_offset < next_offset:
                            next_offset = other_offset

                    # 找到该关系族对应的节点类型索引
                    src_type_idx = None
                    dst_type_idx = None
                    for idx, (t_name, _) in enumerate(sorted_types):
                        if t_name == src_type:
                            src_type_idx = idx
                        if t_name == dst_type:
                            dst_type_idx = idx

                    if src_type_idx is None or dst_type_idx is None:
                        continue

                    # 找到属于该关系族的查询边
                    src_nodes = query_edge_index[0]  # (E_query,)
                    dst_nodes = query_edge_index[1]  # (E_query,)
                    src_types = node_type_ids[src_nodes]  # (E_query,)
                    dst_types = node_type_ids[dst_nodes]  # (E_query,)
                    fam_edge_mask = (src_types == src_type_idx) & (dst_types == dst_type_idx)  # (E_query,)

                    if fam_edge_mask.any():
                        # 允许该关系族范围内的所有子类型（包括 no-edge）
                        edge_family_mask[fam_edge_mask, 0] = 1.0  # no-edge 始终允许
                        for gid in range(offset, next_offset):
                            edge_family_mask[fam_edge_mask, gid] = 1.0

                # 应用mask：将不属于当前边关系族的子类型的logits设为-inf
                # 这样softmax后这些类别的概率为0，不会影响损失计算
                edge_family_mask_inv = 1.0 - edge_family_mask  # (E_query, de)
                sparse_pred.edge_attr = sparse_pred.edge_attr - edge_family_mask_inv * 1e10  # 将不允许的类别设为-inf
        # mask true label for query edges
        # We have the true edge index at time 0, and the query edge index at time t. This function
        # merge the query edges and edge index at time 0, delete repeated one, and retune the mask
        # for the true attr of query edges
        (
            query_mask2,
            true_comp_edge_attr,
            true_comp_edge_index,
        ) = mask_query_graph_from_comp_graph(
            triu_query_edge_index=triu_query_edge_index,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            num_classes=self.out_dims.E,
            heterogeneous=self.heterogeneous,
            for_message_passing=True,  # 训练时用于消息传递，需要双向信息流通
        )

        query_true_edge_attr = true_comp_edge_attr[query_mask2]
        assert (
            true_comp_edge_index[:, query_mask2] - sparse_pred.edge_index == 0
        ).all()

        true_data = utils.SparsePlaceHolder(
            node=data.x,
            charge=data.charge,
            edge_attr=query_true_edge_attr,
            edge_index=sparse_pred.edge_index,
            y=data.y,
            batch=data.batch,
        )
        true_data.collapse()  # Map one-hot to discrete class

        structure_only_global = getattr(self.cfg.model, "structure_only_global", False)
        if structure_only_global:
            raise NotImplementedError(
                "model.structure_only_global is disabled in the simplified hetero edge-prediction path. "
                "Use model.structure_only_global=false."
            )

        loss = self.train_loss.forward(
            pred=sparse_pred,
            true_data=true_data,
            log=step_idx % self.log_every_steps == 0,
        )
        self._record_train_exist_diagnostics(sparse_pred, true_data, sparse_noisy_data)

        closure_w_base = float(getattr(self.cfg.model, "closure_pos_loss_weight", 0.0) or 0.0)
        closure_warmup = int(getattr(self.cfg.model, "closure_pos_loss_warmup_epochs", 0) or 0)
        if closure_warmup > 0:
            closure_factor = min(float(self.current_epoch + 1) / float(closure_warmup), 1.0)
        else:
            closure_factor = 1.0
        closure_w = closure_w_base * closure_factor
        if closure_w > 0:
            loss_closure = self._compute_closure_positive_loss(
                sparse_pred, true_data, data, num_nodes=data.x.shape[0]
            )
            loss = loss + closure_w * loss_closure
            if not hasattr(self, "_epoch_closure_pos_loss_sum"):
                self._epoch_closure_pos_loss_sum = 0.0
                self._epoch_closure_pos_loss_count = 0
                self._epoch_closure_pos_weighted_sum = 0.0
            self._epoch_closure_pos_loss_sum += float(loss_closure.detach().cpu())
            self._epoch_closure_pos_loss_count += 1
            self._epoch_closure_pos_weighted_sum += float((closure_w * loss_closure).detach().cpu())
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/closure_pos_loss",
                    loss_closure.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        count_w_base = float(getattr(self.cfg.model, "edge_count_loss_weight", 0.0) or 0.0)
        count_warmup = int(getattr(self.cfg.model, "edge_count_loss_warmup_epochs", 0) or 0)
        if count_warmup > 0:
            count_factor = min(float(self.current_epoch + 1) / float(count_warmup), 1.0)
        else:
            count_factor = 1.0
        count_w = count_w_base * count_factor
        if count_w > 0:
            loss_count = self._compute_query_edge_count_loss(sparse_pred, true_data)
            loss = loss + count_w * loss_count
            if not hasattr(self, "_epoch_edge_count_loss_sum"):
                self._epoch_edge_count_loss_sum = 0.0
                self._epoch_edge_count_loss_count = 0
                self._epoch_edge_count_weighted_sum = 0.0
            self._epoch_edge_count_loss_sum += float(loss_count.detach().cpu())
            self._epoch_edge_count_loss_count += 1
            self._epoch_edge_count_weighted_sum += float((count_w * loss_count).detach().cpu())
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/edge_count_loss",
                    loss_count.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        degree_w_base = float(getattr(self.cfg.model, "degree_loss_weight", 0.0) or 0.0)
        degree_warmup = int(getattr(self.cfg.model, "degree_loss_warmup_epochs", 0) or 0)
        if degree_warmup > 0:
            degree_factor = min(float(self.current_epoch + 1) / float(degree_warmup), 1.0)
        else:
            degree_factor = 1.0
        degree_w = degree_w_base * degree_factor
        if degree_w > 0:
            if data.x.dim() > 1:
                degree_node_type = data.x.argmax(dim=-1)
            else:
                degree_node_type = data.x.long()
            loss_degree = self._compute_query_degree_loss(
                sparse_pred, true_data, num_nodes=data.x.shape[0], node_type=degree_node_type
            )
            loss = loss + degree_w * loss_degree
            if not hasattr(self, "_epoch_degree_loss_sum"):
                self._epoch_degree_loss_sum = 0.0
                self._epoch_degree_loss_count = 0
                self._epoch_degree_weighted_sum = 0.0
            self._epoch_degree_loss_sum += float(loss_degree.detach().cpu())
            self._epoch_degree_loss_count += 1
            self._epoch_degree_weighted_sum += float((degree_w * loss_degree).detach().cpu())
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/degree_loss",
                    loss_degree.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        training_mode = str(getattr(self.cfg.model, "training_mode", "edge_prediction")).lower()
        if training_mode == "joint_structure":
            node_w = float(getattr(self.cfg.model, "node_dist_loss_weight", 0.0))
            if node_w > 0:
                warmup_epochs = int(getattr(self.cfg.model, "node_dist_warmup_epochs", 0))
                if warmup_epochs > 0:
                    warm_factor = min(float(self.current_epoch + 1) / float(warmup_epochs), 1.0)
                else:
                    warm_factor = 1.0
                include_cond = bool(getattr(self.cfg.model, "node_dist_include_conditional", True))
                cond_w = float(getattr(self.cfg.model, "node_dist_conditional_weight", 0.5))
                loss_node_dist, node_stats = self.train_loss.compute_node_distribution_loss(
                    sparse_pred.node,
                    data.x,
                    include_conditional=include_cond,
                    conditional_weight=cond_w,
                )
                loss = loss + (node_w * warm_factor) * loss_node_dist
                if step_idx % self.log_every_steps == 0:
                    self.log("train/node_dist_loss", loss_node_dist.detach(), on_step=True, prog_bar=False, sync_dist=True)
                    self.log("train/node_dist_global_tv", node_stats["global_tv"].detach(), on_step=True, prog_bar=False, sync_dist=True)
                    self.log("train/node_dist_conditional_tv", node_stats["conditional_tv"].detach(), on_step=True, prog_bar=False, sync_dist=True)

        return {"loss": loss}

    def on_after_backward(self) -> None:
        """Log gradient norm when structure_only_global to verify loss feedback reaches parameters."""
        if not getattr(self.cfg.model, "structure_only_global", False):
            return
        if getattr(self, "log_every_steps", 1) <= 0:
            return
        step = getattr(self, "global_step", 0)
        if step % self.log_every_steps != 0:
            return
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False, sync_dist=True)

    def on_fit_start(self) -> None:
        if self.local_rank == 0:
            # 训练时 main 已创建 WandbLogger 并 setup_wandb，此处仅 test_only 等无 logger 时初始化
            if wandb.run is None:
                utils.setup_wandb(self.cfg)
            # 异质图：将各关系族的正边比例 u1 记录到 wandb，便于监控
            if (
                self.heterogeneous
                and hasattr(self, "dataset_info")
                and hasattr(self.dataset_info, "edge_family_marginals")
                and self.dataset_info.edge_family_marginals
            ):
                u1_dict = {}
                for fam_name, marginals in self.dataset_info.edge_family_marginals.items():
                    if isinstance(marginals, torch.Tensor) and marginals.numel() > 0:
                        u0 = float(marginals[0].item())
                        u1 = max(0.0, min(1.0, 1.0 - u0))
                        u1_dict[f"init/正边比例_u1/{fam_name}"] = u1
                if u1_dict and wandb.run:
                    wandb.log(u1_dict, commit=False)

    def _record_train_exist_diagnostics(self, pred, true_data, sparse_noisy_data):
        """Accumulate epoch-level probability diagnostics for edge existence."""
        with torch.no_grad():
            pred_edge = pred.edge_attr
            target = true_data.edge_attr
            if target.dim() > 1:
                    target = target.argmax(dim=-1)
            if pred_edge.numel() == 0 or target.numel() == 0:
                    return
            pred_edge_flat = pred_edge.reshape(-1, pred_edge.shape[-1])
            true_labels = target.reshape(-1).long().clamp(min=0, max=pred_edge_flat.shape[-1] - 1)
            true_exist = true_labels > 0
            prob = torch.softmax(pred_edge_flat, dim=-1)
            p_exist = 1.0 - prob[:, 0]
            pred_exist = pred_edge_flat.argmax(dim=-1) > 0
            n = float(p_exist.numel())
            if n <= 0:
                    return
            pos_count = int(true_exist.sum().item())
            neg_count = int((~true_exist).sum().item())
            t_float = sparse_noisy_data.get("t_float", None)
            t_mean = float(t_float.float().mean().detach().cpu()) if t_float is not None else 0.0
            if not hasattr(self, "_epoch_diag_count"):
                    self._epoch_diag_count = 0
                    self._epoch_diag_t_sum = 0.0
                    self._epoch_diag_query_pos_ratio_sum = 0.0
                    self._epoch_diag_pred_pos_rate_sum = 0.0
                    self._epoch_diag_pos_p_exist_sum = 0.0
                    self._epoch_diag_pos_p_exist_count = 0
                    self._epoch_diag_neg_p_exist_sum = 0.0
                    self._epoch_diag_neg_p_exist_count = 0
            self._epoch_diag_count += 1
            self._epoch_diag_t_sum += t_mean
            self._epoch_diag_query_pos_ratio_sum += float(pos_count) / n
            self._epoch_diag_pred_pos_rate_sum += float(pred_exist.float().mean().detach().cpu())
            if pos_count > 0:
                    self._epoch_diag_pos_p_exist_sum += float(p_exist[true_exist].mean().detach().cpu())
                    self._epoch_diag_pos_p_exist_count += 1
            if neg_count > 0:
                    self._epoch_diag_neg_p_exist_sum += float(p_exist[~true_exist].mean().detach().cpu())
                    self._epoch_diag_neg_p_exist_count += 1

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.train_loss.reset()
        self._epoch_degree_loss_sum = 0.0
        self._epoch_degree_loss_count = 0
        self._epoch_degree_weighted_sum = 0.0
        self._epoch_edge_count_loss_sum = 0.0
        self._epoch_edge_count_loss_count = 0
        self._epoch_edge_count_weighted_sum = 0.0
        self._epoch_closure_pos_loss_sum = 0.0
        self._epoch_closure_pos_loss_count = 0
        self._epoch_closure_pos_weighted_sum = 0.0
        self._epoch_diag_count = 0
        self._epoch_diag_t_sum = 0.0
        self._epoch_diag_query_pos_ratio_sum = 0.0
        self._epoch_diag_pred_pos_rate_sum = 0.0
        self._epoch_diag_pos_p_exist_sum = 0.0
        self._epoch_diag_pos_p_exist_count = 0
        self._epoch_diag_neg_p_exist_sum = 0.0
        self._epoch_diag_neg_p_exist_count = 0

    def on_train_epoch_end(self) -> None:
        # 只在 epoch 结束时统一从 TrainLossDiscrete 取出本 epoch 的所有训练指标，并一次性 log 到 wandb / Lightning
        epoch_loss = self.train_loss.log_epoch_metrics()
        exist_bce = epoch_loss.get("train_epoch/existence_BCE", -1)
        subtype_ce = epoch_loss.get("train_epoch/subtype_CE", -1)
        pos_acc = epoch_loss.get("train_epoch/existence_pos_acc", -1)
        neg_acc = epoch_loss.get("train_epoch/existence_neg_acc", -1)
        degree_count = int(getattr(self, "_epoch_degree_loss_count", 0) or 0)
        if degree_count > 0:
            degree_l1 = float(getattr(self, "_epoch_degree_loss_sum", 0.0)) / float(degree_count)
            degree_wloss = float(getattr(self, "_epoch_degree_weighted_sum", 0.0)) / float(degree_count)
        else:
            degree_l1 = 0.0
            degree_wloss = 0.0
        count_loss_count = int(getattr(self, "_epoch_edge_count_loss_count", 0) or 0)
        if count_loss_count > 0:
            edge_count_l1 = float(getattr(self, "_epoch_edge_count_loss_sum", 0.0)) / float(count_loss_count)
            edge_count_wloss = float(getattr(self, "_epoch_edge_count_weighted_sum", 0.0)) / float(count_loss_count)
        else:
            edge_count_l1 = 0.0
            edge_count_wloss = 0.0
        closure_count = int(getattr(self, "_epoch_closure_pos_loss_count", 0) or 0)
        if closure_count > 0:
            closure_l1 = float(getattr(self, "_epoch_closure_pos_loss_sum", 0.0)) / float(closure_count)
            closure_wloss = float(getattr(self, "_epoch_closure_pos_weighted_sum", 0.0)) / float(closure_count)
        else:
            closure_l1 = 0.0
            closure_wloss = 0.0
        edge_nll = epoch_loss.get("train_epoch/NLL", -1)
        total_train = edge_nll + degree_wloss + edge_count_wloss + closure_wloss if isinstance(edge_nll, (int, float)) and edge_nll >= 0 else -1
        diag_count = int(getattr(self, "_epoch_diag_count", 0) or 0)
        if diag_count > 0:
            diag_t = float(getattr(self, "_epoch_diag_t_sum", 0.0)) / float(diag_count)
            diag_qpos = float(getattr(self, "_epoch_diag_query_pos_ratio_sum", 0.0)) / float(diag_count)
            diag_pred_pos = float(getattr(self, "_epoch_diag_pred_pos_rate_sum", 0.0)) / float(diag_count)
            pos_p_count = int(getattr(self, "_epoch_diag_pos_p_exist_count", 0) or 0)
            neg_p_count = int(getattr(self, "_epoch_diag_neg_p_exist_count", 0) or 0)
            diag_pos_p = (float(getattr(self, "_epoch_diag_pos_p_exist_sum", 0.0)) / float(pos_p_count)) if pos_p_count > 0 else -1.0
            diag_neg_p = (float(getattr(self, "_epoch_diag_neg_p_exist_sum", 0.0)) / float(neg_p_count)) if neg_p_count > 0 else -1.0
            epoch_loss["train_epoch/t_norm"] = diag_t
            epoch_loss["train_epoch/query_pos_ratio"] = diag_qpos
            epoch_loss["train_epoch/pred_pos_rate"] = diag_pred_pos
            epoch_loss["train_epoch/pos_p_exist_mean"] = diag_pos_p
            epoch_loss["train_epoch/neg_p_exist_mean"] = diag_neg_p
        else:
            diag_t = diag_qpos = diag_pred_pos = diag_pos_p = diag_neg_p = -1.0
        if degree_count > 0:
            epoch_loss["train_epoch/degree_L1"] = degree_l1
            epoch_loss["train_epoch/degree_weighted"] = degree_wloss
        if count_loss_count > 0:
            epoch_loss["train_epoch/edge_count_L1"] = edge_count_l1
            epoch_loss["train_epoch/edge_count_weighted"] = edge_count_wloss
        if closure_count > 0:
            epoch_loss["train_epoch/closure_pos_L1"] = closure_l1
            epoch_loss["train_epoch/closure_pos_weighted"] = closure_wloss
        if (degree_count > 0 or count_loss_count > 0 or closure_count > 0) and total_train >= 0:
            epoch_loss["train_epoch/total_with_aux"] = total_train
        # neg_acc=-1 表示 query 内无负样本（ideal 模式下常见），显示为 N/A
        neg_str = f"{neg_acc:.4f}" if isinstance(neg_acc, (int, float)) and neg_acc >= 0 else "N/A"
        degree_str = f" -- degree_L1: {degree_l1:.6f} -- degree_wloss: {degree_wloss:.6f}" if degree_count > 0 else ""
        edge_count_str = f" -- edge_count_L1: {edge_count_l1:.6f} -- edge_count_wloss: {edge_count_wloss:.6f}" if count_loss_count > 0 else ""
        closure_str = f" -- closure_L1: {closure_l1:.6f} -- closure_wloss: {closure_wloss:.6f}" if closure_count > 0 else ""
        total_str = f" -- total_train: {total_train:.4f}" if (degree_count > 0 or count_loss_count > 0 or closure_count > 0) and total_train >= 0 else ""
        diag_str = (
            f" -- t_norm: {diag_t:.3f} -- q_pos: {diag_qpos:.4f} -- pred_pos: {diag_pred_pos:.4f} "
            f"-- pos_p: {diag_pos_p:.4f} -- neg_p: {diag_neg_p:.4f}"
        ) if diag_count > 0 else ""

        self.print(
            f"Epoch {self.current_epoch} finished: "
            f"exist_BCE: {exist_bce:.4f} -- subtype_CE: {subtype_ce:.4f} -- "
            f"pos_acc: {pos_acc:.4f} -- neg_acc: {neg_str}"
            f"{degree_str}{edge_count_str}{closure_str}{total_str}{diag_str}"
        )

        if wandb.run:
            # 将 epoch 号和本 epoch 的 train_epoch/* 指标一次性记录到 wandb，并显式使用 epoch 作为 step
            log_dict = {"epoch": int(self.current_epoch)}
            log_dict.update(epoch_loss)
            wandb.log(log_dict, step=int(self.current_epoch), commit=True)

        # 供 checkpoint 回调使用（验证关闭时 val/epoch_NLL 不存在，可用 train_epoch/NLL；与 val 同公式，仅取数在 train query 上）
        for key in (
            "train_epoch/existence_BCE",
            "train_epoch/subtype_CE",
            "train_epoch/existence_pos_acc",
            "train_epoch/existence_neg_acc",
            "train_epoch/NLL",
            "train_epoch/degree_L1",
            "train_epoch/degree_weighted",
            "train_epoch/edge_count_L1",
            "train_epoch/edge_count_weighted",
            "train_epoch/closure_pos_L1",
            "train_epoch/closure_pos_weighted",
            "train_epoch/total_with_aux",
            "train_epoch/t_norm",
            "train_epoch/query_pos_ratio",
            "train_epoch/pred_pos_rate",
            "train_epoch/pos_p_exist_mean",
            "train_epoch/neg_p_exist_mean",
        ):
            if key in epoch_loss:
                    self.log(key, epoch_loss[key], sync_dist=True)
        # 非验证 epoch 补 log 上一次的 val/epoch_NLL，避免 ModelCheckpoint(monitor='val/epoch_NLL') 因 key 缺失报警（验证 epoch 由 validation_epoch_end 已 log，不重复）
        check_every = getattr(self.cfg.general, "check_val_every_n_epochs", 1)
        if getattr(self.cfg.general, "enable_validation", True) and (self.current_epoch % check_every != 0):
            self.log("val/epoch_NLL", getattr(self, "_last_val_epoch_nll", float("inf")), sync_dist=True)

        # Optionally export a sampled edge list (denoised graph) at a specific epoch.
        export_epoch = int(getattr(self.cfg.train, "export_edge_list_epoch", -1))
        enable_val_sampling = getattr(self.cfg.general, "enable_val_sampling", False)
        if export_epoch >= 0 and enable_val_sampling:
            if hasattr(self, "local_rank") and self.local_rank != 0:
                return
            if self.current_epoch == export_epoch:
                try:
                    sample = self.sample_batch(
                        batch_id=0,
                        batch_size=1,
                        keep_chain=0,
                        number_chain_steps=self.number_chain_steps,
                        save_final=1,
                    )
                    self._export_edge_list_doc(sample, epoch=self.current_epoch)
                except Exception as exc:
                    print(f"[WARN] export_edge_list_doc failed: {exc}")

    def _export_edge_list_doc(self, generated_graphs, epoch: int) -> None:
        """Optional debug export hook; disabled in the simplified hetero training path."""
        return

    def on_validation_epoch_start(self) -> None:
        self._val_predicted_graphs_list = []
        # 用于边分布图「generate」：在「真实边属于该族」的条件下，统计模型预测的子类型分布，与训练指标一致
        if getattr(self, "dataset_info", None) and getattr(self.dataset_info, "edge_family_marginals", None):
            self._val_edge_pred_counts_by_family = {
                    fam: torch.zeros(len(self.dataset_info.edge_family_marginals[fam]), device=self.device)
                    for fam in self.dataset_info.edge_family_marginals
            }
        else:
            self._val_edge_pred_counts_by_family = {}
        val_metrics = [
            self.val_nll,
            self.val_exist_nll,
            self.val_subtype_nll,
            self.val_exist_kl,
            self.val_subtype_kl,
            self.val_exist_logp,
            self.val_subtype_logp,
            self.val_sampling_metrics,
        ]
        for metric in val_metrics:
            metric.reset()

    def validation_step(self, data, i):
        """Validation uses the same sparse edge-prediction objective as training.

        The older validation path mixed full-graph NLL and sampling-style metrics, which is not
        comparable to focal/block edge-prediction training and produced misleading values.
        """
        data = self.dataset_info.to_one_hot(data)
        with torch.no_grad():
            out = self._training_step_once(data, i, data_is_one_hot=True)
            loss = out["loss"].detach()
        loss_1d = loss.reshape(1)
        self.val_nll(loss_1d)
        self.val_exist_nll(loss_1d)
        zero = torch.zeros_like(loss_1d)
        self.val_subtype_nll(zero)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        def _get(metric):
            ts = getattr(metric, "total_samples", None)
            if ts is None:
                return -1
            try:
                if float(ts.item()) <= 0:
                    return -1
            except Exception:
                return -1
            v = metric.compute()
            try:
                return float(v.item()) if hasattr(v, "item") else float(v)
            except Exception:
                return -1

        total_nll = _get(self.val_nll)
        exist_nll = _get(self.val_exist_nll)
        subtype_nll = _get(self.val_subtype_nll)
        val_nll_value = total_nll if total_nll != -1 else float("inf")
        if val_nll_value < self.best_nll:
            self.best_epoch = self.current_epoch
            self.best_nll = val_nll_value

        if wandb.run:
            wandb.log(
                {
                    "val/epoch_NLL": val_nll_value,
                    "val/edge_loss": val_nll_value,
                    "val/nll_existence": float(exist_nll) if exist_nll != -1 else -1,
                    "val/nll_subtype": float(subtype_nll) if subtype_nll != -1 else -1,
                    "val/best_nll_epoch": int(self.best_epoch),
                    "val/best_nll": float(self.best_nll),
                },
                commit=False,
            )

        exist_str = f"{exist_nll:.4f}" if exist_nll != -1 else "n/a"
        subtype_str = f"{subtype_nll:.4f}" if subtype_nll != -1 else "n/a"
        self.print(
            f"Epoch {self.current_epoch}: Val edge_loss {val_nll_value:.4f} "
            f"(exist {exist_str}, subtype {subtype_str})"
        )
        self.log("val/epoch_NLL", val_nll_value, sync_dist=True)
        self.log("val/edge_loss", val_nll_value, sync_dist=True)
        self._last_val_epoch_nll = val_nll_value

    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        if self.local_rank == 0 and wandb.run is None:
            utils.setup_wandb(
                    self.cfg
            )  # Initialize wandb only when not already in a run (e.g. test_only); train+test reuses run from on_fit_start
        test_metrics = [
            self.test_nll,
            self.test_X_kl,
            self.test_E_kl,
            self.test_X_logp,
            self.test_E_logp,
            self.test_sampling_metrics,
        ]
        if self.use_charge:
            test_metrics.extend([self.test_charge_kl, self.test_charge_logp])
        for metric in test_metrics:
            metric.reset()
        if (
            getattr(self, "local_rank", 0) == 0
            and (
                    getattr(self.train_loss, "structure_loss_type", "legacy") == "graph_metrics"
                    or getattr(self.train_loss, "relation_matrix_loss_weight", 0.0) > 0
                    or getattr(self.train_loss, "metapath2_loss_weight", 0.0) > 0
                    or getattr(self.train_loss, "metapath3_loss_weight", 0.0) > 0
                    or getattr(self.train_loss, "subtype_degree_loss_weight", 0.0) > 0
            )
        ):
            self.print(
                    "[TEST] multi-order structure metrics are aligned in train/val; "
                    "test loop currently reports sampling metrics."
            )

    def test_step(self, data, i):
        pass

    def on_test_epoch_end(self) -> None:
        """Generate sparse samples and compute sampling metrics."""
        enable_test_sampling = getattr(self.cfg.general, "enable_test_sampling", True)
        generated_path = getattr(self.cfg.general, "generated_path", None)
        if not enable_test_sampling and not generated_path:
            if getattr(self, "local_rank", 0) == 0:
                self.print("Test sampling disabled (enable_test_sampling=false), skipping.")
            return

        self._test_intermediate_sample_chunks = {}

        if generated_path:
            self.print("Loading generated samples...")
            with open(generated_path, "rb") as f:
                samples = pickle.load(f)
        else:
            n_to_generate = int(getattr(self.cfg.general, "final_model_samples_to_generate", 1))
            n_to_generate *= int(getattr(self.cfg.general, "test_variance", 1))
            n_to_generate = max(1, math.ceil(n_to_generate / max(getattr(self._trainer, "num_devices", 1), 1)))
            self.print(
                f"Samples to generate: {n_to_generate} for each of the "
                f"{max(getattr(self._trainer, 'num_devices', 1), 1)} devices"
            )
            print(f"Sampling start on GR{self.global_rank}")

            sample_chunks = []
            remaining = n_to_generate
            batch_size = max(1, int(getattr(self.cfg.train, "batch_size", 1)))
            use_fixed_nodes = bool(getattr(self.cfg.general, "cond_edge_gen_fixed_nodes", False))
            use_sample_nodes = bool(getattr(self.cfg.general, "cond_edge_gen_sample_nodes", False))

            while remaining > 0:
                to_generate = min(remaining, batch_size)
                if use_fixed_nodes and hasattr(self.dataset_info, "datamodule"):
                    use_test_graph = getattr(self.cfg.general, "cond_edge_gen_use_test_graph", True)
                    if use_test_graph and hasattr(self.dataset_info.datamodule, "test_dataset"):
                        fixed_single = self.dataset_info.datamodule.test_dataset[0].clone()
                    else:
                        fixed_single = self.dataset_info.datamodule.train_dataset[0].clone()
                    fixed_single = self.dataset_info.to_one_hot(fixed_single)
                    from torch_geometric.data import Batch
                    fixed_batch = Batch.from_data_list([fixed_single.clone() for _ in range(to_generate)]).to(self.device)
                    sampled_batch = self.sample_batch_fixed_nodes(
                        fixed_batch,
                        keep_chain=0,
                        number_chain_steps=self.number_chain_steps,
                        save_final=to_generate,
                    )
                else:
                    sample_num_nodes = getattr(self.cfg.general, "sample_num_nodes", None)
                    if sample_num_nodes is not None:
                        num_nodes = sample_num_nodes
                    elif use_sample_nodes and hasattr(self.dataset_info, "datamodule"):
                        train_data = self.dataset_info.datamodule.train_dataset[0]
                        num_nodes = getattr(train_data, "num_nodes", train_data.x.shape[0] if hasattr(train_data, "x") else 20)
                    else:
                        num_nodes = 20
                    sampled_batch = self.sample_batch(
                        batch_id=n_to_generate - remaining,
                        batch_size=to_generate,
                        num_nodes=num_nodes,
                        save_final=to_generate,
                        keep_chain=0,
                        number_chain_steps=self.number_chain_steps,
                    )
                sample_chunks.append(sampled_batch)
                remaining -= to_generate
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            samples = utils.concat_sparse_graphs(sample_chunks)
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)
            if hasattr(self, "_trainer") and getattr(self._trainer, "num_devices", 1) > 1:
                self._trainer.strategy.barrier()
                all_samples = []
                for rank in range(self._trainer.num_devices):
                    with open(f"generated_samples_rank{rank}.pkl", "rb") as f:
                        all_samples.append(pickle.load(f))
                samples = utils.concat_sparse_graphs(all_samples)
            with open("generated_samples.pkl", "wb") as f:
                pickle.dump(samples, f)

        if getattr(self, "local_rank", 0) == 0:
            print("Computing sampling metrics...")
        test_variance = int(getattr(self, "test_variance", getattr(self.cfg.general, "test_variance", 1)))
        if test_variance <= 1:
            to_log, _ = self.test_sampling_metrics.compute_all_metrics(
                samples, self.current_epoch, local_rank=self.local_rank
            )
            if getattr(self, "local_rank", 0) == 0:
                print("saving results for testing")
                with open(os.path.join(os.getcwd(), f"test_epoch{self.current_epoch}.json"), "w") as file:
                    json.dump(to_log, file)
                table_str = format_structure_metrics_table(to_log, key_prefix="test")
                if table_str.strip():
                    struct_path = os.path.join(os.getcwd(), "test_structure_metrics.txt")
                    with open(struct_path, "w", encoding="utf-8") as f:
                        f.write(table_str)
                    print(f"Structure metrics table saved to {struct_path}")
                print(f"For overall 1 samplings, we have: ")
                print(to_log)
        else:
            split_samples = utils.split_samples(samples, test_variance)
            to_log = {}
            for idx in range(test_variance):
                self.test_sampling_metrics.reset()
                cur_to_log, _ = self.test_sampling_metrics.compute_all_metrics(
                    split_samples[idx], self.current_epoch, local_rank=self.local_rank
                )
                if idx == 0:
                    to_log = {k: [cur_to_log[k]] for k in cur_to_log}
                else:
                    to_log = {k: to_log[k] + [cur_to_log[k]] for k in cur_to_log}
                print(f"For the {idx} th sampling, we have: ")
                print(cur_to_log)
            to_log = {k: (np.mean(v), np.std(v)) for k, v in to_log.items()}
            print(f"For overall {test_variance} samplings, we have: ")
            print(to_log)
        if not generated_path:
            self._compute_and_save_test_intermediate_metrics()
        self.print("Test sampling metrics computed.")

    def apply_sparse_noise(self, data, t_override=None):
        """Sparse forward noising for the simplified hetero edge-prediction path.

        Heterogeneous graphs keep directed canonical edges. Existing edges are diffused inside
        their own relation family and no-edge outcomes are dropped from the explicit noisy graph.
        """
        bs = data.num_graphs
        device = self.device
        if t_override is None:
            t_int = torch.randint(1, self.T + 1, size=(bs, 1), device=device).float()
        else:
            t_int = t_override.to(device).float().view(bs, 1)
        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)

        if self.use_charge:
            prob_charge = data.charge.unsqueeze(1) @ Qtb.charge[data.batch]
            charge_t = prob_charge.squeeze(1).multinomial(1).flatten()
            charge_t = F.one_hot(charge_t, num_classes=self.out_dims.charge).float()
        else:
            charge_t = data.charge

        probN = data.x.unsqueeze(1) @ Qtb.X[data.batch]
        if probN.dim() == 3:
            probN = probN.squeeze(1)
        elif probN.dim() == 1:
            probN = probN.unsqueeze(0)

        if self.heterogeneous and hasattr(self.dataset_info, "type_offsets"):
            type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
            if type_offsets:
                current_node_subtype = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x.long()
                node_type_mask = torch.zeros((current_node_subtype.shape[0], self.out_dims.X), device=device)
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                for i, (t_name, offset) in enumerate(sorted_types):
                    next_offset = sorted_types[i + 1][1] if i + 1 < len(sorted_types) else self.out_dims.X
                    type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)
                    if type_mask.any() and next_offset > offset:
                        node_type_mask[type_mask, offset:next_offset] = 1.0
                probN_masked = probN * node_type_mask
                row_sum = probN_masked.sum(dim=-1, keepdim=True)
                bad = row_sum.squeeze(-1) <= 0
                if bad.any():
                    probN_masked[bad] = probN[bad]
                    row_sum = probN_masked.sum(dim=-1, keepdim=True)
                probN = probN_masked / row_sum.clamp(min=1e-8)

        node_t_ids = probN.multinomial(1).flatten()
        node_t = F.one_hot(node_t_ids, num_classes=self.out_dims.X).float()

        if self.heterogeneous:
            dir_edge_index = data.edge_index
            edge_attr = data.edge_attr.argmax(dim=-1).long() if data.edge_attr.dim() > 1 else data.edge_attr.long()
        else:
            dir_edge_index, edge_attr = utils.undirected_to_directed(data.edge_index, data.edge_attr)
            edge_attr = edge_attr.argmax(dim=-1).long() if edge_attr.dim() > 1 else edge_attr.long()

        num_nodes_per_graph = data.ptr.diff().long()
        noisy_edge_indices = []
        noisy_edge_attrs = []
        sampled_new_total = 0
        has_edge_family = self.heterogeneous and hasattr(data, "edge_family") and data.edge_family is not None
        if dir_edge_index.numel() > 0 and has_edge_family:
            all_family_qt = self.transition_model.get_all_family_Qt_bar(alpha_t_bar, device=device)
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
            id2edge_family = {int(v): k for k, v in edge_family2id.items()}
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {}) or {}
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
            type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
            sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
            type_ranges = {}
            for i, (type_name, type_offset) in enumerate(sorted_types):
                next_offset = sorted_types[i + 1][1] if i + 1 < len(sorted_types) else self.out_dims.X
                type_ranges[type_name] = (int(type_offset), int(next_offset))
            for fam_id, fam_name in id2edge_family.items():
                fam_mask = data.edge_family.long() == fam_id
                if not fam_mask.any() or fam_name not in all_family_qt:
                    continue
                fam_edge_index = dir_edge_index[:, fam_mask]
                fam_attr_global = edge_attr[fam_mask].clone()
                offset = int(edge_family_offsets.get(fam_name, 0))
                fam_attr_local = fam_attr_global.clone()
                nz = fam_attr_local != 0
                if nz.any():
                    fam_attr_local[nz] = fam_attr_local[nz] - offset + 1
                fam_Q = all_family_qt[fam_name].E[data.batch[fam_edge_index[0]]]
                num_states = fam_Q.shape[-1]
                fam_attr_local = fam_attr_local.clamp(0, num_states - 1)
                fam_onehot = F.one_hot(fam_attr_local.long(), num_classes=num_states).float()
                fam_prob = torch.bmm(fam_onehot.unsqueeze(1), fam_Q).squeeze(1)
                fam_sample_local = fam_prob.multinomial(1).flatten()
                fam_sample_global = fam_sample_local.clone()
                nz_sample = fam_sample_global != 0
                if nz_sample.any():
                    fam_sample_global[nz_sample] = fam_sample_global[nz_sample] - 1 + offset
                keep = fam_sample_global != 0
                if keep.any():
                    noisy_edge_indices.append(fam_edge_index[:, keep])
                    noisy_edge_attrs.append(fam_sample_global[keep])

                endpoints = fam_endpoints.get(fam_name, {})
                src_type = endpoints.get("src_type")
                dst_type = endpoints.get("dst_type")
                if src_type in type_ranges and dst_type in type_ranges:
                    s0, s1 = type_ranges[src_type]
                    d0, d1 = type_ranges[dst_type]
                    src_mask = (node_t_ids >= s0) & (node_t_ids < s1)
                    dst_mask = (node_t_ids >= d0) & (node_t_ids < d1)
                    bs_local = int(num_nodes_per_graph.shape[0])
                    num_src = torch.zeros(bs_local, dtype=torch.long, device=device)
                    num_dst = torch.zeros(bs_local, dtype=torch.long, device=device)
                    for b in range(bs_local):
                        batch_mask = data.batch == b
                        num_src[b] = (src_mask & batch_mask).sum()
                        num_dst[b] = (dst_mask & batch_mask).sum()
                    possible = num_src * num_dst
                    if src_type == dst_type:
                        possible = (possible - num_src).clamp(min=0)
                    fam_batch_edge = data.batch[fam_edge_index[0]]
                    existing = torch.zeros(bs_local, dtype=torch.long, device=device)
                    if fam_batch_edge.numel() > 0:
                        unique_b, counts_b = torch.unique(fam_batch_edge, sorted=True, return_counts=True)
                        existing[unique_b] = counts_b.long()
                    num_fam_neg = (possible - existing).clamp(min=0)
                    emerge_prob = all_family_qt[fam_name].E[:, 0, 1:].sum(dim=-1).clamp(0, 1)
                    num_emerge = torch.distributions.binomial.Binomial(
                        num_fam_neg.float(), emerge_prob
                    ).sample().long()
                    if int(num_emerge.max().item()) > 0 and src_mask.any() and dst_mask.any():
                        neg_edge_index = sample_non_existing_edges_batched_heterogeneous(
                            num_edges_to_sample=num_emerge,
                            existing_edge_index=dir_edge_index,
                            num_nodes=num_nodes_per_graph,
                            batch=data.batch,
                            src_mask=src_mask,
                            dst_mask=dst_mask,
                        )
                        if neg_edge_index.numel() > 0:
                            neg_edge_batch = data.batch[neg_edge_index[0]]
                            attr_probs = all_family_qt[fam_name].E[neg_edge_batch, 0, 1:]
                            local_new_attr = torch.multinomial(attr_probs, 1, replacement=True).flatten() + 1
                            global_new_attr = local_new_attr - 1 + offset
                            noisy_edge_indices.append(neg_edge_index.long())
                            noisy_edge_attrs.append(global_new_attr.long().clamp(0, self.out_dims.E - 1))
                            sampled_new_total += int(neg_edge_index.shape[1])
        elif dir_edge_index.numel() > 0:
            edge_batch = data.batch[dir_edge_index[0]]
            edge_onehot = F.one_hot(edge_attr.clamp(0, self.out_dims.E - 1), num_classes=self.out_dims.E).float()
            probE = torch.bmm(edge_onehot.unsqueeze(1), Qtb.E[edge_batch]).squeeze(1)
            sampled_edge_attr = probE.multinomial(1).flatten()
            keep = sampled_edge_attr != 0
            if keep.any():
                noisy_edge_indices.append(dir_edge_index[:, keep])
                noisy_edge_attrs.append(sampled_edge_attr[keep])

        if noisy_edge_indices:
            E_t_index = torch.cat(noisy_edge_indices, dim=1).long()
            E_t_attr_ids = torch.cat(noisy_edge_attrs, dim=0).long().clamp(0, self.out_dims.E - 1)
        else:
            E_t_index = torch.empty((2, 0), dtype=torch.long, device=device)
            E_t_attr_ids = torch.empty((0,), dtype=torch.long, device=device)

        if not self.heterogeneous and E_t_index.numel() > 0:
            E_t_index, E_t_attr_ids = utils.to_undirected(E_t_index, E_t_attr_ids)

        if getattr(self, "local_rank", 0) == 0 and not hasattr(self, "_noise_debug_count"):
            self._noise_debug_count = 0
        if getattr(self, "local_rank", 0) == 0 and self._noise_debug_count < 1:
            t_val = float(t_float.mean().detach().cpu())
            print(
                f"[NOISE] t_norm={t_val:.4f} "
                f"clean_edges={int(dir_edge_index.shape[1])} noisy_edges={int(E_t_index.shape[1])} "
                f"sampled_new={sampled_new_total}"
            )
            self._noise_debug_count += 1

        E_t_attr = F.one_hot(E_t_attr_ids, num_classes=self.out_dims.E).float()
        sparse_noisy_data = {
            "t_int": t_int,
            "t_float": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "node_t": node_t,
            "X_t": node_t,
            "edge_index_t": E_t_index,
            "edge_attr_t": E_t_attr,
            "comp_edge_index_t": None,
            "comp_edge_attr_t": None,
            "y_t": data.y,
            "batch": data.batch,
            "ptr": data.ptr,
            "charge_t": charge_t,
        }
        return sparse_noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, charge, test):
        """Deprecated dense validation objective.

        Validation now uses the sparse edge-prediction objective through validation_step, so
        this old full-graph NLL path is intentionally disabled.
        """
        raise NotImplementedError("compute_val_loss is disabled in the simplified sparse validation path")

    def _sparse_to_dense_edge_labels(self, edge_index, edge_attr, batch, ptr, bs, max_n, device):
        """(batch, ptr) 为图边界；返回 (bs, max_n, max_n) long，0=无边，1..K=子类型。"""
        dense = torch.zeros(bs, max_n, max_n, dtype=torch.long, device=device)
        if edge_index.numel() == 0:
            return dense
        if edge_attr.dim() > 1:
            edge_labels = edge_attr.argmax(dim=-1).long().clamp(0, self.out_dims.E - 1)
        else:
            edge_labels = edge_attr.long().clamp(0, self.out_dims.E - 1)
        b_idx = batch[edge_index[0]]
        offset = ptr[b_idx]
        i_local = (edge_index[0] - offset).clamp(0, max_n - 1)
        j_local = (edge_index[1] - offset).clamp(0, max_n - 1)
        dense[b_idx, i_local, j_local] = edge_labels
        return dense

    def compute_edge_nll_from_discrete(self, pred_E, true_E, node_mask, test=False):
        """在「所有可能边」上由离散预测/真实标签计算 edge NLL（存在性 + 子类型），用于采样式验证。
        pred_E, true_E: (bs, n, n) long，0=无边，1..K=子类型。
        """
        node_mask_bool = node_mask.bool()
        bs, n = node_mask_bool.shape
        eye = torch.eye(n, device=node_mask_bool.device).bool().unsqueeze(0)
        edge_mask = node_mask_bool.unsqueeze(2) & node_mask_bool.unsqueeze(1) & (~eye)
        pred_flat = pred_E[edge_mask].long().clamp(0, self.out_dims.E - 1)
        true_flat = true_E[edge_mask].long().clamp(0, self.out_dims.E - 1)
        if pred_flat.numel() == 0:
            return pred_flat.sum() * 0.0
        eps = 1e-6
        # 存在性：真实 0/1，预测为硬 0/1，平滑为概率后 BCE
        labels_exist = (true_flat > 0).float()
        p_pred_exist = (pred_flat > 0).float() * (1.0 - 2 * eps) + eps
        exist_nll = F.binary_cross_entropy(
            p_pred_exist, labels_exist, reduction="mean"
        )
        # 子类型：仅在真实正边上算 CE；预测为 one-hot 平滑
        pos_mask = true_flat > 0
        num_classes = self.out_dims.E
        if pos_mask.any():
            true_sub = (true_flat[pos_mask] - 1).clamp(0, num_classes - 2)
            pred_sub = (pred_flat[pos_mask] - 1).clamp(0, num_classes - 2)
            K = num_classes - 1
            pred_probs = torch.full((pos_mask.sum().item(), K), eps, device=pred_flat.device, dtype=torch.float)
            pred_probs.scatter_(1, pred_sub.unsqueeze(1), 1.0 - eps * (K - 1))
            subtype_nll = F.nll_loss(
                    torch.log(pred_probs.clamp(min=eps)),
                    true_sub,
                    reduction="mean",
            )
        else:
            subtype_nll = pred_flat.sum() * 0.0
        eps2 = 1e-8
        pos_ratio = pos_mask.float().mean().clamp(min=eps2).detach()
        lambda1 = float(getattr(self.train_loss, "edge_exist_weight", 1.0))
        lambda2 = float(getattr(self.train_loss, "edge_subtype_weight", 1.0)) / float(pos_ratio.item())
        total_nll = lambda1 * exist_nll + lambda2 * subtype_nll
        # logp：存在性 = mean log p(正确类)
        exist_logp = (
            labels_exist * torch.log(p_pred_exist.clamp(min=eps2))
            + (1 - labels_exist) * torch.log((1 - p_pred_exist).clamp(min=eps2))
        ).mean()
        if pos_mask.any():
            subtype_logp = torch.log(
                    pred_probs.gather(1, true_sub.unsqueeze(1)).squeeze(1).clamp(min=eps2)
            ).mean()
        else:
            subtype_logp = pred_flat.sum() * 0.0
        exist_kl = (-exist_logp).detach()
        subtype_kl = (-subtype_logp).detach()
        if test:
            self.test_nll(total_nll.detach().unsqueeze(0))
            self.test_exist_nll(exist_nll.detach().unsqueeze(0))
            self.test_subtype_nll(subtype_nll.detach().unsqueeze(0))
            self.test_exist_kl(exist_kl.detach().unsqueeze(0))
            self.test_subtype_kl(subtype_kl.detach().unsqueeze(0))
            self.test_exist_logp(exist_logp.detach().unsqueeze(0))
            self.test_subtype_logp(subtype_logp.detach().unsqueeze(0))
        else:
            self.val_nll(total_nll.detach().unsqueeze(0))
            self.val_exist_nll(exist_nll.detach().unsqueeze(0))
            self.val_subtype_nll(subtype_nll.detach().unsqueeze(0))
            self.val_exist_kl(exist_kl.detach().unsqueeze(0))
            self.val_subtype_kl(subtype_kl.detach().unsqueeze(0))
            self.val_exist_logp(exist_logp.detach().unsqueeze(0))
            self.val_subtype_logp(subtype_logp.detach().unsqueeze(0))
        return total_nll

    def kl_prior(self, X, E, node_mask, charge):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        # 对于异质图，KL 先验计算使用全局转移矩阵作为近似
        # 因为 KL 先验通常值很小，对训练影响不大
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        if (
            self.heterogeneous
            and getattr(self.dataset_info, "edge_family_marginals", None)
            and getattr(self.dataset_info, "edge_family_offsets", None)
            and getattr(self.dataset_info, "fam_endpoints", None)
            and getattr(self.dataset_info, "type_offsets", None)
        ):
            limit_E = self._build_limit_E_hetero(X, node_mask).type_as(probE)
        else:
            limit_E = (
                    self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
            )

        if self.use_charge:
            prob_charge = charge @ Qtb.charge  # (bs, n, de_out)
            limit_charge = (
                    self.limit_dist.charge[None, None, :]
                    .expand(bs, n, -1)
                    .type_as(prob_charge)
            )
            limit_charge = limit_charge.clone()
        else:
            prob_charge = limit_charge = None

        # Make sure that masked rows do not contribute to the loss
        (
            limit_dist_X,
            limit_dist_E,
            probX,
            probE,
            limit_dist_charge,
            prob_charge,
        ) = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
            true_charge=limit_charge,
            pred_charge=prob_charge,
        )

        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction="none"
        )

        # not all edges are used for loss calculation
        E_mask = torch.logical_or(
            kl_distance_E.sum(-1).isnan(), kl_distance_E.sum(-1).isinf()
        )
        kl_distance_E[E_mask] = 0
        X_mask = torch.logical_or(
            kl_distance_X.sum(-1).isnan(), kl_distance_X.sum(-1).isinf()
        )
        kl_distance_X[X_mask] = 0

        loss = diffusion_utils.sum_except_batch(
            kl_distance_X
        ) + diffusion_utils.sum_except_batch(kl_distance_E)

        # The above code is using the Python debugger module `pdb` to set a breakpoint in the code.
        # When the code is executed, it will pause at this line and allow you to interactively debug
        # the program.

        if self.use_charge:
            kl_distance_charge = F.kl_div(
                    input=prob_charge.log(), target=limit_dist_charge, reduction="none"
            )
            kl_distance_charge[X_mask] = 0
            loss = loss + diffusion_utils.sum_except_batch(kl_distance_charge)

        assert (~loss.isnan()).any()

        return loss

    def _build_limit_E_hetero(self, X, node_mask):
        """Build per-family limit distribution for edges based on node types."""
        device = X.device
        edge_family_marginals = getattr(self.dataset_info, "edge_family_marginals", {})
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
        type_offsets = getattr(self.dataset_info, "type_offsets", {})

        if not edge_family_marginals or not edge_family_offsets or not fam_endpoints or not type_offsets:
            return self.limit_dist.E[None, None, None, :].expand(
                    X.size(0), X.size(1), X.size(1), -1
            )

        # node subtype -> node type id
        subtype_ids = X.argmax(dim=-1)
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_names_ordered = [t for t, _ in sorted_types]
        type_sizes = {}
        for i, (t, off) in enumerate(sorted_types):
            if i + 1 < len(sorted_types):
                    type_sizes[t] = sorted_types[i + 1][1] - off
            else:
                    type_sizes[t] = max(1, self.out_dims.X - off)

        num_types = len(type_names_ordered)
        node_type_ids = subtype_ids.new_full(subtype_ids.shape, -1)
        for tidx, tname in enumerate(type_names_ordered):
            off = type_offsets[tname]
            size = type_sizes.get(tname, 0)
            mask = (subtype_ids >= off) & (subtype_ids < off + size)
            node_type_ids[mask] = tidx
        node_type_ids[~node_mask] = -1

        # map type pair -> family index
        fam_list = sorted(edge_family_marginals.keys())
        fam_index = {f: i for i, f in enumerate(fam_list)}
        fallback_idx = len(fam_list)
        pair_table = torch.full((num_types, num_types), fallback_idx, device=device, dtype=torch.long)
        for fam_name, endpoints in fam_endpoints.items():
            if fam_name not in fam_index:
                    continue
            src_t = endpoints.get("src_type")
            dst_t = endpoints.get("dst_type")
            if src_t in type_names_ordered and dst_t in type_names_ordered:
                    s_idx = type_names_ordered.index(src_t)
                    d_idx = type_names_ordered.index(dst_t)
                    pair_table[s_idx, d_idx] = fam_index[fam_name]

        # build global marginal vectors for each family
        fam_global = []
        for fam_name in fam_list:
            fam_marginals = edge_family_marginals[fam_name]
            if not isinstance(fam_marginals, torch.Tensor):
                    fam_marginals = torch.tensor(fam_marginals, dtype=torch.float, device=device)
            else:
                    fam_marginals = fam_marginals.to(device)
            global_vec = torch.zeros(self.out_dims.E, device=device, dtype=fam_marginals.dtype)
            global_vec[0] = fam_marginals[0] if fam_marginals.numel() > 0 else 0.0
            offset = edge_family_offsets.get(fam_name, 0)
            next_offset = self.out_dims.E
            for _, o in edge_family_offsets.items():
                if o > offset and o < next_offset:
                    next_offset = o
            num_subtypes = max(next_offset - offset, 0)
            for i in range(1, min(num_subtypes + 1, fam_marginals.numel())):
                gid = offset + (i - 1)
                if 0 <= gid < self.out_dims.E:
                    global_vec[gid] = fam_marginals[i]
            fam_global.append(global_vec)

        # add fallback global marginal
        fam_global.append(self.limit_dist.E.to(device))
        fam_global = torch.stack(fam_global, dim=0)  # (F+1, de)

        src = node_type_ids.unsqueeze(2).clamp(min=0)
        dst = node_type_ids.unsqueeze(1).clamp(min=0)
        fam_idx = pair_table[src, dst]
        fam_idx = torch.where(
            (node_type_ids.unsqueeze(2) < 0) | (node_type_ids.unsqueeze(1) < 0),
            torch.full_like(fam_idx, fallback_idx),
            fam_idx,
        )

        return fam_global[fam_idx]

    def compute_Lt(self, X, E, y, charge, pred, noisy_data, node_mask, test):
        """Deprecated dense diffusion objective; sparse edge-prediction training does not use it."""
        raise NotImplementedError("compute_Lt is disabled in the simplified sparse edge-prediction path")

    def reconstruction_logp(self, t, X, E, node_mask, charge):
        """Deprecated dense reconstruction log-probability path."""
        raise NotImplementedError("reconstruction_logp is disabled in the simplified sparse path")

    def forward_sparse(self, sparse_noisy_data):
        node = sparse_noisy_data["node_t"]
        edge_attr = sparse_noisy_data["edge_attr_t"].float()
        edge_index = sparse_noisy_data["edge_index_t"].to(torch.int64)
        y = sparse_noisy_data["y_t"]
        batch = sparse_noisy_data["batch"].long()

        # Guard y dimension to match lin_in_y input (avoid shape mismatch)
        if (
            y is not None
            and hasattr(self.model, "lin_in_y")
            and self.model.lin_in_y is not None
            and y.dim() == 2
        ):
            expected_y = self.model.lin_in_y.in_features
            if y.size(-1) != expected_y:
                if y.size(-1) > expected_y:
                    y = y[:, :expected_y]
                else:
                    pad = y.new_zeros((y.size(0), expected_y - y.size(-1)))
                    y = torch.cat([y, pad], dim=-1)

        # 提取异质图元数据（如果启用）
        if self.heterogeneous and self.model.heterogeneous:
            from sparse_diffusion.utils_heterogeneous import extract_heterogeneous_metadata
            
            type_offsets = getattr(self.dataset_info, "type_offsets", None)
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", None)
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", None)
            num_node_types = len(node_type_names) if node_type_names else 0
            # node 可能已拼接 [node_t, charge, extraX, signnet]，metadata 提取只能使用真实节点状态维度。
            node_for_metadata = node
            if node.dim() > 1 and self.out_dims.X > 0 and node.size(-1) > self.out_dims.X:
                    node_for_metadata = node[:, : self.out_dims.X]
            
            metadata = extract_heterogeneous_metadata(
                    node_t=node_for_metadata,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    type_offsets=type_offsets,
                    node_type_names=node_type_names,
                    edge_family_offsets=edge_family_offsets,
                    fam_endpoints=fam_endpoints,
                    num_node_types=num_node_types,
                    num_edge_classes=self.out_dims.E,
            )
            
            return self.model(
                    node, edge_attr, edge_index, y, batch,
                    node_type_ids=metadata.get("node_type_ids"),
                    node_subtype_ids=metadata.get("node_subtype_ids"),
                    relation_type_ids=metadata.get("relation_type_ids"),
                    edge_family_ids=metadata.get("edge_family_ids"),
            )
        else:
            return self.model(node, edge_attr, edge_index, y, batch)

    def forward(self, noisy_data):
        """
        noisy data contains: node_t, comp_edge_index_t, comp_edge_attr_t, batch
        """
        # build the sparse_noisy_data for the forward function of the sparse model
        sparse_noisy_data = self.compute_extra_data(sparse_noisy_data=noisy_data)

        if self.sign_net and self.cfg.model.extra_features == "all":
            x = self.sign_net(
                    sparse_noisy_data["node_t"],
                    sparse_noisy_data["edge_index_t"],
                    sparse_noisy_data["batch"],
            )
            sparse_noisy_data["node_t"] = torch.hstack([sparse_noisy_data["node_t"], x])

        res = self.forward_sparse(sparse_noisy_data)

        return res

    def _edge_attr_to_discrete(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None or edge_attr.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        if edge_attr.dim() > 1:
            return edge_attr.argmax(dim=-1).long()
        return edge_attr.long()

    def _count_total_triangles_sparse(
        self,
        edge_index: torch.Tensor,
        edge_attr_discrete: torch.Tensor,
        num_nodes: int,
    ) -> int:
        if (
            edge_index is None
            or edge_attr_discrete is None
            or edge_index.numel() == 0
            or edge_attr_discrete.numel() == 0
            or num_nodes <= 2
        ):
            return 0
        exist_mask = edge_attr_discrete != 0
        if not exist_mask.any():
            return 0

        ei = edge_index[:, exist_mask]
        if ei.numel() == 0:
            return 0

        src = ei[0].detach().cpu().tolist()
        dst = ei[1].detach().cpu().tolist()
        neighbors = [set() for _ in range(int(num_nodes))]
        for u, v in zip(src, dst):
            if u == v:
                    continue
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        triangles = 0
        for u in range(int(num_nodes)):
            nu = neighbors[u]
            if len(nu) < 2:
                    continue
            for v in nu:
                if v <= u:
                    continue
                common = nu.intersection(neighbors[v])
                for w in common:
                    if w > v:
                        triangles += 1
        return int(triangles)

    def _replace_edges_with_labels(
        self,
        base_edge_index: torch.Tensor,
        base_edge_attr_discrete: torch.Tensor,
        repl_edge_index: torch.Tensor,
        repl_edge_attr_discrete: torch.Tensor,
    ):
        edge_map = {}
        if base_edge_index is not None and base_edge_index.numel() > 0:
            bu = base_edge_index[0].detach().cpu().tolist()
            bv = base_edge_index[1].detach().cpu().tolist()
            bt = base_edge_attr_discrete.detach().cpu().tolist()
            for u, v, t in zip(bu, bv, bt):
                    edge_map[(int(u), int(v))] = int(t)
        if repl_edge_index is not None and repl_edge_index.numel() > 0:
            ru = repl_edge_index[0].detach().cpu().tolist()
            rv = repl_edge_index[1].detach().cpu().tolist()
            rt = repl_edge_attr_discrete.detach().cpu().tolist()
            for u, v, t in zip(ru, rv, rt):
                    edge_map[(int(u), int(v))] = int(t)

        if len(edge_map) == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            empty_attr = torch.zeros((0,), dtype=torch.long, device=self.device)
            return empty_index, empty_attr

        keys = list(edge_map.keys())
        vals = [edge_map[k] for k in keys]
        new_edge_index = torch.tensor(keys, dtype=torch.long, device=self.device).t().contiguous()
        new_edge_attr = torch.tensor(vals, dtype=torch.long, device=self.device)
        return new_edge_index, new_edge_attr

    def _test_sampling_uses_full_steps(self):
        return bool(getattr(self.cfg.general, "test_sampling_full_steps", True)) and bool(
            getattr(getattr(self, "trainer", None), "testing", False)
        )

    def _sampling_step_size(self):
        if self._test_sampling_uses_full_steps():
            return 1
        step = getattr(self.cfg.general, "sampling_skip", None)
        if step is None:
            step = int(self.cfg.general.skip)
        else:
            step = int(step)
        return max(1, min(int(step), int(self.T)))

    def _test_sampling_metrics_every(self):
        if not bool(getattr(getattr(self, "trainer", None), "testing", False)):
            return 0
        every = int(getattr(self.cfg.general, "test_sampling_metrics_every", 0) or 0)
        return max(0, every)

    def _snapshot_sparse_sample_for_metrics(self, sparse_sampled_data, node_override=None):
        sample = sparse_sampled_data.to_device("cpu")
        if sample.edge_attr.dim() > 1:
            sample.edge_attr = sample.edge_attr.argmax(-1)
        if node_override is not None:
            sample.node = node_override.detach().cpu().long()
        elif sample.node.dim() > 1:
            sample.node = sample.node.argmax(-1)
        else:
            sample.node = sample.node.long().cpu()
        if self.use_charge and getattr(sample, "charge", None) is not None and sample.charge.dim() > 1:
            sample.charge = sample.charge.argmax(-1) - 1
        return sample

    def _record_test_intermediate_sample(self, remaining_step, sparse_sampled_data, node_override=None):
        every = self._test_sampling_metrics_every()
        if every <= 0 or int(remaining_step) % every != 0:
            return
        if not hasattr(self, "_test_intermediate_sample_chunks"):
            self._test_intermediate_sample_chunks = {}
        step_key = int(remaining_step)
        self._test_intermediate_sample_chunks.setdefault(step_key, []).append(
            self._snapshot_sparse_sample_for_metrics(sparse_sampled_data, node_override=node_override)
        )

    def _compute_and_save_test_intermediate_metrics(self):
        chunks_by_step = getattr(self, "_test_intermediate_sample_chunks", None)
        if not chunks_by_step:
            return
        if getattr(self, "local_rank", 0) != 0:
            return
        summary = {}
        for step_key in sorted(chunks_by_step.keys(), reverse=True):
            samples_step = utils.concat_sparse_graphs(chunks_by_step[step_key])
            self.test_sampling_metrics.reset()
            key_suffix = f"/step{step_key:03d}"
            to_log, _ = self.test_sampling_metrics.compute_all_metrics(
                samples_step,
                self.current_epoch,
                local_rank=self.local_rank,
                key_suffix=key_suffix,
                chart_title_suffix=f" step {step_key}",
            )
            summary[str(step_key)] = to_log
            json_path = os.path.join(os.getcwd(), f"test_sampling_step{step_key:03d}.json")
            with open(json_path, "w") as file:
                json.dump(to_log, file)
            table_str = format_structure_metrics_table(to_log, key_prefix=f"test/step{step_key:03d}")
            if table_str.strip():
                table_path = os.path.join(os.getcwd(), f"test_sampling_step{step_key:03d}_structure_metrics.txt")
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(table_str)
            print(f"[TEST-SAMPLING] saved intermediate metrics for remaining step {step_key}: {json_path}")
        with open(os.path.join(os.getcwd(), "test_sampling_intermediate_metrics.json"), "w") as file:
            json.dump(summary, file)
        self.test_sampling_metrics.reset()

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (node_types, charge, positions)
        """
        if num_nodes is None:
            num_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            num_nodes = num_nodes * torch.ones(
                    batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            num_nodes = num_nodes
        num_max = torch.max(num_nodes)

        # Build the masks
        arange = (
            torch.arange(num_max, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        node_mask = arange < num_nodes.unsqueeze(1)

        # Sample noise (z_T): 异质图时用训练集每族平均边数初始化边，使 m 在首步就有合理起点
        if (self.heterogeneous
                    and getattr(self.dataset_info, "edge_family_avg_edge_counts", None)
                    and getattr(self.dataset_info, "fam_endpoints", None)
                    and getattr(self.dataset_info, "type_offsets", None)):
            sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise_heterogeneous(
                    limit_dist=self.limit_dist, node_mask=node_mask, dataset_info=self.dataset_info,
                    out_dims_E=self.out_dims.E, device=self.device
            )
        else:
            sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise(
                    limit_dist=self.limit_dist, node_mask=node_mask
            )

        # Anchor strategy for heterogeneous edge diffusion:
        # keep a fixed node-subtype snapshot from z_T to avoid recomputing per-step
        # relation buckets from changing node states.
        if self.heterogeneous:
            sparse_sampled_data.anchor_node_subtype = (
                    sparse_sampled_data.node.argmax(dim=-1)
                    if sparse_sampled_data.node.dim() > 1
                    else sparse_sampled_data.node.long()
            )
            if str(getattr(self.cfg.model, "sampling_block_mode", "uniform")).lower() == "type_template":
                    self._ensure_hetero_block_templates_from_data()
                    sparse_sampled_data.pseudo_blocks = self._build_type_template_pseudo_blocks(
                    sparse_sampled_data.anchor_node_subtype.long().to(self.device),
                    sparse_sampled_data.batch.long().to(self.device),
                    getattr(self.dataset_info, "type_offsets", {}),
                    )
                    sparse_sampled_data = self._apply_block_marginal_initial_edges(sparse_sampled_data)
                    sparse_sampled_data = self._apply_block_template_initial_edges(sparse_sampled_data)

        assert number_chain_steps < self.T
        chain = utils.SparseChainPlaceHolder(keep_chain=keep_chain)
        
        # 记录采样过程：初始化状态
        sampling_log = []
        trace_triangles = bool(getattr(self.cfg.general, "log_sampling_triangle_trace", False))
        prev_step_triangles = None
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            # 记录初始状态
            init_edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
            init_node_count = sparse_sampled_data.node.shape[0]
            edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
            non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
            init_entry = {
                    'step': self.T,
                    't_norm': 1.0,
                    'num_nodes': init_node_count,
                    'num_edges': init_edge_count,
                    'num_nonzero_edges': non_zero_edges
            }
            if trace_triangles:
                    init_triangles = self._count_total_triangles_sparse(
                    sparse_sampled_data.edge_index,
                    edge_attr_discrete.long(),
                    int(init_node_count),
                    )
                    init_entry["triangles"] = int(init_triangles)
                    prev_step_triangles = init_triangles
            sampling_log.append(init_entry)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        step = self._sampling_step_size()
        time_range = torch.arange(0, self.T, step)
        n_steps = len(time_range)
        disable_tqdm = self.local_rank != 0 if hasattr(self, 'local_rank') else False
        for s_int in tqdm(reversed(time_range), total=n_steps, disable=disable_tqdm):
            s_array = (s_int * torch.ones((batch_size, 1))).to(self.device)
            t_array = s_array + step
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            # print(s_norm, t_norm)

            # Sample z_s
            sparse_sampled_data = self.sample_p_zs_given_zt(
                    s_norm, t_norm, sparse_sampled_data
            )
            
            # 记录采样过程：每一步的状态
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                    edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
                    node_count = sparse_sampled_data.node.shape[0]
                    edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
                    non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
                    step_entry = {
                        'step': s_int,
                        't_norm': t_norm[0].item(),
                        'num_nodes': node_count,
                        'num_edges': edge_count,
                        'num_nonzero_edges': non_zero_edges
                    }
                    if trace_triangles:
                        step_triangles = self._count_total_triangles_sparse(
                            sparse_sampled_data.edge_index,
                            edge_attr_discrete.long(),
                            int(node_count),
                        )
                        step_entry["triangles"] = int(step_triangles)
                        step_entry["delta_triangles"] = int(
                            0 if prev_step_triangles is None else step_triangles - prev_step_triangles
                        )
                        prev_step_triangles = step_triangles
                    sampling_log.append(step_entry)
            self._record_test_intermediate_sample(int(s_int), sparse_sampled_data)
            # keep_chain can be very small, e.g., 1
            if ((s_int * number_chain_steps) % self.T == 0) and (keep_chain != 0):
                    chain.append(sparse_sampled_data)

        # 记录最终状态
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            final_edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
            final_node_count = sparse_sampled_data.node.shape[0]
            edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
            final_non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
            final_entry = {
                'step': 0,
                't_norm': 0.0,
                'num_nodes': final_node_count,
                'num_edges': final_edge_count,
                'num_nonzero_edges': final_non_zero_edges
            }
            if trace_triangles:
                final_triangles = self._count_total_triangles_sparse(
                    sparse_sampled_data.edge_index,
                    edge_attr_discrete.long(),
                    int(final_node_count),
                )
                final_entry["triangles"] = int(final_triangles)
                final_entry["delta_triangles"] = int(
                    0 if prev_step_triangles is None else final_triangles - prev_step_triangles
                )
            sampling_log.append(final_entry)
            # 保存采样记录到文件（统一放到 output/sparse_diffusion 下）
            try:
                import json
                base_dir = utils.get_model_output_root()
                log_dir = os.path.join(
                    base_dir,
                    "sampling_logs",
                    f"{self.cfg.general.name}",
                    f"epoch{self.current_epoch}",
                )
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"batch_{batch_id}_sampling_log.json")
                sampling_log_serializable = []
                for entry in sampling_log:
                    serializable_entry = {
                        'step': int(entry['step']),
                        't_norm': float(entry['t_norm']),
                        'num_nodes': int(entry['num_nodes']),
                        'num_edges': int(entry['num_edges']),
                        'num_nonzero_edges': int(entry['num_nonzero_edges'])
                    }
                    if "triangles" in entry:
                        serializable_entry["triangles"] = int(entry["triangles"])
                    if "delta_triangles" in entry:
                        serializable_entry["delta_triangles"] = int(entry["delta_triangles"])
                    sampling_log_serializable.append(serializable_entry)
                with open(log_file, 'w') as f:
                    json.dump(sampling_log_serializable, f, indent=2)
            except Exception:
                pass
        
        # get generated graphs
        generated_graphs = self._snapshot_sparse_sample_for_metrics(sparse_sampled_data)
        # Visualization is disabled in the simplified sparse path; metrics use the returned graph directly.
        return generated_graphs

    @torch.no_grad()
    def sample_batch_fixed_nodes(
        self,
        fixed_data,
        keep_chain: int = 0,
        number_chain_steps: int = None,
        save_final: int = 1,
    ):
        """固定节点、边从噪声初始化，去噪生成边。用于 DBLP 等「同节点、造边」场景。
        fixed_data: PyG Data，需有 x (one-hot 或离散子类别), batch
        """
        if number_chain_steps is None:
            number_chain_steps = getattr(self.cfg.general, "number_chain_steps", 10)
        node = fixed_data.x
        batch = fixed_data.batch.to(self.device)
        if node.dim() > 1:
            fixed_node_subtypes = node.argmax(dim=-1).long()
        else:
            fixed_node_subtypes = node.long()
        batch_size = int(batch.max().item()) + 1

        if not (
            self.heterogeneous
            and getattr(self.dataset_info, "edge_family_avg_edge_counts", None)
            and getattr(self.dataset_info, "fam_endpoints", None)
            and getattr(self.dataset_info, "type_offsets", None)
        ):
            raise RuntimeError(
                    "sample_batch_fixed_nodes requires heterogeneous mode with "
                    "edge_family_avg_edge_counts, fam_endpoints, type_offsets."
            )
        sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise_heterogeneous_fixed_nodes(
            fixed_node_subtypes=fixed_node_subtypes,
            batch=batch,
            limit_dist=self.limit_dist,
            dataset_info=self.dataset_info,
            out_dims_E=self.out_dims.E,
            device=self.device,
        )
        sparse_sampled_data.anchor_node_subtype = fixed_node_subtypes.to(self.device)
        if str(getattr(self.cfg.model, "sampling_block_mode", "uniform")).lower() == "type_template":
            self._ensure_hetero_block_templates_from_data(fixed_data)
            sparse_sampled_data.pseudo_blocks = self._build_type_template_pseudo_blocks(
                    sparse_sampled_data.anchor_node_subtype.long().to(self.device),
                    sparse_sampled_data.batch.long().to(self.device),
                    getattr(self.dataset_info, "type_offsets", {}),
            )
            sparse_sampled_data = self._apply_block_marginal_initial_edges(sparse_sampled_data)
            sparse_sampled_data = self._apply_block_template_initial_edges(sparse_sampled_data)

        chain = utils.SparseChainPlaceHolder(keep_chain=keep_chain)
        step = self._sampling_step_size()
        time_range = torch.arange(0, self.T, step)
        n_steps = len(time_range)
        verbose_sampling = getattr(self.cfg.general, "verbose_sampling", False)
        disable_tqdm = self.local_rank != 0 if hasattr(self, "local_rank") else False
        from tqdm import tqdm
        for step_index, s_int in enumerate(tqdm(reversed(time_range), total=n_steps, disable=disable_tqdm)):
            if verbose_sampling and hasattr(self, "local_rank") and self.local_rank == 0:
                    self._verbose_step_index = step_index
                    self._verbose_total_steps = n_steps
                    self._verbose_s_float = (s_int * 1.0) / self.T
                    self._verbose_t_float = (s_int + step) * 1.0 / self.T
            s_array = (s_int * torch.ones((batch_size, 1))).to(self.device)
            t_array = s_array + step
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            sparse_sampled_data = self.sample_p_zs_given_zt(s_norm, t_norm, sparse_sampled_data)
            if verbose_sampling and hasattr(self, "local_rank") and self.local_rank == 0:
                    self._verbose_step_index = None
                    self._verbose_total_steps = None
            self._record_test_intermediate_sample(int(s_int), sparse_sampled_data, node_override=fixed_node_subtypes)
            if ((s_int * number_chain_steps) % self.T == 0) and (keep_chain != 0):
                    chain.append(sparse_sampled_data)

        generated_graphs = self._snapshot_sparse_sample_for_metrics(
            sparse_sampled_data, node_override=fixed_node_subtypes
        )
        return generated_graphs

    def sample_node(self, pred_X, p_s_and_t_given_0_X, node_mask):
        # Normalize predictions
        pred_X = F.softmax(pred_X, dim=-1)  # bs, n, d0
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        X_t = diffusion_utils.sample_discrete_node_features(prob_X, node_mask)

        return X_t, prob_X

    def sample_edge(self, pred_E, p_s_and_t_given_0_E, node_mask):
        # Normalize predictions
        bs, n, n, de = pred_E.shape
        pred_E = F.softmax(pred_E, dim=-1)  # bs, n, n, d0
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, de)

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        E_t = diffusion_utils.sample_discrete_edge_features(prob_E, node_mask)

        return E_t, prob_E

    def sample_node_edge(
        self, pred, p_s_and_t_given_0_X, p_s_and_t_given_0_E, node_mask
    ):
        _, prob_X = self.sample_node(pred.X, p_s_and_t_given_0_X, node_mask)
        _, prob_E = self.sample_edge(pred.E, p_s_and_t_given_0_E, node_mask)

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        return sampled_s

    def sample_sparse_node(self, pred_node, p_s_and_t_given_0_X, node_type_mask=None):
        """
        Sample node subtypes, with optional mask to restrict to same node type.
        
        Args:
            pred_node: (N, dx) predicted node logits
            p_s_and_t_given_0_X: (N, dx, dx) transition probabilities
            node_type_mask: (N, dx) mask where 1.0 allows sampling, 0.0 forbids.
                          If None, no restriction (backward compatibility).
        """
        # Normalize predictions
        pred_X = F.softmax(pred_node, dim=-1)  # N, dx
        # Dim of the second tensor: N, dx, dx
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # N, dx, dx
        unnormalized_prob_X = weighted_X.sum(dim=1)  # N, dx
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = (
            1e-5  # TODO: delete/masking?
        )
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # N, dx
        
        # Apply node type mask: restrict each node to its own type's subtypes
        if node_type_mask is not None:
            prob_X = prob_X * node_type_mask
            row_sum = prob_X.sum(dim=-1, keepdim=True)
            all_zero = (row_sum.squeeze(-1) == 0)
            # If all subtypes are masked out, fallback to uniform within current type
            # (This should not happen if mask is correct, but safety check)
            if all_zero.any():
                    # Fallback: use original probabilities for masked nodes
                    prob_X[all_zero] = unnormalized_prob_X[all_zero] / unnormalized_prob_X[all_zero].sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        X_t = prob_X.multinomial(1)[:, 0]

        return X_t

    def sample_sparse_edge(self, pred_edge, p_s_and_t_given_0_E, edge_type_mask=None):
        """
        边预测：可预测空间由关系族决定（edge_type_mask 已按 query_edge_family 限制）。

        - 机构-作者（隶属）：仅 no-edge / 隶属一种状态 → 等价于**存在性预测**；
        - 作者-论文（撰写）：no-edge / 一作 / 二作 / 通信等 → **先预测存在性，再在存在的边上预测子类别**。
        默认分层采样：先「有无边」，再「有边时」采样子类型，与训练时的分层损失一致。
        """
        # Normalize predictions
        pred_E = F.softmax(pred_edge, dim=-1)  # N, d0
        # Dim of the second tensor: N, d0, dt-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # N, d0, dt-1
        unnormalized_prob_E = weighted_E.sum(dim=1)  # N, dt-1
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        
        # 异质图：仅允许 (src_type,dst_type) 在 fam_endpoints 中合法的边类型，避免 Paper->Org 的 author_of 等非法关系
        if edge_type_mask is not None:
            prob_E = prob_E * edge_type_mask
            row_sum = prob_E.sum(dim=-1, keepdim=True)
            all_zero = (row_sum.squeeze(-1) == 0)
            prob_E[all_zero, 0] = 1.0  # 无合法边类型时只允许 no-edge
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # 分层采样：先采样「有无边」，再在「有边」时采样子类型，与训练时的分层损失一致
        no_edge_prob = prob_E[:, 0]  # (N,)
        exist_prob = prob_E[:, 1:].sum(dim=-1)  # (N,)
        u = torch.rand(prob_E.shape[0], device=prob_E.device, dtype=prob_E.dtype)
        has_edge = u < exist_prob
        E_t = torch.zeros(prob_E.shape[0], dtype=torch.long, device=prob_E.device)
        where_has = has_edge.nonzero(as_tuple=True)[0]
        if where_has.numel() > 0:
            sub = prob_E[where_has, 1:]
            sub_sum = sub.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sub = sub / sub_sum
            idx = sub.multinomial(1)[:, 0]  # 0-based 子类型下标
            E_t[where_has] = idx + 1  # 全局 ID：1 ~ d0-1
        return E_t

    def sample_sparse_node_edge(
        self,
        pred_node,
        pred_edge,
        p_s_and_t_given_0_X,
        p_s_and_t_given_0_E,
        pred_charge,
        p_s_and_t_given_0_charge,
        edge_type_mask=None,
        node_type_mask=None,
    ):
        # 采样顺序：先节点后边
        # 节点主类型不变（只允许子类型变化），所以 edge_type_mask（基于主类型的关系族）不需要更新
        sampled_node = self.sample_sparse_node(pred_node, p_s_and_t_given_0_X, node_type_mask).long()
        sampled_edge = self.sample_sparse_edge(pred_edge, p_s_and_t_given_0_E, edge_type_mask).long()

        if pred_charge.size(-1) > 0:
            sampled_charge = self.sample_sparse_node(
                    pred_charge, p_s_and_t_given_0_charge
            ).long()
        else:
            sampled_charge = pred_charge

        return sampled_node, sampled_edge, sampled_charge

    def sample_p_zs_given_zt(self, s_float, t_float, data):
        """One sparse denoising step.

        For type-template block sampling this follows SparseDiff-original's block-round
        dynamics: within one diffusion step, visit pseudo blocks in a random order and
        merge each block's sampled edges before moving to the next block.
        """
        node = data.node
        if node.dim() == 1:
            node_onehot = F.one_hot(node.long().clamp(0, self.out_dims.X - 1), num_classes=self.out_dims.X).float()
        else:
            node_onehot = node.float()
        batch = data.batch.long().to(self.device)
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)
            ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), counts.cumsum(0)])
        else:
            ptr = ptr.long().to(self.device)

        edge_index = data.edge_index.long().to(self.device)
        edge_attr = data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr_ids = edge_attr.long().clamp(0, self.out_dims.E - 1).to(self.device)
            edge_attr_onehot = F.one_hot(edge_attr_ids, num_classes=self.out_dims.E).float()
        else:
            edge_attr_onehot = edge_attr.float().to(self.device)
            edge_attr_ids = edge_attr_onehot.argmax(dim=-1).long()

        anchor_node_subtype = getattr(data, "anchor_node_subtype", None)
        if anchor_node_subtype is None:
            anchor_node_subtype = node_onehot.argmax(dim=-1).long()
        anchor_node_subtype = anchor_node_subtype.long().to(self.device)
        pseudo_blocks = getattr(data, "pseudo_blocks", None)

        query_rounds = []
        use_template_blocks = (
            str(getattr(self.cfg.model, "sampling_block_mode", "uniform")).lower() == "type_template"
            and pseudo_blocks is not None
            and len(pseudo_blocks) > 0
        )
        if use_template_blocks:
            inter_state = {}
            order = torch.randperm(len(pseudo_blocks), device=self.device).tolist()
            for block_idx in order:
                tpl = self._sampling_type_template_query_edges(
                    anchor_node_subtype,
                    batch,
                    [pseudo_blocks[int(block_idx)]],
                    block_ids=[int(block_idx)],
                    inter_state=inter_state,
                    all_blocks=pseudo_blocks,
                )
                if tpl is None:
                    continue
                query_edge_index, query_edge_batch, query_edge_family = tpl
                if query_edge_index is not None and query_edge_index.numel() > 0:
                    query_rounds.append((query_edge_index, query_edge_batch, query_edge_family))
            if query_rounds:
                self._log_block_query_coverage(inter_state)

        if not query_rounds:
            query_edge_index, query_edge_batch = self._sample_heterogeneous_uniform_query_for_sampling(
                anchor_node_subtype, batch, ptr
            )
            if query_edge_index is not None and query_edge_index.numel() > 0:
                query_rounds.append((query_edge_index, query_edge_batch, None))

        if not query_rounds:
            out = data.clone() if hasattr(data, "clone") else data
            out.anchor_node_subtype = anchor_node_subtype
            out.pseudo_blocks = pseudo_blocks
            return out

        base_edge_index = edge_index
        base_edge_attr_ids = edge_attr_ids
        base_edge_attr_onehot = edge_attr_onehot
        cur_edge_index = edge_index
        cur_edge_attr_ids = edge_attr_ids
        cur_edge_attr_onehot = edge_attr_onehot
        use_autoregressive_context = bool(getattr(self.cfg.model, "autoregressive", False))

        for query_edge_index, _query_edge_batch, query_edge_family in query_rounds:
            if query_edge_index is None or query_edge_index.numel() == 0:
                continue
            if use_autoregressive_context:
                fw_edge_index = cur_edge_index
                fw_edge_attr_onehot = cur_edge_attr_onehot
            else:
                fw_edge_index = base_edge_index
                fw_edge_attr_onehot = base_edge_attr_onehot

            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=query_edge_index,
                clean_edge_index=fw_edge_index,
                clean_edge_attr=fw_edge_attr_onehot,
                heterogeneous=self.heterogeneous,
                for_message_passing=True,
                total_num_nodes=node_onehot.shape[0],
            )

            comp_query_edge_index = comp_edge_index[:, query_mask]

            sparse_noisy_data = {
                "node_t": node_onehot,
                "X_t": node_onehot,
                "edge_index_t": fw_edge_index,
                "edge_attr_t": fw_edge_attr_onehot,
                "comp_edge_index_t": comp_edge_index,
                "comp_edge_attr_t": comp_edge_attr,
                "y_t": data.y,
                "batch": batch,
                "ptr": ptr,
                "charge_t": getattr(data, "charge", torch.zeros((node_onehot.shape[0], 0), device=self.device)),
                "t_float": t_float,
            }
            pred = self.forward(sparse_noisy_data)
            comp_query_logits = pred.edge_attr[query_mask]
            if comp_query_logits.numel() == 0:
                continue
            selected_query_edge_index, selected_query_family, query_logits = self._select_original_query_outputs(
                query_edge_index, query_edge_family, comp_query_edge_index, comp_query_logits
            )
            if query_logits.numel() == 0 or selected_query_edge_index.numel() == 0:
                continue
            query_sample = self._sample_edge_labels_hierarchical(query_logits, selected_query_family, selected_query_edge_index)

            cur_edge_index, cur_edge_attr_ids = self._replace_edges_with_labels(
                cur_edge_index,
                cur_edge_attr_ids,
                selected_query_edge_index,
                query_sample,
            )
            keep = cur_edge_attr_ids != 0
            cur_edge_index = cur_edge_index[:, keep]
            cur_edge_attr_ids = cur_edge_attr_ids[keep].clamp(0, self.out_dims.E - 1)
            cur_edge_attr_onehot = F.one_hot(cur_edge_attr_ids, num_classes=self.out_dims.E).float()

        out = utils.SparsePlaceHolder(
            node=node_onehot,
            edge_index=cur_edge_index,
            edge_attr=cur_edge_attr_onehot,
            y=data.y,
            batch=batch,
            charge=getattr(data, "charge", None),
            ptr=ptr,
        )
        out.anchor_node_subtype = anchor_node_subtype
        out.pseudo_blocks = pseudo_blocks
        out = self._apply_block_family_budget_projection(out)
        return out

    def _sample_heterogeneous_uniform_query_for_sampling(self, anchor_node_subtype, batch, ptr):
        """Sample legal directed query edges by relation-family endpoint types."""
        device = self.device
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        edge_family_avg = getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        if not self.heterogeneous or not type_offsets or not fam_endpoints:
            return sample_query_edges(num_nodes_per_graph=ptr.diff(), edge_proportion=self.edge_fraction)

        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_ranges = {}
        for i, (name, offset) in enumerate(sorted_types):
            nxt = sorted_types[i + 1][1] if i + 1 < len(sorted_types) else self.out_dims.X
            type_ranges[name] = (int(offset), int(nxt))

        edge_parts = []
        batch_parts = []
        active_fams = [f for f in fam_endpoints if edge_family_avg.get(f, 0) > 0]
        if not active_fams:
            active_fams = list(fam_endpoints.keys())
        for b in range(int(batch.max().item()) + 1):
            nodes_b = torch.where(batch == b)[0]
            n = int(nodes_b.numel())
            if n <= 1:
                continue
            total_budget = max(1, int(math.ceil(float(self.edge_fraction) * n * (n - 1))))
            per_fam = max(1, int(math.ceil(total_budget / max(len(active_fams), 1))))
            for fam_name in active_fams:
                endpoints = fam_endpoints.get(fam_name, {})
                src_t = endpoints.get("src_type")
                dst_t = endpoints.get("dst_type")
                if src_t not in type_ranges or dst_t not in type_ranges:
                    continue
                s0, s1 = type_ranges[src_t]
                d0, d1 = type_ranges[dst_t]
                src_nodes = nodes_b[(anchor_node_subtype[nodes_b] >= s0) & (anchor_node_subtype[nodes_b] < s1)]
                dst_nodes = nodes_b[(anchor_node_subtype[nodes_b] >= d0) & (anchor_node_subtype[nodes_b] < d1)]
                if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                    continue
                k = min(per_fam, int(src_nodes.numel() * dst_nodes.numel()))
                if src_t == dst_t:
                    k = min(k, max(int(src_nodes.numel() * (src_nodes.numel() - 1)), 0))
                if k <= 0:
                    continue
                src_idx = torch.randint(src_nodes.numel(), (k,), device=device)
                dst_idx = torch.randint(dst_nodes.numel(), (k,), device=device)
                src = src_nodes[src_idx]
                dst = dst_nodes[dst_idx]
                if src_t == dst_t:
                    neq = src != dst
                    src = src[neq]
                    dst = dst[neq]
                    if src.numel() == 0:
                        continue
                edge_parts.append(torch.stack([src, dst], dim=0))
                batch_parts.append(torch.full((src.numel(),), b, dtype=torch.long, device=device))
        if not edge_parts:
            return None, None
        return torch.cat(edge_parts, dim=1), torch.cat(batch_parts, dim=0)

    def compute_sparse_extra_data(self, sparse_noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""
        return utils.SparsePlaceHolder(
            node=sparse_noisy_data["X_t"],
            edge_index=sparse_noisy_data["edge_index_t"],
            edge_attr=sparse_noisy_data["edge_attr_t"],
            y=sparse_noisy_data["y_t"],
        )

    def compute_extra_data(self, sparse_noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input. When extra_features is disabled (null), skip all extra computation and concatenation."""
        node_t = sparse_noisy_data["node_t"].to(device=self.device, dtype=torch.float32)
        comp_edge_attr_t = sparse_noisy_data["comp_edge_attr_t"].to(device=self.device, dtype=torch.float32)
        y_t = sparse_noisy_data["y_t"].to(device=self.device, dtype=torch.float32)
        t_float = sparse_noisy_data["t_float"].to(device=self.device, dtype=torch.float32)

        if self._no_extra_features:
            # 不使用额外特征：不调用 extra_features/domain_features，不分配空张量，不拼接
            if self.use_charge:
                charge_t = sparse_noisy_data["charge_t"]
                if charge_t.dim() == 1:
                    charge_t = charge_t.unsqueeze(-1)
                charge_t = charge_t.to(device=self.device, dtype=torch.float32)
                node = torch.hstack([node_t, charge_t])
            else:
                node = node_t
            y = torch.hstack((y_t, t_float)).float()
            return {
                "node_t": node,
                "edge_index_t": sparse_noisy_data["comp_edge_index_t"],
                "edge_attr_t": comp_edge_attr_t,
                "y_t": y,
                "batch": sparse_noisy_data["batch"],
                "charge_t": sparse_noisy_data["charge_t"],
            }

        # get extra features
        extra_data = self.extra_features(sparse_noisy_data)
        if type(extra_data) == tuple:
            extra_data = extra_data[0]
        extra_mol_data = self.domain_features(sparse_noisy_data)
        if type(extra_mol_data) == tuple:
            extra_mol_data = extra_mol_data[0]

        ptr = sparse_noisy_data["ptr"]
        batch = sparse_noisy_data["batch"]
        n_node = ptr.diff().max()
        node_mask = utils.ptr_to_node_mask(ptr, batch, n_node)

        edge_batch = sparse_noisy_data["batch"][
            sparse_noisy_data["comp_edge_index_t"][0].long()
        ]
        edge_batch = edge_batch.long()
        dense_comp_edge_index = (
            sparse_noisy_data["comp_edge_index_t"]
            - ptr[edge_batch]
            + edge_batch * n_node
        )
        comp_edge_index0 = dense_comp_edge_index[0] % n_node
        comp_edge_index1 = dense_comp_edge_index[1] % n_node

        extraE = extra_data.E[
            edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
        ]

        def _align_extra_X(X, n_node, node_mask):
            if X.dim() < 2:
                return X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]
            if X.dim() == 2:
                return X
            bs, n_max, feat = X.shape[0], X.shape[1], X.shape[2]
            if n_max >= n_node:
                slice_X = X[:, :n_node, :]
            else:
                pad = X.new_zeros(bs, n_node - n_max, feat)
                slice_X = torch.cat([X, pad], dim=1)
            return slice_X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]

        extraX = _align_extra_X(extra_data.X, n_node, node_mask)

        if hasattr(extra_mol_data, 'X'):
            extra_mol_X = _align_extra_X(extra_mol_data.X, n_node, node_mask)
            extra_mol_E = extra_mol_data.E[
                edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
            ]
            extra_mol_y = extra_mol_data.y
        elif hasattr(extra_mol_data, 'node'):
            extra_mol_X = _align_extra_X(extra_mol_data.node, n_node, node_mask)
            if hasattr(extra_mol_data, 'edge_attr') and extra_mol_data.edge_attr is not None:
                ea = extra_mol_data.edge_attr
                num_comp = sparse_noisy_data["comp_edge_index_t"].shape[1]
                if ea.dim() == 2:
                    if ea.shape[0] >= num_comp:
                        extra_mol_E = ea[:num_comp].to(dtype=extraE.dtype, device=extraE.device)
                    else:
                        pad = ea.new_zeros(num_comp - ea.shape[0], ea.shape[1])
                        extra_mol_E = torch.cat([ea, pad], dim=0).to(dtype=extraE.dtype, device=extraE.device)
                else:
                    extra_mol_E = ea[
                        edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
                    ]
            else:
                extra_mol_E = torch.zeros_like(extraE)
            extra_mol_y = extra_mol_data.y
        else:
            extra_mol_X = torch.zeros_like(extraX)
            extra_mol_E = torch.zeros_like(extraE)
            extra_mol_y = torch.zeros_like(extra_data.y)

        extra_mol_X = extra_mol_X.to(dtype=extraX.dtype, device=extraX.device)
        extra_mol_E = extra_mol_E.to(dtype=extraE.dtype, device=extraE.device)
        extra_mol_y = extra_mol_y.to(dtype=extra_data.y.dtype, device=extra_data.y.device)

        extraX_cat = extraX if (extra_mol_X.shape[-1] == 0) else (
            extra_mol_X if (extraX.shape[-1] == 0) else torch.hstack([extra_mol_X, extraX])
        )
        extraE_cat = extraE if (extra_mol_E.shape[-1] == 0) else (
            extra_mol_E if (extraE.shape[-1] == 0) else torch.hstack([extraE, extra_mol_E])
        )
        extra_y_cat = extra_data.y if (extra_mol_y.shape[-1] == 0) else (
            extra_mol_y if (extra_data.y.shape[-1] == 0) else torch.hstack([extra_data.y, extra_mol_y])
        )
        extraX, extraE, extra_y = self.scale_extra_data(extraX_cat, extraE_cat, extra_y_cat)

        if self.use_charge:
            charge_t = sparse_noisy_data["charge_t"]
            if charge_t.dim() == 1:
                charge_t = charge_t.unsqueeze(-1)
            charge_t = charge_t.to(device=self.device, dtype=torch.float32)
            node = torch.hstack([node_t, charge_t, extraX.to(device=self.device, dtype=torch.float32)])
        else:
            node = torch.hstack([node_t, extraX.to(device=self.device, dtype=torch.float32)])

        comp_edge_attr = torch.hstack([comp_edge_attr_t, extraE.to(device=self.device, dtype=torch.float32)])
        y = torch.hstack((y_t, t_float, extra_y.to(device=self.device, dtype=torch.float32))).float()

        # get the input for the forward function
        # TODO: change to PlaceHolder
        extra_sparse_noisy_data = {
            "node_t": node,
            "edge_index_t": sparse_noisy_data["comp_edge_index_t"],
            "edge_attr_t": comp_edge_attr,
            "y_t": y,
            "batch": sparse_noisy_data["batch"],
            "charge_t": sparse_noisy_data["charge_t"],
        }

        return extra_sparse_noisy_data

    def get_scaling_layers(self):
        node_scaling_layer, edge_scaling_layer, graph_scaling_layer = None, None, None
        if self.scaling_layer:
            extra_dim = self.in_dims.X - self.out_dims.X
            if extra_dim > 0:
                node_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )
            extra_dim = self.in_dims.E - self.out_dims.E
            if extra_dim > 0:
                edge_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )
            extra_dim = self.in_dims.y - self.out_dims.y - 1
            if extra_dim > 0:
                graph_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )

        return node_scaling_layer, edge_scaling_layer, graph_scaling_layer

    def scale_extra_data(self, extraX, extraE, extra_y):
        if self.node_scaling_layer is not None:
            extraX = self.node_scaling_layer(extraX.permute(1, 0)).permute(1, 0)
        if self.edge_scaling_layer is not None:
            extraE = self.edge_scaling_layer(extraE.permute(1, 0)).permute(1, 0)
        if self.graph_scaling_layer is not None:
            extra_y = self.graph_scaling_layer(extra_y.permute(1, 0)).permute(1, 0)

        return extraX, extraE, extra_y

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )
