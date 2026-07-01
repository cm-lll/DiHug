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
            and not bool(
                getattr(cfg.model, "train_partition_ensemble", False)
            )
            and not bool(
                getattr(cfg.model, "train_family_staged_queryfree", False)
            )
        )

        self.in_dims = dataset_infos.input_dims
        self.out_dims = dataset_infos.output_dims
        self.cfg = cfg
        self.dataset_info = dataset_infos
        self.sparse_hetero_y_dim = self._sparse_hetero_y_dim()
        self.sparse_family_y_dim = self._sparse_family_y_dim()
        if self.sparse_hetero_y_dim > 0:
            self.in_dims = utils.PlaceHolder(
                X=self.in_dims.X,
                E=self.in_dims.E,
                y=self.in_dims.y + self.sparse_hetero_y_dim,
                charge=self.in_dims.charge,
            )
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

        self.test_variance = cfg.general.test_variance
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
            edge_exist_weight=getattr(cfg.model, "edge_exist_weight", 1.0),
            edge_subtype_weight=getattr(cfg.model, "edge_subtype_weight", 1.2),
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
            use_family_film=getattr(cfg.model, "use_family_film", False),
            use_family_edge_update=getattr(
                cfg.model, "use_family_edge_update", False
            ),
            use_relation_attention_matrix=getattr(
                cfg.model, "use_relation_attention_matrix", False
            ),
            use_relation_message_matrix=getattr(
                cfg.model, "use_relation_message_matrix", False
            ),
            use_family_y_film=getattr(cfg.model, "use_family_y_film", False),
            use_family_y_in_attention=getattr(
                cfg.model, "use_family_y_in_attention", False
            ),
            use_family_y_in_edge_film=getattr(
                cfg.model, "use_family_y_in_edge_film", False
            ),
            family_y_dim=getattr(self, "sparse_family_y_dim", 0),
            family_edge_update_hidden_dim=getattr(
                cfg.model, "family_edge_update_hidden_dim", 128
            ),
            edge_only_model=getattr(cfg.model, "edge_only_model", False),
            use_edge_struct_features=getattr(
                cfg.model, "use_edge_struct_features", False
            ),
            edge_struct_feature_dim=getattr(
                cfg.model, "edge_struct_feature_dim", 8
            ),
            edge_struct_hidden_dim=getattr(
                cfg.model, "edge_struct_hidden_dim", 64
            ),
            edge_struct_residual_scale=getattr(
                cfg.model, "edge_struct_residual_scale", 1.0
            ),
            edge_struct_use_family_y=getattr(
                cfg.model, "edge_struct_use_family_y", False
            ),
        )
        if denoiser_cls is PyHGTDenoiser:
            model_kwargs.update(
                subtype_dim=getattr(cfg.model, "subtype_dim", 32),
                use_edge_phi_fusion=getattr(cfg.model, "use_edge_phi_fusion", True),
                use_dual_softmax=getattr(cfg.model, "use_dual_softmax", True),
                use_query_context_gate=getattr(
                    cfg.model, "use_query_context_gate", False
                ),
                query_context_gate_init=getattr(
                    cfg.model, "query_context_gate_init", 0.2
                ),
                use_two_hop_structure=getattr(
                    cfg.model, "use_two_hop_structure", False
                ),
                use_typed_two_hop_structure=getattr(
                    cfg.model, "use_typed_two_hop_structure", False
                ),
                two_hop_structure_hidden_dim=getattr(
                    cfg.model, "two_hop_structure_hidden_dim", 64
                ),
                two_hop_structure_scale=getattr(
                    cfg.model, "two_hop_structure_scale", 1.0
                ),
                two_hop_structure_schedule=getattr(
                    cfg.model, "two_hop_structure_schedule", "fixed"
                ),
                use_endpoint_role_residual=getattr(
                    cfg.model, "use_endpoint_role_residual", False
                ),
                endpoint_role_hidden_dim=getattr(
                    cfg.model, "endpoint_role_hidden_dim", 64
                ),
                endpoint_role_family_dim=getattr(
                    cfg.model, "endpoint_role_family_dim", 16
                ),
                endpoint_role_scale=getattr(
                    cfg.model, "endpoint_role_scale", 1.0
                ),
                use_family_edge_adapters=getattr(
                    cfg.model, "use_family_edge_adapters", False
                ),
                family_edge_adapter_hidden_dim=getattr(
                    cfg.model, "family_edge_adapter_hidden_dim", 64
                ),
                use_time_film=getattr(cfg.model, "use_time_film", False),
                use_edge_state_update=getattr(cfg.model, "use_edge_state_update", False),
                edge_state_update_mode=getattr(cfg.model, "edge_state_update_mode", "all"),
                edge_input_residual_scale=getattr(
                    cfg.model, "edge_input_residual_scale", 1.0
                ),
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

    def _sparse_hetero_y_dim(self) -> int:
        if not bool(getattr(self.cfg.model, "use_sparse_hetero_y", False)):
            return 0
        if not bool(getattr(self.dataset_info, "heterogeneous", False)):
            return 0
        num_families = len(getattr(self.dataset_info, "edge_family2id", {}) or {})
        num_types = len(getattr(self.dataset_info, "type_offsets", {}) or {})
        if num_families <= 0 or num_types <= 0:
            return 0
        return 2 * num_families + 2 + 2 * num_types

    def _sparse_family_y_dim(self) -> int:
        if not bool(getattr(self.cfg.model, "use_sparse_family_y", False)):
            return 0
        if not bool(getattr(self.dataset_info, "heterogeneous", False)):
            return 0
        num_families = len(getattr(self.dataset_info, "edge_family2id", {}) or {})
        num_types = len(getattr(self.dataset_info, "type_offsets", {}) or {})
        if num_families <= 0 or num_types <= 0:
            return 0
        bins = max(1, int(getattr(self.cfg.model, "sparse_family_y_degree_bins", 5) or 5))
        # family one-hot + src/dst type one-hot + same-type flag + scalar stats
        # + src/dst family-degree mean/std + endpoint degree-bin-pair histogram
        # + current edge subtype distribution.
        return num_families + 2 * num_types + 1 + 7 + 4 + bins * bins + int(self.out_dims.E)

    def _node_type_ids_from_onehot(self, node_onehot: torch.Tensor) -> torch.Tensor:
        node_subtype = (
            node_onehot.argmax(dim=-1).long()
            if node_onehot.dim() > 1
            else node_onehot.long()
        )
        return self._node_type_ids_from_subtype(node_subtype)

    def _compute_sparse_hetero_y(self, sparse_noisy_data) -> torch.Tensor:
        cached = sparse_noisy_data.get("_sparse_hetero_y")
        if cached is not None:
            return cached.to(self.device)
        dim = int(getattr(self, "sparse_hetero_y_dim", 0) or 0)
        batch = sparse_noisy_data["batch"].long().to(self.device)
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        node = sparse_noisy_data["node_t"].to(self.device)
        out = node.new_zeros((bs, dim), dtype=torch.float32)
        def cache_and_return(value):
            sparse_noisy_data["_sparse_hetero_y"] = value
            return value
        if dim <= 0:
            return cache_and_return(out)

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        if not (edge_family2id and fam_endpoints and type_offsets):
            return cache_and_return(out)

        id2family = {
            int(v): str(k)
            for k, v in sorted(edge_family2id.items(), key=lambda item: int(item[1]))
        }
        family_ids = sorted(id2family)
        family_pos = {fam_id: pos for pos, fam_id in enumerate(family_ids)}
        num_families = len(family_ids)
        sorted_types = sorted(((str(k), int(v)) for k, v in type_offsets.items()), key=lambda x: x[1])
        type_name_to_id = {name: idx for idx, (name, _) in enumerate(sorted_types)}
        num_types = len(sorted_types)

        node_type = self._node_type_ids_from_onehot(node).to(self.device)
        edge_index = sparse_noisy_data["edge_index_t"].long().to(self.device)
        edge_attr = sparse_noisy_data["edge_attr_t"].to(self.device)
        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return cache_and_return(out)
        labels = (
            edge_attr.argmax(dim=-1).long()
            if edge_attr.dim() > 1
            else edge_attr.long().reshape(-1)
        )
        visible = labels > 0
        if not visible.any():
            return cache_and_return(out)

        metadata = self._heterogeneous_metadata_for_edges(node, edge_attr, edge_index)
        edge_family = metadata.get("edge_family_ids")
        if edge_family is None:
            return cache_and_return(out)
        edge_family = edge_family.long().to(self.device)
        visible_edge_index = edge_index[:, visible]
        visible_family = edge_family[visible]
        edge_batch = batch[visible_edge_index[0].long()]

        fam_counts = out.new_zeros((bs, num_families))
        for fam_id, pos in family_pos.items():
            mask = visible_family == int(fam_id)
            if not mask.any():
                continue
            fam_counts[:, pos] = torch.bincount(
                edge_batch[mask], minlength=bs
            ).to(out.dtype)

        type_counts = out.new_zeros((bs, num_types))
        for type_id in range(num_types):
            mask = node_type == int(type_id)
            if mask.any():
                type_counts[:, type_id] = torch.bincount(
                    batch[mask], minlength=bs
                ).to(out.dtype)

        fam_possible = out.new_zeros((bs, num_families))
        for fam_id, fam_name in id2family.items():
            pos = family_pos.get(int(fam_id))
            endpoints = fam_endpoints.get(fam_name, {})
            src_type = endpoints.get("src_type")
            dst_type = endpoints.get("dst_type")
            if pos is None or src_type not in type_name_to_id or dst_type not in type_name_to_id:
                continue
            src_count = type_counts[:, type_name_to_id[src_type]]
            dst_count = type_counts[:, type_name_to_id[dst_type]]
            possible = src_count * dst_count
            if src_type == dst_type:
                possible = (possible - src_count).clamp_min(0)
            fam_possible[:, pos] = possible.clamp_min(1)

        fam_density = fam_counts / fam_possible.clamp_min(1)
        fam_log_count = torch.log1p(fam_counts) / torch.log1p(fam_possible).clamp_min(1.0)

        src_type = node_type[visible_edge_index[0].long()]
        dst_type = node_type[visible_edge_index[1].long()]
        valid_type_edge = (src_type >= 0) & (dst_type >= 0)
        same_count = out.new_zeros((bs,))
        cross_count = out.new_zeros((bs,))
        if valid_type_edge.any():
            same_mask = valid_type_edge & (src_type == dst_type)
            cross_mask = valid_type_edge & (src_type != dst_type)
            if same_mask.any():
                same_count = torch.bincount(edge_batch[same_mask], minlength=bs).to(out.dtype)
            if cross_mask.any():
                cross_count = torch.bincount(edge_batch[cross_mask], minlength=bs).to(out.dtype)
        edge_total = (same_count + cross_count).clamp_min(1.0)
        same_cross = torch.stack([same_count / edge_total, cross_count / edge_total], dim=1)

        deg = out.new_zeros((node_type.shape[0],))
        deg.index_add_(0, visible_edge_index[0].long(), out.new_ones(visible_edge_index.shape[1]))
        deg.index_add_(0, visible_edge_index[1].long(), out.new_ones(visible_edge_index.shape[1]))
        type_deg_mean = out.new_zeros((bs, num_types))
        type_deg_std = out.new_zeros((bs, num_types))
        graph_sizes = torch.bincount(batch, minlength=bs).to(out.dtype).clamp_min(1.0)
        for graph_idx in range(bs):
            graph_mask = batch == int(graph_idx)
            scale = graph_sizes[graph_idx].clamp_min(1.0)
            for type_id in range(num_types):
                mask = graph_mask & (node_type == int(type_id))
                if not mask.any():
                    continue
                vals = deg[mask] / scale
                type_deg_mean[graph_idx, type_id] = vals.mean()
                type_deg_std[graph_idx, type_id] = (
                    vals.std(unbiased=False) if vals.numel() > 1 else 0.0
                )

        features = torch.cat(
            [fam_log_count, fam_density, same_cross, type_deg_mean, type_deg_std],
            dim=1,
        )
        return cache_and_return(features)

    def _compute_sparse_family_y(self, sparse_noisy_data) -> torch.Tensor:
        cached = sparse_noisy_data.get("_sparse_family_y")
        if cached is not None:
            return cached.to(self.device)
        dim = int(getattr(self, "sparse_family_y_dim", 0) or 0)
        batch = sparse_noisy_data["batch"].long().to(self.device)
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        num_families = max((max(edge_family2id.values()) + 1) if edge_family2id else 0, 0)
        out = torch.zeros((bs, max(num_families, 1), dim), device=self.device, dtype=torch.float32)

        def cache_and_return(value):
            sparse_noisy_data["_sparse_family_y"] = value
            return value

        if dim <= 0 or num_families <= 0 or not (edge_family2id and fam_endpoints and type_offsets):
            return cache_and_return(out)

        node = sparse_noisy_data["node_t"].to(self.device)
        node_type = self._node_type_ids_from_onehot(node).to(self.device)
        sorted_types = sorted(((str(k), int(v)) for k, v in type_offsets.items()), key=lambda x: x[1])
        type_name_to_id = {name: idx for idx, (name, _) in enumerate(sorted_types)}
        num_types = len(sorted_types)
        bins = max(1, int(getattr(self.cfg.model, "sparse_family_y_degree_bins", 5) or 5))
        edge_index = sparse_noisy_data["edge_index_t"].long().to(self.device)
        edge_attr = sparse_noisy_data["edge_attr_t"].to(self.device)
        labels = (
            edge_attr.argmax(dim=-1).long()
            if edge_attr.dim() > 1
            else edge_attr.long().reshape(-1)
        )
        metadata = self._heterogeneous_metadata_for_edges(node, edge_attr, edge_index)
        edge_family = metadata.get("edge_family_ids")
        edge_family = (
            edge_family.long().to(self.device)
            if edge_family is not None
            else torch.full((edge_index.shape[1],), -1, device=self.device, dtype=torch.long)
        )
        visible = (labels > 0) & (edge_family >= 0)
        edge_batch = batch[edge_index[0].long()] if edge_index.numel() else batch.new_zeros((0,))
        graph_sizes = torch.bincount(batch, minlength=bs).float().to(self.device).clamp_min(1.0)

        type_counts = torch.zeros((bs, num_types), device=self.device, dtype=torch.float32)
        for type_id in range(num_types):
            mask = node_type == int(type_id)
            if mask.any():
                type_counts[:, type_id] = torch.bincount(batch[mask], minlength=bs).float()

        expected_counts = torch.zeros((num_families,), device=self.device, dtype=torch.float32)
        avg_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        for fam_name, fam_id in edge_family2id.items():
            expected_counts[int(fam_id)] = float(avg_counts.get(fam_name, 0.0) or 0.0)

        family_counts = torch.zeros((bs, num_families), device=self.device, dtype=torch.float32)
        if visible.any():
            flat = edge_batch[visible].long() * num_families + edge_family[visible].long().clamp(0, num_families - 1)
            family_counts.reshape(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

        deg_by_family = torch.zeros((bs, num_families, node.shape[0]), device=self.device, dtype=torch.float32)
        if visible.any():
            src = edge_index[0, visible].long()
            dst = edge_index[1, visible].long()
            fam = edge_family[visible].long().clamp(0, num_families - 1)
            gb = edge_batch[visible].long()
            flat = deg_by_family.reshape(-1)
            stride = num_families * node.shape[0]
            flat.index_add_(0, gb * stride + fam * node.shape[0] + src, torch.ones_like(src, dtype=torch.float32))
            flat.index_add_(0, gb * stride + fam * node.shape[0] + dst, torch.ones_like(dst, dtype=torch.float32))

        cursor = 0
        family_eye = torch.eye(num_families, device=self.device, dtype=torch.float32)
        type_eye = torch.eye(num_types, device=self.device, dtype=torch.float32)
        id2family = {int(v): str(k) for k, v in edge_family2id.items()}
        for fam_id in range(num_families):
            fam_name = id2family.get(fam_id, "")
            endpoints = fam_endpoints.get(fam_name, {}) or {}
            src_name = str(endpoints.get("src_type", ""))
            dst_name = str(endpoints.get("dst_type", ""))
            src_type_id = int(type_name_to_id.get(src_name, 0))
            dst_type_id = int(type_name_to_id.get(dst_name, 0))
            same_type = 1.0 if src_type_id == dst_type_id else 0.0
            src_count = type_counts[:, src_type_id] if num_types else torch.ones((bs,), device=self.device)
            dst_count = type_counts[:, dst_type_id] if num_types else torch.ones((bs,), device=self.device)
            possible = (src_count * dst_count).clamp_min(1.0)
            if same_type:
                possible = (possible - src_count).clamp_min(1.0)
            count = family_counts[:, fam_id]
            expected = expected_counts[fam_id].clamp_min(0.0)
            expected_safe = expected.clamp_min(1.0)
            density = count / possible
            expected_density = expected / possible
            log_count = torch.log1p(count) / torch.log1p(possible).clamp_min(1.0)
            count_ratio = (count / expected_safe).clamp(0.0, 10.0) / 10.0
            log_ratio = torch.log((count + 1.0) / (expected + 1.0)).clamp(-5.0, 5.0) / 5.0
            diff_norm = ((count - expected) / possible).clamp(-1.0, 1.0)

            out[:, fam_id, cursor : cursor + num_families] = family_eye[fam_id]
            cursor2 = cursor + num_families
            out[:, fam_id, cursor2 : cursor2 + num_types] = type_eye[src_type_id]
            cursor2 += num_types
            out[:, fam_id, cursor2 : cursor2 + num_types] = type_eye[dst_type_id]
            cursor2 += num_types
            out[:, fam_id, cursor2] = same_type
            cursor2 += 1
            scalars = torch.stack(
                [count / graph_sizes, density, log_count, expected_density, count_ratio, log_ratio, diff_norm],
                dim=1,
            )
            out[:, fam_id, cursor2 : cursor2 + 7] = scalars
            cursor2 += 7

            src_mask_nodes = node_type == src_type_id
            dst_mask_nodes = node_type == dst_type_id
            for graph_idx in range(bs):
                graph_mask = batch == int(graph_idx)
                vals_src = deg_by_family[graph_idx, fam_id, graph_mask & src_mask_nodes]
                vals_dst = deg_by_family[graph_idx, fam_id, graph_mask & dst_mask_nodes]
                scale = graph_sizes[graph_idx].clamp_min(1.0)
                if vals_src.numel():
                    out[graph_idx, fam_id, cursor2] = vals_src.mean() / scale
                    out[graph_idx, fam_id, cursor2 + 1] = vals_src.std(unbiased=False) / scale if vals_src.numel() > 1 else 0.0
                if vals_dst.numel():
                    out[graph_idx, fam_id, cursor2 + 2] = vals_dst.mean() / scale
                    out[graph_idx, fam_id, cursor2 + 3] = vals_dst.std(unbiased=False) / scale if vals_dst.numel() > 1 else 0.0
            cursor2 += 4

            fam_visible = visible & (edge_family == fam_id)
            if fam_visible.any():
                src = edge_index[0, fam_visible].long()
                dst = edge_index[1, fam_visible].long()
                gb = edge_batch[fam_visible].long()
                deg_src = deg_by_family[gb, fam_id, src]
                deg_dst = deg_by_family[gb, fam_id, dst]
                bin_src = torch.bucketize(deg_src, torch.tensor([0.5, 1.5, 3.5, 7.5], device=self.device)).clamp(0, bins - 1)
                bin_dst = torch.bucketize(deg_dst, torch.tensor([0.5, 1.5, 3.5, 7.5], device=self.device)).clamp(0, bins - 1)
                if same_type:
                    lo = torch.minimum(bin_src, bin_dst)
                    hi = torch.maximum(bin_src, bin_dst)
                    bin_src, bin_dst = lo, hi
                pair_flat = gb * (bins * bins) + bin_src * bins + bin_dst
                hist = torch.zeros((bs, bins * bins), device=self.device, dtype=torch.float32)
                hist.reshape(-1).index_add_(0, pair_flat, torch.ones_like(pair_flat, dtype=torch.float32))
                hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
                out[:, fam_id, cursor2 : cursor2 + bins * bins] = hist

                label_hist = torch.zeros((bs, int(self.out_dims.E)), device=self.device, dtype=torch.float32)
                lbl = labels[fam_visible].long().clamp(0, int(self.out_dims.E) - 1)
                label_flat = gb * int(self.out_dims.E) + lbl
                label_hist.reshape(-1).index_add_(0, label_flat, torch.ones_like(label_flat, dtype=torch.float32))
                label_hist = label_hist / label_hist.sum(dim=1, keepdim=True).clamp_min(1.0)
                out[:, fam_id, cursor2 + bins * bins : cursor2 + bins * bins + int(self.out_dims.E)] = label_hist
            cursor = 0

        return cache_and_return(out)

    def _compute_edge_struct_features(self, sparse_noisy_data) -> torch.Tensor:
        cached = sparse_noisy_data.get("_edge_struct_features")
        if cached is not None:
            return cached.to(self.device)
        dim = int(getattr(self.cfg.model, "edge_struct_feature_dim", 8) or 8)
        comp_edge_index = sparse_noisy_data["comp_edge_index_t"].long().to(self.device)
        num_edges = int(comp_edge_index.shape[1])
        out = torch.zeros((num_edges, dim), device=self.device, dtype=torch.float32)

        def cache_and_return(value):
            sparse_noisy_data["_edge_struct_features"] = value
            return value

        if dim <= 0 or num_edges <= 0:
            return cache_and_return(out)

        batch = sparse_noisy_data["batch"].long().to(self.device)
        ptr = sparse_noisy_data.get("ptr")
        if ptr is None:
            counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)
            ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)], dim=0)
        ptr = ptr.long().to(self.device)
        visible_edge_index = sparse_noisy_data["edge_index_t"].long().to(self.device)
        visible_edge_attr = sparse_noisy_data["edge_attr_t"].to(self.device)
        if visible_edge_attr.numel() == 0 or visible_edge_index.numel() == 0:
            return cache_and_return(out)
        labels = (
            visible_edge_attr.argmax(dim=-1).long()
            if visible_edge_attr.dim() > 1
            else visible_edge_attr.long().reshape(-1)
        )
        visible = labels > 0
        if not visible.any():
            return cache_and_return(out)
        visible_edge_index = visible_edge_index[:, visible]

        num_graphs = max(int(ptr.numel()) - 1, 1)
        eps = 1e-6
        for graph_idx in range(num_graphs):
            start = int(ptr[graph_idx].item())
            end = int(ptr[graph_idx + 1].item())
            n = end - start
            if n <= 1:
                continue
            edge_mask = (
                (comp_edge_index[0] >= start)
                & (comp_edge_index[0] < end)
                & (comp_edge_index[1] >= start)
                & (comp_edge_index[1] < end)
            )
            if not edge_mask.any():
                continue
            vis_mask = (
                (visible_edge_index[0] >= start)
                & (visible_edge_index[0] < end)
                & (visible_edge_index[1] >= start)
                & (visible_edge_index[1] < end)
            )
            adj = torch.zeros((n, n), device=self.device, dtype=torch.float32)
            if vis_mask.any():
                src_vis = visible_edge_index[0, vis_mask] - start
                dst_vis = visible_edge_index[1, vis_mask] - start
                adj[src_vis, dst_vis] = 1.0
                adj[dst_vis, src_vis] = 1.0
            adj.fill_diagonal_(0.0)
            deg = adj.sum(dim=1)
            common = adj.matmul(adj)

            idx = torch.where(edge_mask)[0]
            src = comp_edge_index[0, idx] - start
            dst = comp_edge_index[1, idx] - start
            deg_s = deg[src]
            deg_d = deg[dst]
            cn = common[src, dst]
            deg_sum = deg_s + deg_d
            max_log_n = math.log1p(max(n, 1))
            max_log_n2 = math.log1p(max(n * n, 1))
            feats = [
                torch.log1p(cn) / max_log_n,
                cn / torch.sqrt((deg_s * deg_d).clamp_min(eps)),
                torch.log1p(deg_s) / max_log_n,
                torch.log1p(deg_d) / max_log_n,
                (deg_s - deg_d).abs() / (deg_sum + 1.0),
                torch.log1p(deg_s * deg_d) / max_log_n2,
                adj[src, dst],
            ]
            if dim > 7:
                if vis_mask.any():
                    parent = list(range(n))

                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(a, b):
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[rb] = ra

                    src_cpu = (visible_edge_index[0, vis_mask] - start).detach().cpu().tolist()
                    dst_cpu = (visible_edge_index[1, vis_mask] - start).detach().cpu().tolist()
                    for a, b in zip(src_cpu, dst_cpu):
                        if a != b:
                            union(int(a), int(b))
                    comp_id = torch.tensor(
                        [find(i) for i in range(n)],
                        device=self.device,
                        dtype=torch.long,
                    )
                    same_component = (comp_id[src] == comp_id[dst]).to(out.dtype)
                else:
                    same_component = torch.zeros_like(cn)
                feats.append(same_component)
            feat_tensor = torch.stack(feats, dim=1)
            if feat_tensor.size(1) < dim:
                pad = feat_tensor.new_zeros((feat_tensor.size(0), dim - feat_tensor.size(1)))
                feat_tensor = torch.cat([feat_tensor, pad], dim=1)
            out[idx] = feat_tensor[:, :dim]
        return cache_and_return(out)

    def _profile_training_efficiency_enabled(self) -> bool:
        if not bool(
            getattr(self.cfg.general, "profile_training_efficiency", False)
        ):
            return False
        every = max(
            1,
            int(
                getattr(
                    self.cfg.general,
                    "profile_training_efficiency_every_n_epochs",
                    1,
                )
                or 1
            ),
        )
        return int(getattr(self, "current_epoch", 0)) % every == 0

    def _profile_time(self, enabled: bool) -> float:
        if not enabled:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _build_training_two_hop_lookup(
        self,
        sparse_noisy_data,
        num_nodes: int,
    ):
        if not (
            bool(getattr(self.cfg.model, "use_two_hop_structure", False))
            or bool(
                getattr(
                    self.cfg.model, "use_endpoint_role_residual", False
                )
            )
        ):
            return None
        if not isinstance(self.model, PyHGTDenoiser):
            return None
        edge_index = sparse_noisy_data["edge_index_t"].long()
        edge_attr = sparse_noisy_data["edge_attr_t"]
        labels = (
            edge_attr.argmax(dim=-1).long()
            if edge_attr.dim() > 1
            else edge_attr.long().reshape(-1)
        )
        visible_edges = edge_index[:, labels > 0]
        visible_attr = edge_attr[labels > 0]
        node = sparse_noisy_data["node_t"]
        metadata = self._heterogeneous_metadata_for_edges(
            node,
            visible_attr,
            visible_edges,
        )
        return self.model.build_two_hop_structure_lookup(
            context_edge_index=visible_edges,
            num_nodes=int(num_nodes),
            node_type_ids=metadata.get("node_type_ids"),
            edge_family_ids=metadata.get("edge_family_ids"),
            batch=sparse_noisy_data.get("batch"),
        )

    def _two_hop_reliability_factor(self, t_float):
        if not bool(getattr(self.cfg.model, "use_two_hop_structure", False)):
            return None
        schedule = str(
            getattr(self.cfg.model, "two_hop_structure_schedule", "fixed")
            or "fixed"
        ).lower()
        t = t_float.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        if schedule == "fixed":
            return torch.ones_like(t)
        if schedule == "linear_t":
            return 1.0 - t
        if schedule == "quadratic_t":
            return (1.0 - t).square()
        if schedule == "alpha_bar_squared":
            alpha_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
            return alpha_bar.to(dtype=torch.float32).square()
        raise ValueError(
            "Unknown model.two_hop_structure_schedule="
            f"{schedule!r}; expected fixed, linear_t, quadratic_t, "
            "or alpha_bar_squared"
        )

    def _endpoint_role_reliability_factor(self, t_float):
        if not bool(
            getattr(self.cfg.model, "use_endpoint_role_residual", False)
        ):
            return None
        t = t_float.to(
            device=self.device, dtype=torch.float32
        ).clamp(0.0, 1.0)
        alpha_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        return alpha_bar.to(dtype=torch.float32).square()



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
                density = self._family_density_from_marginal(
                    fam_name, marginals, canonical_candidates=True
                )
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
                density = self._family_density_from_marginal(
                    fam_name, marginals, canonical_candidates=True
                )
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

    def _build_type_template_pseudo_blocks(
        self,
        anchor_node_subtype,
        batch,
        type_offsets,
        reference_new_to_old=None,
    ):
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
                # For paired full-pipeline diagnostics, enumerate nodes in the
                # reference coordinate system before drawing the permutation.
                # The same RNG stream then creates exactly mapped pseudo-blocks
                # rather than blocks that merely have the same type counts.
                if reference_new_to_old is not None:
                    ref_ids = reference_new_to_old[nodes].long()
                    nodes = nodes[torch.argsort(ref_ids)]
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

    def _calibrate_edge_logits_for_exist_pos_weight(
        self, logits, query_edge_family=None
    ):
        """Undo the prior shift induced by fixed positive-class BCE weighting."""
        if (
            logits is None
            or logits.numel() == 0
            or not bool(
                getattr(
                    self.cfg.model,
                    "sampling_calibrate_exist_pos_weight",
                    False,
                )
            )
        ):
            return logits

        raw_weight = getattr(self.cfg.model, "exist_pos_weight", None)
        if isinstance(raw_weight, str):
            if (
                getattr(self, "local_rank", 0) == 0
                and not getattr(self, "_logged_sampling_pos_weight_calibration", False)
            ):
                print(
                    "[采样-CALIBRATE] skipped: fixed-logit correction does not "
                    f"support exist_pos_weight={raw_weight!r}"
                )
                self._logged_sampling_pos_weight_calibration = True
            return logits
        try:
            pos_weight = float(raw_weight)
        except (TypeError, ValueError):
            pos_weight = 1.0
        if not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError(
                "sampling_calibrate_exist_pos_weight requires a finite positive "
                f"model.exist_pos_weight, got {raw_weight!r}"
            )

        masked_logits = self._mask_edge_logits_by_query_family(
            logits, query_edge_family
        )
        shift = math.log(pos_weight)
        calibrated = masked_logits.clone()
        calibrated[:, 1:] = calibrated[:, 1:] - shift

        if (
            getattr(self, "local_rank", 0) == 0
            and not getattr(self, "_logged_sampling_pos_weight_calibration", False)
        ):
            before_exist = torch.logsumexp(masked_logits[:, 1:], dim=-1) - masked_logits[:, 0]
            after_exist = torch.logsumexp(calibrated[:, 1:], dim=-1) - calibrated[:, 0]
            before_mean = float(torch.sigmoid(before_exist).mean().detach().cpu())
            after_mean = float(torch.sigmoid(after_exist).mean().detach().cpu())
            print(
                "[采样-CALIBRATE] "
                f"exist_pos_weight={pos_weight:g} shift=log(w)={shift:.6f} "
                f"mean_clean_p_exist={before_mean:.6f}->{after_mean:.6f}"
            )
            self._logged_sampling_pos_weight_calibration = True
        return calibrated

    def _lookup_query_edge_labels(
        self,
        query_edge_index,
        current_edge_index,
        current_edge_labels,
        total_nodes,
    ):
        """Return current E_t labels for canonical query edges; absent edges map to zero."""
        if query_edge_index is None or query_edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        qei = query_edge_index.long().to(self.device)
        qu = torch.minimum(qei[0], qei[1])
        qv = torch.maximum(qei[0], qei[1])
        query_keys = qu * int(total_nodes) + qv
        out = torch.zeros(query_keys.numel(), dtype=torch.long, device=self.device)
        if current_edge_index is None or current_edge_index.numel() == 0:
            return out

        cei = current_edge_index.long().to(self.device)
        cu = torch.minimum(cei[0], cei[1])
        cv = torch.maximum(cei[0], cei[1])
        current_keys = cu * int(total_nodes) + cv
        order = torch.argsort(current_keys)
        sorted_keys = current_keys[order]
        locations = torch.searchsorted(sorted_keys, query_keys)
        valid = locations < sorted_keys.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = sorted_keys[locations[valid]] == query_keys[valid]
        if matched.any():
            labels = current_edge_labels.long().to(self.device)
            out[matched] = labels[order[locations[matched]]]
        return out

    def _edge_reverse_posterior_logits(
        self,
        clean_logits,
        query_edge_family,
        query_edge_batch,
        current_edge_labels,
        s_float,
        t_float,
    ):
        """Convert p(E_0|E_t) logits into log p(E_s|E_t) per relation family."""
        if (
            clean_logits is None
            or clean_logits.numel() == 0
            or query_edge_family is None
            or not bool(getattr(self.cfg.model, "sampling_use_reverse_posterior", False))
        ):
            return clean_logits

        clean_logits = self._mask_edge_logits_by_query_family(
            clean_logits, query_edge_family
        )
        posterior_mix_weight = self._reverse_posterior_mix_weight(
            s_float=s_float,
            t_float=t_float,
        )
        qfam = query_edge_family.long().to(clean_logits.device).reshape(-1)
        qbatch = query_edge_batch.long().to(clean_logits.device).reshape(-1)
        current_edge_labels = current_edge_labels.long().to(clean_logits.device).reshape(-1)
        if (
            qfam.numel() != clean_logits.shape[0]
            or qbatch.numel() != clean_logits.shape[0]
            or current_edge_labels.numel() != clean_logits.shape[0]
        ):
            return clean_logits

        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        alpha_step = (alpha_t_bar / alpha_s_bar.clamp(min=1e-12)).clamp(
            min=1e-8, max=1.0
        )
        beta_step = (1.0 - alpha_step).clamp(min=0.0, max=0.9999)
        family_qt = self.transition_model.get_all_family_Qt(
            beta_step, device=clean_logits.device
        )
        family_qsb = self.transition_model.get_all_family_Qt_bar(
            alpha_s_bar, device=clean_logits.device
        )
        family_qtb = self.transition_model.get_all_family_Qt_bar(
            alpha_t_bar, device=clean_logits.device
        )

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        family_ranges = self._edge_family_label_ranges()
        posterior_logits = torch.full_like(clean_logits, -1e10)
        handled = torch.zeros(clean_logits.shape[0], dtype=torch.bool, device=clean_logits.device)

        for fam_id, fam_name in id2edge_family.items():
            rows = qfam == int(fam_id)
            if (
                not rows.any()
                or fam_name not in family_ranges
                or fam_name not in family_qt
                or fam_name not in family_qsb
                or fam_name not in family_qtb
            ):
                continue
            idx = rows.nonzero(as_tuple=True)[0]
            lo, hi = family_ranges[fam_name]
            local_clean_logits = torch.cat(
                [clean_logits[idx, :1], clean_logits[idx, lo:hi]], dim=-1
            )
            pred_x0 = torch.softmax(local_clean_logits, dim=-1)
            num_states = int(pred_x0.shape[-1])

            labels = current_edge_labels[idx]
            local_t = torch.zeros_like(labels)
            positive = (labels >= int(lo)) & (labels < int(hi))
            local_t[positive] = labels[positive] - int(lo) + 1
            local_t_onehot = F.one_hot(
                local_t.clamp(0, num_states - 1), num_classes=num_states
            ).float()
            local_batch = qbatch[idx]
            posterior_over_x0 = (
                diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                    input_data=local_t_onehot,
                    batch=local_batch,
                    Qt=family_qt[fam_name].E,
                    Qsb=family_qsb[fam_name].E,
                    Qtb=family_qtb[fam_name].E,
                )
            )
            prob_s = (pred_x0.unsqueeze(-1) * posterior_over_x0).sum(dim=1)
            prob_s = prob_s / prob_s.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            if posterior_mix_weight < 1.0:
                prob_s = (
                    (1.0 - posterior_mix_weight) * pred_x0
                    + posterior_mix_weight * prob_s
                )
                prob_s = prob_s / prob_s.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-12)
            local_log = torch.log(prob_s.clamp(min=1e-12))
            posterior_logits[idx, 0] = local_log[:, 0]
            posterior_logits[idx, lo:hi] = local_log[:, 1:]
            handled[idx] = True

        if (~handled).any():
            posterior_logits[~handled] = clean_logits[~handled]
        transition_key = (
            int(round(float(t_float.mean().detach().cpu()) * self.T)),
            int(round(float(s_float.mean().detach().cpu()) * self.T)),
        )
        logged_transitions = getattr(
            self, "_logged_reverse_posterior_transitions", set()
        )
        if (
            getattr(self, "local_rank", 0) == 0
            and transition_key not in logged_transitions
        ):
            beta_mean = float(beta_step.mean().detach().cpu())
            print(
                "[采样-POSTERIOR] enabled "
                f"s={float(s_float.mean().detach().cpu()):.4f} "
                f"t={float(t_float.mean().detach().cpu()):.4f} "
                f"mix={posterior_mix_weight:.4f} "
                f"beta_step={beta_mean:.6f} handled={int(handled.sum().item())}/"
                f"{int(handled.numel())}"
            )
            logged_transitions.add(transition_key)
            self._logged_reverse_posterior_transitions = logged_transitions
        return posterior_logits

    def _reverse_posterior_mix_weight(self, s_float, t_float) -> float:
        configured = getattr(
            self.cfg.model,
            "sampling_reverse_posterior_mix_weights",
            None,
        )
        if configured is None:
            mode = str(
                getattr(
                    self.cfg.model,
                    "sampling_reverse_posterior_mix_mode",
                    "full",
                )
                or "full"
            ).lower()
            scale = float(
                getattr(
                    self.cfg.model,
                    "sampling_reverse_posterior_mix_scale",
                    1.0,
                )
                or 0.0
            )
            if mode == "full":
                return min(max(scale, 0.0), 1.0)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(
                t_normalized=s_float
            )
            reliability = float(alpha_s_bar.mean().detach().cpu())
            if mode == "alpha_bar_s_squared":
                reliability = reliability * reliability
            elif mode != "alpha_bar_s":
                raise ValueError(
                    "Unsupported posterior mix mode: "
                    f"{mode}. Expected full, alpha_bar_s, or "
                    "alpha_bar_s_squared."
                )
            return min(max(scale * reliability, 0.0), 1.0)
        weights = [float(value) for value in list(configured)]
        transitions = self._sampling_time_transitions()
        if len(weights) != len(transitions):
            raise ValueError(
                "model.sampling_reverse_posterior_mix_weights must contain "
                f"one value per reverse transition ({len(transitions)}), "
                f"got {weights}."
            )
        if any(value < 0.0 or value > 1.0 for value in weights):
            raise ValueError(
                "Posterior mix weights must lie in [0, 1], "
                f"got {weights}."
            )
        t_int = int(
            round(float(t_float.mean().detach().cpu()) * self.T)
        )
        s_int = int(
            round(float(s_float.mean().detach().cpu()) * self.T)
        )
        for index, transition in enumerate(transitions):
            if transition == (t_int, s_int):
                return weights[index]
        raise ValueError(
            "Current reverse transition is absent from the configured "
            f"sampling path: {(t_int, s_int)} not in {transitions}."
        )

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
            real_density = self._family_density_from_marginal(
                fam_name, marginals, canonical_candidates=True
            )
            ratio = sampled_rate / max(real_density, 1e-12) if real_density is not None else None
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

    def _family_density_from_marginal(self, fam_name, marginals, canonical_candidates=False):
        if fam_name not in marginals:
            return None
        m = marginals[fam_name]
        try:
            if not torch.is_tensor(m):
                m = torch.tensor(m)
            if m.numel() > 0:
                density = 1.0 - float(m.reshape(-1)[0].detach().cpu().item())
                if canonical_candidates:
                    endpoints = (getattr(self.dataset_info, "fam_endpoints", {}) or {}).get(str(fam_name), {})
                    if (
                        endpoints
                        and str(endpoints.get("src_type")) == str(endpoints.get("dst_type"))
                    ):
                        # Dataset marginals use ordered n*(n-1) pairs for same-type
                        # families, while block/query sampling uses one canonical
                        # undirected candidate per pair: n*(n-1)/2.
                        density *= 2.0
                return max(0.0, min(1.0, density))
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
                density = self._family_density_from_marginal(
                    fam_name, marginals, canonical_candidates=True
                )
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

    def _calibrate_bernoulli_expected_density_by_family(
        self,
        exist_logits,
        has_valid_pos,
        query_edge_family,
    ):
        """Match each family's expected Bernoulli count to its canonical marginal."""
        adjusted_prob = torch.sigmoid(exist_logits).clamp(min=0.0, max=1.0)
        if query_edge_family is None:
            return adjusted_prob
        qfam = query_edge_family.long().to(exist_logits.device).reshape(-1)
        if qfam.numel() != exist_logits.shape[0]:
            return adjusted_prob

        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        marginals = getattr(self.dataset_info, "edge_family_marginals", {}) or {}
        density_by_id = getattr(
            self,
            "_sampling_canonical_family_density_by_id",
            None,
        )
        if density_by_id is None:
            density_by_id = {}
            for fam_id, fam_name in id2edge_family.items():
                density_by_id[int(fam_id)] = self._family_density_from_marginal(
                    fam_name,
                    marginals,
                    canonical_candidates=True,
                )
            self._sampling_canonical_family_density_by_id = density_by_id
        should_log = (
            getattr(self, "local_rank", 0) == 0
            and not getattr(
                self,
                "_logged_sampling_expected_density_calibration",
                False,
            )
        )
        log_records = []
        for fam_id, fam_name in id2edge_family.items():
            mask = (qfam == int(fam_id)) & has_valid_pos
            local_logits = exist_logits[mask]
            n = int(local_logits.numel())
            if n <= 0:
                continue
            density = density_by_id.get(int(fam_id))
            if density is None:
                continue
            target = max(0.0, min(float(n), float(density) * float(n)))
            if target <= 0.0:
                local_prob = torch.zeros_like(local_logits)
                intercept_tensor = local_logits.new_tensor(float("inf"))
            elif target >= float(n):
                local_prob = torch.ones_like(local_logits)
                intercept_tensor = local_logits.new_tensor(float("-inf"))
            else:
                # Solve sum(sigmoid(logit - intercept)) = target. The function
                # is monotone, so bisection is stable even for extreme logits.
                # Keep the complete solve on-device: transferring the expected
                # count to CPU on every iteration makes block-wise sampling
                # dominated by thousands of CUDA synchronizations.
                lo = local_logits.min() - 40.0
                hi = local_logits.max() + 40.0
                target_tensor = local_logits.new_tensor(target)
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    expected = torch.sigmoid(local_logits - mid).sum()
                    move_lo = expected > target_tensor
                    lo = torch.where(move_lo, mid, lo)
                    hi = torch.where(move_lo, hi, mid)
                intercept_tensor = 0.5 * (lo + hi)
                local_prob = torch.sigmoid(local_logits - intercept_tensor)
            adjusted_prob[mask] = local_prob
            if should_log:
                log_records.append(
                    (
                        fam_name,
                        n,
                        target,
                        torch.stack(
                            [
                                torch.sigmoid(local_logits).sum(),
                                local_prob.sum(),
                                intercept_tensor,
                            ]
                        ),
                    )
                )

        adjusted_prob[~has_valid_pos] = 0.0
        if should_log and log_records:
            # One device synchronization for the complete first-round log,
            # never one synchronization per family or bisection iteration.
            log_values = torch.stack([record[3] for record in log_records])
            log_values_cpu = log_values.detach().cpu().tolist()
            log_parts = []
            for (fam_name, n, target, _), values in zip(
                log_records,
                log_values_cpu,
            ):
                raw_expected, calibrated_expected, intercept = values
                intercept_text = (
                    f"{intercept:.4f}" if math.isfinite(intercept) else str(intercept)
                )
                log_parts.append(
                    f"{fam_name}:n={n},raw={raw_expected:.1f},"
                    f"target={target:.1f},cal={calibrated_expected:.1f},"
                    f"b={intercept_text}"
                )
            print("[采样-EXPECTED-K] " + "; ".join(log_parts[:10]))
            self._logged_sampling_expected_density_calibration = True
        return adjusted_prob

    def _merge_conservative_topk_pool(self, old, edge_index, logits, scores, quota, total_nodes):
        """Merge one query round into a bounded, deduplicated online Top-K pool."""
        quota = max(0, int(quota))
        if quota <= 0 or edge_index.numel() == 0:
            return old
        take = min(quota, int(scores.numel()))
        top = torch.topk(scores, k=take, largest=True).indices
        edge_index = edge_index[:, top]
        logits = logits[top]
        scores = scores[top]
        if old is not None:
            edge_index = torch.cat([old["edge_index"], edge_index], dim=1)
            logits = torch.cat([old["logits"], logits], dim=0)
            scores = torch.cat([old["scores"], scores], dim=0)

        order = torch.argsort(scores, descending=True)
        keys = (
            edge_index[0, order].long() * int(total_nodes)
            + edge_index[1, order].long()
        ).detach().cpu().tolist()
        keep_pos = []
        seen = set()
        for pos, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            keep_pos.append(pos)
            if len(keep_pos) >= quota:
                break
        keep_pos = torch.tensor(keep_pos, dtype=torch.long, device=edge_index.device)
        chosen = order[keep_pos]
        return {
            "edge_index": edge_index[:, chosen],
            "logits": logits[chosen],
            "scores": scores[chosen],
        }

    def _update_global_exact_k_candidates(
        self,
        candidate_parts,
        logits,
        query_edge_family,
        query_edge_index,
        query_edge_batch,
        total_nodes,
    ):
        """Cache posterior candidates for one global family exact-K decision."""
        logits = self._mask_edge_logits_by_query_family(logits, query_edge_family)
        pos_logits = logits[:, 1:]
        valid = torch.isfinite(pos_logits).any(dim=-1)
        if not valid.any():
            return
        idx = valid.nonzero(as_tuple=True)[0]
        local_logits = logits[idx]
        raw_scores = (
            torch.logsumexp(local_logits[:, 1:], dim=-1) - local_logits[:, 0]
        )
        edge_index = query_edge_index[:, idx].long()
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        canonical_edge_index = torch.stack([u, v], dim=0)
        edge_keys = u * int(total_nodes) + v

        candidate_parts.append(
            {
                "edge_index": canonical_edge_index,
                "edge_keys": edge_keys,
                "scores": raw_scores,
                "family": query_edge_family[idx].long(),
                "batch": query_edge_batch[idx].long(),
                "logits": local_logits,
            }
        )

    @staticmethod
    def _canonical_edge_key_set(edge_index, total_nodes):
        if edge_index is None or edge_index.numel() == 0:
            return set()
        u = torch.minimum(edge_index[0], edge_index[1]).detach().cpu()
        v = torch.maximum(edge_index[0], edge_index[1]).detach().cpu()
        return set((u * int(total_nodes) + v).tolist())

    def _degree_role_reference_context(self, total_nodes, device):
        cached = getattr(self, "_degree_role_reference_cache", None)
        if cached is not None and int(cached["num_nodes"]) == int(total_nodes):
            return cached
        datamodule = getattr(self.dataset_info, "datamodule", None)
        dataset = (
            getattr(datamodule, "train_dataset", None)
            if datamodule is not None
            else None
        )
        if dataset is None or len(dataset) == 0:
            return None
        reference = dataset[0]
        node_type = getattr(reference, "node_type", None)
        if node_type is None:
            return None
        node_type = node_type.long().to(device)
        edge_index = reference.edge_index.long().to(device)
        degree = torch.zeros(int(total_nodes), dtype=torch.float, device=device)
        if edge_index.numel():
            endpoints = torch.cat([edge_index[0], edge_index[1]])
            degree.scatter_add_(
                0, endpoints, torch.ones_like(endpoints, dtype=torch.float)
            )
        num_types = int(node_type.max().item()) + 1
        p50 = torch.zeros(num_types, dtype=torch.float, device=device)
        p80 = torch.zeros(num_types, dtype=torch.float, device=device)
        max_real = torch.zeros(num_types, dtype=torch.float, device=device)
        for type_id in range(num_types):
            values = degree[node_type == type_id]
            if values.numel():
                p50[type_id] = torch.quantile(values, 0.5)
                p80[type_id] = torch.quantile(values, 0.8)
                max_real[type_id] = values.max()
        cached = {
            "num_nodes": int(total_nodes),
            "node_type": node_type,
            "p50": p50,
            "p80": p80,
            "max_real": max_real,
        }
        self._degree_role_reference_cache = cached
        return cached

    def _log_exact_k_ranking_intervention(
        self,
        clean_selected_edges,
        mixed_selected_edges,
        current_edge_index,
        candidate_parts,
        total_nodes,
        s_float,
        t_float,
    ):
        clean = self._canonical_edge_key_set(
            clean_selected_edges, total_nodes
        )
        mixed = self._canonical_edge_key_set(
            mixed_selected_edges, total_nodes
        )
        current = self._canonical_edge_key_set(
            current_edge_index, total_nodes
        )
        key_to_family = {}
        for part in candidate_parts:
            keys = part["edge_keys"].detach().cpu().tolist()
            families = part["family"].detach().cpu().tolist()
            for key, family in zip(keys, families):
                key_to_family[int(key)] = int(family)
        family_stats = {}
        edge_family2id = getattr(
            self.dataset_info, "edge_family2id", {}
        ) or {}
        for family_name, family_id in edge_family2id.items():
            clean_family = {
                key for key in clean
                if key_to_family.get(int(key)) == int(family_id)
            }
            mixed_family = {
                key for key in mixed
                if key_to_family.get(int(key)) == int(family_id)
            }
            family_union = clean_family | mixed_family
            family_stats[str(family_name)] = {
                "jaccard": (
                    len(clean_family & mixed_family)
                    / max(len(family_union), 1)
                ),
                "intervention": len(clean_family ^ mixed_family) // 2,
            }

        degree = torch.zeros(
            int(total_nodes), dtype=torch.float, device=current_edge_index.device
        )
        if current_edge_index.numel():
            endpoints = torch.cat(
                [current_edge_index[0], current_edge_index[1]]
            ).long()
            degree.scatter_add_(
                0,
                endpoints,
                torch.ones_like(endpoints, dtype=torch.float),
            )
        role_reference = self._degree_role_reference_context(
            total_nodes, current_edge_index.device
        )
        intervention_rows = []
        if role_reference is not None:
            node_type = role_reference["node_type"]
            p50 = role_reference["p50"]
            p80 = role_reference["p80"]

            def role_id(node):
                type_id = int(node_type[node].item())
                value = float(degree[node].item())
                if value <= float(p50[type_id].item()):
                    return 0
                if value <= float(p80[type_id].item()):
                    return 1
                return 2

            role_names = ("low", "mid", "high")
            id2family = {
                int(value): str(key)
                for key, value in edge_family2id.items()
            }
            for family_id, family_name in sorted(id2family.items()):
                clean_family = {
                    key for key in clean
                    if key_to_family.get(int(key)) == family_id
                }
                mixed_family = {
                    key for key in mixed
                    if key_to_family.get(int(key)) == family_id
                }
                same_type = False
                endpoints = getattr(
                    self.dataset_info, "fam_endpoints", {}
                ).get(family_name, {})
                same_type = (
                    endpoints.get("src_type")
                    == endpoints.get("dst_type")
                )
                for src_role in range(3):
                    for dst_role in range(3):
                        if same_type and src_role > dst_role:
                            continue

                        def keys_for_roles(keys):
                            selected = set()
                            for key in keys:
                                src = int(key) // int(total_nodes)
                                dst = int(key) % int(total_nodes)
                                left, right = role_id(src), role_id(dst)
                                if same_type and left > right:
                                    left, right = right, left
                                if left == src_role and right == dst_role:
                                    selected.add(key)
                            return selected

                        clean_role = keys_for_roles(clean_family)
                        mixed_role = keys_for_roles(mixed_family)
                        clean_existing = clean_role & current
                        mixed_existing = mixed_role & current
                        clean_new = clean_role - current
                        mixed_new = mixed_role - current
                        intervention_rows.append(
                            {
                                "family": family_name,
                                "src_role": role_names[src_role],
                                "dst_role": role_names[dst_role],
                                "clean_selected": len(clean_role),
                                "mixed_selected": len(mixed_role),
                                "retained_extra": max(
                                    len(mixed_existing)
                                    - len(clean_existing),
                                    0,
                                ),
                                "addition_blocked": max(
                                    len(clean_new) - len(mixed_new),
                                    0,
                                ),
                            }
                        )

        def degree_role_stats(selected_edges):
            if selected_edges is None or selected_edges.numel() == 0:
                return 0.0, 0.0
            src_degree = degree[selected_edges[0].long()]
            dst_degree = degree[selected_edges[1].long()]
            return (
                float((src_degree * dst_degree).mean().detach().cpu()),
                float(
                    (src_degree - dst_degree)
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                ),
            )

        clean_degree_product, clean_degree_diff = degree_role_stats(
            clean_selected_edges
        )
        mixed_degree_product, mixed_degree_diff = degree_role_stats(
            mixed_selected_edges
        )
        union = clean | mixed
        intersection = clean & mixed
        intervention = len(clean ^ mixed) // 2
        clean_retained = len(clean & current)
        mixed_retained = len(mixed & current)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(
            t_normalized=s_float
        )
        mix_weight = self._reverse_posterior_mix_weight(
            s_float=s_float, t_float=t_float
        )
        print(
            "[采样-RANK-INTERVENTION] "
            f"t={float(t_float.mean().detach().cpu()):.4f} "
            f"s={float(s_float.mean().detach().cpu()):.4f} "
            f"alpha_bar_s={float(alpha_s_bar.mean().detach().cpu()):.6f} "
            f"lambda={mix_weight:.6f} "
            f"jaccard={len(intersection) / max(len(union), 1):.6f} "
            f"intervention={intervention} "
            f"clean_keep/add/del={clean_retained}/"
            f"{len(clean-current)}/{len(current-clean)} "
            f"mix_keep/add/del={mixed_retained}/"
            f"{len(mixed-current)}/{len(current-mixed)} "
            f"posterior_extra_keep={max(mixed_retained-clean_retained, 0)} "
            f"posterior_blocked_add={max(len(clean-current)-len(mixed-current), 0)} "
            f"degree_product={clean_degree_product:.4f}->{mixed_degree_product:.4f} "
            f"degree_diff={clean_degree_diff:.4f}->{mixed_degree_diff:.4f} "
            f"families={json.dumps(family_stats, ensure_ascii=False)}"
        )
        if intervention_rows:
            record = {
                "t": float(t_float.mean().detach().cpu()),
                "s": float(s_float.mean().detach().cpu()),
                "alpha_bar_s": float(alpha_s_bar.mean().detach().cpu()),
                "lambda": mix_weight,
                "rows": intervention_rows,
            }
            with open(
                "posterior_intervention.jsonl",
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    def _finalize_global_exact_k_candidates(
        self,
        data,
        candidate_parts,
        selection_mode,
        sampling_step,
        apply_connectivity_repair=False,
        log_selection=True,
    ):
        """Deduplicate candidates, then select exact family quotas globally."""
        if not candidate_parts:
            return None, None

        edge_index = torch.cat(
            [part["edge_index"] for part in candidate_parts], dim=1
        )
        edge_keys = torch.cat([part["edge_keys"] for part in candidate_parts])
        raw_scores = torch.cat([part["scores"] for part in candidate_parts])
        families = torch.cat([part["family"] for part in candidate_parts])
        edge_batch = torch.cat([part["batch"] for part in candidate_parts])
        logits = torch.cat([part["logits"] for part in candidate_parts], dim=0)
        candidate_count_before = int(raw_scores.numel())

        # Stable canonical ordering makes duplicate resolution and random draws
        # independent of the random block traversal order.
        order = torch.argsort(edge_keys)
        edge_index = edge_index[:, order]
        edge_keys = edge_keys[order]
        raw_scores = raw_scores[order]
        families = families[order]
        edge_batch = edge_batch[order]
        logits = logits[order]

        # A canonical edge may appear in multiple block queries. Keep the
        # occurrence with the highest posterior existence logit, then add at
        # most one Gumbel draw to that unique edge.
        _, inverse = torch.unique_consecutive(edge_keys, return_inverse=True)
        num_unique = int(inverse[-1].item()) + 1 if inverse.numel() else 0
        max_scores = torch.full(
            (num_unique,),
            -float("inf"),
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )
        max_scores.scatter_reduce_(
            0, inverse, raw_scores, reduce="amax", include_self=True
        )
        positions = torch.arange(
            raw_scores.numel(), dtype=torch.long, device=raw_scores.device
        )
        sentinel = int(raw_scores.numel())
        first_max = torch.full(
            (num_unique,),
            sentinel,
            dtype=torch.long,
            device=raw_scores.device,
        )
        is_max = raw_scores == max_scores[inverse]
        first_max.scatter_reduce_(
            0,
            inverse,
            torch.where(
                is_max,
                positions,
                torch.full_like(positions, sentinel),
            ),
            reduce="amin",
            include_self=True,
        )
        chosen = first_max[first_max < sentinel]
        edge_index = edge_index[:, chosen]
        edge_keys = edge_keys[chosen]
        raw_scores = raw_scores[chosen]
        families = families[chosen]
        edge_batch = edge_batch[chosen]
        logits = logits[chosen]
        candidate_count_after = int(raw_scores.numel())

        mode = str(selection_mode).lower()
        base_seed = self._current_sampling_seed()
        degree_pair_bias = None
        if mode in ("gumbel_exact_k_degree_pair", "deterministic_exact_k_degree_pair"):
            degree_pair_bias = self._degree_pair_exact_k_score_bias(edge_index, families)
            if degree_pair_bias is not None:
                raw_scores = raw_scores + degree_pair_bias.to(
                    device=raw_scores.device,
                    dtype=raw_scores.dtype,
                )
        structure_guidance_bias = self._sampling_structure_guidance_bias(
            data=data,
            edge_index=edge_index,
            families=families,
            edge_batch=edge_batch,
            sampling_step=sampling_step,
        )
        if structure_guidance_bias is not None:
            raw_scores = raw_scores + structure_guidance_bias.to(
                device=raw_scores.device,
                dtype=raw_scores.dtype,
            )

        if mode in (
            "gumbel_exact_k",
            "gumbel_exact_k_degree_pair",
            "gumbel_exact_k_degree_pair_quota",
        ):
            temperature = max(
                float(
                    getattr(
                        self.cfg.model,
                        "sampling_gumbel_temperature",
                        1.0,
                    )
                    or 1.0
                ),
                1e-8,
            )
            gumbel_mode = str(
                getattr(
                    self.cfg.general,
                    "equivariance_gumbel_mode",
                    "position",
                )
                or "position"
            ).lower()
            seed_offset = int(
                getattr(
                    self.cfg.general,
                    "equivariance_gumbel_seed_offset",
                    0,
                )
                or 0
            )
            # Production mode uses IID draws in canonical-array order. The
            # mapped-reference diagnostic instead draws a reference-coordinate
            # NxN table and transports values through the supplied permutation.
            # Thus overlapping semantic edges receive the same random value
            # even when candidate sets or current node IDs differ.
            gumbel_generator = torch.Generator(device="cpu")
            gumbel_generator.manual_seed(
                base_seed + 1000003 * int(sampling_step) + seed_offset
            )
            reference_new_to_old = getattr(
                data, "equivariance_reference_new_to_old", None
            )
            if gumbel_mode == "mapped_reference":
                total_nodes = int(data.node.shape[0])
                if reference_new_to_old is None:
                    reference_new_to_old = torch.arange(
                        total_nodes, dtype=torch.long, device=edge_index.device
                    )
                else:
                    reference_new_to_old = reference_new_to_old.long().to(
                        edge_index.device
                    )
                ref_edges = reference_new_to_old[edge_index.long()]
                ref_u = torch.minimum(ref_edges[0], ref_edges[1])
                ref_v = torch.maximum(ref_edges[0], ref_edges[1])
                ref_keys = ref_u * total_nodes + ref_v
                uniform_table = torch.rand(
                    (total_nodes * total_nodes,),
                    generator=gumbel_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                uniform = uniform_table[ref_keys.detach().cpu()]
            elif gumbel_mode in ("position", "independent"):
                uniform = torch.rand(
                    raw_scores.shape,
                    generator=gumbel_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
            else:
                raise ValueError(
                    "Unsupported general.equivariance_gumbel_mode: "
                    f"{gumbel_mode}"
                )
            uniform = uniform.clamp_(1e-8, 1.0 - 1e-8)
            gumbel = (-torch.log(-torch.log(uniform))).to(
                device=raw_scores.device,
                dtype=raw_scores.dtype,
            )
            selection_scores = raw_scores / temperature + gumbel
        elif mode in (
            "deterministic_exact_k",
            "deterministic_exact_k_degree_pair",
            "deterministic_exact_k_degree_pair_quota",
        ):
            temperature = None
            selection_scores = raw_scores
        else:
            raise ValueError(f"Unsupported global exact-K mode: {selection_mode}")

        avg_counts = (
            getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        )
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
        graph_ids = edge_batch.unique(sorted=True)
        selected_parts = []
        log_parts = []
        unfilled = {}
        if mode in ("gumbel_exact_k_degree_pair_quota", "deterministic_exact_k_degree_pair_quota"):
            quota_result = self._select_exact_k_with_degree_pair_quotas(
                selection_scores=selection_scores,
                edge_index=edge_index,
                families=families,
                edge_batch=edge_batch,
                avg_counts=avg_counts,
                id2edge_family=id2edge_family,
            )
            if quota_result[0] is not None:
                selected_parts, log_parts, unfilled = quota_result

        if not selected_parts:
            for graph_idx_tensor in graph_ids:
                graph_idx = int(graph_idx_tensor.item())
                for fam_id, fam_name in id2edge_family.items():
                    mask = (edge_batch == graph_idx) & (families == int(fam_id))
                    local_idx = mask.nonzero(as_tuple=True)[0]
                    target = max(
                        0,
                        int(round(float(avg_counts.get(fam_name, 0.0) or 0.0))),
                    )
                    take = min(target, int(local_idx.numel()))
                    if take > 0:
                        top_local = torch.topk(
                            selection_scores[local_idx],
                            k=take,
                            largest=True,
                        ).indices
                        selected_parts.append(local_idx[top_local])
                    if take != target:
                        unfilled[(graph_idx, fam_name)] = (target, take)
                    if getattr(self, "local_rank", 0) == 0 and target > 0:
                        log_parts.append(
                            f"g{graph_idx}/{fam_name}:target={target},"
                            f"candidates={int(local_idx.numel())},selected={take}"
                        )

        if selected_parts:
            selected = torch.cat(selected_parts)
            # Stable endpoint order gives subtype sampling common random numbers
            # across temperatures and deterministic/Gumbel modes.
            selected = selected[torch.argsort(edge_keys[selected])]
        else:
            selected = torch.empty(
                0, dtype=torch.long, device=edge_index.device
            )
        if log_selection and getattr(self, "local_rank", 0) == 0:
            temp_text = "deterministic" if temperature is None else f"{temperature:g}"
            print(
                "[采样-EXACT-K] "
                f"mode={mode} temperature={temp_text} "
                f"candidates={candidate_count_before}->{candidate_count_after} "
                f"selected={int(selected.numel())} "
                f"unfilled={unfilled} "
                + "; ".join(log_parts[:10])
            )

        if apply_connectivity_repair and selected.numel() > 0:
            selected = self._repair_global_exact_k_connectivity(
                data=data,
                edge_index=edge_index,
                raw_scores=raw_scores,
                families=families,
                edge_batch=edge_batch,
                selected=selected,
            )

        if selected.numel() == 0:
            labels = torch.empty(
                0, dtype=torch.long, device=edge_index.device
            )
        else:
            subtype_prob_cpu = torch.softmax(
                logits[selected, 1:].detach().float().cpu(),
                dim=-1,
            )
            if str(
                getattr(
                    self.cfg.general,
                    "equivariance_gumbel_mode",
                    "position",
                )
                or "position"
            ).lower() == "mapped_reference":
                total_nodes = int(data.node.shape[0])
                reference_new_to_old = getattr(
                    data, "equivariance_reference_new_to_old", None
                )
                if reference_new_to_old is None:
                    reference_new_to_old = torch.arange(total_nodes)
                reference_new_to_old = reference_new_to_old.long().cpu()
                selected_edges_cpu = edge_index[:, selected].long().detach().cpu()
                ref_edges = reference_new_to_old[selected_edges_cpu]
                ref_u = torch.minimum(ref_edges[0], ref_edges[1])
                ref_v = torch.maximum(ref_edges[0], ref_edges[1])
                ref_keys = ref_u * total_nodes + ref_v
                subtype_generator = torch.Generator(device="cpu")
                subtype_generator.manual_seed(
                    base_seed + 2000003 * int(sampling_step) + 7919
                )
                subtype_uniform_table = torch.rand(
                    (total_nodes * total_nodes,),
                    generator=subtype_generator,
                    dtype=torch.float32,
                )
                subtype_uniform = subtype_uniform_table[ref_keys].unsqueeze(1)
                labels = (
                    (subtype_uniform > subtype_prob_cpu.cumsum(dim=1))
                    .sum(dim=1)
                    .clamp(max=subtype_prob_cpu.shape[1] - 1)
                    .long()
                    .to(edge_index.device)
                    + 1
                )
            else:
                subtype_generator = torch.Generator(device="cpu")
                subtype_generator.manual_seed(
                    base_seed + 2000003 * int(sampling_step) + 7919
                    + int(
                        getattr(
                            self.cfg.general,
                            "equivariance_gumbel_seed_offset",
                            0,
                        )
                        or 0
                    )
                )
                labels = (
                    torch.multinomial(
                        subtype_prob_cpu,
                        num_samples=1,
                        replacement=True,
                        generator=subtype_generator,
                    )
                    .flatten()
                    .long()
                    .to(edge_index.device)
                    + 1
                )
        return edge_index[:, selected], labels

    def _repair_global_exact_k_connectivity(
        self,
        data,
        edge_index,
        raw_scores,
        families,
        edge_batch,
        selected,
    ):
        """Minimally connect an exact-K graph with quota-preserving exchanges."""
        device = edge_index.device
        edge_index_cpu = edge_index.long().detach().cpu()
        scores_cpu = raw_scores.float().detach().cpu()
        families_cpu = families.long().detach().cpu()
        edge_batch_cpu = edge_batch.long().detach().cpu()
        selected_idx_cpu = selected.long().detach().cpu()
        candidate_count = int(scores_cpu.numel())

        selected_mask = torch.zeros(candidate_count, dtype=torch.bool)
        selected_mask[selected_idx_cpu] = True
        batch_cpu = data.batch.long().detach().cpu()
        total_nodes = int(data.node.shape[0])
        graph_ids = batch_cpu.unique(sorted=True).tolist()

        family_targets = {}
        for graph_idx, fam_id in zip(
            edge_batch_cpu[selected_mask].tolist(),
            families_cpu[selected_mask].tolist(),
        ):
            key = (int(graph_idx), int(fam_id))
            family_targets[key] = family_targets.get(key, 0) + 1

        parent = list(range(total_nodes))
        rank = [0] * total_nodes
        component_size = [1] * total_nodes

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            root_left, root_right = find(left), find(right)
            if root_left == root_right:
                return False
            if rank[root_left] < rank[root_right]:
                root_left, root_right = root_right, root_left
            parent[root_right] = root_left
            component_size[root_left] += component_size[root_right]
            if rank[root_left] == rank[root_right]:
                rank[root_left] += 1
            return True

        # A maximum-posterior spanning forest identifies selected edges that
        # cannot be deleted without breaking already established connectivity.
        forest_mask = torch.zeros(candidate_count, dtype=torch.bool)
        selected_order = selected_idx_cpu[
            torch.argsort(scores_cpu[selected_idx_cpu], descending=True)
        ]
        for pos in selected_order.tolist():
            u = int(edge_index_cpu[0, pos].item())
            v = int(edge_index_cpu[1, pos].item())
            if union(u, v):
                forest_mask[pos] = True

        def component_stats():
            stats = {}
            for graph_idx in graph_ids:
                nodes = torch.where(batch_cpu == int(graph_idx))[0].tolist()
                roots = [find(int(node)) for node in nodes]
                stats[int(graph_idx)] = {
                    "components": len(set(roots)),
                    "lcc": max(
                        (component_size[find(int(node))] for node in nodes),
                        default=0,
                    ),
                }
            return stats

        before = component_stats()
        current_components = {
            graph_idx: int(info["components"])
            for graph_idx, info in before.items()
        }
        before_triangles = self._count_total_triangles_sparse(
            edge_index_cpu[:, selected_mask],
            torch.ones(int(selected_mask.sum().item()), dtype=torch.long),
            total_nodes,
        )

        # Only non-forest selected edges are safe deletion candidates. Within
        # each graph/family pool, remove the lowest posterior score first.
        delete_pools = {}
        for key in family_targets:
            graph_idx, fam_id = key
            mask = (
                selected_mask
                & (~forest_mask)
                & (edge_batch_cpu == int(graph_idx))
                & (families_cpu == int(fam_id))
            )
            pool = mask.nonzero(as_tuple=True)[0]
            if pool.numel() > 0:
                pool = pool[torch.argsort(scores_cpu[pool], descending=False)]
            delete_pools[key] = pool.tolist()

        max_swaps = int(
            getattr(self.cfg.model, "sampling_exact_k_repair_max_swaps", 0) or 0
        )
        repair_added = []
        repair_removed = []
        score_delta = 0.0
        infeasible_families = set()
        unselected = (~selected_mask).nonzero(as_tuple=True)[0]
        bridge_order = unselected[
            torch.argsort(scores_cpu[unselected], descending=True)
        ]
        for pos in bridge_order.tolist():
            if max_swaps > 0 and len(repair_added) >= max_swaps:
                break
            graph_idx = int(edge_batch_cpu[pos].item())
            if current_components.get(graph_idx, 1) <= 1:
                continue
            u = int(edge_index_cpu[0, pos].item())
            v = int(edge_index_cpu[1, pos].item())
            if find(u) == find(v):
                continue

            key = (graph_idx, int(families_cpu[pos].item()))
            pool = delete_pools.get(key, [])
            while pool and not selected_mask[pool[0]]:
                pool.pop(0)
            if not pool:
                infeasible_families.add(key)
                continue

            removed = pool.pop(0)
            selected_mask[removed] = False
            selected_mask[pos] = True
            forest_mask[pos] = True
            union(u, v)
            current_components[graph_idx] = max(
                1, current_components.get(graph_idx, 1) - 1
            )
            repair_added.append(pos)
            repair_removed.append(removed)
            score_delta += float(
                scores_cpu[pos].item() - scores_cpu[removed].item()
            )
            if all(count <= 1 for count in current_components.values()):
                break

        after = component_stats()
        after_triangles = self._count_total_triangles_sparse(
            edge_index_cpu[:, selected_mask],
            torch.ones(int(selected_mask.sum().item()), dtype=torch.long),
            total_nodes,
        )
        unresolved = {
            graph_idx: info["components"]
            for graph_idx, info in after.items()
            if info["components"] > 1
        }

        final_counts = {}
        for graph_idx, fam_id in zip(
            edge_batch_cpu[selected_mask].tolist(),
            families_cpu[selected_mask].tolist(),
        ):
            key = (int(graph_idx), int(fam_id))
            final_counts[key] = final_counts.get(key, 0) + 1
        quota_mismatch = {
            str(key): (int(target), int(final_counts.get(key, 0)))
            for key, target in family_targets.items()
            if int(target) != int(final_counts.get(key, 0))
        }
        if quota_mismatch:
            raise RuntimeError(
                f"exact-K connectivity repair changed family quotas: {quota_mismatch}"
            )
        if int(selected_mask.sum().item()) != int(selected_idx_cpu.numel()):
            raise RuntimeError(
                "exact-K connectivity repair changed the total selected edge count"
            )

        if getattr(self, "local_rank", 0) == 0:
            before_components = {g: v["components"] for g, v in before.items()}
            after_components = {g: v["components"] for g, v in after.items()}
            before_lcc = {g: v["lcc"] for g, v in before.items()}
            after_lcc = {g: v["lcc"] for g, v in after.items()}
            print(
                "[采样-EXACT-K-REPAIR] "
                f"components={before_components}->{after_components} "
                f"lcc={before_lcc}->{after_lcc} "
                f"swaps={len(repair_added)} "
                f"score_delta={score_delta:.6f} "
                f"triangles={before_triangles}->{after_triangles} "
                f"infeasible_family_count={len(infeasible_families)} "
                f"unresolved_components={unresolved}"
            )

        repaired = selected_mask.nonzero(as_tuple=True)[0]
        # Restore stable endpoint order before subtype sampling.
        repaired_keys = (
            edge_index_cpu[0, repaired] * total_nodes
            + edge_index_cpu[1, repaired]
        )
        repaired = repaired[torch.argsort(repaired_keys)]
        return repaired.to(device=device)

    def _update_topk_density_repair_candidates(
        self,
        candidate_parts,
        logits,
        query_edge_family,
        query_edge_index,
        query_edge_batch,
        total_nodes,
    ):
        """Cache compact final-step candidates for global Top-K and connectivity repair."""
        logits = self._mask_edge_logits_by_query_family(logits, query_edge_family)
        pos_logits = logits[:, 1:]
        valid = torch.isfinite(pos_logits).any(dim=-1)
        if not valid.any():
            return

        idx = valid.nonzero(as_tuple=True)[0]
        no_edge_logit = logits[idx, 0]
        valid_pos_logits = pos_logits[idx]
        scores = torch.logsumexp(valid_pos_logits, dim=-1) - no_edge_logit
        temp = max(
            float(getattr(self.cfg.model, "sampling_exist_temperature", 1.0) or 1.0),
            1e-6,
        )
        bias = float(getattr(self.cfg.model, "sampling_exist_logit_bias", 0.0) or 0.0)
        scores = scores / temp - bias

        edge_index = query_edge_index[:, idx].long()
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        edge_keys = u * int(total_nodes) + v
        labels = valid_pos_logits.argmax(dim=-1).long() + 1
        candidate_parts.append(
            {
                "edge_index": edge_index.detach().cpu(),
                "edge_keys": edge_keys.detach().cpu(),
                "scores": scores.detach().float().cpu(),
                "family": query_edge_family[idx].long().detach().cpu(),
                "batch": query_edge_batch[idx].long().detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )

    def _finalize_topk_density_repair_candidates(
        self,
        data,
        candidate_parts,
        base_edge_index,
        base_edge_labels,
    ):
        """Minimally repair the ordinary final-step Top-K graph."""
        if not candidate_parts:
            return None, None

        edge_index = torch.cat([part["edge_index"] for part in candidate_parts], dim=1)
        edge_keys = torch.cat([part["edge_keys"] for part in candidate_parts])
        scores = torch.cat([part["scores"] for part in candidate_parts])
        families = torch.cat([part["family"] for part in candidate_parts])
        edge_batch = torch.cat([part["batch"] for part in candidate_parts])
        labels = torch.cat([part["labels"] for part in candidate_parts])
        candidate_count_before = int(scores.numel())

        # Keep the highest-scoring orientation/occurrence of each canonical edge.
        _, inverse = torch.unique(edge_keys, sorted=False, return_inverse=True)
        num_unique = int(inverse.max().item()) + 1 if inverse.numel() else 0
        max_scores = torch.full((num_unique,), -float("inf"), dtype=scores.dtype)
        max_scores.scatter_reduce_(0, inverse, scores, reduce="amax", include_self=True)
        is_max = scores == max_scores[inverse]
        positions = torch.arange(scores.numel(), dtype=torch.long)
        sentinel = int(scores.numel())
        first_max = torch.full((num_unique,), sentinel, dtype=torch.long)
        first_max.scatter_reduce_(
            0,
            inverse,
            torch.where(is_max, positions, torch.full_like(positions, sentinel)),
            reduce="amin",
            include_self=True,
        )
        chosen = first_max[first_max < sentinel]
        edge_index = edge_index[:, chosen]
        edge_keys = edge_keys[chosen]
        scores = scores[chosen]
        families = families[chosen]
        edge_batch = edge_batch[chosen]
        labels = labels[chosen]
        candidate_count = int(scores.numel())

        selected = torch.zeros(candidate_count, dtype=torch.bool)
        base_edge_index = base_edge_index.long().detach().cpu()
        base_edge_labels = base_edge_labels.long().detach().cpu()
        base_u = torch.minimum(base_edge_index[0], base_edge_index[1])
        base_v = torch.maximum(base_edge_index[0], base_edge_index[1])
        base_keys = base_u * int(data.node.shape[0]) + base_v

        # Map the already-sampled ordinary Top-K graph into the full candidate pool.
        key_order = torch.argsort(edge_keys)
        sorted_candidate_keys = edge_keys[key_order]
        locations = torch.searchsorted(sorted_candidate_keys, base_keys)
        valid_location = locations < sorted_candidate_keys.numel()
        matched = torch.zeros_like(valid_location)
        matched[valid_location] = (
            sorted_candidate_keys[locations[valid_location]] == base_keys[valid_location]
        )
        if not matched.all():
            missing = int((~matched).sum().item())
            raise RuntimeError(
                f"topk_density_repair missing {missing} selected edges from final candidate pool"
            )
        base_candidate_idx = key_order[locations]
        selected[base_candidate_idx] = True
        # Preserve the subtype labels sampled by the original topk_density path.
        labels[base_candidate_idx] = base_edge_labels

        family_targets = {}
        for graph_idx, fam_id in zip(
            edge_batch[selected].tolist(), families[selected].tolist()
        ):
            key = (int(graph_idx), int(fam_id))
            family_targets[key] = family_targets.get(key, 0) + 1
        graph_ids = edge_batch.unique(sorted=True).tolist()

        total_nodes = int(data.node.shape[0])
        batch = data.batch.long().detach().cpu()
        parent = list(range(total_nodes))
        rank = [0] * total_nodes
        comp_size = [1] * total_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            comp_size[ra] += comp_size[rb]
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        forest = torch.zeros(candidate_count, dtype=torch.bool)
        selected_idx = selected.nonzero(as_tuple=True)[0]
        selected_order = selected_idx[torch.argsort(scores[selected_idx], descending=True)]
        for pos in selected_order.tolist():
            u = int(edge_index[0, pos].item())
            v = int(edge_index[1, pos].item())
            if union(u, v):
                forest[pos] = True

        def component_stats():
            counts = {}
            for graph_idx in graph_ids:
                nodes = torch.where(batch == int(graph_idx))[0].tolist()
                roots = [find(int(node)) for node in nodes]
                counts[int(graph_idx)] = {
                    "components": len(set(roots)),
                    "lcc": max((comp_size[find(int(node))] for node in nodes), default=0),
                }
            return counts

        before = component_stats()
        current_components = {
            graph_idx: int(info["components"]) for graph_idx, info in before.items()
        }
        delete_pools = {}
        for key in family_targets:
            graph_idx, fam_id = key
            mask = (
                selected
                & (~forest)
                & (edge_batch == int(graph_idx))
                & (families == int(fam_id))
            )
            pool = mask.nonzero(as_tuple=True)[0]
            if pool.numel() > 0:
                pool = pool[torch.argsort(scores[pool], descending=False)]
            delete_pools[key] = pool.tolist()

        repair_added = []
        repair_removed = []
        repair_score_delta = 0.0
        infeasible_families = set()
        unselected_idx = (~selected).nonzero(as_tuple=True)[0]
        repair_order = unselected_idx[torch.argsort(scores[unselected_idx], descending=True)]
        for pos in repair_order.tolist():
            graph_idx = int(edge_batch[pos].item())
            if current_components.get(graph_idx, 1) <= 1:
                continue
            u = int(edge_index[0, pos].item())
            v = int(edge_index[1, pos].item())
            if find(u) == find(v):
                continue
            key = (graph_idx, int(families[pos].item()))
            pool = delete_pools.get(key, [])
            while pool and not selected[pool[0]]:
                pool.pop(0)
            if not pool:
                infeasible_families.add(key)
                continue
            removed = pool.pop(0)
            selected[removed] = False
            selected[pos] = True
            forest[pos] = True
            union(u, v)
            current_components[graph_idx] = max(
                1, current_components.get(graph_idx, 1) - 1
            )
            repair_added.append(pos)
            repair_removed.append(removed)
            repair_score_delta += float(scores[pos].item() - scores[removed].item())
            if all(count <= 1 for count in current_components.values()):
                break

        after = component_stats()
        unresolved = {
            graph_idx: info["components"]
            for graph_idx, info in after.items()
            if info["components"] > 1
        }

        # Assert that every atomic exchange preserved the ordinary Top-K quotas exactly.
        final_counts = {}
        for graph_idx, fam_id in zip(
            edge_batch[selected].tolist(), families[selected].tolist()
        ):
            key = (int(graph_idx), int(fam_id))
            final_counts[key] = final_counts.get(key, 0) + 1
        quota_mismatch = {
            str(key): (int(family_targets[key]), int(final_counts.get(key, 0)))
            for key in family_targets
            if int(family_targets[key]) != int(final_counts.get(key, 0))
        }
        if quota_mismatch:
            raise RuntimeError(
                f"topk_density_repair changed family quotas: {quota_mismatch}"
            )

        if getattr(self, "local_rank", 0) == 0:
            before_components = {g: v["components"] for g, v in before.items()}
            after_components = {g: v["components"] for g, v in after.items()}
            before_lcc = {g: v["lcc"] for g, v in before.items()}
            after_lcc = {g: v["lcc"] for g, v in after.items()}
            print(
                "[采样-REPAIR] "
                f"components={before_components}->{after_components} "
                f"lcc={before_lcc}->{after_lcc} "
                f"added={len(repair_added)} removed={len(repair_removed)} "
                f"score_delta={repair_score_delta:.6f} "
                f"infeasible_family_count={len(infeasible_families)} "
                f"unresolved_components={unresolved} "
                f"candidates={candidate_count_before}->{candidate_count}"
            )

        final_idx = selected.nonzero(as_tuple=True)[0]
        return (
            edge_index[:, final_idx].to(self.device),
            labels[final_idx].long().to(self.device),
        )

    def _prepare_connectivity_candidates(self, sparse_sampled_data):
        """Preserve initialized edges as same-family fallback candidates."""
        selection = str(getattr(self.cfg.model, "sampling_edge_selection", "") or "").lower()
        pseudo_blocks = getattr(sparse_sampled_data, "pseudo_blocks", None)
        if selection != "connectivity_topk" or not pseudo_blocks:
            return sparse_sampled_data

        edge_index = sparse_sampled_data.edge_index.long().to(self.device)
        edge_attr = sparse_sampled_data.edge_attr
        edge_labels = edge_attr.argmax(dim=-1).long() if edge_attr.dim() > 1 else edge_attr.long()
        edge_labels = edge_labels.to(self.device)
        batch = sparse_sampled_data.batch.long().to(self.device)
        num_nodes = int(sparse_sampled_data.node.shape[0])
        block_of = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)
        for block_idx, block_nodes in enumerate(pseudo_blocks):
            if block_nodes is not None and block_nodes.numel() > 0:
                block_of[block_nodes.long().to(self.device)] = int(block_idx)

        label_to_fam = self._label_to_family_lookup().to(self.device)
        valid = (edge_labels > 0) & (edge_labels < int(label_to_fam.numel()))
        edge_family = torch.full_like(edge_labels, -1)
        edge_family[valid] = label_to_fam[edge_labels[valid]]
        edge_batch = batch[edge_index[0]]

        family_targets = {}
        avg_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
        bs = int(batch.max().item()) + 1 if batch.numel() else 1
        for fam_name, fam_id in edge_family2id.items():
            target = max(0, int(round(float(avg_counts.get(fam_name, 0.0) or 0.0))))
            for graph_idx in range(bs):
                family_targets[(int(graph_idx), int(fam_id))] = target

        sparse_sampled_data.connectivity_init_edge_index = edge_index[:, valid]
        sparse_sampled_data.connectivity_init_edge_labels = edge_labels[valid]
        sparse_sampled_data.connectivity_init_family = edge_family[valid]
        sparse_sampled_data.connectivity_init_batch = edge_batch[valid]
        sparse_sampled_data.connectivity_block_of = block_of
        sparse_sampled_data.connectivity_family_targets = family_targets
        return sparse_sampled_data

    def _update_connectivity_candidate_pools(
        self,
        pools,
        stats,
        data,
        logits,
        query_edge_family,
        query_edge_index,
        query_edge_batch,
        total_nodes,
    ):
        """Online-cache family Top-K and block-pair/family Top-m candidates."""
        logits = self._mask_edge_logits_by_query_family(logits, query_edge_family)
        no_edge_logit = logits[:, 0]
        pos_logits = logits[:, 1:]
        valid = torch.isfinite(pos_logits).any(dim=-1)
        scores = torch.logsumexp(pos_logits, dim=-1) - no_edge_logit
        temp = max(float(getattr(self.cfg.model, "sampling_exist_temperature", 1.0) or 1.0), 1e-6)
        bias = float(getattr(self.cfg.model, "sampling_exist_logit_bias", 0.0) or 0.0)
        scores = scores / temp - bias
        qfam = query_edge_family.long().to(logits.device).reshape(-1)
        qbatch = query_edge_batch.long().to(logits.device).reshape(-1)
        block_of = data.connectivity_block_of.long().to(logits.device)
        bu = block_of[query_edge_index[0].long()]
        bv = block_of[query_edge_index[1].long()]
        pair_topm = max(1, int(getattr(self.cfg.model, "sampling_connectivity_pair_topm", 32) or 32))
        target_map = getattr(data, "connectivity_family_targets", {}) or {}
        stats["candidate_count_before"] = int(stats.get("candidate_count_before", 0)) + int(valid.sum().item())

        for (graph_idx, fam_id), target in target_map.items():
            mask = valid & (qbatch == int(graph_idx)) & (qfam == int(fam_id))
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            family_key = ("family", int(graph_idx), int(fam_id))
            pools[family_key] = self._merge_conservative_topk_pool(
                pools.get(family_key),
                query_edge_index[:, idx],
                logits[idx],
                scores[idx],
                max(1, int(target)),
                total_nodes,
            )

            pair_a = torch.minimum(bu[idx], bv[idx])
            pair_b = torch.maximum(bu[idx], bv[idx])
            pair_ids = torch.stack([pair_a, pair_b], dim=1).unique(dim=0)
            for pair in pair_ids:
                a, b = int(pair[0].item()), int(pair[1].item())
                pmask = (pair_a == a) & (pair_b == b)
                pidx = idx[pmask]
                pair_key = ("pair", int(graph_idx), int(fam_id), a, b)
                pools[pair_key] = self._merge_conservative_topk_pool(
                    pools.get(pair_key),
                    query_edge_index[:, pidx],
                    logits[pidx],
                    scores[pidx],
                    pair_topm,
                    total_nodes,
                )

    def _finalize_connectivity_candidate_pools(self, data, pools, stats):
        """Greedy family-constrained maximum-spanning forest, then family Top-K fill."""
        total_nodes = int(data.node.shape[0])
        batch = data.batch.long().to(self.device)
        target_map = getattr(data, "connectivity_family_targets", {}) or {}

        # Merge family and pair caches by canonical endpoint key, retaining max score.
        candidate_map = {}
        for key, item in pools.items():
            graph_idx = int(key[1])
            fam_id = int(key[2])
            for idx in range(int(item["scores"].numel())):
                u = int(item["edge_index"][0, idx].item())
                v = int(item["edge_index"][1, idx].item())
                edge_key = u * total_nodes + v
                score = float(item["scores"][idx].item())
                old = candidate_map.get(edge_key)
                if old is None or score > old["score"]:
                    candidate_map[edge_key] = {
                        "u": u,
                        "v": v,
                        "graph": graph_idx,
                        "family": fam_id,
                        "score": score,
                        "logits": item["logits"][idx],
                    }
        candidates = sorted(candidate_map.values(), key=lambda x: x["score"], reverse=True)
        stats["candidate_count_after_cache"] = len(candidates)

        parent = list(range(total_nodes))
        rank = [0] * total_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        graph_nodes = {}
        for graph_idx in range(int(batch.max().item()) + 1 if batch.numel() else 1):
            graph_nodes[graph_idx] = torch.where(batch == graph_idx)[0].detach().cpu().tolist()
        components_before = {g: len(nodes) for g, nodes in graph_nodes.items()}
        quota_used = {key: 0 for key in target_map}
        selected_keys = set()
        selected_model = []

        # Family-constrained greedy maximum-spanning forest.
        for cand in candidates:
            quota_key = (cand["graph"], cand["family"])
            if quota_used.get(quota_key, 0) >= int(target_map.get(quota_key, 0)):
                continue
            if union(cand["u"], cand["v"]):
                edge_key = cand["u"] * total_nodes + cand["v"]
                selected_keys.add(edge_key)
                selected_model.append(cand)
                quota_used[quota_key] = quota_used.get(quota_key, 0) + 1

        # Fallback: initialized same-family candidates that still bridge components.
        fallback = []
        init_ei = data.connectivity_init_edge_index.long().to(self.device)
        init_labels = data.connectivity_init_edge_labels.long().to(self.device)
        init_fam = data.connectivity_init_family.long().to(self.device)
        init_batch = data.connectivity_init_batch.long().to(self.device)
        perm = torch.randperm(init_labels.numel(), device=self.device)
        for pos in perm.detach().cpu().tolist():
            u, v = int(init_ei[0, pos].item()), int(init_ei[1, pos].item())
            graph_idx, fam_id = int(init_batch[pos].item()), int(init_fam[pos].item())
            quota_key = (graph_idx, fam_id)
            edge_key = u * total_nodes + v
            if edge_key in selected_keys or quota_used.get(quota_key, 0) >= int(target_map.get(quota_key, 0)):
                continue
            if union(u, v):
                selected_keys.add(edge_key)
                fallback.append((u, v, int(init_labels[pos].item()), graph_idx, fam_id))
                quota_used[quota_key] = quota_used.get(quota_key, 0) + 1

        skeleton_count = len(selected_model) + len(fallback)
        skeleton_quota_used = {str(k): int(v) for k, v in quota_used.items() if int(v) > 0}
        components_after = {}
        for graph_idx, nodes in graph_nodes.items():
            components_after[graph_idx] = len({find(int(n)) for n in nodes})

        # Fill each family's remaining exact quota by descending model score.
        for cand in candidates:
            quota_key = (cand["graph"], cand["family"])
            if quota_used.get(quota_key, 0) >= int(target_map.get(quota_key, 0)):
                continue
            edge_key = cand["u"] * total_nodes + cand["v"]
            if edge_key in selected_keys:
                continue
            selected_keys.add(edge_key)
            selected_model.append(cand)
            quota_used[quota_key] = quota_used.get(quota_key, 0) + 1

        # Fill any remaining family quota from initialized same-family edges.
        for pos in perm.detach().cpu().tolist():
            u, v = int(init_ei[0, pos].item()), int(init_ei[1, pos].item())
            graph_idx, fam_id = int(init_batch[pos].item()), int(init_fam[pos].item())
            quota_key = (graph_idx, fam_id)
            edge_key = u * total_nodes + v
            if edge_key in selected_keys or quota_used.get(quota_key, 0) >= int(target_map.get(quota_key, 0)):
                continue
            selected_keys.add(edge_key)
            fallback.append((u, v, int(init_labels[pos].item()), graph_idx, fam_id))
            quota_used[quota_key] = quota_used.get(quota_key, 0) + 1

        out_edges = []
        out_labels = []
        for cand in selected_model:
            logits = self._mask_edge_logits_by_query_family(
                cand["logits"].unsqueeze(0),
                torch.tensor([cand["family"]], dtype=torch.long, device=self.device),
            )
            label = int(torch.softmax(logits[:, 1:], dim=-1).multinomial(1).item()) + 1
            out_edges.append((cand["u"], cand["v"]))
            out_labels.append(label)
        for u, v, label, _graph_idx, _fam_id in fallback:
            out_edges.append((u, v))
            out_labels.append(label)

        unfilled = {
            str(key): max(0, int(target_map[key]) - int(quota_used.get(key, 0)))
            for key in target_map
            if int(quota_used.get(key, 0)) < int(target_map[key])
        }
        stats.update(
            {
                "skeleton_edges": skeleton_count,
                "components_before": components_before,
                "components_after": components_after,
                "family_quota_used_by_skeleton": skeleton_quota_used,
                "fallback_edges": len(fallback),
                "unfilled_family_quota": unfilled,
            }
        )
        if getattr(self, "local_rank", 0) == 0:
            print(
                "[采样-CONNECT] "
                f"skeleton_edges={skeleton_count} components={components_before}->{components_after} "
                f"fallback_edges={len(fallback)} unfilled={sum(unfilled.values())} "
                f"candidates={stats.get('candidate_count_before', 0)}->{stats.get('candidate_count_after_cache', 0)}"
            )
            print(f"[采样-CONNECT-FAMILY] {stats['family_quota_used_by_skeleton']}")

        if not out_edges:
            return None, None
        return (
            torch.tensor(out_edges, dtype=torch.long, device=self.device).t().contiguous(),
            torch.tensor(out_labels, dtype=torch.long, device=self.device),
        )


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
        family_infos = []
        if qfam.numel() == exist_prob.shape[0] and id2edge_family:
            for fam_id, fam_name in id2edge_family.items():
                mask = (qfam == int(fam_id)) & has_valid_pos
                n = int(mask.sum().item())
                if n <= 0:
                    continue
                mean_conf = float(exist_prob[mask].mean().item())
                k = int(round(base_frac * max(mean_conf, 1e-6) * n))
                k = max(1, min(k, n))
                family_infos.append((int(fam_id), str(fam_name), mask, n, k))
            balance_mode = str(
                getattr(
                    self.cfg.model,
                    "sampling_topk_structure_family_balance",
                    "none",
                )
                or "none"
            ).lower()
            if balance_mode in {"equal", "equal_quota"} and family_infos:
                target_k = int(
                    round(
                        sum(int(info[4]) for info in family_infos)
                        / float(len(family_infos))
                    )
                )
                family_infos = [
                    (fam_id, fam_name, mask, n, max(1, min(int(target_k), int(n))))
                    for fam_id, fam_name, mask, n, _k in family_infos
                ]
            elif balance_mode in {"target", "target_count", "family_target", "avg_count"} and family_infos:
                avg_counts = (
                    getattr(self.dataset_info, "edge_family_avg_edge_counts", None)
                    or getattr(self.dataset_info, "edge_family_avg_counts", None)
                    or {}
                )
                target_infos = []
                for fam_id, fam_name, mask, n, k in family_infos:
                    target = avg_counts.get(fam_name, None)
                    if target is None and fam_name.startswith("link_"):
                        target = avg_counts.get(fam_name[len("link_"):], None)
                    if target is None:
                        target_k = int(k)
                    else:
                        target_k = int(round(float(target)))
                    target_infos.append(
                        (fam_id, fam_name, mask, n, max(0, min(int(target_k), int(n))))
                    )
                family_infos = [
                    info
                    for info in target_infos
                    if int(info[4]) > 0
                ]
            for fam_id, fam_name, mask, n, k in family_infos:
                local_idx = mask.nonzero(as_tuple=True)[0]
                top_local = torch.topk(exist_prob[local_idx], k=k, largest=True).indices
                has_edge[local_idx[top_local]] = True
                used_mask |= mask
            if (
                getattr(self, "local_rank", 0) == 0
                and family_infos
                and not getattr(self, "_logged_sampling_topk_structure_balance", False)
            ):
                print(
                    "[采样-TOPK-STRUCTURE] "
                    f"balance={balance_mode} "
                    + "; ".join(
                        f"{fam_name}:n={n},k={k}"
                        for _fam_id, fam_name, _mask, n, k in family_infos
                    )
                )
                self._logged_sampling_topk_structure_balance = True
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
                density = self._family_density_from_marginal(
                    fam_name, marginals, canonical_candidates=True
                )
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

    def _degree_pair_exact_k_score_bias(self, edge_index, families, strength_override=None):
        """Return a per-candidate log-prior bias for global exact-K selection.

        Unlike ``topk_degree_pair``, this does not choose edges directly. It only
        nudges the global exact-K ranking toward the reference graph's
        family-specific endpoint degree-bin-pair distribution, so family quotas
        and total edge count remain controlled by exact-K.
        """
        strength = (
            float(strength_override)
            if strength_override is not None
            else float(getattr(self.cfg.model, "sampling_degree_pair_strength", 0.0) or 0.0)
        )
        if strength <= 0.0 or edge_index is None or edge_index.numel() == 0:
            return None
        ref = self._ensure_sampling_degree_pair_reference()
        degree_bins = ref.get("degree_bins") if ref else None
        pair_counts = ref.get("pair_counts", {}) if ref else {}
        family_totals = ref.get("family_totals", {}) if ref else {}
        if degree_bins is None or degree_bins.numel() == 0 or not pair_counts:
            return None

        device = edge_index.device
        dbins = degree_bins.to(device)
        max_node = int(dbins.numel())
        valid_endpoint = (
            (edge_index[0] >= 0)
            & (edge_index[0] < max_node)
            & (edge_index[1] >= 0)
            & (edge_index[1] < max_node)
        )
        if not bool(valid_endpoint.any()):
            return None

        fam = families.long().to(device).reshape(-1)
        bin_u = torch.zeros_like(fam)
        bin_v = torch.zeros_like(fam)
        bin_u[valid_endpoint] = dbins[edge_index[0, valid_endpoint]]
        bin_v[valid_endpoint] = dbins[edge_index[1, valid_endpoint]]
        pair_a = torch.minimum(bin_u, bin_v)
        pair_b = torch.maximum(bin_u, bin_v)
        num_bins = max(2, int(getattr(self.cfg.model, "sampling_degree_pair_bins", 5) or 5))
        pair_id = pair_a * int(num_bins) + pair_b

        eps = 1e-8
        clip = float(getattr(self.cfg.model, "sampling_degree_pair_bias_clip", 4.0) or 4.0)
        bias = torch.zeros((fam.numel(),), dtype=torch.float32, device=device)
        log_parts = []
        for fam_id, ref_pairs in pair_counts.items():
            fam_id = int(fam_id)
            mask = (fam == fam_id) & valid_endpoint
            n = int(mask.sum().item())
            total = int(family_totals.get(fam_id, 0) or 0)
            if n <= 0 or total <= 0:
                continue
            local_pair_id = pair_id[mask]
            cand_counts = torch.bincount(local_pair_id, minlength=num_bins * num_bins).float()
            local_bias = torch.zeros((num_bins * num_bins,), dtype=torch.float32, device=device)
            for pair_key, cnt in ref_pairs.items():
                a, b = int(pair_key[0]), int(pair_key[1])
                pid = a * int(num_bins) + b
                if pid < 0 or pid >= int(local_bias.numel()):
                    continue
                target_share = float(cnt) / float(max(1, total))
                cand_share = float(cand_counts[pid].item()) / float(max(1, n))
                local_bias[pid] = math.log((target_share + eps) / (cand_share + eps))
            if clip > 0:
                local_bias = local_bias.clamp(min=-clip, max=clip)
            bias[mask] = local_bias[local_pair_id] * strength
            if getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_exactk_degree_pair_bias", False):
                log_parts.append(
                    f"fam={fam_id}:n={n},pairs={len(ref_pairs)},"
                    f"bias=[{float(local_bias.min().item()):.3f},{float(local_bias.max().item()):.3f}]"
                )
        if log_parts and getattr(self, "local_rank", 0) == 0 and not getattr(self, "_logged_sampling_exactk_degree_pair_bias", False):
            print(
                "[采样-EXACT-K-DEGPAIR] "
                f"strength={strength:.4g} clip={clip:.4g} "
                + "; ".join(log_parts[:10])
            )
            self._logged_sampling_exactk_degree_pair_bias = True
        return bias

    def _sampling_structure_guidance_factor(self, sampling_step):
        remaining_frac = float(sampling_step) / float(max(int(self.T), 1))
        lo = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_min_step_frac",
                0.0,
            )
            or 0.0
        )
        hi = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_max_step_frac",
                1.0,
            )
            or 1.0
        )
        if remaining_frac < lo or remaining_frac > hi:
            return 0.0
        power = float(
            getattr(self.cfg.model, "sampling_structure_guidance_power", 1.0)
            or 1.0
        )
        return max(0.0, min(1.0, remaining_frac)) ** max(power, 0.0)

    def _sampling_structure_guidance_bias(
        self,
        data,
        edge_index,
        families,
        edge_batch,
        sampling_step,
    ):
        closure_w = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_closure_weight",
                0.0,
            )
            or 0.0
        )
        connect_w = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_connect_weight",
                0.0,
            )
            or 0.0
        )
        dp_w = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_degree_pair_weight",
                0.0,
            )
            or 0.0
        )
        assort_w = float(
            getattr(
                self.cfg.model,
                "sampling_structure_guidance_assort_weight",
                0.0,
            )
            or 0.0
        )
        factor = self._sampling_structure_guidance_factor(sampling_step)
        if (
            factor <= 0.0
            or edge_index is None
            or edge_index.numel() == 0
            or max(abs(closure_w), abs(connect_w), abs(dp_w), abs(assort_w)) <= 0.0
        ):
            return None

        device = edge_index.device
        bias = torch.zeros((edge_index.shape[1],), dtype=torch.float32, device=device)
        comp_feature = None
        closure_feature = None
        current_edge_index = getattr(data, "edge_index", None)
        current_edge_attr = getattr(data, "edge_attr", None)
        if current_edge_index is not None and current_edge_attr is not None:
            cur_ei = current_edge_index.long().to(device)
            cur_attr = current_edge_attr.to(device)
            if cur_attr.dim() > 1:
                cur_labels = cur_attr.argmax(dim=-1).long()
            else:
                cur_labels = cur_attr.long().reshape(-1)
            visible = cur_labels > 0
            cur_ei = cur_ei[:, visible] if cur_ei.numel() else cur_ei
            node_batch = getattr(data, "batch", None)
            if node_batch is None:
                node_batch = torch.zeros(
                    (int(data.node.shape[0]),), dtype=torch.long, device=device
                )
            else:
                node_batch = node_batch.long().to(device)
            graph_ids = edge_batch.unique(sorted=True)
            closure_feature = torch.zeros_like(bias)
            comp_feature = torch.zeros_like(bias)
            for graph_idx_tensor in graph_ids:
                graph_idx = int(graph_idx_tensor.item())
                nodes = torch.where(node_batch == graph_idx)[0]
                if nodes.numel() <= 1:
                    continue
                local_of = torch.full(
                    (node_batch.shape[0],),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                local_of[nodes] = torch.arange(nodes.numel(), device=device)
                cand_mask = edge_batch == graph_idx
                if not cand_mask.any():
                    continue
                vis_mask = (
                    (cur_ei.numel() > 0)
                    & (node_batch[cur_ei[0].long()] == graph_idx)
                    & (node_batch[cur_ei[1].long()] == graph_idx)
                )
                n = int(nodes.numel())
                adj = torch.zeros((n, n), dtype=torch.float32, device=device)
                if bool(vis_mask.any()):
                    src_vis = local_of[cur_ei[0, vis_mask].long()]
                    dst_vis = local_of[cur_ei[1, vis_mask].long()]
                    valid_vis = (src_vis >= 0) & (dst_vis >= 0)
                    src_vis = src_vis[valid_vis]
                    dst_vis = dst_vis[valid_vis]
                    adj[src_vis, dst_vis] = 1.0
                    adj[dst_vis, src_vis] = 1.0
                adj.fill_diagonal_(0.0)
                common = adj.matmul(adj)
                idx = cand_mask.nonzero(as_tuple=True)[0]
                src = local_of[edge_index[0, idx].long()]
                dst = local_of[edge_index[1, idx].long()]
                valid = (src >= 0) & (dst >= 0)
                if valid.any():
                    local_score = torch.zeros((idx.numel(),), dtype=torch.float32, device=device)
                    cn = common[src[valid], dst[valid]]
                    local_score[valid] = torch.log1p(cn) / math.log1p(max(n, 1))
                    closure_feature[idx] = local_score
                if (connect_w != 0.0) and bool(vis_mask.any()):
                    parent = list(range(n))

                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(a, b):
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[rb] = ra

                    src_cpu = src_vis.detach().cpu().tolist() if bool(vis_mask.any()) else []
                    dst_cpu = dst_vis.detach().cpu().tolist() if bool(vis_mask.any()) else []
                    for a, b in zip(src_cpu, dst_cpu):
                        if a != b:
                            union(int(a), int(b))
                    comp_id = torch.tensor(
                        [find(i) for i in range(n)],
                        dtype=torch.long,
                        device=device,
                    )
                    valid_comp = (src >= 0) & (dst >= 0)
                    local_bridge = torch.zeros((idx.numel(),), dtype=torch.float32, device=device)
                    if valid_comp.any():
                        local_bridge[valid_comp] = (
                            comp_id[src[valid_comp]] != comp_id[dst[valid_comp]]
                        ).to(torch.float32)
                    comp_feature[idx] = local_bridge

        if closure_feature is not None and closure_w != 0.0:
            bias = bias + float(closure_w * factor) * closure_feature
        if comp_feature is not None and connect_w != 0.0:
            bias = bias + float(connect_w * factor) * comp_feature

        if dp_w != 0.0:
            dp_bias = self._degree_pair_exact_k_score_bias(
                edge_index,
                families,
                strength_override=float(dp_w * factor),
            )
            if dp_bias is not None:
                bias = bias + dp_bias.to(device=device, dtype=bias.dtype)

        if assort_w != 0.0:
            ref = self._ensure_sampling_degree_pair_reference()
            degree_bins = ref.get("degree_bins") if ref else None
            if degree_bins is not None and degree_bins.numel() > 0:
                dbins = degree_bins.to(device)
                valid = (
                    (edge_index[0] >= 0)
                    & (edge_index[0] < dbins.numel())
                    & (edge_index[1] >= 0)
                    & (edge_index[1] < dbins.numel())
                )
                if valid.any():
                    num_bins = max(
                        2,
                        int(
                            getattr(
                                self.cfg.model,
                                "sampling_degree_pair_bins",
                                5,
                            )
                            or 5
                        ),
                    )
                    bu = torch.zeros_like(families.long())
                    bv = torch.zeros_like(families.long())
                    bu[valid] = dbins[edge_index[0, valid]]
                    bv[valid] = dbins[edge_index[1, valid]]
                    diff = (bu.float() - bv.float()).abs() / float(max(num_bins - 1, 1))
                    target = getattr(
                        self.cfg.model,
                        "sampling_structure_guidance_assort_target",
                        None,
                    )
                    try:
                        target_val = None if target is None else float(target)
                    except (TypeError, ValueError):
                        target_val = None
                    # PubMed target is negative; null defaults to disassortative
                    # because current experiments aim to reduce positive assortativity.
                    direction = -1.0 if target_val is None or target_val < 0 else 1.0
                    assort_feature = diff if direction < 0 else (1.0 - diff)
                    bias = bias + float(assort_w * factor) * assort_feature.to(bias.dtype)

        if (
            getattr(self, "local_rank", 0) == 0
            and not getattr(self, "_logged_sampling_structure_guidance", False)
        ):
            print(
                "[采样-STRUCT-GUIDE] "
                f"step={sampling_step} factor={factor:.4g} "
                f"closure={closure_w:.4g} connect={connect_w:.4g} "
                f"degree_pair={dp_w:.4g} assort={assort_w:.4g} "
                f"bias=[{float(bias.min().item()):.4g},{float(bias.max().item()):.4g}]"
            )
            self._logged_sampling_structure_guidance = True
        return bias

    def _degree_pair_ids_for_candidates(self, edge_index, families):
        ref = self._ensure_sampling_degree_pair_reference()
        degree_bins = ref.get("degree_bins") if ref else None
        pair_counts = ref.get("pair_counts", {}) if ref else {}
        family_totals = ref.get("family_totals", {}) if ref else {}
        if degree_bins is None or degree_bins.numel() == 0 or not pair_counts:
            return None
        device = edge_index.device
        dbins = degree_bins.to(device)
        max_node = int(dbins.numel())
        valid_endpoint = (
            (edge_index[0] >= 0)
            & (edge_index[0] < max_node)
            & (edge_index[1] >= 0)
            & (edge_index[1] < max_node)
        )
        if not bool(valid_endpoint.any()):
            return None
        fam = families.long().to(device).reshape(-1)
        bin_u = torch.zeros_like(fam)
        bin_v = torch.zeros_like(fam)
        bin_u[valid_endpoint] = dbins[edge_index[0, valid_endpoint]]
        bin_v[valid_endpoint] = dbins[edge_index[1, valid_endpoint]]
        pair_a = torch.minimum(bin_u, bin_v)
        pair_b = torch.maximum(bin_u, bin_v)
        num_bins = max(2, int(getattr(self.cfg.model, "sampling_degree_pair_bins", 5) or 5))
        pair_id = pair_a * int(num_bins) + pair_b
        return {
            "pair_id": pair_id,
            "valid": valid_endpoint,
            "pair_counts": pair_counts,
            "family_totals": family_totals,
            "num_bins": int(num_bins),
        }

    def _select_exact_k_with_degree_pair_quotas(
        self,
        selection_scores,
        edge_index,
        families,
        edge_batch,
        avg_counts,
        id2edge_family,
    ):
        """Select exact family quotas, with optional within-family degree-pair quotas."""
        pair_info = self._degree_pair_ids_for_candidates(edge_index, families)
        if pair_info is None:
            return None, [], {}

        pair_id = pair_info["pair_id"]
        valid_pair = pair_info["valid"]
        pair_counts = pair_info["pair_counts"]
        family_totals = pair_info["family_totals"]
        num_bins = int(pair_info["num_bins"])
        strength = float(getattr(self.cfg.model, "sampling_degree_pair_strength", 1.0) or 1.0)
        strength = max(0.0, min(1.0, strength))
        min_bin_edges = max(0, int(getattr(self.cfg.model, "sampling_degree_pair_min_bin_edges", 1) or 1))
        graph_ids = edge_batch.unique(sorted=True)
        selected_parts = []
        log_parts = []
        unfilled = {}

        for graph_idx_tensor in graph_ids:
            graph_idx = int(graph_idx_tensor.item())
            for fam_id, fam_name in id2edge_family.items():
                fam_id = int(fam_id)
                mask = (edge_batch == graph_idx) & (families == fam_id)
                local_idx = mask.nonzero(as_tuple=True)[0]
                target = max(
                    0,
                    int(round(float(avg_counts.get(fam_name, 0.0) or 0.0))),
                )
                take = min(target, int(local_idx.numel()))
                if take <= 0:
                    if take != target:
                        unfilled[(graph_idx, fam_name)] = (target, take)
                    continue

                local_selected = torch.zeros_like(mask)
                pair_target = int(round(float(take) * strength))
                pair_target = max(0, min(pair_target, take))
                assigned = 0
                pair_infos = []
                fam_total = float(max(1, int(family_totals.get(fam_id, 0))))
                if pair_target > 0 and fam_id in pair_counts:
                    for pair_key, cnt in pair_counts.get(fam_id, {}).items():
                        a, b = int(pair_key[0]), int(pair_key[1])
                        pid = a * num_bins + b
                        pmask = mask & valid_pair & (pair_id == pid)
                        pn = int(pmask.sum().item())
                        if pn <= 0:
                            continue
                        exact = float(pair_target) * float(cnt) / fam_total
                        quota = int(math.floor(exact))
                        if cnt >= min_bin_edges and exact > 0.0:
                            quota = max(1, quota)
                        quota = max(0, min(quota, pn))
                        pair_infos.append(
                            {
                                "mask": pmask,
                                "quota": quota,
                                "frac": exact - math.floor(exact),
                                "n": pn,
                            }
                        )
                        assigned += quota
                    if assigned > pair_target:
                        excess = assigned - pair_target
                        for info in sorted(pair_infos, key=lambda x: x["frac"]):
                            if excess <= 0:
                                break
                            drop = min(int(info["quota"]), excess)
                            info["quota"] -= drop
                            excess -= drop

                    for info in pair_infos:
                        quota = int(info["quota"])
                        if quota <= 0:
                            continue
                        pidx = info["mask"].nonzero(as_tuple=True)[0]
                        top_local = torch.topk(
                            selection_scores[pidx],
                            k=min(quota, int(pidx.numel())),
                            largest=True,
                        ).indices
                        local_selected[pidx[top_local]] = True

                selected_count = int(local_selected.sum().item())
                if selected_count < take:
                    remaining = mask & (~local_selected)
                    ridx = remaining.nonzero(as_tuple=True)[0]
                    add = min(take - selected_count, int(ridx.numel()))
                    if add > 0:
                        top_local = torch.topk(
                            selection_scores[ridx],
                            k=add,
                            largest=True,
                        ).indices
                        local_selected[ridx[top_local]] = True
                selected_count = int(local_selected.sum().item())
                if selected_count > 0:
                    selected_parts.append(local_selected.nonzero(as_tuple=True)[0])
                if selected_count != target:
                    unfilled[(graph_idx, fam_name)] = (target, selected_count)
                if getattr(self, "local_rank", 0) == 0 and target > 0:
                    log_parts.append(
                        f"g{graph_idx}/{fam_name}:target={target},"
                        f"pair_target={pair_target},candidates={int(local_idx.numel())},"
                        f"selected={selected_count}"
                    )
        return selected_parts, log_parts, unfilled

    def _sample_edge_labels_hierarchical(
        self,
        logits,
        query_edge_family=None,
        query_edge_index=None,
        selection_override=None,
    ):
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
        selection = str(
            selection_override
            or getattr(self.cfg.model, "sampling_edge_selection", "bernoulli")
            or "bernoulli"
        ).lower()
        has_edge = torch.zeros_like(exist_prob, dtype=torch.bool)
        if (
            selection == "bernoulli_expected_density"
            and query_edge_family is not None
        ):
            exist_prob = self._calibrate_bernoulli_expected_density_by_family(
                exist_logits,
                has_valid_pos,
                query_edge_family,
            )
            has_edge = (torch.rand_like(exist_prob) < exist_prob) & has_valid_pos
        elif selection in ("topk_density", "topk_density_repair") and query_edge_family is not None:
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

    def _canonicalize_partition_ensemble_query_pool(
        self, query_rounds, total_nodes
    ):
        """Deduplicate one shared query pool before constructing either view."""
        edges, batches, families = [], [], []
        for edge_index, edge_batch, edge_family in query_rounds:
            if (
                edge_index is None
                or edge_index.numel() == 0
                or edge_family is None
            ):
                continue
            edges.append(edge_index.long())
            batches.append(edge_batch.long())
            families.append(edge_family.long())
        if not edges:
            return None
        edge_index = torch.cat(edges, dim=1)
        edge_batch = torch.cat(batches)
        edge_family = torch.cat(families)
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        edge_index = torch.stack([u, v], dim=0)
        num_families = max(
            1,
            len(getattr(self.dataset_info, "edge_family2id", {}) or {}),
        )
        composite = (
            (u * int(total_nodes) + v) * int(num_families)
            + edge_family.clamp_min(0)
        )
        order = torch.argsort(composite)
        composite = composite[order]
        keep = torch.ones(
            composite.shape[0], dtype=torch.bool, device=composite.device
        )
        if composite.numel() > 1:
            keep[1:] = composite[1:] != composite[:-1]
        chosen = order[keep]
        return (
            edge_index[:, chosen],
            edge_batch[chosen],
            edge_family[chosen],
        )

    def _current_gt_hetero_metis_blocks(
        self,
        edge_index,
        edge_attr_ids,
        anchor_node_subtype,
        batch,
    ):
        """Build graph-aware blocks from the currently visible G_t only."""
        from sparse_diffusion.graph_partition.connected_blocks import (
            hetero_metis_blocks_from_graph,
        )

        edge_family_offsets = (
            getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        )
        edge_family2id = (
            getattr(self.dataset_info, "edge_family2id", {}) or {}
        )
        offset_items = sorted(
            (int(offset), str(name))
            for name, offset in edge_family_offsets.items()
        )
        edge_family = torch.zeros_like(edge_attr_ids, dtype=torch.long)
        for idx, (offset, family_name) in enumerate(offset_items):
            next_offset = (
                offset_items[idx + 1][0]
                if idx + 1 < len(offset_items)
                else int(self.out_dims.E)
            )
            mask = (edge_attr_ids >= offset) & (edge_attr_ids < next_offset)
            edge_family[mask] = int(edge_family2id.get(family_name, 0))

        rho = self._rho_for_partition()
        rel_balance_power = float(
            getattr(
                self.cfg.model,
                "hetero_metis_relation_balance_power",
                0.5,
            )
        )
        refine_degree_balance = bool(
            getattr(
                self.cfg.model,
                "hetero_metis_refine_degree_balance",
                False,
            )
        )
        refine_max_iter = int(
            getattr(self.cfg.model, "hetero_metis_refine_max_iter", 200) or 0
        )
        refine_preserve_high_high = bool(
            getattr(
                self.cfg.model,
                "hetero_metis_refine_preserve_high_high",
                False,
            )
        )
        refine_preserve_high_quantile = float(
            getattr(
                self.cfg.model,
                "hetero_metis_refine_preserve_high_quantile",
                0.8,
            )
            or 0.8
        )
        refine_preserve_penalty_weight = float(
            getattr(
                self.cfg.model,
                "hetero_metis_refine_preserve_penalty_weight",
                0.0,
            )
            or 0.0
        )
        all_blocks = []
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        for graph_idx in range(graph_count):
            nodes = torch.where(batch == graph_idx)[0]
            if nodes.numel() == 0:
                continue
            local_of = torch.full(
                (anchor_node_subtype.shape[0],),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            local_of[nodes] = torch.arange(
                nodes.numel(), dtype=torch.long, device=self.device
            )
            mask = (
                (batch[edge_index[0].long()] == graph_idx)
                & (batch[edge_index[1].long()] == graph_idx)
            )
            local_edges = local_of[edge_index[:, mask].long()]
            local_family = edge_family[mask]
            blocks = hetero_metis_blocks_from_graph(
                local_edges,
                local_family,
                int(nodes.numel()),
                rho,
                relation_balance_power=rel_balance_power,
                node_type_local=anchor_node_subtype[nodes],
                refine_degree_balance=refine_degree_balance,
                refine_max_iter=refine_max_iter,
                refine_preserve_high_high=refine_preserve_high_high,
                refine_preserve_high_quantile=refine_preserve_high_quantile,
                refine_preserve_penalty_weight=refine_preserve_penalty_weight,
            )
            if not blocks:
                blocks = [list(range(int(nodes.numel())))]
            for block in blocks:
                local_ids = torch.tensor(
                    block, dtype=torch.long, device=self.device
                )
                all_blocks.append(nodes[local_ids])
        return all_blocks

    def _partition_query_pool_into_rounds(
        self, query_pool, blocks, total_nodes
    ):
        """Assign every canonical query exactly once using its first endpoint."""
        edge_index, edge_batch, edge_family = query_pool
        block_of = torch.full(
            (int(total_nodes),), -1, dtype=torch.long, device=self.device
        )
        for block_id, nodes in enumerate(blocks):
            if nodes is not None and nodes.numel() > 0:
                block_of[nodes.long()] = int(block_id)
        missing = block_of < 0
        if missing.any():
            block_of[missing] = 0
        owner = block_of[edge_index[0].long()]
        rounds = []
        for block_id in range(max(1, len(blocks))):
            idx = (owner == block_id).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                rounds.append(
                    (
                        edge_index[:, idx],
                        edge_batch[idx],
                        edge_family[idx],
                    )
                )
        return rounds

    def _partition_context_edges(
        self,
        edge_index,
        edge_attr,
        blocks,
        total_nodes,
        target_edge_count=None,
        random_seed=0,
    ):
        """Select a fixed-size context without using endpoint IDs as priority."""
        block_of = torch.full(
            (int(total_nodes),), -1, dtype=torch.long, device=self.device
        )
        for block_id, nodes in enumerate(blocks):
            if nodes is not None and nodes.numel() > 0:
                block_of[nodes.long()] = int(block_id)
        missing = block_of < 0
        if missing.any():
            block_of[missing] = 0
        src = edge_index[0].long()
        dst = edge_index[1].long()
        within = block_of[src] == block_of[dst]
        within_idx = within.nonzero(as_tuple=True)[0]
        cross_idx = (~within).nonzero(as_tuple=True)[0]
        if target_edge_count is None:
            target_edge_count = int(within_idx.numel()) + int(
                round(
                    float(
                        getattr(
                            self.cfg.model,
                            "partition_context_cross_edge_ratio",
                            0.1,
                        )
                        or 0.0
                    )
                    * int(cross_idx.numel())
                )
            )
        target_edge_count = max(
            1, min(int(target_edge_count), int(edge_index.shape[1]))
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))

        take_intra = min(int(within_idx.numel()), target_edge_count)
        if take_intra > 0:
            intra_order = torch.randperm(
                within_idx.numel(), generator=generator, device="cpu"
            )[:take_intra].to(within_idx.device)
            selected_intra = within_idx[intra_order]
        else:
            selected_intra = within_idx[:0]
        remaining = target_edge_count - int(selected_intra.numel())
        take_cross = min(int(cross_idx.numel()), remaining)
        if take_cross > 0:
            cross_order = torch.randperm(
                cross_idx.numel(), generator=generator, device="cpu"
            )[:take_cross].to(cross_idx.device)
            selected_cross = cross_idx[cross_order]
        else:
            selected_cross = cross_idx[:0]
        chosen = torch.cat([selected_intra, selected_cross])
        if chosen.numel() < target_edge_count:
            selected_mask = torch.zeros(
                edge_index.shape[1], dtype=torch.bool, device=self.device
            )
            selected_mask[chosen] = True
            leftover = (~selected_mask).nonzero(as_tuple=True)[0]
            need = min(
                target_edge_count - int(chosen.numel()), int(leftover.numel())
            )
            if need > 0:
                extra_order = torch.randperm(
                    leftover.numel(), generator=generator, device="cpu"
                )[:need].to(leftover.device)
                chosen = torch.cat([chosen, leftover[extra_order]])
        chosen = chosen.sort().values
        stats = self._partition_context_stats(
            edge_index[:, chosen],
            edge_attr[chosen],
            block_of,
            int(total_nodes),
        )
        return edge_index[:, chosen], edge_attr[chosen], stats

    def _partition_intra_count(self, edge_index, blocks, total_nodes):
        block_of = torch.full(
            (int(total_nodes),), -1, dtype=torch.long, device=self.device
        )
        for block_id, nodes in enumerate(blocks):
            if nodes is not None and nodes.numel() > 0:
                block_of[nodes.long()] = int(block_id)
        block_of[block_of < 0] = 0
        return int(
            (
                block_of[edge_index[0].long()]
                == block_of[edge_index[1].long()]
            )
            .sum()
            .item()
        )

    def _shared_partition_context_budget(
        self, edge_index, metis_blocks, random_blocks, total_nodes
    ):
        configured = int(
            getattr(
                self.cfg.model, "partition_context_edge_budget", 0
            )
            or 0
        )
        total_edges = int(edge_index.shape[1])
        if configured > 0:
            return max(1, min(configured, total_edges))
        metis_intra = self._partition_intra_count(
            edge_index, metis_blocks, total_nodes
        )
        random_intra = self._partition_intra_count(
            edge_index, random_blocks, total_nodes
        )
        base = max(metis_intra, random_intra)
        ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.cfg.model,
                        "partition_context_cross_edge_ratio",
                        0.1,
                    )
                    or 0.0
                ),
            ),
        )
        return max(
            1,
            min(total_edges, base + int(round(ratio * (total_edges - base)))),
        )

    def _partition_context_stats(
        self, edge_index, edge_attr, block_of, total_nodes
    ):
        src, dst = edge_index[0].long(), edge_index[1].long()
        intra = block_of[src] == block_of[dst]
        degree = torch.zeros(
            int(total_nodes), dtype=torch.float32, device=edge_index.device
        )
        ones = torch.ones(src.numel(), device=edge_index.device)
        degree.scatter_add_(0, src, ones)
        degree.scatter_add_(0, dst, ones)
        mean_degree = float(degree.mean().detach().cpu())
        degree_cv = float(
            degree.std(unbiased=False).detach().cpu()
            / max(mean_degree, 1e-8)
        )
        labels = (
            edge_attr.argmax(dim=-1).long()
            if edge_attr.dim() > 1
            else edge_attr.long()
        )
        family_counts = {}
        for family_name, (lo, hi) in self._edge_family_label_ranges().items():
            family_counts[family_name] = int(
                ((labels >= int(lo)) & (labels < int(hi))).sum().item()
            )
        return {
            "total": int(src.numel()),
            "intra": int(intra.sum().item()),
            "inter": int((~intra).sum().item()),
            "degree_cv": degree_cv,
            "family_counts": family_counts,
        }

    def _build_type_balanced_random_blocks(
        self, anchor_node_subtype, batch, num_blocks
    ):
        """Random blocks with balanced counts for every parent node type."""
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        sorted_types = sorted(type_offsets.items(), key=lambda item: item[1])
        num_blocks = max(1, int(num_blocks))
        all_blocks = []
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        for graph_idx in range(graph_count):
            graph_blocks = [[] for _ in range(num_blocks)]
            for type_idx, (_, offset) in enumerate(sorted_types):
                next_offset = (
                    sorted_types[type_idx + 1][1]
                    if type_idx + 1 < len(sorted_types)
                    else self.out_dims.X
                )
                nodes = torch.where(
                    (batch == graph_idx)
                    & (anchor_node_subtype >= int(offset))
                    & (anchor_node_subtype < int(next_offset))
                )[0]
                if nodes.numel() == 0:
                    continue
                nodes = nodes[
                    torch.randperm(nodes.numel(), device=nodes.device)
                ]
                for block_id, chunk in enumerate(
                    torch.tensor_split(nodes, num_blocks)
                ):
                    if chunk.numel() > 0:
                        graph_blocks[block_id].append(chunk)
            for parts in graph_blocks:
                if parts:
                    all_blocks.append(torch.cat(parts).long())
        return all_blocks

    def _canonical_training_query_pool(
        self, edge_index, edge_batch, total_nodes
    ):
        if edge_index is None or edge_index.numel() == 0:
            return None
        u = torch.minimum(edge_index[0].long(), edge_index[1].long())
        v = torch.maximum(edge_index[0].long(), edge_index[1].long())
        valid = u != v
        u, v, edge_batch = u[valid], v[valid], edge_batch.long()[valid]
        keys = (
            edge_batch * int(total_nodes * total_nodes)
            + u * int(total_nodes)
            + v
        )
        order = torch.argsort(keys)
        keys = keys[order]
        keep = torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device)
        if keys.numel() > 1:
            keep[1:] = keys[1:] != keys[:-1]
        chosen = order[keep]
        return torch.stack([u[chosen], v[chosen]], dim=0), edge_batch[chosen]

    def _query_family_from_endpoints(self, edge_index, node_subtype):
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        edge_family2id = (
            getattr(self.dataset_info, "edge_family2id", {}) or {}
        )
        sorted_types = sorted(type_offsets.items(), key=lambda item: item[1])
        node_type = torch.full_like(node_subtype.long(), -1)
        type_name_to_id = {}
        for type_id, (type_name, offset) in enumerate(sorted_types):
            next_offset = (
                sorted_types[type_id + 1][1]
                if type_id + 1 < len(sorted_types)
                else self.out_dims.X
            )
            node_type[
                (node_subtype >= int(offset))
                & (node_subtype < int(next_offset))
            ] = int(type_id)
            type_name_to_id[str(type_name)] = int(type_id)
        pair_to_family = {}
        for family_name, endpoints in fam_endpoints.items():
            src_name = str(endpoints.get("src_type"))
            dst_name = str(endpoints.get("dst_type"))
            if (
                src_name in type_name_to_id
                and dst_name in type_name_to_id
                and family_name in edge_family2id
            ):
                pair_to_family[
                    (type_name_to_id[src_name], type_name_to_id[dst_name])
                ] = int(edge_family2id[family_name])
        result = torch.full(
            (edge_index.shape[1],),
            -1,
            dtype=torch.long,
            device=edge_index.device,
        )
        src_types = node_type[edge_index[0].long()]
        dst_types = node_type[edge_index[1].long()]
        for (src_type, dst_type), family_id in pair_to_family.items():
            mask = (src_types == src_type) & (dst_types == dst_type)
            if mask.any():
                result[mask] = int(family_id)
        return result

    def _node_type_ids_from_subtype(self, node_subtype):
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        sorted_types = sorted(type_offsets.items(), key=lambda item: item[1])
        node_type = torch.full_like(node_subtype.long(), -1)
        for type_id, (_type_name, offset) in enumerate(sorted_types):
            next_offset = (
                sorted_types[type_id + 1][1]
                if type_id + 1 < len(sorted_types)
                else self.out_dims.X
            )
            node_type[
                (node_subtype >= int(offset))
                & (node_subtype < int(next_offset))
            ] = int(type_id)
        return node_type

    def _same_type_edge_mask(self, edge_index, node_subtype):
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        edge_index = edge_index.long()
        node_type = self._node_type_ids_from_subtype(node_subtype.long()).to(edge_index.device)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]
        return (src_type >= 0) & (src_type == dst_type)

    def _filter_same_type_edges(self, edge_index, edge_attr, node_subtype):
        if edge_index is None or edge_index.numel() == 0:
            return edge_index, edge_attr
        mask = self._same_type_edge_mask(edge_index.long(), node_subtype.long())
        return edge_index[:, mask], edge_attr[mask]

    def _training_loss_from_encoded_queries(
        self,
        encoded,
        data,
        sparse_noisy_data,
        query_edge_index,
        query_edge_batch,
        node_subtype,
        step_idx,
        log_metrics=True,
        balance_loss_by_query_family=False,
        return_predictions=False,
    ):
        total_nodes = int(data.x.shape[0])
        current_labels = self._lookup_query_edge_labels(
            query_edge_index,
            sparse_noisy_data["edge_index_t"],
            sparse_noisy_data["edge_attr_t"].argmax(dim=-1).long(),
            total_nodes,
        )
        query_family = self._query_family_from_endpoints(
            query_edge_index, node_subtype
        )
        decode_labels = self._query_decode_labels_for_mode(
            current_labels,
            getattr(self.cfg.model, "train_queryfree_query_state", "current"),
        )
        logits = self._decode_query_logits(
            encoded,
            query_edge_index,
            decode_labels,
            data.batch,
            query_family,
        )
        logits = self._mask_edge_logits_by_query_family(logits, query_family)
        true_labels = self._lookup_query_edge_labels(
            query_edge_index,
            data.edge_index,
            data.edge_attr.argmax(dim=-1).long(),
            total_nodes,
        )
        pred = utils.SparsePlaceHolder(
            node=encoded["node_logits"],
            edge_attr=logits,
            edge_index=query_edge_index,
            y=encoded["y_out"],
            batch=data.batch,
            charge=encoded["charge_logits"],
        )
        true_data = utils.SparsePlaceHolder(
            node=data.x,
            charge=data.charge,
            edge_attr=true_labels,
            edge_index=query_edge_index,
            y=data.y,
            batch=data.batch,
        )
        if bool(balance_loss_by_query_family):
            loss = self._balanced_query_family_loss(
                pred=pred,
                true_data=true_data,
                query_family=query_family,
                log_metrics=bool(log_metrics),
                step_idx=int(step_idx),
            )
        else:
            loss = self.train_loss.forward(
                pred=pred,
                true_data=true_data,
                log=bool(log_metrics) and step_idx % self.log_every_steps == 0,
            )
        if bool(log_metrics):
            self._record_train_exist_diagnostics(
                pred, true_data, sparse_noisy_data
            )
        if not return_predictions:
            return loss
        return loss, {
            "logits": logits.detach(),
            "true_labels": true_labels.detach(),
            "current_labels": current_labels.detach(),
            "query_family": query_family.detach(),
        }

    def _balanced_query_family_loss(
        self,
        pred,
        true_data,
        query_family,
        log_metrics,
        step_idx,
    ):
        """Mean of per-family query losses with per-family clamped pos_weight."""
        if query_family is None or query_family.numel() == 0:
            return self.train_loss.forward(pred=pred, true_data=true_data, log=False)
        unique_families = torch.unique(query_family.long())
        losses = []
        old_pos_weight = getattr(self.train_loss, "exist_pos_weight", None)
        max_pos_weight = float(
            getattr(
                self.cfg.model,
                "same_type_internal_pos_weight_max",
                getattr(self.cfg.model, "exist_pos_weight", 20),
            )
            or 20
        )
        min_pos_weight = float(
            getattr(self.cfg.model, "same_type_internal_pos_weight_min", 1.0) or 1.0
        )
        max_pos_weight = max(min_pos_weight, max_pos_weight)
        family_stats = {}
        try:
            target = true_data.edge_attr
            if target.dim() > 1:
                target = target.argmax(dim=-1)
            true_exist = target.reshape(-1).long() > 0
            for fam_id_t in unique_families:
                fam_id = int(fam_id_t.item())
                fam_mask = query_family.long() == fam_id
                if not fam_mask.any():
                    continue
                fam_exist = true_exist[fam_mask]
                pos = int(fam_exist.sum().item())
                neg = int((~fam_exist).sum().item())
                if pos > 0:
                    pos_weight = float(neg) / float(max(pos, 1))
                    pos_weight = max(min_pos_weight, min(max_pos_weight, pos_weight))
                else:
                    pos_weight = min_pos_weight
                self.train_loss.exist_pos_weight = pos_weight
                losses.append(
                    self.train_loss.forward(
                        pred=pred,
                        true_data=true_data,
                        log=False,
                        query_mask=fam_mask,
                    )
                )
                family_stats[fam_id] = {
                    "queries": int(fam_mask.sum().item()),
                    "pos": pos,
                    "neg": neg,
                    "pos_weight": pos_weight,
                }
        finally:
            self.train_loss.exist_pos_weight = old_pos_weight
        if not losses:
            return pred.edge_attr.sum() * 0.0
        loss = torch.stack(losses).mean()
        if (
            log_metrics
            and step_idx % self.log_every_steps == 0
            and getattr(self, "local_rank", 0) == 0
        ):
            print(
                "[TRAIN-FAMILY-BALANCED] "
                + json.dumps(family_stats, ensure_ascii=False)
            )
            self.log(
                "train/family_balanced_loss",
                loss.detach(),
                on_step=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def _predict_clean_logits_for_query_rounds(
        self,
        data,
        query_rounds,
        node_onehot,
        batch,
        ptr,
        base_edge_index,
        base_edge_attr_onehot,
        t_float,
        context_blocks=None,
        context_target_edge_count=None,
        context_random_seed=0,
    ):
        """Run one frozen-G_t view and return clean logits aligned by edge key."""
        total_nodes = int(node_onehot.shape[0])
        if context_blocks is None:
            context_edge_index = base_edge_index
            context_edge_attr = base_edge_attr_onehot
        else:
            context_edge_index, context_edge_attr, context_stats = (
                self._partition_context_edges(
                    base_edge_index,
                    base_edge_attr_onehot,
                    context_blocks,
                    total_nodes,
                    target_edge_count=context_target_edge_count,
                    random_seed=context_random_seed,
                )
            )
        if context_blocks is None:
            context_stats = {
                "total": int(context_edge_index.shape[1]),
                "intra": 0,
                "inter": int(context_edge_index.shape[1]),
                "degree_cv": float("nan"),
                "family_counts": {},
            }
        encoded = self._encode_context_only(
            data=data,
            node_onehot=node_onehot,
            context_edge_index=context_edge_index,
            context_edge_attr=context_edge_attr,
            batch=batch,
            ptr=ptr,
            t_float=t_float,
        )
        parts = []
        for query_edge_index, query_edge_batch, query_edge_family in query_rounds:
            current_labels = self._lookup_query_edge_labels(
                query_edge_index,
                base_edge_index,
                base_edge_attr_onehot.argmax(dim=-1).long(),
                total_nodes,
            )
            decode_labels = self._query_decode_labels_for_mode(
                current_labels,
                getattr(
                    self.cfg.model,
                    "sampling_queryfree_query_state",
                    "current",
                ),
            )
            selected_edges = query_edge_index
            selected_families = query_edge_family
            selected_logits = self._decode_query_logits(
                encoded,
                query_edge_index,
                decode_labels,
                batch,
                query_edge_family,
                edge_input_residual_scale=getattr(
                    self.cfg.model,
                    "sampling_edge_input_residual_scale",
                    None,
                ),
            )
            if selected_logits.numel() == 0:
                continue
            u = torch.minimum(selected_edges[0], selected_edges[1])
            v = torch.maximum(selected_edges[0], selected_edges[1])
            num_families = max(
                1,
                len(getattr(self.dataset_info, "edge_family2id", {}) or {}),
            )
            parts.append(
                {
                    "keys": (
                        (u * total_nodes + v) * int(num_families)
                        + selected_families.clamp_min(0)
                    ),
                    "edge_index": torch.stack([u, v], dim=0),
                    "batch": batch[selected_edges[0].long()],
                    "family": selected_families,
                    "logits": selected_logits,
                }
            )
        if not parts:
            return None
        keys = torch.cat([part["keys"] for part in parts])
        order = torch.argsort(keys)
        return {
            "keys": keys[order],
            "edge_index": torch.cat(
                [part["edge_index"] for part in parts], dim=1
            )[:, order],
            "batch": torch.cat([part["batch"] for part in parts])[order],
            "family": torch.cat([part["family"] for part in parts])[order],
            "logits": torch.cat([part["logits"] for part in parts], dim=0)[
                order
            ],
            "context_stats": context_stats,
        }

    def _log_partition_ensemble_diagnostics(
        self, metis_view, random_view, fused_logits, t_float, s_float
    ):
        metis_scores = (
            torch.logsumexp(metis_view["logits"][:, 1:], dim=-1)
            - metis_view["logits"][:, 0]
        )
        random_scores = (
            torch.logsumexp(random_view["logits"][:, 1:], dim=-1)
            - random_view["logits"][:, 0]
        )
        fused_scores = (
            torch.logsumexp(fused_logits[:, 1:], dim=-1) - fused_logits[:, 0]
        )
        max_n = max(
            1000,
            int(
                getattr(
                    self.cfg.model,
                    "sampling_partition_ensemble_diag_max_candidates",
                    100000,
                )
                or 100000
            ),
        )
        n = int(metis_scores.numel())
        if n > max_n:
            sample_idx = torch.linspace(
                0, n - 1, steps=max_n, device=metis_scores.device
            ).long()
        else:
            sample_idx = torch.arange(n, device=metis_scores.device)
        m_np = metis_scores[sample_idx].detach().float().cpu().numpy()
        r_np = random_scores[sample_idx].detach().float().cpu().numpy()
        m_rank = self._rankdata_average(m_np)
        r_rank = self._rankdata_average(r_np)
        spearman = (
            float(np.corrcoef(m_rank, r_rank)[0, 1])
            if np.std(m_rank) > 0 and np.std(r_rank) > 0
            else float("nan")
        )

        avg_counts = (
            getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        )
        edge_family2id = (
            getattr(self.dataset_info, "edge_family2id", {}) or {}
        )
        jaccard_mr_num = jaccard_mr_den = 0
        jaccard_fm_num = jaccard_fm_den = 0
        jaccard_fr_num = jaccard_fr_den = 0
        for family_name, family_id in edge_family2id.items():
            mask = metis_view["family"] == int(family_id)
            idx = mask.nonzero(as_tuple=True)[0]
            target = min(
                int(idx.numel()),
                max(0, int(round(float(avg_counts.get(family_name, 0.0))))),
            )
            if target <= 0:
                continue
            m_top = set(
                idx[
                    torch.topk(metis_scores[idx], k=target).indices
                ].detach().cpu().tolist()
            )
            r_top = set(
                idx[
                    torch.topk(random_scores[idx], k=target).indices
                ].detach().cpu().tolist()
            )
            f_top = set(
                idx[
                    torch.topk(fused_scores[idx], k=target).indices
                ].detach().cpu().tolist()
            )
            jaccard_mr_num += len(m_top & r_top)
            jaccard_mr_den += len(m_top | r_top)
            jaccard_fm_num += len(f_top & m_top)
            jaccard_fm_den += len(f_top | m_top)
            jaccard_fr_num += len(f_top & r_top)
            jaccard_fr_den += len(f_top | r_top)
        result = {
            "t": float(t_float.mean().detach().cpu()),
            "s": float(s_float.mean().detach().cpu()),
            "candidates_metis": int(metis_view["keys"].numel()),
            "candidates_random": int(random_view["keys"].numel()),
            "context_metis": metis_view.get("context_stats", {}),
            "context_random": random_view.get("context_stats", {}),
            "missing_between_views": int(
                (metis_view["keys"] != random_view["keys"]).sum().item()
            ),
            "clean_exist_spearman": spearman,
            "topk_jaccard_metis_random": float(
                jaccard_mr_num / max(1, jaccard_mr_den)
            ),
            "topk_jaccard_fused_metis": float(
                jaccard_fm_num / max(1, jaccard_fm_den)
            ),
            "topk_jaccard_fused_random": float(
                jaccard_fr_num / max(1, jaccard_fr_den)
            ),
        }
        print("[采样-PARTITION-ENSEMBLE] " + json.dumps(result, ensure_ascii=False))

    def _sample_partition_ensemble_exact_k(
        self,
        data,
        query_rounds,
        random_blocks,
        node_onehot,
        anchor_node_subtype,
        batch,
        ptr,
        edge_index,
        edge_attr_ids,
        edge_attr_onehot,
        s_float,
        t_float,
        is_final_sampling_step,
        selection_mode,
    ):
        """Fuse two clean-logit partition views before one reverse posterior."""
        mode = str(
            getattr(
                self.cfg.model,
                "sampling_partition_ensemble",
                "off",
            )
            or "off"
        ).lower()
        total_nodes = int(node_onehot.shape[0])
        query_pool = self._canonicalize_partition_ensemble_query_pool(
            query_rounds, total_nodes
        )
        if query_pool is None:
            return None
        metis_blocks = self._current_gt_hetero_metis_blocks(
            edge_index,
            edge_attr_ids,
            anchor_node_subtype,
            batch,
        )
        if not random_blocks:
            random_blocks = self._build_type_template_pseudo_blocks(
                anchor_node_subtype,
                batch,
                getattr(self.dataset_info, "type_offsets", {}),
            )
        metis_rounds = self._partition_query_pool_into_rounds(
            query_pool, metis_blocks, total_nodes
        )
        random_rounds = self._partition_query_pool_into_rounds(
            query_pool, random_blocks, total_nodes
        )
        context_budget = self._shared_partition_context_budget(
            edge_index, metis_blocks, random_blocks, total_nodes
        )
        sampling_step = int(
            round(
                float(t_float.mean().detach().cpu().item())
                * float(getattr(self.cfg.model, "diffusion_steps", 100))
            )
        )
        seed_base = self._current_sampling_seed()
        metis_view = None
        random_view = None
        if mode in ("metis_only", "mean"):
            metis_view = self._predict_clean_logits_for_query_rounds(
                data,
                metis_rounds,
                node_onehot,
                batch,
                ptr,
                edge_index,
                edge_attr_onehot,
                t_float,
                context_blocks=metis_blocks,
                context_target_edge_count=context_budget,
                context_random_seed=seed_base + 3000017 * sampling_step + 17,
            )
        if mode in ("random_only", "mean"):
            random_view = self._predict_clean_logits_for_query_rounds(
                data,
                random_rounds,
                node_onehot,
                batch,
                ptr,
                edge_index,
                edge_attr_onehot,
                t_float,
                context_blocks=random_blocks,
                context_target_edge_count=context_budget,
                context_random_seed=seed_base + 3000017 * sampling_step + 31,
            )
        if mode == "metis_only":
            fused_view = metis_view
        elif mode == "random_only":
            fused_view = random_view
        elif mode == "mean":
            if metis_view is None or random_view is None:
                raise RuntimeError("Both partition views are required for mean fusion")
            if (
                metis_view["keys"].shape != random_view["keys"].shape
                or not torch.equal(metis_view["keys"], random_view["keys"])
                or not torch.equal(
                    metis_view["family"], random_view["family"]
                )
            ):
                raise RuntimeError(
                    "Partition views did not cover the identical canonical query pool"
                )
            weight = min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            self.cfg.model,
                            "sampling_partition_ensemble_metis_weight",
                            0.5,
                        )
                    ),
                ),
            )
            fused_logits = (
                weight * metis_view["logits"]
                + (1.0 - weight) * random_view["logits"]
            )
            fused_view = dict(metis_view)
            fused_view["logits"] = fused_logits
            if getattr(self, "local_rank", 0) == 0:
                self._log_partition_ensemble_diagnostics(
                    metis_view,
                    random_view,
                    fused_logits,
                    t_float,
                    s_float,
                )
        else:
            raise ValueError(f"Unsupported partition ensemble mode: {mode}")
        if fused_view is None:
            return None

        clean_logits = self._calibrate_edge_logits_for_exist_pos_weight(
            fused_view["logits"], fused_view["family"]
        )
        current_labels = self._lookup_query_edge_labels(
            fused_view["edge_index"],
            edge_index,
            edge_attr_ids,
            total_nodes,
        )
        posterior_logits = self._edge_reverse_posterior_logits(
            clean_logits,
            fused_view["family"],
            fused_view["batch"],
            current_labels,
            s_float,
            t_float,
        )
        candidate_parts = []
        self._update_global_exact_k_candidates(
            candidate_parts,
            posterior_logits,
            fused_view["family"],
            fused_view["edge_index"],
            fused_view["batch"],
            total_nodes,
        )
        selected_edges, selected_labels = self._finalize_global_exact_k_candidates(
            data,
            candidate_parts,
            selection_mode,
            sampling_step,
            apply_connectivity_repair=(
                bool(is_final_sampling_step)
                and bool(
                    getattr(
                        self.cfg.model,
                        "sampling_exact_k_connectivity_repair",
                        False,
                    )
                )
            ),
        )
        if selected_edges is None:
            selected_edges = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            selected_labels = torch.empty(
                (0,), dtype=torch.long, device=self.device
            )
        out = utils.SparsePlaceHolder(
            node=node_onehot,
            edge_index=selected_edges,
            edge_attr=F.one_hot(
                selected_labels.clamp(0, self.out_dims.E - 1),
                num_classes=self.out_dims.E,
            ).float(),
            y=data.y,
            batch=batch,
            charge=getattr(data, "charge", None),
            ptr=ptr,
        )
        out.anchor_node_subtype = anchor_node_subtype
        out.pseudo_blocks = random_blocks
        for attr_name in (
            "connectivity_init_edge_index",
            "connectivity_init_edge_labels",
            "connectivity_init_family",
            "connectivity_init_batch",
            "connectivity_block_of",
            "connectivity_family_targets",
            "edge_score_diag_reference_edge_index",
            "forward_equivariance_diag_node_type",
            "equivariance_reference_new_to_old",
        ):
            if hasattr(data, attr_name):
                setattr(out, attr_name, getattr(data, attr_name))
        return out

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
        refine_preserve_high_high = bool(
            getattr(self.cfg.model, "hetero_metis_refine_preserve_high_high", False)
        )
        refine_preserve_high_quantile = float(
            getattr(self.cfg.model, "hetero_metis_refine_preserve_high_quantile", 0.8) or 0.8
        )
        refine_preserve_penalty_weight = float(
            getattr(self.cfg.model, "hetero_metis_refine_preserve_penalty_weight", 0.0) or 0.0
        )
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
                bool(refine_preserve_high_high),
                round(float(refine_preserve_high_quantile), 8),
                round(float(refine_preserve_penalty_weight), 8),
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
                    refine_preserve_high_high=refine_preserve_high_high,
                    refine_preserve_high_quantile=refine_preserve_high_quantile,
                    refine_preserve_penalty_weight=refine_preserve_penalty_weight,
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
            metadata_key = ("hetero_metis_metadata",) + cache_key
            block_metadata = self._true_block_cache.get(metadata_key)
            if block_metadata is None:
                all_block_nodes = [
                    batch_nodes[
                        torch.tensor(b, dtype=torch.long, device=self.device)
                    ]
                    for b in blocks
                ]
                node_block_id = torch.full(
                    (int(data.x.shape[0]),),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                for cached_block_id, cached_nodes in enumerate(all_block_nodes):
                    node_block_id[cached_nodes] = int(cached_block_id)
                type_node_pools = {}
                block_type_node_pools = {}
                batch_mask = data.batch == graph_idx
                for type_name, type_offset in sorted_types:
                    type_size = int(type_sizes.get(type_name, 0))
                    type_mask = (
                        (node_t >= int(type_offset))
                        & (node_t < int(type_offset) + type_size)
                        & batch_mask
                    )
                    type_nodes = torch.where(type_mask)[0]
                    type_node_pools[str(type_name)] = type_nodes
                    for cached_block_id in range(len(all_block_nodes)):
                        block_type_node_pools[
                            (int(cached_block_id), str(type_name))
                        ] = type_nodes[
                            node_block_id[type_nodes] == int(cached_block_id)
                        ]
                block_metadata = {
                    "all_block_nodes": all_block_nodes,
                    "node_block_id": node_block_id,
                    "type_node_pools": type_node_pools,
                    "block_type_node_pools": block_type_node_pools,
                }
                self._true_block_cache[metadata_key] = block_metadata
            else:
                all_block_nodes = block_metadata["all_block_nodes"]
            block_nodes = all_block_nodes[block_id]
            train_inter_state = {}
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

                batch_src_nodes = block_metadata["type_node_pools"].get(
                    str(src_type),
                    torch.empty(0, dtype=torch.long, device=self.device),
                )
                batch_dst_nodes = block_metadata["type_node_pools"].get(
                    str(dst_type),
                    torch.empty(0, dtype=torch.long, device=self.device),
                )
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

                block_src_nodes = block_metadata[
                    "block_type_node_pools"
                ].get(
                    (int(block_id), str(src_type)),
                    torch.empty(0, dtype=torch.long, device=self.device),
                )
                block_dst_nodes = block_metadata[
                    "block_type_node_pools"
                ].get(
                    (int(block_id), str(dst_type)),
                    torch.empty(0, dtype=torch.long, device=self.device),
                )
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

    def _auxiliary_time_factor(self, sparse_noisy_data, power: float) -> float:
        """Return a stable scalar high-noise emphasis factor from t/T."""
        try:
            power = float(power or 0.0)
        except Exception:
            power = 0.0
        if power <= 0.0:
            return 1.0
        t_float = None
        if isinstance(sparse_noisy_data, dict):
            t_float = sparse_noisy_data.get("t_float", None)
        if t_float is None:
            return 1.0
        try:
            t_val = float(torch.as_tensor(t_float).detach().float().mean().cpu())
        except Exception:
            return 1.0
        t_val = max(0.0, min(1.0, t_val))
        return float(t_val ** power)

    def _warmup_weight(self, base_weight: float, warmup_epochs: int, t_factor: float = 1.0) -> float:
        base_weight = float(base_weight or 0.0)
        if base_weight <= 0.0:
            return 0.0
        warmup_epochs = int(warmup_epochs or 0)
        if warmup_epochs > 0:
            warm = min(float(self.current_epoch + 1) / float(warmup_epochs), 1.0)
        else:
            warm = 1.0
        return base_weight * warm * float(t_factor)

    def _query_family_and_target_exist(self, pred, true_data, clean_data):
        pred_edge = pred.edge_attr
        edge_index = pred.edge_index
        target = true_data.edge_attr
        if (
            pred_edge.numel() == 0
            or edge_index.numel() == 0
            or target.numel() == 0
            or not self.heterogeneous
        ):
            return None, None
        _node_type_ids, query_family = self._infer_query_family_from_clean_node_types(
            edge_index.long(),
            clean_data.x,
        )
        if query_family is None or query_family.numel() != pred_edge.shape[0]:
            return None, None
        if target.dim() > 1:
            target = target.argmax(dim=-1)
        target_exist = (target.reshape(-1).long().to(pred_edge.device) > 0).to(
            dtype=pred_edge.dtype
        )
        return query_family.to(pred_edge.device).long().reshape(-1), target_exist

    def _global_aux_stats_from_block(self, pred, true_data, clean_data):
        query_family, target_exist = self._query_family_and_target_exist(
            pred, true_data, clean_data
        )
        if query_family is None:
            return None
        pred_edge = pred.edge_attr
        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        pred_exist = (1.0 - torch.softmax(logits, dim=-1)[:, 0]).clamp(
            min=0.0, max=1.0
        )
        stats = {
            "family": {},
            "degree": {},
            "num_pairs": 0,
            "device": pred_edge.device,
            "dtype": pred_edge.dtype,
        }
        for fam in torch.unique(query_family):
            fam_id = int(fam.detach().cpu().item())
            mask = query_family == fam_id
            stats["family"][fam_id] = {
                "n": float(mask.sum().item()),
                "pred": float(pred_exist[mask].sum().detach().cpu()),
                "target": float(target_exist[mask].sum().detach().cpu()),
            }

        pair_info = self._degree_pair_ids_for_candidates(pred.edge_index.long(), query_family)
        if pair_info is None:
            return stats
        pair_id = pair_info["pair_id"].to(pred_edge.device).long()
        valid_pair = pair_info["valid"].to(pred_edge.device)
        num_bins = int(pair_info["num_bins"])
        num_pairs = int(num_bins * num_bins)
        stats["num_pairs"] = num_pairs
        for fam in torch.unique(query_family):
            fam_id = int(fam.detach().cpu().item())
            mask = (query_family == fam_id) & valid_pair
            if not mask.any():
                continue
            fam_pair_ids = pair_id[mask].clamp(min=0, max=num_pairs - 1)
            pred_mass = pred_edge.new_zeros(num_pairs)
            target_mass = pred_edge.new_zeros(num_pairs)
            pred_mass.scatter_add_(0, fam_pair_ids, pred_exist[mask])
            target_mass.scatter_add_(0, fam_pair_ids, target_exist[mask])
            stats["degree"][fam_id] = {
                "pred": pred_mass.detach().cpu(),
                "target": target_mass.detach().cpu(),
            }
        return stats

    def _merge_global_aux_stats(self, base, add):
        if add is None:
            return base
        if base is None:
            return add
        for fam_id, vals in add.get("family", {}).items():
            cur = base["family"].setdefault(fam_id, {"n": 0.0, "pred": 0.0, "target": 0.0})
            cur["n"] += float(vals.get("n", 0.0))
            cur["pred"] += float(vals.get("pred", 0.0))
            cur["target"] += float(vals.get("target", 0.0))
        base["num_pairs"] = max(int(base.get("num_pairs", 0)), int(add.get("num_pairs", 0)))
        for fam_id, vals in add.get("degree", {}).items():
            if fam_id not in base["degree"]:
                base["degree"][fam_id] = {
                    "pred": vals["pred"].clone(),
                    "target": vals["target"].clone(),
                }
            else:
                base["degree"][fam_id]["pred"] = base["degree"][fam_id]["pred"] + vals["pred"]
                base["degree"][fam_id]["target"] = base["degree"][fam_id]["target"] + vals["target"]
        return base

    def _prepare_streaming_global_aux(self, stats, sparse_noisy_data):
        if not stats:
            return None
        device = stats.get("device", self.device)
        dtype = stats.get("dtype", torch.float32)
        prepared = {"family_coef": {}, "degree_grad": {}, "stats": stats}

        family_w = self._warmup_weight(
            float(getattr(self.cfg.model, "family_count_loss_weight", 0.0) or 0.0),
            int(getattr(self.cfg.model, "family_count_loss_warmup_epochs", 0) or 0),
            self._auxiliary_time_factor(
                sparse_noisy_data,
                float(getattr(self.cfg.model, "family_count_loss_t_power", 0.0) or 0.0),
            ),
        )
        loss_type = str(getattr(self.cfg.model, "family_count_loss_type", "l1") or "l1").lower()
        min_edges = int(getattr(self.cfg.model, "family_count_loss_min_query_edges", 32) or 0)
        active = [
            (fam_id, vals)
            for fam_id, vals in stats.get("family", {}).items()
            if float(vals.get("n", 0.0)) >= min_edges
        ]
        if family_w > 0 and active:
            denom_fams = float(len(active))
            for fam_id, vals in active:
                n = max(float(vals["n"]), 1.0)
                diff = float(vals["pred"] / n - vals["target"] / n)
                if loss_type == "mse":
                    coef = 2.0 * diff / n
                else:
                    coef = (1.0 if diff >= 0.0 else -1.0) / n
                prepared["family_coef"][fam_id] = family_w * coef / denom_fams

        degree_w = self._warmup_weight(
            float(getattr(self.cfg.model, "degree_pair_dist_loss_weight", 0.0) or 0.0),
            int(getattr(self.cfg.model, "degree_pair_dist_loss_warmup_epochs", 0) or 0),
            self._auxiliary_time_factor(
                sparse_noisy_data,
                float(getattr(self.cfg.model, "degree_pair_dist_loss_t_power", 0.0) or 0.0),
            ),
        )
        if degree_w <= 0:
            return prepared
        eps = 1e-8
        loss_type = str(getattr(self.cfg.model, "degree_pair_dist_loss_type", "js") or "js").lower()
        min_edges = int(getattr(self.cfg.model, "degree_pair_dist_min_query_edges", 32) or 0)
        active_degree = []
        for fam_id, vals in stats.get("degree", {}).items():
            target_mass = vals["target"].float()
            pred_mass = vals["pred"].float()
            if float((target_mass.sum() + pred_mass.sum()).item()) <= 0:
                continue
            if float(stats.get("family", {}).get(fam_id, {}).get("n", 0.0)) < min_edges:
                continue
            active_degree.append((fam_id, pred_mass, target_mass))
        if not active_degree:
            return prepared
        denom_fams = float(len(active_degree))
        for fam_id, pred_mass_cpu, target_mass_cpu in active_degree:
            pred_mass = pred_mass_cpu.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
            target_mass = target_mass_cpu.to(device=device, dtype=dtype)
            if target_mass.sum() <= eps or pred_mass.sum() <= eps:
                continue
            pred_dist = pred_mass / pred_mass.sum().clamp(min=eps)
            target_dist = target_mass / target_mass.sum().clamp(min=eps)
            if loss_type == "l1":
                small_loss = (pred_dist - target_dist).abs().sum()
            elif loss_type == "mse":
                small_loss = (pred_dist - target_dist).pow(2).sum()
            elif loss_type == "kl":
                small_loss = (
                    target_dist * ((target_dist + eps).log() - (pred_dist + eps).log())
                ).sum()
            else:
                mix = 0.5 * (pred_dist + target_dist)
                small_loss = 0.5 * (
                    (target_dist * ((target_dist + eps).log() - (mix + eps).log())).sum()
                    + (pred_dist * ((pred_dist + eps).log() - (mix + eps).log())).sum()
                )
            grad = torch.autograd.grad(small_loss, pred_mass, retain_graph=False)[0]
            prepared["degree_grad"][fam_id] = (degree_w / denom_fams) * grad.detach()
        return prepared

    def _streaming_global_aux_loss_for_block(self, pred, true_data, clean_data, prepared):
        if not prepared:
            return pred.edge_attr.sum() * 0.0
        query_family, _target_exist = self._query_family_and_target_exist(pred, true_data, clean_data)
        if query_family is None:
            return pred.edge_attr.sum() * 0.0
        pred_edge = pred.edge_attr
        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        pred_exist = (1.0 - torch.softmax(logits, dim=-1)[:, 0]).clamp(min=0.0, max=1.0)
        loss = pred_edge.sum() * 0.0
        for fam_id, coef in prepared.get("family_coef", {}).items():
            mask = query_family == int(fam_id)
            if mask.any():
                loss = loss + float(coef) * pred_exist[mask].sum()
        if prepared.get("degree_grad"):
            pair_info = self._degree_pair_ids_for_candidates(pred.edge_index.long(), query_family)
            if pair_info is not None:
                pair_id = pair_info["pair_id"].to(pred_edge.device).long()
                valid_pair = pair_info["valid"].to(pred_edge.device)
                num_pairs = int(pair_info["num_bins"]) ** 2
                for fam_id, grad in prepared.get("degree_grad", {}).items():
                    mask = (query_family == int(fam_id)) & valid_pair
                    if not mask.any():
                        continue
                    fam_pair_ids = pair_id[mask].clamp(min=0, max=num_pairs - 1)
                    grad_dev = grad.to(device=pred_edge.device, dtype=pred_edge.dtype)
                    loss = loss + (grad_dev[fam_pair_ids] * pred_exist[mask]).sum()
        return loss

    @staticmethod
    def _js_divergence_from_masses(pred_mass, target_mass, eps=1e-8):
        pred_sum = pred_mass.sum().clamp(min=eps)
        target_sum = target_mass.sum().clamp(min=eps)
        pred_dist = pred_mass / pred_sum
        target_dist = target_mass / target_sum
        mix = 0.5 * (pred_dist + target_dist)
        return 0.5 * (
            (pred_dist * ((pred_dist + eps).log() - (mix + eps).log())).sum()
            + (target_dist * ((target_dist + eps).log() - (mix + eps).log())).sum()
        )

    def _compute_query_topk_structure_probe(self, pred, true_data, clean_data, num_nodes):
        pred_edge = pred.edge_attr
        edge_index = pred.edge_index
        min_query = int(
            getattr(self.cfg.model, "train_structure_probe_min_query_edges", 256) or 0
        )
        if (
            pred_edge.numel() == 0
            or edge_index.numel() == 0
            or pred_edge.shape[0] < min_query
            or not self.heterogeneous
        ):
            return None
        query_family, target_exist = self._query_family_and_target_exist(
            pred, true_data, clean_data
        )
        if query_family is None:
            return None
        k = int(target_exist.sum().detach().item())
        n = int(target_exist.numel())
        if k <= 0 or n <= 0:
            return None
        k = min(k, n)
        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        exist_scores = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        selected = torch.zeros(n, dtype=torch.bool, device=pred_edge.device)
        top_idx = torch.topk(exist_scores, k=k, largest=True).indices
        selected[top_idx] = True

        target_bool = target_exist > 0.5
        true_in_topk = target_exist[selected].sum()
        precision = true_in_topk / float(max(k, 1))
        recall = true_in_topk / target_exist.sum().clamp(min=1.0)

        fam_ids = torch.unique(query_family)
        family_sel = pred_edge.new_zeros(int(fam_ids.numel()))
        family_target = pred_edge.new_zeros(int(fam_ids.numel()))
        for pos, fam in enumerate(fam_ids):
            mask = query_family == fam
            family_sel[pos] = selected[mask].to(dtype=pred_edge.dtype).sum()
            family_target[pos] = target_exist[mask].sum()
        family_l1 = (
            family_sel / family_sel.sum().clamp(min=1.0)
            - family_target / family_target.sum().clamp(min=1.0)
        ).abs().sum() * 0.5

        degree_pair_js = None
        pair_info = self._degree_pair_ids_for_candidates(edge_index.long(), query_family)
        if pair_info is not None:
            pair_id = pair_info["pair_id"].to(pred_edge.device).long()
            valid_pair = pair_info["valid"].to(pred_edge.device)
            num_pairs = int(pair_info["num_bins"]) ** 2
            fam_count = int(query_family.max().detach().item()) + 1 if query_family.numel() else 0
            total_bins = max(1, fam_count * num_pairs)
            flat_ids = (query_family.clamp(min=0) * num_pairs + pair_id.clamp(min=0, max=num_pairs - 1)).long()
            valid = valid_pair & (query_family >= 0)
            if valid.any():
                sel_mass = pred_edge.new_zeros(total_bins)
                tgt_mass = pred_edge.new_zeros(total_bins)
                sel_mass.scatter_add_(
                    0,
                    flat_ids[valid].clamp(min=0, max=total_bins - 1),
                    selected[valid].to(dtype=pred_edge.dtype),
                )
                tgt_mass.scatter_add_(
                    0,
                    flat_ids[valid].clamp(min=0, max=total_bins - 1),
                    target_exist[valid],
                )
                if sel_mass.sum() > 0 and tgt_mass.sum() > 0:
                    degree_pair_js = self._js_divergence_from_masses(sel_mass, tgt_mass)

        closure_mass_rel = None
        closure_mean_gap = None
        common = self._clean_common_neighbor_scores(
            edge_index=edge_index.long(),
            full_data=clean_data,
            num_nodes=int(num_nodes),
            device=pred_edge.device,
            dtype=pred_edge.dtype,
        )
        if common is not None and common.numel() == n:
            selected_mass = common[selected].sum()
            target_mass = (common * target_exist).sum()
            closure_mass_rel = (
                (selected_mass - target_mass).abs()
                / target_mass.abs().clamp(min=1.0)
            )
            selected_mean = common[selected].mean() if selected.any() else common.sum() * 0.0
            target_mean = common[target_bool].mean() if target_bool.any() else common.sum() * 0.0
            closure_mean_gap = selected_mean - target_mean

        return {
            "topk_precision": float(precision.detach().cpu()),
            "topk_recall": float(recall.detach().cpu()),
            "family_l1": float(family_l1.detach().cpu()),
            "degree_pair_js": (
                float(degree_pair_js.detach().cpu())
                if degree_pair_js is not None
                else None
            ),
            "closure_mass_rel": (
                float(closure_mass_rel.detach().cpu())
                if closure_mass_rel is not None
                else None
            ),
            "closure_mean_gap": (
                float(closure_mean_gap.detach().cpu())
                if closure_mean_gap is not None
                else None
            ),
            "k": float(k),
        }

    def _record_query_topk_structure_probe(self, pred, true_data, clean_data):
        every = int(getattr(self.cfg.model, "train_structure_probe_every_epochs", 0) or 0)
        if every <= 0 or int(self.current_epoch) % every != 0:
            return
        with torch.no_grad():
            stats = self._compute_query_topk_structure_probe(
                pred,
                true_data,
                clean_data,
                num_nodes=clean_data.x.shape[0],
            )
        if not stats:
            return
        if not hasattr(self, "_epoch_struct_probe_count"):
            self._epoch_struct_probe_count = 0
            self._epoch_struct_probe_topk_precision_sum = 0.0
            self._epoch_struct_probe_topk_recall_sum = 0.0
            self._epoch_struct_probe_family_l1_sum = 0.0
            self._epoch_struct_probe_degree_pair_js_sum = 0.0
            self._epoch_struct_probe_degree_pair_js_count = 0
            self._epoch_struct_probe_closure_mass_rel_sum = 0.0
            self._epoch_struct_probe_closure_mass_rel_count = 0
            self._epoch_struct_probe_closure_mean_gap_sum = 0.0
            self._epoch_struct_probe_closure_mean_gap_count = 0
            self._epoch_struct_probe_k_sum = 0.0
        self._epoch_struct_probe_count += 1
        self._epoch_struct_probe_topk_precision_sum += float(stats["topk_precision"])
        self._epoch_struct_probe_topk_recall_sum += float(stats["topk_recall"])
        self._epoch_struct_probe_family_l1_sum += float(stats["family_l1"])
        self._epoch_struct_probe_k_sum += float(stats["k"])
        if stats.get("degree_pair_js") is not None:
            self._epoch_struct_probe_degree_pair_js_sum += float(stats["degree_pair_js"])
            self._epoch_struct_probe_degree_pair_js_count += 1
        if stats.get("closure_mass_rel") is not None:
            self._epoch_struct_probe_closure_mass_rel_sum += float(stats["closure_mass_rel"])
            self._epoch_struct_probe_closure_mass_rel_count += 1
        if stats.get("closure_mean_gap") is not None:
            self._epoch_struct_probe_closure_mean_gap_sum += float(stats["closure_mean_gap"])
            self._epoch_struct_probe_closure_mean_gap_count += 1

    def _compute_family_count_loss(self, pred, true_data, clean_data):
        """Match family-wise expected edge density on the current query set."""
        pred_edge = pred.edge_attr
        edge_index = pred.edge_index
        target = true_data.edge_attr
        if (
            pred_edge.numel() == 0
            or edge_index.numel() == 0
            or target.numel() == 0
            or not self.heterogeneous
        ):
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}
        _node_type_ids, query_family = self._infer_query_family_from_clean_node_types(
            edge_index.long(),
            clean_data.x,
        )
        if query_family is None or query_family.numel() != pred_edge.shape[0]:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}
        if target.dim() > 1:
            target = target.argmax(dim=-1)
        query_family = query_family.to(pred_edge.device).long().reshape(-1)
        target_exist = (target.reshape(-1).long().to(pred_edge.device) > 0).to(
            dtype=pred_edge.dtype
        )
        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        prob = torch.softmax(logits, dim=-1)
        pred_exist = (1.0 - prob[:, 0]).clamp(min=0.0, max=1.0)
        min_edges = int(getattr(self.cfg.model, "family_count_loss_min_query_edges", 32) or 0)
        loss_type = str(getattr(self.cfg.model, "family_count_loss_type", "l1") or "l1").lower()
        losses = []
        for fam in torch.unique(query_family):
            mask = query_family == fam
            n = int(mask.sum().item())
            if n < min_edges:
                continue
            pred_density = pred_exist[mask].mean()
            target_density = target_exist[mask].mean()
            diff = pred_density - target_density.detach()
            losses.append(diff.pow(2) if loss_type == "mse" else diff.abs())
        if not losses:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}
        return torch.stack(losses).mean(), {"active_families": float(len(losses))}

    def _compute_degree_pair_distribution_loss(self, pred, clean_data, true_data=None):
        """Match family-wise endpoint degree-bin-pair distributions on query edges.

        This is a training-only soft teacher for the diagnostic oracle used by
        ``*_degree_pair_quota`` sampling. It compares the predicted soft edge
        mass in each relation family against the clean graph's endpoint
        degree-bin-pair histogram, but only over pair bins represented in the
        current query set.
        """
        pred_edge = pred.edge_attr
        edge_index = pred.edge_index
        if pred_edge.numel() == 0 or edge_index.numel() == 0 or not self.heterogeneous:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}

        _node_type_ids, query_family = self._infer_query_family_from_clean_node_types(
            edge_index.long(),
            clean_data.x,
        )
        if query_family is None or query_family.numel() != pred_edge.shape[0]:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}
        query_family = query_family.to(pred_edge.device).long().reshape(-1)
        pair_info = self._degree_pair_ids_for_candidates(edge_index.long(), query_family)
        if pair_info is None:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}

        pair_id = pair_info["pair_id"].to(pred_edge.device).long()
        valid_pair = pair_info["valid"].to(pred_edge.device)
        pair_counts = pair_info["pair_counts"]
        num_bins = int(pair_info["num_bins"])
        num_pairs = int(num_bins * num_bins)
        eps = 1e-8
        min_edges = int(getattr(self.cfg.model, "degree_pair_dist_min_query_edges", 32) or 0)
        loss_type = str(
            getattr(self.cfg.model, "degree_pair_dist_loss_type", "js")
            or "js"
        ).lower()
        target_smoothing = float(
            getattr(self.cfg.model, "degree_pair_dist_target_smoothing", 0.0)
            or 0.0
        )
        target_smoothing = max(0.0, min(1.0, target_smoothing))
        target_mode = str(
            getattr(self.cfg.model, "degree_pair_dist_target", "block") or "block"
        ).lower()
        if target_mode not in {"block", "reference"}:
            target_mode = "block"
        if target_mode == "block" and true_data is None:
            target_mode = "reference"
        block_target_exist = None
        if target_mode == "block" and true_data is not None:
            target = true_data.edge_attr
            if target.dim() > 1:
                target = target.argmax(dim=-1)
            block_target_exist = (
                target.reshape(-1).long().to(pred_edge.device) > 0
            ).to(dtype=pred_edge.dtype)

        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        prob = torch.softmax(logits, dim=-1)
        pred_exist = (1.0 - prob[:, 0]).clamp(min=0.0)
        losses = []
        for fam in torch.unique(query_family):
            fam_id = int(fam.detach().cpu().item())
            if fam_id < 0 or fam_id not in pair_counts:
                continue
            mask = (query_family == fam_id) & valid_pair
            n = int(mask.sum().item())
            if n < min_edges:
                continue
            fam_pair_ids = pair_id[mask].clamp(min=0, max=num_pairs - 1)
            pred_mass = pred_edge.new_zeros(num_pairs)
            pred_mass.scatter_add_(0, fam_pair_ids, pred_exist[mask])
            if pred_mass.sum() <= eps:
                continue

            available = torch.zeros(num_pairs, dtype=torch.bool, device=pred_edge.device)
            available[fam_pair_ids.unique()] = True
            target_mass = pred_edge.new_zeros(num_pairs)
            if target_mode == "block" and block_target_exist is not None:
                target_mass.scatter_add_(0, fam_pair_ids, block_target_exist[mask])
            else:
                for pair_key, cnt in pair_counts.get(fam_id, {}).items():
                    a, b = int(pair_key[0]), int(pair_key[1])
                    pid = a * num_bins + b
                    if 0 <= pid < num_pairs:
                        target_mass[pid] = float(cnt)
                target_mass = target_mass * available.to(dtype=target_mass.dtype)
            if target_mass.sum() <= eps:
                continue
            if target_smoothing > 0:
                smooth = available.to(dtype=target_mass.dtype)
                smooth = smooth / smooth.sum().clamp(min=1.0)
                target_dist = target_mass / target_mass.sum().clamp(min=eps)
                target_dist = (1.0 - target_smoothing) * target_dist + target_smoothing * smooth
            else:
                target_dist = target_mass / target_mass.sum().clamp(min=eps)
            pred_dist = pred_mass / pred_mass.sum().clamp(min=eps)

            if loss_type == "l1":
                losses.append((pred_dist - target_dist).abs().sum())
            elif loss_type == "mse":
                losses.append((pred_dist - target_dist).pow(2).sum())
            elif loss_type == "kl":
                losses.append(
                    (target_dist * ((target_dist + eps).log() - (pred_dist + eps).log())).sum()
                )
            else:
                mix = 0.5 * (pred_dist + target_dist)
                losses.append(
                    0.5
                    * (
                        (target_dist * ((target_dist + eps).log() - (mix + eps).log())).sum()
                        + (pred_dist * ((pred_dist + eps).log() - (mix + eps).log())).sum()
                    )
                )

        if not losses:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0}
        return torch.stack(losses).mean(), {"active_families": float(len(losses))}

    def _clean_common_neighbor_scores(self, edge_index, full_data, num_nodes, device, dtype):
        full_edge_index = getattr(full_data, "edge_index", None)
        full_edge_attr = getattr(full_data, "edge_attr", None)
        if full_edge_index is None or full_edge_index.numel() == 0 or edge_index.numel() == 0:
            return None
        src_full = full_edge_index[0].long().to(device)
        dst_full = full_edge_index[1].long().to(device)
        if full_edge_attr is not None and full_edge_attr.numel() > 0:
            if full_edge_attr.dim() > 1:
                full_labels = full_edge_attr.argmax(dim=-1).to(device)
            else:
                full_labels = full_edge_attr.long().to(device)
            full_pos = full_labels.reshape(-1) > 0
            src_full = src_full[full_pos]
            dst_full = dst_full[full_pos]
        valid = (
            (src_full >= 0)
            & (src_full < num_nodes)
            & (dst_full >= 0)
            & (dst_full < num_nodes)
            & (src_full != dst_full)
        )
        src_full = src_full[valid]
        dst_full = dst_full[valid]
        if src_full.numel() == 0:
            return None
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        adj[src_full, dst_full] = True
        adj[dst_full, src_full] = True
        q_src = edge_index[0].long().to(device)
        q_dst = edge_index[1].long().to(device)
        valid_q = (
            (q_src >= 0)
            & (q_src < num_nodes)
            & (q_dst >= 0)
            & (q_dst < num_nodes)
            & (q_src != q_dst)
        )
        common = torch.zeros(edge_index.shape[1], dtype=dtype, device=device)
        if valid_q.any():
            valid_idx = valid_q.nonzero(as_tuple=False).reshape(-1)
            chunk_size = int(
                getattr(
                    self.cfg.model,
                    "closure_rank_common_neighbor_chunk_size",
                    65536,
                )
                or 65536
            )
            chunk_size = max(1, chunk_size)
            for start in range(0, int(valid_idx.numel()), chunk_size):
                idx = valid_idx[start : start + chunk_size]
                common[idx] = (
                    adj[q_src[idx]] & adj[q_dst[idx]]
                ).sum(dim=-1).to(dtype=dtype)
        return common

    def _compute_closure_ranking_loss(self, pred, clean_data, num_nodes):
        """Pairwise ranking for structural closure opportunity, independent of edge ID."""
        pred_edge = pred.edge_attr
        edge_index = pred.edge_index
        if pred_edge.numel() == 0 or edge_index.numel() == 0 or not self.heterogeneous:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0, "pairs": 0.0}
        _node_type_ids, query_family = self._infer_query_family_from_clean_node_types(
            edge_index.long(),
            clean_data.x,
        )
        if query_family is None or query_family.numel() != pred_edge.shape[0]:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0, "pairs": 0.0}
        common = self._clean_common_neighbor_scores(
            edge_index=edge_index.long(),
            full_data=clean_data,
            num_nodes=int(num_nodes),
            device=pred_edge.device,
            dtype=pred_edge.dtype,
        )
        if common is None:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0, "pairs": 0.0}
        if bool(getattr(self.cfg.model, "closure_rank_score_log1p", True)):
            rank_score = torch.log1p(common)
        else:
            rank_score = common
        min_common = float(
            getattr(self.cfg.model, "closure_rank_min_common_neighbors", 1.0) or 0.0
        )
        top_q = max(
            0.0,
            min(1.0, float(getattr(self.cfg.model, "closure_rank_top_quantile", 0.8) or 0.8)),
        )
        bottom_q = max(
            0.0,
            min(1.0, float(getattr(self.cfg.model, "closure_rank_bottom_quantile", 0.5) or 0.5)),
        )
        min_edges = int(getattr(self.cfg.model, "closure_rank_min_query_edges", 64) or 0)
        max_pairs = int(getattr(self.cfg.model, "closure_rank_pairs_per_family", 2048) or 2048)
        margin = float(getattr(self.cfg.model, "closure_rank_margin", 0.0) or 0.0)
        query_family = query_family.to(pred_edge.device).long().reshape(-1)
        logits = pred_edge.reshape(-1, pred_edge.shape[-1])
        exist_logits = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        losses = []
        pair_total = 0
        for fam in torch.unique(query_family):
            fam_mask = query_family == fam
            if int(fam_mask.sum().item()) < min_edges:
                continue
            fam_score = rank_score[fam_mask]
            fam_logits = exist_logits[fam_mask]
            positive_struct = fam_score >= float(math.log1p(min_common) if bool(getattr(self.cfg.model, "closure_rank_score_log1p", True)) else min_common)
            if not positive_struct.any():
                continue
            high_threshold = torch.quantile(fam_score[positive_struct].detach().float(), top_q).to(
                device=pred_edge.device, dtype=pred_edge.dtype
            )
            low_threshold = torch.quantile(fam_score.detach().float(), bottom_q).to(
                device=pred_edge.device, dtype=pred_edge.dtype
            )
            good_idx = torch.where(fam_score >= high_threshold)[0]
            bad_idx = torch.where(fam_score <= low_threshold)[0]
            if good_idx.numel() == 0 or bad_idx.numel() == 0:
                continue
            pairs = min(max_pairs, int(good_idx.numel()), int(bad_idx.numel()))
            if pairs <= 0:
                continue
            if good_idx.numel() > pairs:
                good_idx = good_idx[torch.randperm(good_idx.numel(), device=pred_edge.device)[:pairs]]
            if bad_idx.numel() > pairs:
                bad_idx = bad_idx[torch.randperm(bad_idx.numel(), device=pred_edge.device)[:pairs]]
            good_logits = fam_logits[good_idx]
            bad_logits = fam_logits[bad_idx]
            losses.append(F.softplus(margin - (good_logits - bad_logits)).mean())
            pair_total += pairs
        if not losses:
            zero = pred_edge.sum() * 0.0
            return zero, {"active_families": 0.0, "pairs": 0.0}
        return torch.stack(losses).mean(), {
            "active_families": float(len(losses)),
            "pairs": float(pair_total),
        }

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

    def _infer_query_family_from_clean_node_types(self, edge_index, clean_node):
        """Vectorized family inference from clean endpoint node types."""
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        edge_family_offsets = (
            getattr(self.dataset_info, "edge_family_offsets", {}) or {}
        )
        if not type_offsets or not fam_endpoints or not edge_family_offsets:
            return None, None
        if clean_node.dim() > 1:
            node_subtype = clean_node.argmax(dim=-1).long()
        else:
            node_subtype = clean_node.long().reshape(-1)
        sorted_types = sorted(type_offsets.items(), key=lambda kv: kv[1])
        node_type_ids = node_subtype.new_full(node_subtype.shape, -1)
        type_name_to_id = {}
        for type_id, (type_name, offset) in enumerate(sorted_types):
            type_name_to_id[str(type_name)] = type_id
            if type_id + 1 < len(sorted_types):
                next_offset = int(sorted_types[type_id + 1][1])
                mask = (node_subtype >= int(offset)) & (
                    node_subtype < next_offset
                )
            else:
                mask = node_subtype >= int(offset)
            node_type_ids[mask] = type_id
        num_types = len(sorted_types)
        if num_types <= 0:
            return None, None
        pair_to_family = node_subtype.new_full((num_types, num_types), -1)
        for fam_id, (fam_name, _) in enumerate(
            sorted(edge_family_offsets.items(), key=lambda kv: kv[1])
        ):
            endpoints = fam_endpoints.get(str(fam_name), {}) or {}
            src_name = str(endpoints.get("src_type", ""))
            dst_name = str(endpoints.get("dst_type", ""))
            if src_name not in type_name_to_id or dst_name not in type_name_to_id:
                continue
            src_type = type_name_to_id[src_name]
            dst_type = type_name_to_id[dst_name]
            pair_to_family[src_type, dst_type] = fam_id
            # Query edges are canonical in most training paths, but positive
            # augmentation and undirected utilities can still present the
            # reverse endpoint order. Treat both as the same family.
            pair_to_family[dst_type, src_type] = fam_id
        src_type = node_type_ids[edge_index[0].long()].clamp(min=0)
        dst_type = node_type_ids[edge_index[1].long()].clamp(min=0)
        query_family = pair_to_family[src_type, dst_type]
        invalid = (
            node_type_ids[edge_index[0].long()] < 0
        ) | (node_type_ids[edge_index[1].long()] < 0)
        if invalid.any():
            query_family = query_family.clone()
            query_family[invalid] = -1
        return node_type_ids, query_family

    def _compute_family_role_ranking_loss(
        self,
        pred,
        true_data,
        clean_data,
        sparse_noisy_data,
    ):
        """Soft family-role ranking loss on the current query set.

        For each relation family, compare the endpoint degree-role matrix
        induced by predicted existence rankings against the clean positive
        query edges. The predicted mass is obtained by a family-local softmax
        over existence logits and scaled to the number of clean positives in
        that family, so the objective supervises ranking/role allocation rather
        than probability calibration.
        """
        if pred.edge_attr.numel() == 0 or pred.edge_index.numel() == 0:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {"active_families": 0.0, "reliability": 0.0}
        if not self.heterogeneous:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {"active_families": 0.0, "reliability": 0.0}

        reliability = sparse_noisy_data.get("endpoint_role_scale_factor")
        if reliability is None:
            t_float = sparse_noisy_data.get("t_float")
            if t_float is not None:
                reliability = self._endpoint_role_reliability_factor(t_float)
        if reliability is None:
            reliability_value = 1.0
        else:
            reliability_value = float(
                reliability.detach().float().mean().clamp(0.0, 1.0).cpu()
            )
        min_reliability = float(
            getattr(self.cfg.model, "family_role_loss_min_reliability", 0.0)
            or 0.0
        )
        if reliability_value < min_reliability:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {
                "active_families": 0.0,
                "reliability": reliability_value,
            }

        node = clean_data.x
        if node.dim() > 1:
            clean_node_labels = node.argmax(dim=-1).long()
        else:
            clean_node_labels = node.long()
        node_type_ids, query_family = self._infer_query_family_from_clean_node_types(
            pred.edge_index.long(),
            clean_data.x,
        )
        if node_type_ids is None or query_family is None:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {"active_families": 0.0, "reliability": reliability_value}
        node_type_ids = node_type_ids.to(pred.edge_attr.device).long()
        query_family = query_family.to(pred.edge_attr.device).long().reshape(-1)
        if query_family.numel() != pred.edge_attr.shape[0]:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {"active_families": 0.0, "reliability": reliability_value}

        clean_edge_attr = clean_data.edge_attr
        if clean_edge_attr.dim() > 1:
            clean_labels = clean_edge_attr.argmax(dim=-1).long()
        else:
            clean_labels = clean_edge_attr.long().reshape(-1)
        clean_pos = clean_labels > 0
        clean_edges = clean_data.edge_index[:, clean_pos].long()
        degree = pred.edge_attr.new_zeros(clean_node_labels.shape[0])
        if clean_edges.numel() > 0:
            ones = pred.edge_attr.new_ones(clean_edges.shape[1])
            degree.scatter_add_(0, clean_edges[0].to(degree.device), ones)
            degree.scatter_add_(0, clean_edges[1].to(degree.device), ones)
        log_degree = torch.log1p(degree)

        batch = getattr(clean_data, "batch", None)
        if batch is None:
            batch = torch.zeros_like(clean_node_labels, device=pred.edge_attr.device)
        else:
            batch = batch.to(pred.edge_attr.device).long()
        node_type_ids = node_type_ids.to(pred.edge_attr.device)
        roles = torch.ones_like(clean_node_labels, device=pred.edge_attr.device)
        for graph_id in torch.unique(batch):
            graph_mask = batch == graph_id
            for type_id in torch.unique(node_type_ids[graph_mask]):
                mask = graph_mask & (node_type_ids == type_id)
                vals = log_degree[mask]
                if vals.numel() == 0:
                    continue
                if vals.numel() == 1:
                    roles[mask] = 1
                    continue
                q50 = torch.quantile(vals.float(), 0.5)
                q80 = torch.quantile(vals.float(), 0.8)
                r = torch.ones_like(vals, dtype=torch.long)
                r[vals <= q50] = 0
                r[vals > q80] = 2
                roles[mask] = r

        src = pred.edge_index[0].long()
        dst = pred.edge_index[1].long()
        src_role = roles[src]
        dst_role = roles[dst]
        same_type = node_type_ids[src] == node_type_ids[dst]
        role_a = src_role.clone()
        role_b = dst_role.clone()
        if same_type.any():
            lo = torch.minimum(src_role[same_type], dst_role[same_type])
            hi = torch.maximum(src_role[same_type], dst_role[same_type])
            role_a[same_type] = lo
            role_b[same_type] = hi
        role_pair = (role_a * 3 + role_b).long().clamp(0, 8)

        target = true_data.edge_attr
        if target.dim() > 1:
            target = target.argmax(dim=-1).long()
        else:
            target = target.long().reshape(-1)
        true_exist = (target > 0).to(dtype=pred.edge_attr.dtype)
        logits = pred.edge_attr.reshape(-1, pred.edge_attr.shape[-1])
        exist_logits = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        temperature = max(
            float(getattr(self.cfg.model, "family_role_loss_temperature", 1.0) or 1.0),
            1e-4,
        )
        eps = 1e-8
        losses = []
        for fam in torch.unique(query_family):
            fam_int = int(fam.detach().cpu().item())
            if fam_int < 0:
                continue
            fam_mask = query_family == fam
            target_count = true_exist[fam_mask].sum()
            if target_count <= 0:
                continue
            pred_weights = (
                torch.softmax(exist_logits[fam_mask] / temperature, dim=0)
                * target_count.detach()
            )
            target_weights = true_exist[fam_mask]
            fam_role = role_pair[fam_mask]
            pred_mass = pred.edge_attr.new_zeros(9)
            target_mass = pred.edge_attr.new_zeros(9)
            pred_mass.scatter_add_(0, fam_role, pred_weights)
            target_mass.scatter_add_(0, fam_role, target_weights)
            pred_dist = pred_mass / pred_mass.sum().clamp(min=eps)
            target_dist = target_mass / target_mass.sum().clamp(min=eps)
            mix = 0.5 * (pred_dist + target_dist)
            jsd = 0.5 * (
                (target_dist * ((target_dist + eps).log() - (mix + eps).log())).sum()
                + (pred_dist * ((pred_dist + eps).log() - (mix + eps).log())).sum()
            )
            losses.append(jsd)

        if not losses:
            zero = pred.edge_attr.sum() * 0.0
            return zero, {"active_families": 0.0, "reliability": reliability_value}
        return torch.stack(losses).mean(), {
            "active_families": float(len(losses)),
            "reliability": reliability_value,
        }

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
        if bool(getattr(self.cfg.model, "train_family_staged_queryfree", False)):
            return self._training_step_family_staged_queryfree(data, i)

        if bool(getattr(self.cfg.model, "train_partition_ensemble", False)):
            return self._training_step_partition_ensemble(data, i)

        if bool(getattr(self.cfg.model, "train_all_blocks_per_noise", False)):
            if data.edge_index.numel() == 0:
                if hasattr(self, 'local_rank') and self.local_rank == 0:
                    print("Found a batch with no edges. Skipping.")
                return None

            if not self.use_block_query:
                return self._training_step_once(data, i)

            opt = self.optimizers()
            profile_efficiency = self._profile_training_efficiency_enabled()
            if profile_efficiency and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            profile_start = self._profile_time(profile_efficiency)
            data_one_hot = self.dataset_info.to_one_hot(data)
            t_override = self._training_t_override(data_one_hot, i)
            sparse_noisy_data = self.apply_sparse_noise(data_one_hot, t_override=t_override)
            noise_done = self._profile_time(profile_efficiency)
            lookup_start = noise_done
            two_hop_lookup = self._build_training_two_hop_lookup(
                sparse_noisy_data,
                num_nodes=int(data_one_hot.x.shape[0]),
            )
            if two_hop_lookup is not None:
                sparse_noisy_data["two_hop_structure_lookup"] = two_hop_lookup
            lookup_done = self._profile_time(profile_efficiency)

            block_count = int(getattr(self.cfg.model, "train_all_blocks_count", 0) or 0)
            if block_count <= 0:
                ef = max(float(getattr(self, "edge_fraction", 1.0) or 1.0), 1e-8)
                block_count = max(1, int(math.ceil(1.0 / ef)))
            step_mode = str(
                getattr(self.cfg.model, "train_all_blocks_step_mode", "sequential")
                or "sequential"
            ).lower()
            if step_mode not in {
                "sequential",
                "accumulate",
                "accumulate_global",
                "accumulate_streaming_global",
            }:
                raise ValueError(
                    "model.train_all_blocks_step_mode must be 'sequential', "
                    "'accumulate', 'accumulate_global', or "
                    f"'accumulate_streaming_global', got {step_mode!r}"
                )

            block_order = torch.arange(block_count, device=self.device)
            if bool(getattr(self.cfg.model, "train_all_blocks_shuffle", True)) and block_count > 1:
                block_order = block_order[torch.randperm(block_count, device=self.device)]

            losses_detached = []
            query_build_time = 0.0
            comp_graph_time = 0.0
            hgt_forward_time = 0.0
            loss_build_time = 0.0
            backward_time = 0.0
            optimizer_time = 0.0
            query_edges_total = 0
            if step_mode in {"accumulate", "accumulate_global", "accumulate_streaming_global"}:
                opt.zero_grad()
            accumulate_divisor = max(1, int(block_order.numel()))
            global_pred_edges = []
            global_pred_edge_indices = []
            global_true_edges = []
            streaming_global_aux = None
            if step_mode == "accumulate_streaming_global":
                streaming_stats = None
                with torch.no_grad():
                    for block_id_t in block_order:
                        stat_out = self._training_step_once(
                            data_one_hot,
                            i,
                            data_is_one_hot=True,
                            sparse_noisy_data_override=dict(sparse_noisy_data),
                            forced_block_id=int(block_id_t.item()),
                            skip_auxiliary_structure_losses=True,
                            return_query_outputs=True,
                            record_structure_probe=False,
                        )
                        if stat_out is None or "pred_edge_attr" not in stat_out:
                            continue
                        pred_stat = utils.SparsePlaceHolder(
                            node=data_one_hot.x,
                            charge=data_one_hot.charge,
                            edge_attr=stat_out["pred_edge_attr"],
                            edge_index=stat_out["pred_edge_index"],
                            y=data_one_hot.y,
                            batch=data_one_hot.batch,
                        )
                        true_stat = utils.SparsePlaceHolder(
                            node=data_one_hot.x,
                            charge=data_one_hot.charge,
                            edge_attr=stat_out["true_edge_attr"],
                            edge_index=stat_out["pred_edge_index"],
                            y=data_one_hot.y,
                            batch=data_one_hot.batch,
                        )
                        streaming_stats = self._merge_global_aux_stats(
                            streaming_stats,
                            self._global_aux_stats_from_block(
                                pred_stat, true_stat, data_one_hot
                            ),
                        )
                streaming_global_aux = self._prepare_streaming_global_aux(
                    streaming_stats, sparse_noisy_data
                )
            for block_id_t in block_order:
                if step_mode == "sequential":
                    opt.zero_grad()
                out = self._training_step_once(
                    data_one_hot,
                    i,
                    data_is_one_hot=True,
                    sparse_noisy_data_override=dict(sparse_noisy_data),
                    forced_block_id=int(block_id_t.item()),
                    skip_auxiliary_structure_losses=(
                        step_mode in {"accumulate_global", "accumulate_streaming_global"}
                    ),
                    return_query_outputs=(
                        step_mode in {"accumulate_global", "accumulate_streaming_global"}
                    ),
                )
                if out is None:
                    continue
                loss_piece = out["loss"]
                forward_done = self._profile_time(profile_efficiency)
                if step_mode == "accumulate_global":
                    if "pred_edge_attr" in out:
                        global_pred_edges.append(out["pred_edge_attr"])
                        global_pred_edge_indices.append(out["pred_edge_index"])
                        global_true_edges.append(out["true_edge_attr"])
                    backward_done = forward_done
                elif step_mode == "accumulate_streaming_global":
                    aux_block_loss = loss_piece.sum() * 0.0
                    if "pred_edge_attr" in out:
                        pred_block = utils.SparsePlaceHolder(
                            node=data_one_hot.x,
                            charge=data_one_hot.charge,
                            edge_attr=out["pred_edge_attr"],
                            edge_index=out["pred_edge_index"],
                            y=data_one_hot.y,
                            batch=data_one_hot.batch,
                        )
                        true_block = utils.SparsePlaceHolder(
                            node=data_one_hot.x,
                            charge=data_one_hot.charge,
                            edge_attr=out["true_edge_attr"],
                            edge_index=out["pred_edge_index"],
                            y=data_one_hot.y,
                            batch=data_one_hot.batch,
                        )
                        aux_block_loss = self._streaming_global_aux_loss_for_block(
                            pred_block,
                            true_block,
                            data_one_hot,
                            streaming_global_aux,
                        )
                        closure_rank_w = self._warmup_weight(
                            float(getattr(self.cfg.model, "closure_rank_loss_weight", 0.0) or 0.0),
                            int(getattr(self.cfg.model, "closure_rank_loss_warmup_epochs", 0) or 0),
                            self._auxiliary_time_factor(
                                sparse_noisy_data,
                                float(getattr(self.cfg.model, "closure_rank_loss_t_power", 1.0) or 0.0),
                            ),
                        )
                        if closure_rank_w > 0:
                            loss_closure_rank, closure_rank_stats = self._compute_closure_ranking_loss(
                                pred_block,
                                data_one_hot,
                                num_nodes=data_one_hot.x.shape[0],
                            )
                            aux_block_loss = aux_block_loss + (
                                closure_rank_w * loss_closure_rank / float(accumulate_divisor)
                            )
                            if not hasattr(self, "_epoch_closure_rank_loss_sum"):
                                self._epoch_closure_rank_loss_sum = 0.0
                                self._epoch_closure_rank_loss_count = 0
                                self._epoch_closure_rank_weighted_sum = 0.0
                                self._epoch_closure_rank_active_sum = 0.0
                                self._epoch_closure_rank_pairs_sum = 0.0
                            self._epoch_closure_rank_loss_sum += float(
                                loss_closure_rank.detach().cpu()
                            )
                            self._epoch_closure_rank_loss_count += 1
                            self._epoch_closure_rank_weighted_sum += float(
                                (closure_rank_w * loss_closure_rank).detach().cpu()
                            )
                            self._epoch_closure_rank_active_sum += float(
                                closure_rank_stats.get("active_families", 0.0)
                            )
                            self._epoch_closure_rank_pairs_sum += float(
                                closure_rank_stats.get("pairs", 0.0)
                            )
                    backward_loss = loss_piece / float(accumulate_divisor) + aux_block_loss
                    self.manual_backward(backward_loss)
                    backward_done = self._profile_time(profile_efficiency)
                    loss_piece = loss_piece.detach() + aux_block_loss.detach()
                else:
                    backward_loss = (
                        loss_piece / float(accumulate_divisor)
                        if step_mode == "accumulate"
                        else loss_piece
                    )
                    self.manual_backward(backward_loss)
                    backward_done = self._profile_time(profile_efficiency)
                if step_mode == "sequential":
                    clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
                    if clip_val > 0:
                        self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
                    opt.step()
                    optimizer_done = self._profile_time(profile_efficiency)
                else:
                    optimizer_done = backward_done
                perf = out.get("perf", {})
                query_build_time += float(perf.get("query_build_s", 0.0))
                comp_graph_time += float(perf.get("comp_graph_s", 0.0))
                hgt_forward_time += float(perf.get("hgt_forward_s", 0.0))
                loss_build_time += float(perf.get("loss_build_s", 0.0))
                backward_time += backward_done - forward_done
                optimizer_time += optimizer_done - backward_done
                query_edges_total += int(out.get("query_edges", 0))
                losses_detached.append(loss_piece.detach())

            if not losses_detached:
                return None
            if step_mode == "accumulate_global":
                global_loss = torch.stack(losses_detached).mean()
                if global_pred_edges and global_true_edges:
                    pred_global = utils.SparsePlaceHolder(
                        node=data_one_hot.x,
                        charge=data_one_hot.charge,
                        edge_attr=torch.cat(global_pred_edges, dim=0),
                        edge_index=torch.cat(global_pred_edge_indices, dim=1),
                        y=data_one_hot.y,
                        batch=data_one_hot.batch,
                    )
                    true_global = utils.SparsePlaceHolder(
                        node=data_one_hot.x,
                        charge=data_one_hot.charge,
                        edge_attr=torch.cat(global_true_edges, dim=0),
                        edge_index=pred_global.edge_index,
                        y=data_one_hot.y,
                        batch=data_one_hot.batch,
                    )

                    family_count_w = self._warmup_weight(
                        float(getattr(self.cfg.model, "family_count_loss_weight", 0.0) or 0.0),
                        int(getattr(self.cfg.model, "family_count_loss_warmup_epochs", 0) or 0),
                        self._auxiliary_time_factor(
                            sparse_noisy_data,
                            float(getattr(self.cfg.model, "family_count_loss_t_power", 0.0) or 0.0),
                        ),
                    )
                    if family_count_w > 0:
                        loss_family_count, family_count_stats = self._compute_family_count_loss(
                            pred_global, true_global, data_one_hot
                        )
                        global_loss = global_loss + family_count_w * loss_family_count
                        if not hasattr(self, "_epoch_family_count_loss_sum"):
                            self._epoch_family_count_loss_sum = 0.0
                            self._epoch_family_count_loss_count = 0
                            self._epoch_family_count_weighted_sum = 0.0
                            self._epoch_family_count_active_sum = 0.0
                        self._epoch_family_count_loss_sum += float(loss_family_count.detach().cpu())
                        self._epoch_family_count_loss_count += 1
                        self._epoch_family_count_weighted_sum += float(
                            (family_count_w * loss_family_count).detach().cpu()
                        )
                        self._epoch_family_count_active_sum += float(
                            family_count_stats.get("active_families", 0.0)
                        )

                    degree_pair_w = self._warmup_weight(
                        float(getattr(self.cfg.model, "degree_pair_dist_loss_weight", 0.0) or 0.0),
                        int(getattr(self.cfg.model, "degree_pair_dist_loss_warmup_epochs", 0) or 0),
                        self._auxiliary_time_factor(
                            sparse_noisy_data,
                            float(getattr(self.cfg.model, "degree_pair_dist_loss_t_power", 0.0) or 0.0),
                        ),
                    )
                    if degree_pair_w > 0:
                        loss_degree_pair, degree_pair_stats = self._compute_degree_pair_distribution_loss(
                            pred_global, data_one_hot, true_global
                        )
                        global_loss = global_loss + degree_pair_w * loss_degree_pair
                        if not hasattr(self, "_epoch_degree_pair_dist_loss_sum"):
                            self._epoch_degree_pair_dist_loss_sum = 0.0
                            self._epoch_degree_pair_dist_loss_count = 0
                            self._epoch_degree_pair_dist_weighted_sum = 0.0
                            self._epoch_degree_pair_dist_active_sum = 0.0
                        self._epoch_degree_pair_dist_loss_sum += float(loss_degree_pair.detach().cpu())
                        self._epoch_degree_pair_dist_loss_count += 1
                        self._epoch_degree_pair_dist_weighted_sum += float(
                            (degree_pair_w * loss_degree_pair).detach().cpu()
                        )
                        self._epoch_degree_pair_dist_active_sum += float(
                            degree_pair_stats.get("active_families", 0.0)
                        )

                    closure_rank_w = self._warmup_weight(
                        float(getattr(self.cfg.model, "closure_rank_loss_weight", 0.0) or 0.0),
                        int(getattr(self.cfg.model, "closure_rank_loss_warmup_epochs", 0) or 0),
                        self._auxiliary_time_factor(
                            sparse_noisy_data,
                            float(getattr(self.cfg.model, "closure_rank_loss_t_power", 1.0) or 0.0),
                        ),
                    )
                    if closure_rank_w > 0:
                        loss_closure_rank, closure_rank_stats = self._compute_closure_ranking_loss(
                            pred_global, data_one_hot, num_nodes=data_one_hot.x.shape[0]
                        )
                        global_loss = global_loss + closure_rank_w * loss_closure_rank
                        if not hasattr(self, "_epoch_closure_rank_loss_sum"):
                            self._epoch_closure_rank_loss_sum = 0.0
                            self._epoch_closure_rank_loss_count = 0
                            self._epoch_closure_rank_weighted_sum = 0.0
                            self._epoch_closure_rank_active_sum = 0.0
                            self._epoch_closure_rank_pairs_sum = 0.0
                        self._epoch_closure_rank_loss_sum += float(loss_closure_rank.detach().cpu())
                        self._epoch_closure_rank_loss_count += 1
                        self._epoch_closure_rank_weighted_sum += float(
                            (closure_rank_w * loss_closure_rank).detach().cpu()
                        )
                        self._epoch_closure_rank_active_sum += float(
                            closure_rank_stats.get("active_families", 0.0)
                        )
                        self._epoch_closure_rank_pairs_sum += float(
                            closure_rank_stats.get("pairs", 0.0)
                        )

                backward_start = self._profile_time(profile_efficiency)
                self.manual_backward(global_loss)
                backward_done = self._profile_time(profile_efficiency)
                backward_time += backward_done - backward_start

            if step_mode in {"accumulate", "accumulate_global", "accumulate_streaming_global"}:
                opt_start = self._profile_time(profile_efficiency)
                clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
                if clip_val > 0:
                    self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
                opt.step()
                opt_done = self._profile_time(profile_efficiency)
                optimizer_time += opt_done - opt_start
            if profile_efficiency and getattr(self, "local_rank", 0) == 0:
                peak_allocated = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                )
                peak_reserved = (
                    int(torch.cuda.max_memory_reserved(self.device))
                    if self.device.type == "cuda"
                    else 0
                )
                lookup = sparse_noisy_data.get("two_hop_structure_lookup") or {}
                print(
                    "[TRAIN-PERF] "
                    f"epoch={int(self.current_epoch)} mode={step_mode} "
                    f"blocks={len(losses_detached)} "
                    f"queries={query_edges_total} "
                    f"noise_s={noise_done - profile_start:.4f} "
                    f"twohop_build_s={lookup_done - lookup_start:.4f} "
                    f"query_build_s={query_build_time:.4f} "
                    f"comp_graph_s={comp_graph_time:.4f} "
                    f"hgt_forward_s={hgt_forward_time:.4f} "
                    f"loss_build_s={loss_build_time:.4f} "
                    f"backward_s={backward_time:.4f} "
                    f"optimizer_s={optimizer_time:.4f} "
                    f"adj_nnz={int(lookup.get('adjacency_nnz', 0))} "
                    f"twohop_nnz={int(lookup.get('two_hop_nnz', 0))} "
                    f"peak_alloc_mb={peak_allocated / (1024 ** 2):.1f} "
                    f"peak_reserved_mb={peak_reserved / (1024 ** 2):.1f}"
                )
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

    def _training_step_partition_ensemble(self, data, i):
        """One G_t, one canonical query pool, two partition views, one update."""
        if data.edge_index.numel() == 0:
            return None
        if not self.heterogeneous:
            raise ValueError("train_partition_ensemble requires heterogeneous data")

        opt = self.optimizers()
        opt.zero_grad()
        data_one_hot = self.dataset_info.to_one_hot(data)
        t_override = self._training_t_override(data_one_hot, i)
        sparse_noisy_data = self.apply_sparse_noise(
            data_one_hot, t_override=t_override
        )
        node_subtype = (
            data_one_hot.x.argmax(dim=-1).long()
            if data_one_hot.x.dim() > 1
            else data_one_hot.x.long()
        )
        batch = data_one_hot.batch.long()
        ptr = data_one_hot.ptr.long()
        raw_query, raw_batch = (
            self._sample_heterogeneous_uniform_query_for_sampling(
                node_subtype, batch, ptr
            )
        )
        query_pool = self._canonical_training_query_pool(
            raw_query, raw_batch, int(data_one_hot.x.shape[0])
        )
        if query_pool is None:
            return None
        query_edge_index, query_edge_batch = query_pool

        noisy_edge_attr = sparse_noisy_data["edge_attr_t"]
        noisy_edge_ids = (
            noisy_edge_attr.argmax(dim=-1).long()
            if noisy_edge_attr.dim() > 1
            else noisy_edge_attr.long()
        )
        num_blocks = int(
            getattr(
                self.cfg.model,
                "train_partition_ensemble_num_blocks",
                0,
            )
            or 0
        )
        if num_blocks <= 0:
            num_blocks = max(
                1,
                int(
                    math.ceil(
                        1.0
                        / max(
                            float(getattr(self, "edge_fraction", 1.0)),
                            1e-8,
                        )
                    )
                ),
            )
        metis_blocks = self._current_gt_hetero_metis_blocks(
            sparse_noisy_data["edge_index_t"].long(),
            noisy_edge_ids,
            node_subtype,
            batch,
        )
        random_blocks = self._build_type_balanced_random_blocks(
            node_subtype, batch, num_blocks
        )
        context_budget = self._shared_partition_context_budget(
            sparse_noisy_data["edge_index_t"].long(),
            metis_blocks,
            random_blocks,
            int(data_one_hot.x.shape[0]),
        )
        # Families are not required for training-target construction; this
        # placeholder only lets the shared partition helper retain alignment.
        placeholder_family = torch.zeros(
            query_edge_index.shape[1], dtype=torch.long, device=self.device
        )
        packed_pool = (
            query_edge_index,
            query_edge_batch,
            placeholder_family,
        )
        metis_rounds = self._partition_query_pool_into_rounds(
            packed_pool, metis_blocks, int(data_one_hot.x.shape[0])
        )
        random_rounds = self._partition_query_pool_into_rounds(
            packed_pool, random_blocks, int(data_one_hot.x.shape[0])
        )
        metis_weight = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.cfg.model,
                        "train_partition_ensemble_metis_weight",
                        0.5,
                    )
                ),
            ),
        )
        total_queries = max(1, int(query_edge_index.shape[1]))
        weighted_terms = []
        view_stats = []
        seed_base = self._current_sampling_seed()
        global_step = int(getattr(self, "global_step", 0))
        for view_idx, (view_name, rounds, blocks, view_weight) in enumerate((
            ("metis", metis_rounds, metis_blocks, metis_weight),
            ("random", random_rounds, random_blocks, 1.0 - metis_weight),
        )):
            used = 0
            if view_weight <= 0:
                view_stats.append(f"{view_name}=0")
                continue
            context_edge_index, context_edge_attr, context_stats = (
                self._partition_context_edges(
                    sparse_noisy_data["edge_index_t"].long(),
                    sparse_noisy_data["edge_attr_t"].float(),
                    blocks,
                    int(data_one_hot.x.shape[0]),
                    target_edge_count=context_budget,
                    random_seed=(
                        seed_base + 4000037 * global_step + 97 * view_idx
                    ),
                )
            )
            encoded = self._encode_context_only(
                data=data_one_hot,
                node_onehot=sparse_noisy_data["node_t"],
                context_edge_index=context_edge_index,
                context_edge_attr=context_edge_attr,
                batch=batch,
                ptr=ptr,
                t_float=sparse_noisy_data["t_float"],
            )
            for round_edges, round_batch, _ in rounds:
                if round_edges.numel() == 0:
                    continue
                loss_piece = self._training_loss_from_encoded_queries(
                    encoded=encoded,
                    data=data_one_hot,
                    sparse_noisy_data=sparse_noisy_data,
                    query_edge_index=round_edges,
                    query_edge_batch=round_batch,
                    node_subtype=node_subtype,
                    step_idx=int(i),
                )
                round_count = int(round_edges.shape[1])
                scale = view_weight * float(round_count) / float(total_queries)
                weighted_terms.append(loss_piece * scale)
                used += round_count
            view_stats.append(
                f"{view_name}={used}/ctx={context_stats['total']}"
                f"/intra={context_stats['intra']}"
                f"/inter={context_stats['inter']}"
                f"/degree_cv={context_stats['degree_cv']:.3f}"
                f"/families={context_stats['family_counts']}"
            )

        if not weighted_terms:
            return None
        total_loss = torch.stack(weighted_terms).sum()
        self.manual_backward(total_loss)
        clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
        if clip_val > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )
        opt.step()
        if (
            getattr(self, "local_rank", 0) == 0
            and not getattr(self, "_logged_train_partition_ensemble", False)
        ):
            print(
                "[TRAIN-PARTITION-ENSEMBLE] "
                f"queries={total_queries} blocks={num_blocks} "
                f"metis_weight={metis_weight:.3f} "
                + " ".join(view_stats)
            )
            self._logged_train_partition_ensemble = True
        return {"loss": total_loss.detach()}

    def _iter_family_staged_query_chunks(
        self,
        node_subtype: torch.Tensor,
        batch: torch.Tensor,
        chunk_size: int,
    ):
        """Yield full legal candidate chunks grouped as intra- then cross-family tasks."""
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {}) or {}
        edge_family_avg = (
            getattr(self.dataset_info, "edge_family_avg_edge_counts", {}) or {}
        )
        if not (self.heterogeneous and type_offsets and fam_endpoints):
            return

        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_ranges = {}
        for idx, (name, offset) in enumerate(sorted_types):
            nxt = (
                sorted_types[idx + 1][1]
                if idx + 1 < len(sorted_types)
                else self.out_dims.X
            )
            type_ranges[str(name)] = (int(offset), int(nxt))

        families = []
        for fam_name, endpoints in fam_endpoints.items():
            if float(edge_family_avg.get(fam_name, 0.0) or 0.0) <= 0.0:
                continue
            src_t = str(endpoints.get("src_type"))
            dst_t = str(endpoints.get("dst_type"))
            if src_t not in type_ranges or dst_t not in type_ranges:
                continue
            stage = 0 if src_t == dst_t else 1
            families.append((stage, str(fam_name), src_t, dst_t))
        families.sort(key=lambda item: (item[0], item[1]))

        chunk_size = max(1024, int(chunk_size))
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        for stage, fam_name, src_t, dst_t in families:
            s0, s1 = type_ranges[src_t]
            d0, d1 = type_ranges[dst_t]
            same_type = src_t == dst_t
            for graph_idx in range(graph_count):
                graph_mask = batch == int(graph_idx)
                src_nodes = torch.where(
                    graph_mask
                    & (node_subtype >= int(s0))
                    & (node_subtype < int(s1))
                )[0]
                dst_nodes = torch.where(
                    graph_mask
                    & (node_subtype >= int(d0))
                    & (node_subtype < int(d1))
                )[0]
                if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                    continue
                if same_type:
                    pair_index = torch.triu_indices(
                        src_nodes.numel(),
                        src_nodes.numel(),
                        offset=1,
                        device=self.device,
                    )
                    total = int(pair_index.shape[1])
                    for start in range(0, total, chunk_size):
                        sub = pair_index[:, start : start + chunk_size]
                        if sub.numel() == 0:
                            continue
                        edge_index = torch.stack(
                            [src_nodes[sub[0]], src_nodes[sub[1]]], dim=0
                        )
                        edge_batch = torch.full(
                            (edge_index.shape[1],),
                            int(graph_idx),
                            dtype=torch.long,
                            device=self.device,
                        )
                        yield stage, fam_name, edge_index, edge_batch
                else:
                    total = int(src_nodes.numel() * dst_nodes.numel())
                    for start in range(0, total, chunk_size):
                        end = min(total, start + chunk_size)
                        flat = torch.arange(start, end, device=self.device)
                        edge_index = torch.stack(
                            [
                                src_nodes[flat // dst_nodes.numel()],
                                dst_nodes[flat % dst_nodes.numel()],
                            ],
                            dim=0,
                        )
                        edge_batch = torch.full(
                            (edge_index.shape[1],),
                            int(graph_idx),
                            dtype=torch.long,
                            device=self.device,
                        )
                        yield stage, fam_name, edge_index, edge_batch

    def _training_step_family_staged_queryfree(
        self,
        data,
        i,
        do_backward: bool = True,
        log_metrics: bool = False,
    ):
        """Encode full visible G_t once, then decode intra-family and cross-family tasks.

        Query candidates are decoder-only; they never enter HGT message passing.
        Stage 0 contains same-type families, stage 1 contains cross-type families.
        """
        if data.edge_index.numel() == 0:
            return None
        if not self.heterogeneous:
            raise ValueError("train_family_staged_queryfree requires heterogeneous data")

        opt = self.optimizers() if do_backward else None
        if do_backward:
            opt.zero_grad()
        data_one_hot = self.dataset_info.to_one_hot(data)
        t_override = self._training_t_override(data_one_hot, i)
        sparse_noisy_data = self.apply_sparse_noise(
            data_one_hot, t_override=t_override
        )
        node_subtype = (
            data_one_hot.x.argmax(dim=-1).long()
            if data_one_hot.x.dim() > 1
            else data_one_hot.x.long()
        )
        batch = data_one_hot.batch.long()
        ptr = data_one_hot.ptr.long()
        encoded = self._encode_context_only(
            data=data_one_hot,
            node_onehot=sparse_noisy_data["node_t"],
            context_edge_index=sparse_noisy_data["edge_index_t"].long(),
            context_edge_attr=sparse_noisy_data["edge_attr_t"].float(),
            batch=batch,
            ptr=ptr,
            t_float=sparse_noisy_data["t_float"],
        )

        chunk_size = int(
            getattr(
                self.cfg.model,
                "train_family_staged_query_chunk_size",
                65536,
            )
            or 65536
        )
        max_chunks = int(
            getattr(self.cfg.model, "train_family_staged_max_chunks", 0) or 0
        )
        balance_by_family = bool(
            getattr(self.cfg.model, "train_family_staged_balance_by_family", True)
        )
        chunk_meta = []
        stage_counts = [0, 0]
        fam_counts = {}
        for stage, fam_name, query_edge_index, query_edge_batch in (
            self._iter_family_staged_query_chunks(
                node_subtype=node_subtype,
                batch=batch,
                chunk_size=chunk_size,
            )
        ):
            if query_edge_index.numel() == 0:
                continue
            count = int(query_edge_index.shape[1])
            chunk_meta.append((int(stage), str(fam_name), count))
            stage_counts[int(stage)] += count
            fam_counts[fam_name] = int(fam_counts.get(fam_name, 0)) + count
            if max_chunks > 0 and len(chunk_meta) >= max_chunks:
                break

        total_queries = int(sum(stage_counts))
        if total_queries <= 0 or not chunk_meta:
            return None
        active_families = max(len(fam_counts), 1)
        total_chunks = len(chunk_meta)
        chunk_count = 0
        total_loss_metric = None
        for stage, fam_name, query_edge_index, query_edge_batch in (
            self._iter_family_staged_query_chunks(
                node_subtype=node_subtype,
                batch=batch,
                chunk_size=chunk_size,
            )
        ):
            if query_edge_index.numel() == 0:
                continue
            if max_chunks > 0 and chunk_count >= max_chunks:
                break
            loss_piece = self._training_loss_from_encoded_queries(
                encoded=encoded,
                data=data_one_hot,
                sparse_noisy_data=sparse_noisy_data,
                query_edge_index=query_edge_index,
                query_edge_batch=query_edge_batch,
                node_subtype=node_subtype,
                step_idx=int(i),
                log_metrics=bool(log_metrics),
                balance_loss_by_query_family=bool(balance_by_family),
            )
            count = int(query_edge_index.shape[1])
            if balance_by_family:
                scale = (
                    float(count)
                    / float(max(int(fam_counts.get(fam_name, 0)), 1))
                    / float(active_families)
                )
            else:
                scale = float(count) / float(total_queries)
            scaled_loss = loss_piece * scale
            chunk_count += 1
            if do_backward:
                retain_graph = chunk_count < total_chunks
                self.manual_backward(scaled_loss, retain_graph=retain_graph)
            total_loss_metric = (
                scaled_loss.detach()
                if total_loss_metric is None
                else total_loss_metric + scaled_loss.detach()
            )
        clip_val = float(getattr(self.cfg.train, "clip_grad", 0.0) or 0.0)
        if do_backward and clip_val > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )
        if do_backward:
            opt.step()
        if total_loss_metric is None:
            return None
        if (
            getattr(self, "local_rank", 0) == 0
            and not getattr(self, "_logged_train_family_staged_queryfree", False)
        ):
            print(
                "[TRAIN-FAMILY-STAGED] "
                f"queries={total_queries} chunks={chunk_count} "
                f"intra={stage_counts[0]} cross={stage_counts[1]} "
                f"balance_by_family={balance_by_family} "
                f"families={fam_counts}"
            )
            self._logged_train_family_staged_queryfree = True
        return {"loss": total_loss_metric.detach()}

    def _training_step_once(
        self,
        data,
        i,
        data_is_one_hot=False,
        sparse_noisy_data_override=None,
        forced_block_id=None,
        t_override=None,
        query_edge_override=None,
        query_batch_override=None,
        skip_auxiliary_structure_losses=False,
        return_query_outputs=False,
        record_structure_probe=True,
    ):
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
        if "two_hop_structure_lookup" not in sparse_noisy_data:
            training_lookup = self._build_training_two_hop_lookup(
                sparse_noisy_data,
                num_nodes=int(data.x.shape[0]),
            )
            if training_lookup is not None:
                sparse_noisy_data["two_hop_structure_lookup"] = training_lookup
        profile_efficiency = self._profile_training_efficiency_enabled()
        query_build_start = self._profile_time(profile_efficiency)
        # ----- 随机选边用途 2/2：构造 query 边 -----
        # 遵循「全部可能边」概念：每次只对全部可能边中的一块做预测。query = 该块（比例 k=edge_fraction）。
        # 异质图：按族在族内可能边上采样 k*num_fam_possible_edges；同质图：全局 k*num_edges。
        # comp = 加噪后的显式边 + 本块 query 边；loss 仅在本块 query 上计算。
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
            triu_query_edge_index, query_edge_batch = (
                query_edge_override,
                query_batch_override,
            )
            if self.use_block_query and triu_query_edge_index is None:
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
        query_build_done = self._profile_time(profile_efficiency)

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
        comp_graph_done = self._profile_time(profile_efficiency)

        # pass sparse comp_graph to dense comp_graph for ease calculation
        sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
        sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
        sparse_pred = self.forward(sparse_noisy_data)
        forward_done = self._profile_time(profile_efficiency)

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
        if record_structure_probe:
            self._record_query_topk_structure_probe(sparse_pred, true_data, data)

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

        family_role_w_base = float(
            getattr(self.cfg.model, "family_role_loss_weight", 0.0) or 0.0
        )
        family_role_warmup = int(
            getattr(self.cfg.model, "family_role_loss_warmup_epochs", 0) or 0
        )
        if family_role_warmup > 0:
            family_role_factor = min(
                float(self.current_epoch + 1) / float(family_role_warmup),
                1.0,
            )
        else:
            family_role_factor = 1.0
        family_role_w = family_role_w_base * family_role_factor
        if family_role_w > 0:
            loss_family_role, family_role_stats = (
                self._compute_family_role_ranking_loss(
                    sparse_pred, true_data, data, sparse_noisy_data
                )
            )
            loss = loss + family_role_w * loss_family_role
            if not hasattr(self, "_epoch_family_role_loss_sum"):
                self._epoch_family_role_loss_sum = 0.0
                self._epoch_family_role_loss_count = 0
                self._epoch_family_role_weighted_sum = 0.0
                self._epoch_family_role_reliability_sum = 0.0
                self._epoch_family_role_active_sum = 0.0
            self._epoch_family_role_loss_sum += float(
                loss_family_role.detach().cpu()
            )
            self._epoch_family_role_loss_count += 1
            self._epoch_family_role_weighted_sum += float(
                (family_role_w * loss_family_role).detach().cpu()
            )
            self._epoch_family_role_reliability_sum += float(
                family_role_stats.get("reliability", 0.0)
            )
            self._epoch_family_role_active_sum += float(
                family_role_stats.get("active_families", 0.0)
            )
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/family_role_loss",
                    loss_family_role.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "train/family_role_weight",
                    torch.as_tensor(family_role_w, device=self.device),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        family_count_w_base = float(
            getattr(self.cfg.model, "family_count_loss_weight", 0.0) or 0.0
        )
        family_count_w = self._warmup_weight(
            family_count_w_base,
            int(getattr(self.cfg.model, "family_count_loss_warmup_epochs", 0) or 0),
            self._auxiliary_time_factor(
                sparse_noisy_data,
                float(getattr(self.cfg.model, "family_count_loss_t_power", 0.0) or 0.0),
            ),
        )
        if family_count_w > 0 and not skip_auxiliary_structure_losses:
            loss_family_count, family_count_stats = self._compute_family_count_loss(
                sparse_pred, true_data, data
            )
            loss = loss + family_count_w * loss_family_count
            if not hasattr(self, "_epoch_family_count_loss_sum"):
                self._epoch_family_count_loss_sum = 0.0
                self._epoch_family_count_loss_count = 0
                self._epoch_family_count_weighted_sum = 0.0
                self._epoch_family_count_active_sum = 0.0
            self._epoch_family_count_loss_sum += float(
                loss_family_count.detach().cpu()
            )
            self._epoch_family_count_loss_count += 1
            self._epoch_family_count_weighted_sum += float(
                (family_count_w * loss_family_count).detach().cpu()
            )
            self._epoch_family_count_active_sum += float(
                family_count_stats.get("active_families", 0.0)
            )
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/family_count_loss",
                    loss_family_count.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "train/family_count_weight",
                    torch.as_tensor(family_count_w, device=self.device),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        degree_pair_w_base = float(
            getattr(self.cfg.model, "degree_pair_dist_loss_weight", 0.0) or 0.0
        )
        degree_pair_w = self._warmup_weight(
            degree_pair_w_base,
            int(getattr(self.cfg.model, "degree_pair_dist_loss_warmup_epochs", 0) or 0),
            self._auxiliary_time_factor(
                sparse_noisy_data,
                float(getattr(self.cfg.model, "degree_pair_dist_loss_t_power", 0.0) or 0.0),
            ),
        )
        if degree_pair_w > 0 and not skip_auxiliary_structure_losses:
            loss_degree_pair, degree_pair_stats = (
                self._compute_degree_pair_distribution_loss(sparse_pred, data, true_data)
            )
            loss = loss + degree_pair_w * loss_degree_pair
            if not hasattr(self, "_epoch_degree_pair_dist_loss_sum"):
                self._epoch_degree_pair_dist_loss_sum = 0.0
                self._epoch_degree_pair_dist_loss_count = 0
                self._epoch_degree_pair_dist_weighted_sum = 0.0
                self._epoch_degree_pair_dist_active_sum = 0.0
            self._epoch_degree_pair_dist_loss_sum += float(
                loss_degree_pair.detach().cpu()
            )
            self._epoch_degree_pair_dist_loss_count += 1
            self._epoch_degree_pair_dist_weighted_sum += float(
                (degree_pair_w * loss_degree_pair).detach().cpu()
            )
            self._epoch_degree_pair_dist_active_sum += float(
                degree_pair_stats.get("active_families", 0.0)
            )
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/degree_pair_dist_loss",
                    loss_degree_pair.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "train/degree_pair_dist_weight",
                    torch.as_tensor(degree_pair_w, device=self.device),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        closure_rank_w_base = float(
            getattr(self.cfg.model, "closure_rank_loss_weight", 0.0) or 0.0
        )
        closure_rank_w = self._warmup_weight(
            closure_rank_w_base,
            int(getattr(self.cfg.model, "closure_rank_loss_warmup_epochs", 0) or 0),
            self._auxiliary_time_factor(
                sparse_noisy_data,
                float(getattr(self.cfg.model, "closure_rank_loss_t_power", 1.0) or 0.0),
            ),
        )
        if closure_rank_w > 0 and not skip_auxiliary_structure_losses:
            loss_closure_rank, closure_rank_stats = self._compute_closure_ranking_loss(
                sparse_pred, data, num_nodes=data.x.shape[0]
            )
            loss = loss + closure_rank_w * loss_closure_rank
            if not hasattr(self, "_epoch_closure_rank_loss_sum"):
                self._epoch_closure_rank_loss_sum = 0.0
                self._epoch_closure_rank_loss_count = 0
                self._epoch_closure_rank_weighted_sum = 0.0
                self._epoch_closure_rank_active_sum = 0.0
                self._epoch_closure_rank_pairs_sum = 0.0
            self._epoch_closure_rank_loss_sum += float(
                loss_closure_rank.detach().cpu()
            )
            self._epoch_closure_rank_loss_count += 1
            self._epoch_closure_rank_weighted_sum += float(
                (closure_rank_w * loss_closure_rank).detach().cpu()
            )
            self._epoch_closure_rank_active_sum += float(
                closure_rank_stats.get("active_families", 0.0)
            )
            self._epoch_closure_rank_pairs_sum += float(
                closure_rank_stats.get("pairs", 0.0)
            )
            if step_idx % self.log_every_steps == 0:
                self.log(
                    "train/closure_rank_loss",
                    loss_closure_rank.detach(),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "train/closure_rank_weight",
                    torch.as_tensor(closure_rank_w, device=self.device),
                    on_step=True,
                    prog_bar=False,
                    sync_dist=True,
                )

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

        loss_build_done = self._profile_time(profile_efficiency)
        out = {
            "loss": loss,
            "query_edges": int(triu_query_edge_index.shape[1]),
            "perf": {
                "query_build_s": query_build_done - query_build_start,
                "comp_graph_s": comp_graph_done - query_build_done,
                "hgt_forward_s": forward_done - comp_graph_done,
                "loss_build_s": loss_build_done - forward_done,
            },
        }
        if return_query_outputs:
            out["pred_edge_attr"] = sparse_pred.edge_attr
            out["pred_edge_index"] = sparse_pred.edge_index
            out["true_edge_attr"] = true_data.edge_attr
        return out

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
        self._epoch_family_role_loss_sum = 0.0
        self._epoch_family_role_loss_count = 0
        self._epoch_family_role_weighted_sum = 0.0
        self._epoch_family_role_reliability_sum = 0.0
        self._epoch_family_role_active_sum = 0.0
        self._epoch_family_count_loss_sum = 0.0
        self._epoch_family_count_loss_count = 0
        self._epoch_family_count_weighted_sum = 0.0
        self._epoch_family_count_active_sum = 0.0
        self._epoch_degree_pair_dist_loss_sum = 0.0
        self._epoch_degree_pair_dist_loss_count = 0
        self._epoch_degree_pair_dist_weighted_sum = 0.0
        self._epoch_degree_pair_dist_active_sum = 0.0
        self._epoch_closure_rank_loss_sum = 0.0
        self._epoch_closure_rank_loss_count = 0
        self._epoch_closure_rank_weighted_sum = 0.0
        self._epoch_closure_rank_active_sum = 0.0
        self._epoch_closure_rank_pairs_sum = 0.0
        self._epoch_diag_count = 0
        self._epoch_diag_t_sum = 0.0
        self._epoch_diag_query_pos_ratio_sum = 0.0
        self._epoch_diag_pred_pos_rate_sum = 0.0
        self._epoch_diag_pos_p_exist_sum = 0.0
        self._epoch_diag_pos_p_exist_count = 0
        self._epoch_diag_neg_p_exist_sum = 0.0
        self._epoch_diag_neg_p_exist_count = 0
        self._epoch_struct_probe_count = 0
        self._epoch_struct_probe_topk_precision_sum = 0.0
        self._epoch_struct_probe_topk_recall_sum = 0.0
        self._epoch_struct_probe_family_l1_sum = 0.0
        self._epoch_struct_probe_degree_pair_js_sum = 0.0
        self._epoch_struct_probe_degree_pair_js_count = 0
        self._epoch_struct_probe_closure_mass_rel_sum = 0.0
        self._epoch_struct_probe_closure_mass_rel_count = 0
        self._epoch_struct_probe_closure_mean_gap_sum = 0.0
        self._epoch_struct_probe_closure_mean_gap_count = 0
        self._epoch_struct_probe_k_sum = 0.0

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
        family_role_count = int(getattr(self, "_epoch_family_role_loss_count", 0) or 0)
        if family_role_count > 0:
            family_role_loss = float(getattr(self, "_epoch_family_role_loss_sum", 0.0)) / float(family_role_count)
            family_role_wloss = float(getattr(self, "_epoch_family_role_weighted_sum", 0.0)) / float(family_role_count)
            family_role_rel = float(getattr(self, "_epoch_family_role_reliability_sum", 0.0)) / float(family_role_count)
            family_role_active = float(getattr(self, "_epoch_family_role_active_sum", 0.0)) / float(family_role_count)
        else:
            family_role_loss = 0.0
            family_role_wloss = 0.0
            family_role_rel = 0.0
            family_role_active = 0.0
        family_count_count = int(getattr(self, "_epoch_family_count_loss_count", 0) or 0)
        if family_count_count > 0:
            family_count_loss = float(getattr(self, "_epoch_family_count_loss_sum", 0.0)) / float(family_count_count)
            family_count_wloss = float(getattr(self, "_epoch_family_count_weighted_sum", 0.0)) / float(family_count_count)
            family_count_active = float(getattr(self, "_epoch_family_count_active_sum", 0.0)) / float(family_count_count)
        else:
            family_count_loss = 0.0
            family_count_wloss = 0.0
            family_count_active = 0.0
        degree_pair_count = int(getattr(self, "_epoch_degree_pair_dist_loss_count", 0) or 0)
        if degree_pair_count > 0:
            degree_pair_loss = float(getattr(self, "_epoch_degree_pair_dist_loss_sum", 0.0)) / float(degree_pair_count)
            degree_pair_wloss = float(getattr(self, "_epoch_degree_pair_dist_weighted_sum", 0.0)) / float(degree_pair_count)
            degree_pair_active = float(getattr(self, "_epoch_degree_pair_dist_active_sum", 0.0)) / float(degree_pair_count)
        else:
            degree_pair_loss = 0.0
            degree_pair_wloss = 0.0
            degree_pair_active = 0.0
        closure_rank_count = int(getattr(self, "_epoch_closure_rank_loss_count", 0) or 0)
        if closure_rank_count > 0:
            closure_rank_loss = float(getattr(self, "_epoch_closure_rank_loss_sum", 0.0)) / float(closure_rank_count)
            closure_rank_wloss = float(getattr(self, "_epoch_closure_rank_weighted_sum", 0.0)) / float(closure_rank_count)
            closure_rank_active = float(getattr(self, "_epoch_closure_rank_active_sum", 0.0)) / float(closure_rank_count)
            closure_rank_pairs = float(getattr(self, "_epoch_closure_rank_pairs_sum", 0.0)) / float(closure_rank_count)
        else:
            closure_rank_loss = 0.0
            closure_rank_wloss = 0.0
            closure_rank_active = 0.0
            closure_rank_pairs = 0.0
        struct_probe_count = int(getattr(self, "_epoch_struct_probe_count", 0) or 0)
        if struct_probe_count > 0:
            struct_probe_precision = float(
                getattr(self, "_epoch_struct_probe_topk_precision_sum", 0.0)
            ) / float(struct_probe_count)
            struct_probe_recall = float(
                getattr(self, "_epoch_struct_probe_topk_recall_sum", 0.0)
            ) / float(struct_probe_count)
            struct_probe_family_l1 = float(
                getattr(self, "_epoch_struct_probe_family_l1_sum", 0.0)
            ) / float(struct_probe_count)
            struct_probe_k = float(
                getattr(self, "_epoch_struct_probe_k_sum", 0.0)
            ) / float(struct_probe_count)
        else:
            struct_probe_precision = 0.0
            struct_probe_recall = 0.0
            struct_probe_family_l1 = 0.0
            struct_probe_k = 0.0
        struct_probe_dp_count = int(
            getattr(self, "_epoch_struct_probe_degree_pair_js_count", 0) or 0
        )
        struct_probe_degree_pair_js = (
            float(getattr(self, "_epoch_struct_probe_degree_pair_js_sum", 0.0))
            / float(struct_probe_dp_count)
            if struct_probe_dp_count > 0
            else 0.0
        )
        struct_probe_closure_count = int(
            getattr(self, "_epoch_struct_probe_closure_mass_rel_count", 0) or 0
        )
        struct_probe_closure_mass_rel = (
            float(getattr(self, "_epoch_struct_probe_closure_mass_rel_sum", 0.0))
            / float(struct_probe_closure_count)
            if struct_probe_closure_count > 0
            else 0.0
        )
        struct_probe_gap_count = int(
            getattr(self, "_epoch_struct_probe_closure_mean_gap_count", 0) or 0
        )
        struct_probe_closure_mean_gap = (
            float(getattr(self, "_epoch_struct_probe_closure_mean_gap_sum", 0.0))
            / float(struct_probe_gap_count)
            if struct_probe_gap_count > 0
            else 0.0
        )
        edge_nll = epoch_loss.get("train_epoch/NLL", -1)
        any_aux = (
            degree_count > 0
            or count_loss_count > 0
            or closure_count > 0
            or family_role_count > 0
            or family_count_count > 0
            or degree_pair_count > 0
            or closure_rank_count > 0
        )
        total_train = edge_nll + degree_wloss + edge_count_wloss + closure_wloss + family_role_wloss + family_count_wloss + degree_pair_wloss + closure_rank_wloss if isinstance(edge_nll, (int, float)) and edge_nll >= 0 else -1
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
        if family_role_count > 0:
            epoch_loss["train_epoch/family_role_loss"] = family_role_loss
            epoch_loss["train_epoch/family_role_weighted"] = family_role_wloss
            epoch_loss["train_epoch/family_role_reliability"] = family_role_rel
            epoch_loss["train_epoch/family_role_active_families"] = family_role_active
        if family_count_count > 0:
            epoch_loss["train_epoch/family_count_loss"] = family_count_loss
            epoch_loss["train_epoch/family_count_weighted"] = family_count_wloss
            epoch_loss["train_epoch/family_count_active_families"] = family_count_active
        if degree_pair_count > 0:
            epoch_loss["train_epoch/degree_pair_dist_loss"] = degree_pair_loss
            epoch_loss["train_epoch/degree_pair_dist_weighted"] = degree_pair_wloss
            epoch_loss["train_epoch/degree_pair_dist_active_families"] = degree_pair_active
        if closure_rank_count > 0:
            epoch_loss["train_epoch/closure_rank_loss"] = closure_rank_loss
            epoch_loss["train_epoch/closure_rank_weighted"] = closure_rank_wloss
            epoch_loss["train_epoch/closure_rank_active_families"] = closure_rank_active
            epoch_loss["train_epoch/closure_rank_pairs"] = closure_rank_pairs
        if struct_probe_count > 0:
            epoch_loss["train_epoch/struct_probe_topk_precision"] = struct_probe_precision
            epoch_loss["train_epoch/struct_probe_topk_recall"] = struct_probe_recall
            epoch_loss["train_epoch/struct_probe_family_l1"] = struct_probe_family_l1
            epoch_loss["train_epoch/struct_probe_degree_pair_js"] = struct_probe_degree_pair_js
            epoch_loss["train_epoch/struct_probe_closure_mass_rel"] = struct_probe_closure_mass_rel
            epoch_loss["train_epoch/struct_probe_closure_mean_gap"] = struct_probe_closure_mean_gap
            epoch_loss["train_epoch/struct_probe_k"] = struct_probe_k
        if any_aux and total_train >= 0:
            epoch_loss["train_epoch/total_with_aux"] = total_train
        # neg_acc=-1 表示 query 内无负样本（ideal 模式下常见），显示为 N/A
        neg_str = f"{neg_acc:.4f}" if isinstance(neg_acc, (int, float)) and neg_acc >= 0 else "N/A"
        degree_str = f" -- degree_L1: {degree_l1:.6f} -- degree_wloss: {degree_wloss:.6f}" if degree_count > 0 else ""
        edge_count_str = f" -- edge_count_L1: {edge_count_l1:.6f} -- edge_count_wloss: {edge_count_wloss:.6f}" if count_loss_count > 0 else ""
        closure_str = f" -- closure_L1: {closure_l1:.6f} -- closure_wloss: {closure_wloss:.6f}" if closure_count > 0 else ""
        family_role_str = (
            f" -- family_role_loss: {family_role_loss:.6f}"
            f" -- family_role_wloss: {family_role_wloss:.6f}"
            f" -- family_role_rel: {family_role_rel:.3f}"
            f" -- family_role_fams: {family_role_active:.1f}"
        ) if family_role_count > 0 else ""
        family_count_str = (
            f" -- family_count: {family_count_loss:.6f}"
            f" -- family_count_wloss: {family_count_wloss:.6f}"
            f" -- family_count_fams: {family_count_active:.1f}"
        ) if family_count_count > 0 else ""
        degree_pair_str = (
            f" -- degree_pair_dist: {degree_pair_loss:.6f}"
            f" -- degree_pair_wloss: {degree_pair_wloss:.6f}"
            f" -- degree_pair_fams: {degree_pair_active:.1f}"
        ) if degree_pair_count > 0 else ""
        closure_rank_str = (
            f" -- closure_rank: {closure_rank_loss:.6f}"
            f" -- closure_rank_wloss: {closure_rank_wloss:.6f}"
            f" -- closure_rank_fams: {closure_rank_active:.1f}"
            f" -- closure_rank_pairs: {closure_rank_pairs:.0f}"
        ) if closure_rank_count > 0 else ""
        struct_probe_str = (
            f" -- struct_probe[topK_p={struct_probe_precision:.4f}"
            f", famL1={struct_probe_family_l1:.4f}"
            f", dpJS={struct_probe_degree_pair_js:.4f}"
            f", closureRel={struct_probe_closure_mass_rel:.4f}"
            f", closureGap={struct_probe_closure_mean_gap:.4f}]"
        ) if struct_probe_count > 0 else ""
        total_str = f" -- total_train: {total_train:.4f}" if any_aux and total_train >= 0 else ""
        diag_str = (
            f" -- t_norm: {diag_t:.3f} -- q_pos: {diag_qpos:.4f} -- pred_pos: {diag_pred_pos:.4f} "
            f"-- pos_p: {diag_pos_p:.4f} -- neg_p: {diag_neg_p:.4f}"
        ) if diag_count > 0 else ""
        gate_str = ""
        if bool(getattr(self.cfg.model, "use_query_context_gate", False)):
            gate_values = [
                float(torch.sigmoid(layer.query_context_gate_logit).mean().detach().cpu())
                for layer in getattr(self.model, "gcs", [])
                if hasattr(layer, "query_context_gate_logit")
            ]
            if gate_values:
                gate_str = " -- query_gate: " + "/".join(
                    f"{value:.3f}" for value in gate_values
                )
                for layer_idx, value in enumerate(gate_values):
                    epoch_loss[f"train_epoch/query_gate_layer_{layer_idx}"] = value
        if bool(getattr(self.cfg.model, "use_two_hop_structure", False)):
            base_residual = float(
                getattr(self.model, "last_two_hop_base_residual_mean", 0.0)
            )
            structure_factor = float(
                getattr(self.model, "last_two_hop_scale_factor_mean", 1.0)
            )
            effective_scale = float(
                getattr(self.model, "last_two_hop_effective_scale_mean", 0.0)
            )
            effective_residual = float(
                getattr(
                    self.model,
                    "last_two_hop_effective_residual_mean",
                    getattr(self.model, "last_two_hop_residual_mean", 0.0),
                )
            )
            gate_str += (
                f" -- two_hop_factor: {structure_factor:.6f}"
                f" -- two_hop_base_residual: {base_residual:.6f}"
                f" -- two_hop_effective_scale: {effective_scale:.6f}"
                f" -- two_hop_effective_residual: {effective_residual:.6f}"
            )
            epoch_loss[
                "train_epoch/two_hop_base_residual_mean"
            ] = base_residual
            epoch_loss[
                "train_epoch/two_hop_scale_factor_mean"
            ] = structure_factor
            epoch_loss[
                "train_epoch/two_hop_effective_scale_mean"
            ] = effective_scale
            epoch_loss[
                "train_epoch/two_hop_effective_residual_mean"
            ] = effective_residual
        if bool(
            getattr(self.cfg.model, "use_endpoint_role_residual", False)
        ):
            role_base_residual = float(
                getattr(
                    self.model,
                    "last_endpoint_role_base_residual_mean",
                    0.0,
                )
            )
            role_effective_scale = float(
                getattr(
                    self.model,
                    "last_endpoint_role_effective_scale_mean",
                    0.0,
                )
            )
            role_effective_residual = float(
                getattr(
                    self.model,
                    "last_endpoint_role_effective_residual_mean",
                    0.0,
                )
            )
            gate_str += (
                f" -- endpoint_role_base_residual: {role_base_residual:.6f}"
                f" -- endpoint_role_effective_scale: {role_effective_scale:.6f}"
                f" -- endpoint_role_effective_residual: {role_effective_residual:.6f}"
            )
            epoch_loss[
                "train_epoch/endpoint_role_base_residual_mean"
            ] = role_base_residual
            epoch_loss[
                "train_epoch/endpoint_role_effective_scale_mean"
            ] = role_effective_scale
            epoch_loss[
                "train_epoch/endpoint_role_effective_residual_mean"
            ] = role_effective_residual

        self.print(
            f"Epoch {self.current_epoch} finished: "
            f"exist_BCE: {exist_bce:.4f} -- subtype_CE: {subtype_ce:.4f} -- "
            f"pos_acc: {pos_acc:.4f} -- neg_acc: {neg_str}"
            f"{degree_str}{edge_count_str}{closure_str}{family_role_str}"
            f"{family_count_str}{degree_pair_str}{closure_rank_str}"
            f"{struct_probe_str}{total_str}{diag_str}{gate_str}"
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
            "train_epoch/family_count_loss",
            "train_epoch/family_count_weighted",
            "train_epoch/degree_pair_dist_loss",
            "train_epoch/degree_pair_dist_weighted",
            "train_epoch/closure_rank_loss",
            "train_epoch/closure_rank_weighted",
            "train_epoch/struct_probe_topk_precision",
            "train_epoch/struct_probe_topk_recall",
            "train_epoch/struct_probe_family_l1",
            "train_epoch/struct_probe_degree_pair_js",
            "train_epoch/struct_probe_closure_mass_rel",
            "train_epoch/struct_probe_closure_mean_gap",
            "train_epoch/struct_probe_k",
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
        self._val_same_type_forward_metrics = []
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
        with torch.no_grad():
            if bool(getattr(self.cfg.model, "train_family_staged_queryfree", False)):
                out = self._training_step_family_staged_queryfree(
                    data,
                    i,
                    do_backward=False,
                    log_metrics=False,
                )
                if out is None:
                    return None
                loss = out["loss"].detach()
            else:
                data = self.dataset_info.to_one_hot(data)
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

        def _mean_nested_metrics(items):
            if not items:
                return {}
            out = {}
            type_names = sorted(set().union(*(item.keys() for item in items)))
            for type_name in type_names:
                out[type_name] = {}
                metric_names = sorted(
                    set().union(
                        *(
                            (item.get(type_name, {}) or {}).keys()
                            for item in items
                        )
                    )
                )
                for metric_name in metric_names:
                    vals = []
                    for item in items:
                        value = (item.get(type_name, {}) or {}).get(metric_name)
                        if value is None:
                            continue
                        try:
                            value = float(value)
                        except Exception:
                            continue
                        if math.isfinite(value):
                            vals.append(value)
                    out[type_name][metric_name] = (
                        float(np.mean(vals)) if vals else None
                    )
            return out

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
        if getattr(self, "local_rank", 0) == 0:
            forward_metrics = _mean_nested_metrics(
                getattr(self, "_val_same_type_forward_metrics", [])
            )
            if forward_metrics:
                table_str = self._format_same_type_internal_metrics_table(
                    forward_metrics,
                    prefix=(
                        "Validation same-type forward top-K reconstruction "
                        f"metrics at epoch {self.current_epoch}"
                    ),
                )
                if table_str:
                    print(table_str)
                forward_path = os.path.join(
                    os.getcwd(),
                    f"val_same_type_forward_metrics_epoch{self.current_epoch}.json",
                )
                with open(forward_path, "w", encoding="utf-8") as file:
                    json.dump(forward_metrics, file, ensure_ascii=False, indent=2)
                print(f"Validation same-type forward metrics saved to {forward_path}")
                if wandb.run:
                    flat_metrics = {}
                    for type_name, metrics in forward_metrics.items():
                        for metric_name, value in (metrics or {}).items():
                            if value is not None and math.isfinite(float(value)):
                                flat_metrics[
                                    f"val_same_type_forward/{type_name}/{metric_name}"
                                ] = float(value)
                    if flat_metrics:
                        wandb.log(flat_metrics, step=int(self.current_epoch), commit=False)
        self._last_val_epoch_nll = val_nll_value
        self._maybe_run_validation_sampling()

    @torch.no_grad()
    def _maybe_run_validation_sampling(self) -> None:
        if not bool(getattr(self.cfg.general, "enable_val_sampling", False)):
            return
        sample_every = max(1, int(getattr(self.cfg.general, "sample_every_val", 1) or 1))
        current_epoch = int(getattr(self, "current_epoch", 0))
        if current_epoch % sample_every != 0:
            return
        if getattr(self, "local_rank", 0) != 0:
            return

        n_to_generate = max(1, int(getattr(self.cfg.general, "samples_to_generate", 1) or 1))
        n_to_generate = min(n_to_generate, max(1, int(getattr(self.cfg.train, "batch_size", 1) or 1)))
        self.print(
            f"Validation sampling: generating {n_to_generate} sample(s) at epoch {current_epoch}"
        )
        use_fixed_nodes = bool(getattr(self.cfg.general, "cond_edge_gen_fixed_nodes", False))
        use_sample_nodes = bool(getattr(self.cfg.general, "cond_edge_gen_sample_nodes", False))
        if use_fixed_nodes and hasattr(self.dataset_info, "datamodule"):
            use_test_graph = getattr(self.cfg.general, "cond_edge_gen_use_test_graph", True)
            if use_test_graph and hasattr(self.dataset_info.datamodule, "test_dataset"):
                fixed_single = self.dataset_info.datamodule.test_dataset[0].clone()
            else:
                fixed_single = self.dataset_info.datamodule.train_dataset[0].clone()
            fixed_single = self._permute_fixed_graph_within_node_types(
                fixed_single,
                getattr(self.cfg.general, "fixed_node_type_permutation_seed", None),
            )
            fixed_single = self.dataset_info.to_one_hot(fixed_single)
            from torch_geometric.data import Batch

            fixed_batch = Batch.from_data_list(
                [fixed_single.clone() for _ in range(n_to_generate)]
            ).to(self.device)
            samples = self.sample_batch_fixed_nodes(
                fixed_batch,
                keep_chain=0,
                number_chain_steps=self.number_chain_steps,
                save_final=n_to_generate,
            )
        else:
            sample_num_nodes = getattr(self.cfg.general, "sample_num_nodes", None)
            if sample_num_nodes is not None:
                num_nodes = sample_num_nodes
            elif use_sample_nodes and hasattr(self.dataset_info, "datamodule"):
                train_data = self.dataset_info.datamodule.train_dataset[0]
                num_nodes = getattr(
                    train_data,
                    "num_nodes",
                    train_data.x.shape[0] if hasattr(train_data, "x") else 20,
                )
            else:
                num_nodes = 20
            samples = self.sample_batch(
                batch_id=0,
                batch_size=n_to_generate,
                num_nodes=num_nodes,
                save_final=n_to_generate,
                keep_chain=0,
                number_chain_steps=self.number_chain_steps,
            )

        to_log, _ = self.val_sampling_metrics.compute_all_metrics(
            samples, current_epoch, local_rank=self.local_rank
        )
        with open(os.path.join(os.getcwd(), f"val_epoch{current_epoch}.json"), "w") as file:
            json.dump(to_log, file)
        internal_metrics = self._compute_same_type_internal_sampling_metrics(samples)
        if internal_metrics:
            internal_path = os.path.join(
                os.getcwd(),
                f"val_same_type_internal_metrics_epoch{current_epoch}.json",
            )
            with open(internal_path, "w", encoding="utf-8") as file:
                json.dump(internal_metrics, file, ensure_ascii=False, indent=2)
            table_str = self._format_same_type_internal_metrics_table(
                internal_metrics,
                prefix="Validation same-type internal metrics",
            )
            if table_str:
                print(table_str)
            self.print(f"Validation same-type internal metrics saved to {internal_path}")
        self.val_sampling_metrics.reset()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

    def _current_sampling_seed(self) -> int:
        active_seed = getattr(self, "_active_test_sampling_seed", None)
        if active_seed is not None:
            return int(active_seed)
        return int(getattr(self.cfg.train, "seed", 0) or 0)

    def _explicit_test_sampling_seeds(self):
        configured = getattr(
            self.cfg.general, "test_sampling_seeds", None
        )
        if configured is None:
            return None
        seeds = [int(seed) for seed in list(configured)]
        if not seeds:
            return None
        test_variance = int(
            getattr(self.cfg.general, "test_variance", 1)
        )
        if len(seeds) != test_variance:
            raise ValueError(
                "general.test_sampling_seeds must contain exactly "
                f"general.test_variance={test_variance} seeds, got {seeds}."
            )
        final_samples = int(
            getattr(
                self.cfg.general,
                "final_model_samples_to_generate",
                1,
            )
        )
        if final_samples != 1:
            raise ValueError(
                "Explicit paired test_sampling_seeds currently require "
                "general.final_model_samples_to_generate=1."
            )
        num_devices = max(
            int(getattr(self._trainer, "num_devices", 1)), 1
        )
        if num_devices != 1:
            raise ValueError(
                "Explicit paired test_sampling_seeds currently require "
                "single-device testing."
            )
        return seeds

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
            explicit_sampling_seeds = self._explicit_test_sampling_seeds()
            if explicit_sampling_seeds is not None:
                n_to_generate = len(explicit_sampling_seeds)
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

            generated_count = 0
            try:
                while remaining > 0:
                    if explicit_sampling_seeds is not None:
                        to_generate = 1
                        sample_seed = explicit_sampling_seeds[
                            generated_count
                        ]
                        pl.seed_everything(sample_seed, workers=True)
                        self._active_test_sampling_seed = sample_seed
                        self.print(
                            f"[TEST-SEED] sample_index={generated_count} "
                            f"seed={sample_seed}"
                        )
                    else:
                        to_generate = min(remaining, batch_size)
                    if use_fixed_nodes and hasattr(self.dataset_info, "datamodule"):
                        use_test_graph = getattr(self.cfg.general, "cond_edge_gen_use_test_graph", True)
                        if use_test_graph and hasattr(self.dataset_info.datamodule, "test_dataset"):
                            fixed_single = self.dataset_info.datamodule.test_dataset[0].clone()
                        else:
                            fixed_single = self.dataset_info.datamodule.train_dataset[0].clone()
                        fixed_single = self._permute_fixed_graph_within_node_types(
                            fixed_single,
                            getattr(
                                self.cfg.general,
                                "fixed_node_type_permutation_seed",
                                None,
                            ),
                        )
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
                            batch_id=generated_count,
                            batch_size=to_generate,
                            num_nodes=num_nodes,
                            save_final=to_generate,
                            keep_chain=0,
                            number_chain_steps=self.number_chain_steps,
                        )
                    sample_chunks.append(sampled_batch)
                    remaining -= to_generate
                    generated_count += to_generate
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            finally:
                self._active_test_sampling_seed = None

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
                if getattr(self, "local_rank", 0) == 0:
                    cur_internal_metrics = self._compute_same_type_internal_sampling_metrics(
                        split_samples[idx]
                    )
                    table_str = self._format_same_type_internal_metrics_table(
                        cur_internal_metrics,
                        prefix=f"Same-type internal metrics for sampling {idx}",
                    )
                    if table_str:
                        print(table_str)
            to_log = {k: (np.mean(v), np.std(v)) for k, v in to_log.items()}
            print(f"For overall {test_variance} samplings, we have: ")
            print(to_log)
        if getattr(self, "local_rank", 0) == 0:
            internal_metrics = self._compute_same_type_internal_sampling_metrics(samples)
            if internal_metrics:
                internal_path = os.path.join(
                    os.getcwd(),
                    f"test_same_type_internal_metrics_epoch{self.current_epoch}.json",
                )
                with open(internal_path, "w", encoding="utf-8") as file:
                    json.dump(internal_metrics, file, ensure_ascii=False, indent=2)
                table_str = self._format_same_type_internal_metrics_table(
                    internal_metrics,
                    prefix=f"Overall same-type internal metrics for {test_variance} samplings",
                )
                if table_str:
                    print(table_str)
                print(f"Same-type internal metrics saved to {internal_path}")
                if wandb.run:
                    flat_metrics = {}
                    for type_name, metrics in internal_metrics.items():
                        if not isinstance(metrics, dict):
                            continue
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                                flat_metrics[
                                    f"test_same_type_internal/{type_name}/{metric_name}"
                                ] = float(value)
                    if flat_metrics:
                        wandb.log(flat_metrics, step=int(self.current_epoch), commit=False)
        if not generated_path:
            self._compute_and_save_test_intermediate_metrics()
        self.print("Test sampling metrics computed.")

    def _reference_graph_for_same_type_metrics(self):
        datamodule = getattr(self.dataset_info, "datamodule", None)
        if datamodule is None:
            return None
        for dataset_name in ("test_dataset", "train_dataset", "val_dataset"):
            dataset = getattr(datamodule, dataset_name, None)
            if dataset is not None and len(dataset) > 0:
                return dataset[0]
        return None

    def _node_type_ranges(self):
        type_offsets = getattr(self.dataset_info, "type_offsets", {}) or {}
        if not type_offsets:
            return []
        sorted_types = sorted(
            ((str(name), int(offset)) for name, offset in type_offsets.items()),
            key=lambda item: item[1],
        )
        ranges = []
        for idx, (type_name, offset) in enumerate(sorted_types):
            next_offset = (
                int(sorted_types[idx + 1][1])
                if idx + 1 < len(sorted_types)
                else int(self.out_dims.X)
            )
            ranges.append((type_name, int(offset), int(next_offset)))
        return ranges

    @staticmethod
    def _json_metric_value(value):
        try:
            value = float(value)
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _same_type_internal_graph_stats(self, data):
        try:
            import networkx as nx
        except Exception as exc:
            if getattr(self, "local_rank", 0) == 0:
                print(f"[WARN] networkx unavailable for same-type metrics: {exc}")
            return {}
        if data is None:
            return {}
        node_state = getattr(data, "x", None)
        if node_state is None:
            node_state = getattr(data, "node", None)
        if node_state is None:
            return {}
        if node_state.dim() > 1:
            node_state = node_state.argmax(dim=-1)
        node_state = node_state.long().detach().cpu()
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index = edge_index.long().detach().cpu()
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            edge_labels = torch.ones(edge_index.shape[1], dtype=torch.long)
        elif edge_attr.dim() > 1:
            edge_labels = edge_attr.argmax(dim=-1).long().detach().cpu()
        else:
            edge_labels = edge_attr.long().detach().cpu()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(node_state.shape[0], dtype=torch.long)
        else:
            batch = batch.long().detach().cpu()
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        type_ranges = self._node_type_ranges()
        stats = {}
        for type_name, offset, next_offset in type_ranges:
            per_graph = {
                "nodes": [],
                "edges": [],
                "avg_degree": [],
                "clustering": [],
                "triangles": [],
                "degree_assortativity": [],
            }
            for graph_idx in range(graph_count):
                node_mask = (
                    (batch == int(graph_idx))
                    & (node_state >= int(offset))
                    & (node_state < int(next_offset))
                )
                nodes = torch.where(node_mask)[0]
                graph = nx.Graph()
                graph.add_nodes_from(range(int(nodes.numel())))
                if nodes.numel() > 0 and edge_index.numel() > 0:
                    local_id = {
                        int(node.item()): local_idx
                        for local_idx, node in enumerate(nodes)
                    }
                    src = edge_index[0]
                    dst = edge_index[1]
                    edge_mask = (
                        (edge_labels > 0)
                        & node_mask[src]
                        & node_mask[dst]
                        & (batch[src] == int(graph_idx))
                        & (batch[dst] == int(graph_idx))
                    )
                    for u, v in edge_index[:, edge_mask].t().tolist():
                        if int(u) == int(v):
                            continue
                        graph.add_edge(local_id[int(u)], local_id[int(v)])
                per_graph["nodes"].append(float(graph.number_of_nodes()))
                per_graph["edges"].append(float(graph.number_of_edges()))
                per_graph["avg_degree"].append(
                    float(2.0 * graph.number_of_edges() / max(graph.number_of_nodes(), 1))
                )
                if graph.number_of_nodes() > 0:
                    per_graph["clustering"].append(float(nx.average_clustering(graph)))
                    per_graph["triangles"].append(
                        float(sum(nx.triangles(graph).values()) / 3.0)
                    )
                else:
                    per_graph["clustering"].append(float("nan"))
                    per_graph["triangles"].append(float("nan"))
                try:
                    assortativity = nx.degree_assortativity_coefficient(graph)
                except Exception:
                    assortativity = float("nan")
                per_graph["degree_assortativity"].append(
                    float(assortativity) if assortativity is not None else float("nan")
                )
            stats[type_name] = {}
            for metric_name, values in per_graph.items():
                arr = np.asarray(values, dtype=float)
                if arr.size == 0 or np.all(np.isnan(arr)):
                    stats[type_name][metric_name] = None
                else:
                    stats[type_name][metric_name] = self._json_metric_value(
                        np.nanmean(arr)
                    )
        return stats

    def _compute_same_type_internal_sampling_metrics(self, samples):
        reference = self._reference_graph_for_same_type_metrics()
        return self._compute_same_type_internal_metrics_between(reference, samples)

    def _compute_same_type_internal_metrics_between(self, reference, samples):
        real_stats = self._same_type_internal_graph_stats(reference)
        gen_stats = self._same_type_internal_graph_stats(samples)
        if not real_stats or not gen_stats:
            return {}
        result = {}
        for type_name in sorted(set(real_stats) | set(gen_stats)):
            result[type_name] = {}
            metric_names = sorted(
                set(real_stats.get(type_name, {})) | set(gen_stats.get(type_name, {}))
            )
            for metric_name in metric_names:
                real_value = real_stats.get(type_name, {}).get(metric_name)
                gen_value = gen_stats.get(type_name, {}).get(metric_name)
                result[type_name][f"{metric_name}_real"] = real_value
                result[type_name][f"{metric_name}_gen"] = gen_value
                if real_value is None or gen_value is None:
                    result[type_name][f"{metric_name}_abs_gap"] = None
                else:
                    result[type_name][f"{metric_name}_abs_gap"] = abs(
                        float(gen_value) - float(real_value)
                    )
        return result

    def _same_type_internal_forward_reconstruction_metrics(
        self,
        data_one_hot,
        query_edge_index,
        pred_info,
    ):
        if pred_info is None:
            return {}
        logits = pred_info.get("logits")
        true_labels = pred_info.get("true_labels")
        query_family = pred_info.get("query_family")
        if (
            logits is None
            or true_labels is None
            or query_family is None
            or logits.numel() == 0
        ):
            return {}
        logits = logits.detach()
        true_labels = true_labels.detach().long().reshape(-1)
        query_family = query_family.detach().long().reshape(-1)
        query_edge_index = query_edge_index.detach().long()
        if logits.shape[0] != query_edge_index.shape[1]:
            return {}
        if logits.shape[-1] <= 1:
            exist_scores = logits[:, 0]
        else:
            exist_scores = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        selected_edges = []
        selected_labels = []
        for fam_id_t in torch.unique(query_family):
            fam_mask = query_family == fam_id_t
            pos_count = int(((true_labels > 0) & fam_mask).sum().item())
            if pos_count <= 0:
                continue
            fam_idx = torch.where(fam_mask)[0]
            if fam_idx.numel() == 0:
                continue
            k = min(pos_count, int(fam_idx.numel()))
            top_local = torch.topk(exist_scores[fam_idx], k=k, largest=True).indices
            top_idx = fam_idx[top_local]
            selected_edges.append(query_edge_index[:, top_idx])
            if logits.shape[-1] > 1:
                labels = logits[top_idx, 1:].argmax(dim=-1).long() + 1
            else:
                labels = torch.ones((top_idx.numel(),), dtype=torch.long, device=logits.device)
            selected_labels.append(labels)
        if selected_edges:
            edge_index = torch.cat(selected_edges, dim=1)
            edge_attr = torch.cat(selected_labels, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0,), dtype=torch.long, device=self.device)
        pred_graph = utils.SparsePlaceHolder(
            node=data_one_hot.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data_one_hot.y,
            batch=data_one_hot.batch,
            charge=getattr(data_one_hot, "charge", None),
            ptr=getattr(data_one_hot, "ptr", None),
        )
        return self._compute_same_type_internal_metrics_between(data_one_hot, pred_graph)

    @staticmethod
    def _format_same_type_internal_metrics_table(metrics, prefix="Same-type internal metrics"):
        if not metrics:
            return ""
        rows = []
        header = (
            f"{'type':<10} {'edges real/gen':>18} {'tri real/gen':>18} "
            f"{'clust real/gen':>21} {'assort real/gen':>22}"
        )
        rows.append(prefix)
        rows.append(header)
        rows.append("-" * len(header))

        def _fmt(value, digits=4):
            if value is None:
                return "nan"
            try:
                value = float(value)
            except Exception:
                return "nan"
            if not math.isfinite(value):
                return "nan"
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return f"{value:.{digits}f}"

        for type_name in sorted(metrics.keys()):
            item = metrics.get(type_name, {}) or {}
            rows.append(
                f"{type_name:<10} "
                f"{_fmt(item.get('edges_real'), 1):>8}/{_fmt(item.get('edges_gen'), 1):<8} "
                f"{_fmt(item.get('triangles_real'), 1):>8}/{_fmt(item.get('triangles_gen'), 1):<8} "
                f"{_fmt(item.get('clustering_real')):>9}/{_fmt(item.get('clustering_gen')):<9} "
                f"{_fmt(item.get('degree_assortativity_real')):>10}/{_fmt(item.get('degree_assortativity_gen')):<10}"
            )
        return "\n".join(rows)

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
                    two_hop_structure_lookup=sparse_noisy_data.get(
                        "two_hop_structure_lookup"
                    ),
                    two_hop_scale_factor=sparse_noisy_data.get(
                        "two_hop_scale_factor"
                    ),
                    endpoint_role_scale_factor=sparse_noisy_data.get(
                        "endpoint_role_scale_factor"
                    ),
                    edge_input_residual_scale=sparse_noisy_data.get(
                        "edge_input_residual_scale"
                    ),
                    family_y_table=sparse_noisy_data.get("family_y_t"),
                    edge_struct_features=sparse_noisy_data.get(
                        "edge_struct_features_t"
                    ),
            )
        else:
            return self.model(
                node,
                edge_attr,
                edge_index,
                y,
                batch,
                family_y_table=sparse_noisy_data.get("family_y_t"),
                edge_struct_features=sparse_noisy_data.get(
                    "edge_struct_features_t"
                ),
            )

    def _heterogeneous_metadata_for_edges(
        self, node, edge_attr, edge_index
    ):
        from sparse_diffusion.utils_heterogeneous import (
            extract_heterogeneous_metadata,
        )

        node_for_metadata = node
        if (
            node.dim() > 1
            and self.out_dims.X > 0
            and node.size(-1) > self.out_dims.X
        ):
            node_for_metadata = node[:, : self.out_dims.X]
        return extract_heterogeneous_metadata(
            node_t=node_for_metadata,
            edge_attr=edge_attr,
            edge_index=edge_index,
            type_offsets=getattr(self.dataset_info, "type_offsets", None),
            node_type_names=getattr(self.dataset_info, "node_type_names", []),
            edge_family_offsets=getattr(
                self.dataset_info, "edge_family_offsets", None
            ),
            fam_endpoints=getattr(self.dataset_info, "fam_endpoints", None),
            num_node_types=len(
                getattr(self.dataset_info, "node_type_names", []) or []
            ),
            num_edge_classes=self.out_dims.E,
        )

    def _encode_context_only(
        self,
        data,
        node_onehot,
        context_edge_index,
        context_edge_attr,
        batch,
        ptr,
        t_float,
    ):
        """Run HGT message passing on visible G_t context edges only."""
        context_edge_index, context_edge_attr = utils.to_undirected(
            context_edge_index, context_edge_attr
        )
        noisy = {
            "node_t": node_onehot,
            "X_t": node_onehot,
            "edge_index_t": context_edge_index,
            "edge_attr_t": context_edge_attr,
            "comp_edge_index_t": context_edge_index,
            "comp_edge_attr_t": context_edge_attr,
            "y_t": data.y,
            "batch": batch,
            "ptr": ptr,
            "charge_t": getattr(
                data,
                "charge",
                torch.zeros((node_onehot.shape[0], 0), device=self.device),
            ),
            "t_float": t_float,
        }
        prepared = self.compute_extra_data(noisy)
        if self.sign_net and self.cfg.model.extra_features == "all":
            sign = self.sign_net(
                prepared["node_t"],
                prepared["edge_index_t"],
                prepared["batch"],
            )
            prepared["node_t"] = torch.hstack([prepared["node_t"], sign])
        metadata = self._heterogeneous_metadata_for_edges(
            prepared["node_t"],
            prepared["edge_attr_t"],
            prepared["edge_index_t"],
        )
        structure_lookup = None
        if (
            bool(getattr(self.cfg.model, "use_two_hop_structure", False))
            or bool(
                getattr(
                    self.cfg.model, "use_endpoint_role_residual", False
                )
            )
        ):
            structure_lookup = self.model.build_two_hop_structure_lookup(
                context_edge_index=prepared["edge_index_t"],
                num_nodes=int(prepared["node_t"].shape[0]),
                node_type_ids=metadata.get("node_type_ids"),
                edge_family_ids=metadata.get("edge_family_ids"),
                batch=prepared["batch"],
            )
        return self.model.encode_context(
            X=prepared["node_t"],
            edge_attr=prepared["edge_attr_t"],
            edge_index=prepared["edge_index_t"],
            y=prepared["y_t"],
            batch=prepared["batch"],
            node_type_ids=metadata.get("node_type_ids"),
            node_subtype_ids=metadata.get("node_subtype_ids"),
            relation_type_ids=metadata.get("relation_type_ids"),
            edge_family_ids=metadata.get("edge_family_ids"),
            two_hop_structure_lookup=structure_lookup,
            two_hop_scale_factor=prepared.get("two_hop_scale_factor"),
            endpoint_role_scale_factor=prepared.get(
                "endpoint_role_scale_factor"
            ),
        )

    def _decode_query_logits(
        self,
        encoded,
        query_edge_index,
        query_current_labels,
        batch,
        query_edge_family=None,
        edge_input_residual_scale=None,
    ):
        query_attr = F.one_hot(
            query_current_labels.long().clamp(0, self.out_dims.E - 1),
            num_classes=self.out_dims.E,
        ).float()
        return self.model.decode_queries(
            encoded=encoded,
            query_edge_attr=query_attr,
            query_edge_index=query_edge_index.long(),
            batch=batch.long(),
            query_edge_family_ids=query_edge_family,
            edge_input_residual_scale=edge_input_residual_scale,
        )

    @staticmethod
    def _query_decode_labels_for_mode(current_labels, mode):
        mode = str(mode or "current").lower()
        if mode in ("none", "no_edge", "zero"):
            return torch.zeros_like(current_labels)
        return current_labels

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

    def _permute_fixed_graph_within_node_types(self, data, permutation_seed):
        """Synchronously relabel node-aligned tensors and edges within each type."""
        if permutation_seed is None:
            return data
        if not hasattr(data, "node_type"):
            raise ValueError(
                "general.fixed_node_type_permutation_seed requires data.node_type"
            )
        out = data.clone()
        node_type = data.node_type.long().detach().cpu()
        num_nodes = int(node_type.numel())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(permutation_seed))

        new_to_old = torch.arange(num_nodes, dtype=torch.long)
        for type_id in torch.unique(node_type, sorted=True).tolist():
            positions = torch.where(node_type == int(type_id))[0]
            shuffled = positions[
                torch.randperm(positions.numel(), generator=generator)
            ]
            new_to_old[positions] = shuffled
        old_to_new = torch.empty_like(new_to_old)
        old_to_new[new_to_old] = torch.arange(num_nodes, dtype=torch.long)

        data_keys = data.keys() if callable(data.keys) else data.keys
        for key in data_keys:
            value = getattr(data, key)
            if (
                torch.is_tensor(value)
                and value.dim() > 0
                and int(value.shape[0]) == num_nodes
                and key != "edge_index"
            ):
                setattr(out, key, value[new_to_old.to(value.device)])
        out.edge_index = old_to_new.to(data.edge_index.device)[
            data.edge_index.long()
        ]
        if getattr(self, "local_rank", 0) == 0:
            moved = int((new_to_old != torch.arange(num_nodes)).sum().item())
            checksum = int(
                (
                    (torch.arange(num_nodes, dtype=torch.long) + 1)
                    * (new_to_old + 1)
                ).sum().item()
            )
            print(
                "[采样-PERMUTE] "
                f"within_type_seed={int(permutation_seed)} "
                f"moved={moved}/{num_nodes} checksum={checksum}"
            )
        return out

    @staticmethod
    def _build_within_type_permutation(node_type, permutation_seed):
        node_type = node_type.long().detach().cpu()
        num_nodes = int(node_type.numel())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(permutation_seed))
        new_to_old = torch.arange(num_nodes, dtype=torch.long)
        for type_id in torch.unique(node_type, sorted=True).tolist():
            positions = torch.where(node_type == int(type_id))[0]
            new_to_old[positions] = positions[
                torch.randperm(positions.numel(), generator=generator)
            ]
        old_to_new = torch.empty_like(new_to_old)
        old_to_new[new_to_old] = torch.arange(num_nodes, dtype=torch.long)
        return old_to_new, new_to_old

    def _log_forward_equivariance_diag(self, stats, t_float, s_float):
        if not stats or int(stats.get("count", 0)) <= 0:
            return
        original = torch.cat(stats["exist_original"])
        permuted = torch.cat(stats["exist_permuted"])
        rank_original = self._rankdata_average(original.numpy())
        rank_permuted = self._rankdata_average(permuted.numpy())
        if np.std(rank_original) > 0 and np.std(rank_permuted) > 0:
            spearman = float(np.corrcoef(rank_original, rank_permuted)[0, 1])
        else:
            spearman = float("nan")
        result = {
            "t": float(t_float.mean().detach().cpu()),
            "s": float(s_float.mean().detach().cpu()),
            "candidate_count": int(stats["count"]),
            "raw_logits_mean_abs_error": float(
                stats["raw_abs_sum"] / max(1, stats["raw_element_count"])
            ),
            "raw_logits_max_abs_error": float(stats["raw_abs_max"]),
            "exist_logit_mean_abs_error": float(
                stats["exist_abs_sum"] / max(1, stats["count"])
            ),
            "exist_logit_max_abs_error": float(stats["exist_abs_max"]),
            "exist_probability_mean_abs_error": float(
                stats["prob_abs_sum"] / max(1, stats["count"])
            ),
            "exist_probability_max_abs_error": float(stats["prob_abs_max"]),
            "exist_score_spearman": spearman,
        }
        print("[采样-FORWARD-EQUIV] " + json.dumps(result, ensure_ascii=False))
        with open(
            os.path.join(os.getcwd(), "forward_equivariance_diag.jsonl"),
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _log_exactk_equivariance_diag(
        self,
        original_edges,
        permuted_edges,
        new_to_old,
        total_nodes,
        t_float,
        s_float,
    ):
        if original_edges is None or permuted_edges is None:
            return
        original = original_edges.long().detach().cpu()
        mapped = new_to_old[permuted_edges.long().detach().cpu()]
        original_u = torch.minimum(original[0], original[1])
        original_v = torch.maximum(original[0], original[1])
        mapped_u = torch.minimum(mapped[0], mapped[1])
        mapped_v = torch.maximum(mapped[0], mapped[1])
        original_keys = set(
            (original_u * int(total_nodes) + original_v).tolist()
        )
        mapped_keys = set((mapped_u * int(total_nodes) + mapped_v).tolist())
        intersection = len(original_keys & mapped_keys)
        union = len(original_keys | mapped_keys)
        result = {
            "t": float(t_float.mean().detach().cpu()),
            "s": float(s_float.mean().detach().cpu()),
            "original_edges": len(original_keys),
            "permuted_edges": len(mapped_keys),
            "intersection": intersection,
            "jaccard": float(intersection / max(1, union)),
            "symmetric_difference": len(original_keys ^ mapped_keys),
        }
        print("[采样-EXACT-K-EQUIV] " + json.dumps(result, ensure_ascii=False))
        with open(
            os.path.join(os.getcwd(), "exactk_equivariance_diag.jsonl"),
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    @staticmethod
    def _rankdata_average(values):
        """Average ranks for ties, implemented without scipy."""
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    @classmethod
    def _spearman_corr(cls, left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.size < 2 or right.size != left.size:
            return float("nan")
        left_rank = cls._rankdata_average(left)
        right_rank = cls._rankdata_average(right)
        if left_rank.std() <= 0 or right_rank.std() <= 0:
            return float("nan")
        return float(np.corrcoef(left_rank, right_rank)[0, 1])

    def _finalize_edge_score_structure_diag(
        self,
        candidate_parts,
        reference_edge_index,
        total_nodes,
        t_float,
        s_float,
    ):
        """Compare clean-edge logits with real-edge structural importance."""
        if not candidate_parts or reference_edge_index is None:
            return
        edge_index = torch.cat(
            [part["edge_index"].detach().cpu() for part in candidate_parts], dim=1
        ).long()
        scores = torch.cat(
            [part["scores"].detach().float().cpu() for part in candidate_parts]
        )
        family_parts = [
            part.get("family")
            for part in candidate_parts
            if part.get("family") is not None
        ]
        families = None
        if len(family_parts) == len(candidate_parts):
            families = torch.cat(
                [part.detach().cpu().long() for part in family_parts]
            )
        keys = edge_index[0] * int(total_nodes) + edge_index[1]

        # Keep the maximum model score for duplicate block-query occurrences.
        order = torch.argsort(keys)
        edge_index = edge_index[:, order]
        keys = keys[order]
        scores = scores[order]
        if families is not None:
            families = families[order]
        _, inverse = torch.unique_consecutive(keys, return_inverse=True)
        num_unique = int(inverse[-1].item()) + 1 if inverse.numel() else 0
        max_scores = torch.full((num_unique,), -float("inf"), dtype=scores.dtype)
        max_scores.scatter_reduce_(0, inverse, scores, reduce="amax", include_self=True)
        positions = torch.arange(scores.numel(), dtype=torch.long)
        sentinel = int(scores.numel())
        first_max = torch.full((num_unique,), sentinel, dtype=torch.long)
        first_max.scatter_reduce_(
            0,
            inverse,
            torch.where(
                scores == max_scores[inverse],
                positions,
                torch.full_like(positions, sentinel),
            ),
            reduce="amin",
            include_self=True,
        )
        chosen = first_max[first_max < sentinel]
        edge_index = edge_index[:, chosen]
        keys = keys[chosen]
        scores = scores[chosen]
        if families is not None:
            families = families[chosen]

        ref = reference_edge_index.long().detach().cpu()
        ref_u = torch.minimum(ref[0], ref[1])
        ref_v = torch.maximum(ref[0], ref[1])
        ref_keep = ref_u != ref_v
        ref_u, ref_v = ref_u[ref_keep], ref_v[ref_keep]
        ref_keys = torch.unique(ref_u * int(total_nodes) + ref_v)
        real_mask = torch.isin(keys, ref_keys)
        real_idx = real_mask.nonzero(as_tuple=True)[0]
        negative_idx = (~real_mask).nonzero(as_tuple=True)[0]

        max_negatives = int(
            getattr(
                self.cfg.general,
                "edge_score_structure_diag_max_negatives",
                100000,
            )
            or 100000
        )
        max_negatives = max(0, max_negatives)
        if negative_idx.numel() > max_negatives:
            seed = self._current_sampling_seed()
            t_int = int(round(float(t_float.mean().detach().cpu()) * self.T))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 3000017 * t_int)
            negative_idx = negative_idx[
                torch.randperm(negative_idx.numel(), generator=generator)[
                    :max_negatives
                ]
            ]
        sample_idx = torch.cat([real_idx, negative_idx])
        sample_edges = edge_index[:, sample_idx]
        sample_scores = scores[sample_idx].numpy()
        sample_labels = real_mask[sample_idx].numpy().astype(np.int64)

        neighbors = [set() for _ in range(int(total_nodes))]
        for u, v in zip(ref_u.tolist(), ref_v.tolist()):
            neighbors[int(u)].add(int(v))
            neighbors[int(v)].add(int(u))
        degrees = np.asarray([len(value) for value in neighbors], dtype=np.float64)
        sample_u = sample_edges[0].tolist()
        sample_v = sample_edges[1].tolist()
        common_neighbors = np.asarray(
            [len(neighbors[int(u)].intersection(neighbors[int(v)]))
             for u, v in zip(sample_u, sample_v)],
            dtype=np.float64,
        )
        degree_sum = np.asarray(
            [degrees[int(u)] + degrees[int(v)] for u, v in zip(sample_u, sample_v)],
            dtype=np.float64,
        )

        pos_scores = sample_scores[sample_labels == 1]
        neg_scores = sample_scores[sample_labels == 0]
        if pos_scores.size and neg_scores.size:
            combined_scores = np.concatenate([pos_scores, neg_scores])
            combined_labels = np.concatenate(
                [np.ones(pos_scores.size), np.zeros(neg_scores.size)]
            )
            score_ranks = self._rankdata_average(combined_scores)
            pos_rank_sum = score_ranks[combined_labels == 1].sum()
            auc = float(
                (
                    pos_rank_sum
                    - pos_scores.size * (pos_scores.size + 1) / 2.0
                )
                / (pos_scores.size * neg_scores.size)
            )
        else:
            auc = float("nan")

        top_k = min(int(ref_keys.numel()), int(scores.numel()))
        top_idx = torch.topk(scores, k=top_k, largest=True).indices if top_k else torch.empty(0, dtype=torch.long)
        top_real = real_mask[top_idx]
        top_edges = edge_index[:, top_idx]
        top_common = np.asarray(
            [
                len(neighbors[int(u)].intersection(neighbors[int(v)]))
                for u, v in zip(top_edges[0].tolist(), top_edges[1].tolist())
            ],
            dtype=np.float64,
        )

        positive_common = common_neighbors[sample_labels == 1]
        positive_score = sample_scores[sample_labels == 1]
        positive_groups = {}
        for label, mask in (
            ("cn0", positive_common == 0),
            ("cn1", positive_common == 1),
            ("cn2plus", positive_common >= 2),
        ):
            positive_groups[label] = {
                "count": int(mask.sum()),
                "mean_score": (
                    float(positive_score[mask].mean()) if mask.any() else float("nan")
                ),
            }

        per_family = {}
        if families is not None and families.numel() == scores.numel():
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {}) or {}
            id2edge_family = {int(v): str(k) for k, v in edge_family2id.items()}
            for fam_id_tensor in torch.unique(families, sorted=True):
                fam_id = int(fam_id_tensor.item())
                fam_mask = families == fam_id
                fam_idx = fam_mask.nonzero(as_tuple=True)[0]
                if fam_idx.numel() == 0:
                    continue
                fam_real_mask = real_mask[fam_idx]
                fam_real_count = int(fam_real_mask.sum().item())
                fam_neg_count = int((~fam_real_mask).sum().item())
                fam_scores = scores[fam_idx].numpy()
                fam_labels = fam_real_mask.numpy().astype(np.int64)
                fam_auc = float("nan")
                if fam_real_count > 0 and fam_neg_count > 0:
                    fam_ranks = self._rankdata_average(fam_scores)
                    fam_pos_rank_sum = fam_ranks[fam_labels == 1].sum()
                    fam_auc = float(
                        (
                            fam_pos_rank_sum
                            - fam_real_count * (fam_real_count + 1) / 2.0
                        )
                        / (fam_real_count * fam_neg_count)
                    )
                fam_top_k = min(fam_real_count, int(fam_idx.numel()))
                if fam_top_k > 0:
                    fam_top_local = torch.topk(
                        scores[fam_idx], k=fam_top_k, largest=True
                    ).indices
                    fam_top_idx = fam_idx[fam_top_local]
                    fam_top_real = real_mask[fam_top_idx]
                    fam_top_edges = edge_index[:, fam_top_idx]
                    fam_top_common = np.asarray(
                        [
                            len(neighbors[int(u)].intersection(neighbors[int(v)]))
                            for u, v in zip(
                                fam_top_edges[0].tolist(),
                                fam_top_edges[1].tolist(),
                            )
                        ],
                        dtype=np.float64,
                    )
                    fam_top_recall = float(
                        fam_top_real.sum().item() / max(fam_real_count, 1)
                    )
                    fam_top_common_mean = (
                        float(fam_top_common.mean())
                        if fam_top_common.size
                        else float("nan")
                    )
                else:
                    fam_top_recall = float("nan")
                    fam_top_common_mean = float("nan")
                fam_pos_scores = fam_scores[fam_labels == 1]
                fam_neg_scores = fam_scores[fam_labels == 0]
                per_family[id2edge_family.get(fam_id, str(fam_id))] = {
                    "candidate_count": int(fam_idx.numel()),
                    "real_candidate_count": fam_real_count,
                    "negative_candidate_count": fam_neg_count,
                    "real_vs_negative_auc": fam_auc,
                    "mean_score_real": (
                        float(fam_pos_scores.mean())
                        if fam_pos_scores.size
                        else float("nan")
                    ),
                    "mean_score_negative": (
                        float(fam_neg_scores.mean())
                        if fam_neg_scores.size
                        else float("nan")
                    ),
                    "top_k": fam_top_k,
                    "top_k_real_recall": fam_top_recall,
                    "top_k_mean_common_neighbors": fam_top_common_mean,
                }

        result = {
            "t": float(t_float.mean().detach().cpu()),
            "s": float(s_float.mean().detach().cpu()),
            "candidate_count": int(scores.numel()),
            "real_candidate_count": int(real_idx.numel()),
            "sampled_negative_count": int(negative_idx.numel()),
            "real_vs_negative_auc": auc,
            "mean_score_real": float(pos_scores.mean()) if pos_scores.size else float("nan"),
            "mean_score_negative": float(neg_scores.mean()) if neg_scores.size else float("nan"),
            "spearman_score_common_neighbors_all": self._spearman_corr(
                sample_scores, common_neighbors
            ),
            "spearman_score_common_neighbors_real": self._spearman_corr(
                positive_score, positive_common
            ),
            "spearman_score_degree_sum_all": self._spearman_corr(
                sample_scores, degree_sum
            ),
            "top_k": top_k,
            "top_k_real_recall": (
                float(top_real.sum().item()) / max(int(ref_keys.numel()), 1)
            ),
            "top_k_mean_common_neighbors": (
                float(top_common.mean()) if top_common.size else float("nan")
            ),
            "positive_score_by_triangle_contribution": positive_groups,
            "per_family": per_family,
        }
        print("[采样-EDGE-SCORE-DIAG] " + json.dumps(result, ensure_ascii=False))
        output_path = os.path.join(
            os.getcwd(),
            "edge_score_structure_diag.jsonl",
        )
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

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

    def _sampling_time_transitions(self):
        """Return explicit (t, s) reverse transitions in integer diffusion time."""
        configured = getattr(self.cfg.general, "sampling_time_schedule", None)
        is_testing = bool(getattr(getattr(self, "trainer", None), "testing", False))
        if configured is not None and is_testing:
            schedule = [int(value) for value in list(configured)]
            if len(schedule) < 2:
                raise ValueError(
                    "general.sampling_time_schedule must contain at least [T, 0]"
                )
            if schedule[0] != int(self.T) or schedule[-1] != 0:
                raise ValueError(
                    "general.sampling_time_schedule must start at "
                    f"T={self.T} and end at 0, got {schedule}"
                )
            if any(value < 0 or value > int(self.T) for value in schedule):
                raise ValueError(
                    "general.sampling_time_schedule values must lie in "
                    f"[0, {self.T}], got {schedule}"
                )
            if any(left <= right for left, right in zip(schedule, schedule[1:])):
                raise ValueError(
                    "general.sampling_time_schedule must be strictly decreasing, "
                    f"got {schedule}"
                )
            transitions = list(zip(schedule[:-1], schedule[1:]))
            if getattr(self, "local_rank", 0) == 0:
                print(f"[采样-SCHEDULE] explicit={schedule}")
            return transitions

        step = self._sampling_step_size()
        s_values = list(reversed(range(0, int(self.T), int(step))))
        return [
            (min(int(self.T), int(s_value) + int(step)), int(s_value))
            for s_value in s_values
        ]

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
                    sparse_sampled_data = self._prepare_connectivity_candidates(sparse_sampled_data)

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
        transitions = self._sampling_time_transitions()
        n_steps = len(transitions)
        disable_tqdm = self.local_rank != 0 if hasattr(self, 'local_rank') else False
        for step_index, (t_int, s_int) in enumerate(
            tqdm(transitions, total=n_steps, disable=disable_tqdm)
        ):
            s_array = (s_int * torch.ones((batch_size, 1))).to(self.device)
            t_array = (t_int * torch.ones((batch_size, 1))).to(self.device)
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            # print(s_norm, t_norm)

            # Sample z_s
            sparse_sampled_data = self.sample_p_zs_given_zt(
                    s_norm,
                    t_norm,
                    sparse_sampled_data,
                    is_final_sampling_step=(step_index == n_steps - 1),
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
        equivariance_partition_mode = str(
            getattr(
                self.cfg.general,
                "equivariance_partition_mode",
                "current",
            )
            or "current"
        ).lower()
        equivariance_initial_noise_mode = str(
            getattr(
                self.cfg.general,
                "equivariance_initial_noise_mode",
                "current",
            )
            or "current"
        ).lower()
        equivariance_gumbel_mode = str(
            getattr(
                self.cfg.general,
                "equivariance_gumbel_mode",
                "position",
            )
            or "position"
        ).lower()
        permutation_seed = getattr(
            self.cfg.general, "fixed_node_type_permutation_seed", None
        )
        reference_new_to_old = None
        reference_old_to_new = None
        needs_reference_mapping = (
            equivariance_partition_mode == "mapped_reference"
            or equivariance_initial_noise_mode == "mapped_reference"
            or equivariance_gumbel_mode == "mapped_reference"
        )
        if needs_reference_mapping:
            if permutation_seed is None:
                reference_old_to_new = torch.arange(
                    fixed_node_subtypes.shape[0],
                    dtype=torch.long,
                    device=self.device,
                )
                reference_new_to_old = reference_old_to_new
            else:
                reference_old_to_new, reference_new_to_old = (
                    self._build_within_type_permutation(
                        fixed_data.node_type,
                        int(permutation_seed),
                    )
                )
                reference_old_to_new = reference_old_to_new.to(self.device)
                reference_new_to_old = reference_new_to_old.to(self.device)
            sparse_sampled_data.equivariance_reference_new_to_old = (
                reference_new_to_old
            )

        if equivariance_initial_noise_mode == "mapped_reference":
            if permutation_seed is not None:
                # z_T was generated in reference-coordinate RNG order. Move it
                # into the relabelled coordinate system so the posterior sees a
                # paired current state from the first denoising step onward.
                sparse_sampled_data.edge_index = reference_old_to_new[
                    sparse_sampled_data.edge_index.long()
                ]
                mapped_u = torch.minimum(
                    sparse_sampled_data.edge_index[0],
                    sparse_sampled_data.edge_index[1],
                )
                mapped_v = torch.maximum(
                    sparse_sampled_data.edge_index[0],
                    sparse_sampled_data.edge_index[1],
                )
                sparse_sampled_data.edge_index = torch.stack(
                    [mapped_u, mapped_v], dim=0
                )
            if getattr(self, "local_rank", 0) == 0:
                print(
                    "[采样-EQUIV-L3] "
                    f"partition={equivariance_partition_mode} "
                    "initial_noise=mapped_reference "
                    f"gumbel={equivariance_gumbel_mode} "
                    f"permutation_seed={permutation_seed}"
                )
        elif equivariance_initial_noise_mode != "current":
            raise ValueError(
                "Unsupported general.equivariance_initial_noise_mode: "
                f"{equivariance_initial_noise_mode}"
            )
        if equivariance_partition_mode not in ("current", "mapped_reference"):
            raise ValueError(
                "Unsupported general.equivariance_partition_mode: "
                f"{equivariance_partition_mode}"
            )
        if bool(getattr(self.cfg.general, "forward_equivariance_diag", False)):
            sparse_sampled_data.forward_equivariance_diag_node_type = (
                fixed_data.node_type.long().detach().to(self.device)
            )
        if bool(getattr(self.cfg.general, "edge_score_structure_diag", False)):
            sparse_sampled_data.edge_score_diag_reference_edge_index = (
                fixed_data.edge_index.long().detach().to(self.device)
            )
        if str(getattr(self.cfg.model, "sampling_block_mode", "uniform")).lower() == "type_template":
            template_source = fixed_data
            if (
                equivariance_partition_mode == "mapped_reference"
                and permutation_seed is not None
            ):
                # Undo the external relabelling only for METIS/template
                # construction. The resulting reference templates are then
                # transported to current IDs by the pseudo-block builder.
                template_source = fixed_data.clone()
                num_nodes = int(fixed_node_subtypes.shape[0])
                data_keys = (
                    fixed_data.keys()
                    if callable(fixed_data.keys)
                    else fixed_data.keys
                )
                for key in data_keys:
                    value = getattr(fixed_data, key)
                    if (
                        torch.is_tensor(value)
                        and value.dim() > 0
                        and int(value.shape[0]) == num_nodes
                        and key != "edge_index"
                    ):
                        setattr(
                            template_source,
                            key,
                            value[reference_old_to_new.to(value.device)],
                        )
                template_source.edge_index = reference_new_to_old.to(
                    fixed_data.edge_index.device
                )[fixed_data.edge_index.long()]
                ref_u = torch.minimum(
                    template_source.edge_index[0],
                    template_source.edge_index[1],
                )
                ref_v = torch.maximum(
                    template_source.edge_index[0],
                    template_source.edge_index[1],
                )
                template_source.edge_index = torch.stack([ref_u, ref_v], dim=0)
            self._ensure_hetero_block_templates_from_data(template_source)
            sparse_sampled_data.pseudo_blocks = self._build_type_template_pseudo_blocks(
                    sparse_sampled_data.anchor_node_subtype.long().to(self.device),
                    sparse_sampled_data.batch.long().to(self.device),
                    getattr(self.dataset_info, "type_offsets", {}),
                    reference_new_to_old=(
                        reference_new_to_old
                        if equivariance_partition_mode == "mapped_reference"
                        else None
                    ),
            )
            sparse_sampled_data = self._apply_block_marginal_initial_edges(sparse_sampled_data)
            sparse_sampled_data = self._apply_block_template_initial_edges(sparse_sampled_data)
            sparse_sampled_data = self._prepare_connectivity_candidates(sparse_sampled_data)

        chain = utils.SparseChainPlaceHolder(keep_chain=keep_chain)
        transitions = self._sampling_time_transitions()
        n_steps = len(transitions)
        verbose_sampling = getattr(self.cfg.general, "verbose_sampling", False)
        disable_tqdm = self.local_rank != 0 if hasattr(self, "local_rank") else False
        from tqdm import tqdm
        for step_index, (t_int, s_int) in enumerate(
            tqdm(transitions, total=n_steps, disable=disable_tqdm)
        ):
            if verbose_sampling and hasattr(self, "local_rank") and self.local_rank == 0:
                    self._verbose_step_index = step_index
                    self._verbose_total_steps = n_steps
                    self._verbose_s_float = (s_int * 1.0) / self.T
                    self._verbose_t_float = (t_int * 1.0) / self.T
            s_array = (s_int * torch.ones((batch_size, 1))).to(self.device)
            t_array = (t_int * torch.ones((batch_size, 1))).to(self.device)
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            sparse_sampled_data = self.sample_p_zs_given_zt(
                s_norm,
                t_norm,
                sparse_sampled_data,
                is_final_sampling_step=(step_index == n_steps - 1),
            )
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

    def sample_p_zs_given_zt(
        self,
        s_float,
        t_float,
        data,
        is_final_sampling_step=False,
    ):
        """One sparse denoising step.

        For type-template block sampling this follows SparseDiff-original's block-round
        dynamics: within one diffusion step, visit pseudo blocks in a random order and
        merge each block's sampled edges before moving to the next block.
        """
        verbose_sampling = bool(getattr(self.cfg.general, "verbose_sampling", False))
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
            not query_rounds
            and str(getattr(self.cfg.model, "sampling_block_mode", "uniform")).lower() == "type_template"
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

        partition_ensemble_mode = str(
            getattr(
                self.cfg.model,
                "sampling_partition_ensemble",
                "off",
            )
            or "off"
        ).lower()
        if partition_ensemble_mode != "off":
            ensemble_selection_mode = str(
                getattr(
                    self.cfg.model,
                    "sampling_edge_selection",
                    "gumbel_exact_k",
                )
                or "gumbel_exact_k"
            ).lower()
            if ensemble_selection_mode not in (
                "gumbel_exact_k",
                "deterministic_exact_k",
                "gumbel_exact_k_degree_pair",
                "deterministic_exact_k_degree_pair",
                "gumbel_exact_k_degree_pair_quota",
                "deterministic_exact_k_degree_pair_quota",
            ):
                raise ValueError(
                    "sampling_partition_ensemble requires gumbel_exact_k "
                    "or deterministic_exact_k"
                )
            ensemble_out = self._sample_partition_ensemble_exact_k(
                data=data,
                query_rounds=query_rounds,
                random_blocks=pseudo_blocks,
                node_onehot=node_onehot,
                anchor_node_subtype=anchor_node_subtype,
                batch=batch,
                ptr=ptr,
                edge_index=edge_index,
                edge_attr_ids=edge_attr_ids,
                edge_attr_onehot=edge_attr_onehot,
                s_float=s_float,
                t_float=t_float,
                is_final_sampling_step=is_final_sampling_step,
                selection_mode=ensemble_selection_mode,
            )
            if ensemble_out is not None:
                return ensemble_out

        base_edge_index = edge_index
        base_edge_attr_ids = edge_attr_ids
        base_edge_attr_onehot = edge_attr_onehot
        sampling_structure_lookup = None
        if isinstance(self.model, PyHGTDenoiser) and (
            bool(getattr(self.cfg.model, "use_two_hop_structure", False))
            or bool(
                getattr(
                    self.cfg.model, "use_endpoint_role_residual", False
                )
            )
        ):
            visible = base_edge_attr_ids > 0
            visible_edge_index = base_edge_index[:, visible]
            visible_edge_attr = base_edge_attr_onehot[visible]
            visible_metadata = self._heterogeneous_metadata_for_edges(
                node_onehot,
                visible_edge_attr,
                visible_edge_index,
            )
            sampling_structure_lookup = (
                self.model.build_two_hop_structure_lookup(
                    context_edge_index=visible_edge_index,
                    num_nodes=int(node_onehot.shape[0]),
                    node_type_ids=visible_metadata.get("node_type_ids"),
                    edge_family_ids=visible_metadata.get("edge_family_ids"),
                    batch=batch,
                )
            )
        cur_edge_index = edge_index
        cur_edge_attr_ids = edge_attr_ids
        cur_edge_attr_onehot = edge_attr_onehot
        use_autoregressive_context = bool(getattr(self.cfg.model, "autoregressive", False))
        selection_mode = str(
            getattr(self.cfg.model, "sampling_edge_selection", "bernoulli") or "bernoulli"
        ).lower()
        use_connectivity_topk = (
            selection_mode == "connectivity_topk"
            and query_rounds
            and query_rounds[0][2] is not None
            and hasattr(data, "connectivity_family_targets")
            and not use_autoregressive_context
        )
        use_global_exact_k = (
            selection_mode in (
                "gumbel_exact_k",
                "deterministic_exact_k",
                "gumbel_exact_k_degree_pair",
                "deterministic_exact_k_degree_pair",
                "gumbel_exact_k_degree_pair_quota",
                "deterministic_exact_k_degree_pair_quota",
            )
            and query_rounds
            and query_rounds[0][2] is not None
            and not use_autoregressive_context
        )
        use_queryfree_global_exact_k = (
            use_global_exact_k
            and isinstance(self.model, PyHGTDenoiser)
            and bool(getattr(self.cfg.model, "sampling_queryfree_decode", False))
        )
        use_exact_k_connectivity_repair = (
            use_global_exact_k
            and bool(is_final_sampling_step)
            and bool(
                getattr(
                    self.cfg.model,
                    "sampling_exact_k_connectivity_repair",
                    False,
                )
            )
        )
        repair_final_only = bool(
            getattr(self.cfg.model, "sampling_connectivity_repair_final_only", True)
        )
        use_topk_density_repair = (
            selection_mode == "topk_density_repair"
            and (bool(is_final_sampling_step) or not repair_final_only)
            and query_rounds
            and query_rounds[0][2] is not None
            and not use_autoregressive_context
        )
        connectivity_pools = {}
        connectivity_stats = {}
        repair_candidate_parts = []
        global_exact_k_candidate_parts = []
        clean_exact_k_candidate_parts = []
        use_ranking_intervention_diag = (
            use_global_exact_k
            and bool(
                getattr(
                    self.cfg.model,
                    "sampling_ranking_intervention_diag",
                    False,
                )
            )
        )
        use_forward_equivariance_diag = bool(
            getattr(self.cfg.general, "forward_equivariance_diag", False)
        ) and hasattr(data, "forward_equivariance_diag_node_type")
        if use_queryfree_global_exact_k and use_forward_equivariance_diag:
            raise ValueError(
                "sampling_queryfree_decode does not currently support "
                "forward_equivariance_diag."
            )
        use_exactk_equivariance_diag = (
            use_forward_equivariance_diag
            and bool(getattr(self.cfg.general, "exactk_equivariance_diag", False))
            and use_global_exact_k
        )
        paired_exact_k_candidate_parts = []
        forward_equivariance_stats = None
        diag_old_to_new = None
        diag_new_to_old = None
        if use_forward_equivariance_diag:
            diag_old_to_new, diag_new_to_old = self._build_within_type_permutation(
                data.forward_equivariance_diag_node_type,
                int(
                    getattr(
                        self.cfg.general,
                        "forward_equivariance_permutation_seed",
                        0,
                    )
                    or 0
                ),
            )
            diag_old_to_new = diag_old_to_new.to(self.device)
            diag_new_to_old = diag_new_to_old.to(self.device)
            forward_equivariance_stats = {
                "count": 0,
                "raw_element_count": 0,
                "raw_abs_sum": 0.0,
                "raw_abs_max": 0.0,
                "exist_abs_sum": 0.0,
                "exist_abs_max": 0.0,
                "prob_abs_sum": 0.0,
                "prob_abs_max": 0.0,
                "exist_original": [],
                "exist_permuted": [],
            }
        use_edge_score_structure_diag = bool(
            getattr(self.cfg.general, "edge_score_structure_diag", False)
        ) and hasattr(data, "edge_score_diag_reference_edge_index")
        edge_score_structure_diag_parts = []
        total_nodes = int(node_onehot.shape[0])
        step_diag = {
            "query_count": 0,
            "raw_clean_p_sum": 0.0,
            "calibrated_clean_p_sum": 0.0,
            "posterior_p_sum": 0.0,
            "current_edge_p_sum": 0.0,
            "current_edge_p_count": 0,
            "current_nonedge_p_sum": 0.0,
            "current_nonedge_p_count": 0,
            "current_query_pos": 0,
            "sampled_query_pos": 0,
            "query_added": 0,
            "query_removed": 0,
            "query_retained": 0,
        }
        queryfree_encoded = None
        if use_queryfree_global_exact_k:
            queryfree_encoded = self._encode_context_only(
                data=data,
                node_onehot=node_onehot,
                context_edge_index=base_edge_index.long(),
                context_edge_attr=base_edge_attr_onehot.float(),
                batch=batch,
                ptr=ptr,
                t_float=t_float,
            )
            if (
                getattr(self, "local_rank", 0) == 0
                and not getattr(self, "_logged_sampling_queryfree_decode", False)
            ):
                print(
                    "[采样-QUERYFREE] global exact-K encodes current full G_t "
                    "once and decodes query candidates without query message passing"
                )
                self._logged_sampling_queryfree_decode = True

        for query_edge_index, _query_edge_batch, query_edge_family in query_rounds:
            if query_edge_index is None or query_edge_index.numel() == 0:
                continue
            if use_autoregressive_context:
                fw_edge_index = cur_edge_index
                fw_edge_attr_onehot = cur_edge_attr_onehot
            else:
                fw_edge_index = base_edge_index
                fw_edge_attr_onehot = base_edge_attr_onehot

            sampling_residual_scale = getattr(
                self.cfg.model, "sampling_edge_input_residual_scale", None
            )
            selected_query_edge_index = None
            selected_query_family = None
            selected_query_batch = None
            current_query_labels = None
            query_logits = None
            paired_comp_query_logits = None
            paired_query_logits = None
            if use_queryfree_global_exact_k:
                selected_query_edge_index = query_edge_index.long()
                selected_query_family = query_edge_family.long()
                selected_query_batch = batch[selected_query_edge_index[0].long()]
                current_query_labels = self._lookup_query_edge_labels(
                    selected_query_edge_index,
                    fw_edge_index,
                    fw_edge_attr_onehot.argmax(dim=-1).long(),
                    total_nodes,
                )
                decode_query_labels = self._query_decode_labels_for_mode(
                    current_query_labels,
                    getattr(
                        self.cfg.model,
                        "sampling_queryfree_query_state",
                        "current",
                    ),
                )
                query_logits = self._decode_query_logits(
                    queryfree_encoded,
                    selected_query_edge_index,
                    decode_query_labels,
                    batch,
                    selected_query_family,
                    edge_input_residual_scale=sampling_residual_scale,
                )
            else:
                query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                    triu_query_edge_index=query_edge_index,
                    clean_edge_index=fw_edge_index,
                    clean_edge_attr=fw_edge_attr_onehot,
                    heterogeneous=self.heterogeneous,
                    for_message_passing=True,
                    total_num_nodes=node_onehot.shape[0],
                )
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
                if sampling_residual_scale is not None:
                    sparse_noisy_data["edge_input_residual_scale"] = float(
                        sampling_residual_scale
                    )
                if sampling_structure_lookup is not None:
                    sparse_noisy_data["two_hop_structure_lookup"] = (
                        sampling_structure_lookup
                    )
                pred = self.forward(sparse_noisy_data)
                comp_query_logits = pred.edge_attr[query_mask]
                comp_query_edge_index = comp_edge_index[:, query_mask]
                if use_forward_equivariance_diag:
                    diag_charge = getattr(data, "charge", None)
                    if diag_charge is None:
                        diag_charge = torch.zeros(
                            (node_onehot.shape[0], 0), device=self.device
                        )
                    paired_sparse_noisy_data = {
                        "node_t": node_onehot[diag_new_to_old],
                        "X_t": node_onehot[diag_new_to_old],
                        "edge_index_t": diag_old_to_new[fw_edge_index],
                        "edge_attr_t": fw_edge_attr_onehot,
                        "comp_edge_index_t": diag_old_to_new[comp_edge_index],
                        "comp_edge_attr_t": comp_edge_attr,
                        "y_t": data.y,
                        "batch": batch[diag_new_to_old],
                        "ptr": ptr,
                        "charge_t": diag_charge[diag_new_to_old],
                        "t_float": t_float,
                    }
                    if sampling_residual_scale is not None:
                        paired_sparse_noisy_data["edge_input_residual_scale"] = float(
                            sampling_residual_scale
                        )
                    if sampling_structure_lookup is not None:
                        paired_visible = fw_edge_attr_onehot.argmax(dim=-1).long() > 0
                        paired_visible_edge_index = diag_old_to_new[
                            fw_edge_index[:, paired_visible]
                        ]
                        paired_visible_edge_attr = fw_edge_attr_onehot[paired_visible]
                        paired_metadata = self._heterogeneous_metadata_for_edges(
                            paired_sparse_noisy_data["node_t"],
                            paired_visible_edge_attr,
                            paired_visible_edge_index,
                        )
                        paired_sparse_noisy_data["two_hop_structure_lookup"] = (
                            self.model.build_two_hop_structure_lookup(
                                context_edge_index=paired_visible_edge_index,
                                num_nodes=int(node_onehot.shape[0]),
                                node_type_ids=paired_metadata.get("node_type_ids"),
                                edge_family_ids=paired_metadata.get(
                                    "edge_family_ids"
                                ),
                                batch=paired_sparse_noisy_data["batch"],
                            )
                        )
                    paired_pred = self.forward(paired_sparse_noisy_data)
                    paired_comp_query_logits = paired_pred.edge_attr[query_mask]
                    if paired_comp_query_logits.shape != comp_query_logits.shape:
                        raise RuntimeError(
                            "paired forward changed query-logit shape: "
                            f"{tuple(comp_query_logits.shape)} vs "
                            f"{tuple(paired_comp_query_logits.shape)}"
                        )
                    raw_diff = (
                        comp_query_logits - paired_comp_query_logits
                    ).abs()
                    original_exist = (
                        torch.logsumexp(comp_query_logits[:, 1:], dim=-1)
                        - comp_query_logits[:, 0]
                    )
                    paired_exist = (
                        torch.logsumexp(paired_comp_query_logits[:, 1:], dim=-1)
                        - paired_comp_query_logits[:, 0]
                    )
                    exist_diff = (original_exist - paired_exist).abs()
                    prob_diff = (
                        torch.sigmoid(original_exist)
                        - torch.sigmoid(paired_exist)
                    ).abs()
                    forward_equivariance_stats["count"] += int(
                        original_exist.numel()
                    )
                    forward_equivariance_stats["raw_element_count"] += int(
                        raw_diff.numel()
                    )
                    forward_equivariance_stats["raw_abs_sum"] += float(
                        raw_diff.sum().detach().cpu()
                    )
                    forward_equivariance_stats["raw_abs_max"] = max(
                        forward_equivariance_stats["raw_abs_max"],
                        float(raw_diff.max().detach().cpu()),
                    )
                    forward_equivariance_stats["exist_abs_sum"] += float(
                        exist_diff.sum().detach().cpu()
                    )
                    forward_equivariance_stats["exist_abs_max"] = max(
                        forward_equivariance_stats["exist_abs_max"],
                        float(exist_diff.max().detach().cpu()),
                    )
                    forward_equivariance_stats["prob_abs_sum"] += float(
                        prob_diff.sum().detach().cpu()
                    )
                    forward_equivariance_stats["prob_abs_max"] = max(
                        forward_equivariance_stats["prob_abs_max"],
                        float(prob_diff.max().detach().cpu()),
                    )
                    forward_equivariance_stats["exist_original"].append(
                        original_exist.detach().float().cpu()
                    )
                    forward_equivariance_stats["exist_permuted"].append(
                        paired_exist.detach().float().cpu()
                    )
                if comp_query_logits.numel() == 0:
                    continue
                selected_query_edge_index, selected_query_family, query_logits = self._select_original_query_outputs(
                    query_edge_index, query_edge_family, comp_query_edge_index, comp_query_logits
                )
                if use_forward_equivariance_diag:
                    (
                        paired_selected_query_edge_index,
                        paired_selected_query_family,
                        paired_query_logits,
                    ) = self._select_original_query_outputs(
                        diag_old_to_new[query_edge_index],
                        query_edge_family,
                        diag_old_to_new[comp_query_edge_index],
                        paired_comp_query_logits,
                    )
                    if (
                        paired_query_logits.shape != query_logits.shape
                        or not torch.equal(
                            paired_selected_query_family, selected_query_family
                        )
                    ):
                        raise RuntimeError(
                            "paired query selection changed candidate alignment"
                        )
            if query_logits.numel() == 0 or selected_query_edge_index.numel() == 0:
                continue
            if selected_query_batch is None:
                selected_query_batch = batch[selected_query_edge_index[0].long()]
            if current_query_labels is None:
                current_query_labels = self._lookup_query_edge_labels(
                    selected_query_edge_index,
                    fw_edge_index,
                    fw_edge_attr_onehot.argmax(dim=-1).long(),
                    total_nodes,
                )
            if verbose_sampling:
                raw_masked = self._mask_edge_logits_by_query_family(
                    query_logits, selected_query_family
                )
                raw_exist_logits = (
                    torch.logsumexp(raw_masked[:, 1:], dim=-1) - raw_masked[:, 0]
                )
            query_logits = self._calibrate_edge_logits_for_exist_pos_weight(
                query_logits,
                selected_query_family,
            )
            if use_ranking_intervention_diag:
                self._update_global_exact_k_candidates(
                    clean_exact_k_candidate_parts,
                    query_logits,
                    selected_query_family,
                    selected_query_edge_index,
                    selected_query_batch,
                    total_nodes,
                )
            if use_forward_equivariance_diag:
                paired_query_logits = (
                    self._calibrate_edge_logits_for_exist_pos_weight(
                        paired_query_logits,
                        paired_selected_query_family,
                    )
                )
            if use_edge_score_structure_diag:
                diag_masked = self._mask_edge_logits_by_query_family(
                    query_logits, selected_query_family
                )
                diag_scores = (
                    torch.logsumexp(diag_masked[:, 1:], dim=-1)
                    - diag_masked[:, 0]
                )
                diag_u = torch.minimum(
                    selected_query_edge_index[0].long(),
                    selected_query_edge_index[1].long(),
                )
                diag_v = torch.maximum(
                    selected_query_edge_index[0].long(),
                    selected_query_edge_index[1].long(),
                )
                edge_score_structure_diag_parts.append(
                    {
                        "edge_index": torch.stack([diag_u, diag_v], dim=0),
                        "scores": diag_scores,
                        "family": selected_query_family.detach(),
                    }
                )
            if verbose_sampling:
                calibrated_exist_logits = (
                    torch.logsumexp(query_logits[:, 1:], dim=-1) - query_logits[:, 0]
                )
            query_logits = self._edge_reverse_posterior_logits(
                query_logits,
                selected_query_family,
                selected_query_batch,
                current_query_labels,
                s_float,
                t_float,
            )
            if use_exactk_equivariance_diag:
                paired_current_query_labels = self._lookup_query_edge_labels(
                    paired_selected_query_edge_index,
                    diag_old_to_new[fw_edge_index],
                    fw_edge_attr_onehot.argmax(dim=-1).long(),
                    total_nodes,
                )
                paired_query_logits = self._edge_reverse_posterior_logits(
                    paired_query_logits,
                    paired_selected_query_family,
                    selected_query_batch,
                    paired_current_query_labels,
                    s_float,
                    t_float,
                )
            if verbose_sampling:
                posterior_masked = self._mask_edge_logits_by_query_family(
                    query_logits, selected_query_family
                )
                posterior_exist_logits = (
                    torch.logsumexp(posterior_masked[:, 1:], dim=-1)
                    - posterior_masked[:, 0]
                )
                query_count = int(query_logits.shape[0])
                step_diag["query_count"] += query_count
                step_diag["raw_clean_p_sum"] += float(
                    torch.sigmoid(raw_exist_logits).sum().detach().cpu()
                )
                step_diag["calibrated_clean_p_sum"] += float(
                    torch.sigmoid(calibrated_exist_logits).sum().detach().cpu()
                )
                step_diag["posterior_p_sum"] += float(
                    torch.sigmoid(posterior_exist_logits).sum().detach().cpu()
                )
                posterior_exist_prob = torch.sigmoid(posterior_exist_logits)
                current_pos_mask = current_query_labels > 0
                current_nonpos_mask = ~current_pos_mask
                if current_pos_mask.any():
                    step_diag["current_edge_p_sum"] += float(
                        posterior_exist_prob[current_pos_mask].sum().detach().cpu()
                    )
                    step_diag["current_edge_p_count"] += int(
                        current_pos_mask.sum().item()
                    )
                if current_nonpos_mask.any():
                    step_diag["current_nonedge_p_sum"] += float(
                        posterior_exist_prob[current_nonpos_mask].sum().detach().cpu()
                    )
                    step_diag["current_nonedge_p_count"] += int(
                        current_nonpos_mask.sum().item()
                    )
            if use_connectivity_topk:
                self._update_connectivity_candidate_pools(
                    connectivity_pools,
                    connectivity_stats,
                    data,
                    query_logits,
                    selected_query_family,
                    selected_query_edge_index,
                    selected_query_batch,
                    total_nodes,
                )
                continue
            if use_global_exact_k:
                self._update_global_exact_k_candidates(
                    global_exact_k_candidate_parts,
                    query_logits,
                    selected_query_family,
                    selected_query_edge_index,
                    selected_query_batch,
                    total_nodes,
                )
                if use_exactk_equivariance_diag:
                    self._update_global_exact_k_candidates(
                        paired_exact_k_candidate_parts,
                        paired_query_logits,
                        paired_selected_query_family,
                        paired_selected_query_edge_index,
                        selected_query_batch,
                        total_nodes,
                    )
                continue
            if use_topk_density_repair:
                self._update_topk_density_repair_candidates(
                    repair_candidate_parts,
                    query_logits,
                    selected_query_family,
                    selected_query_edge_index,
                    selected_query_batch,
                    total_nodes,
                )
            selection_override = None
            if (
                bool(getattr(self.cfg.model, "sampling_use_reverse_posterior", False))
                and bool(
                    getattr(
                        self.cfg.model,
                        "sampling_reverse_posterior_stochastic_steps",
                        True,
                    )
                )
                and not bool(is_final_sampling_step)
            ):
                configured_selection = str(
                    getattr(
                        self.cfg.model,
                        "sampling_edge_selection",
                        "bernoulli",
                    )
                    or "bernoulli"
                ).lower()
                selection_override = (
                    "bernoulli_expected_density"
                    if configured_selection == "bernoulli_expected_density"
                    else "bernoulli"
                )
            query_sample = self._sample_edge_labels_hierarchical(
                query_logits,
                selected_query_family,
                selected_query_edge_index,
                selection_override=selection_override,
            )
            if verbose_sampling:
                current_pos = current_query_labels > 0
                sampled_pos = query_sample > 0
                step_diag["current_query_pos"] += int(current_pos.sum().item())
                step_diag["sampled_query_pos"] += int(sampled_pos.sum().item())
                step_diag["query_added"] += int(
                    ((~current_pos) & sampled_pos).sum().item()
                )
                step_diag["query_removed"] += int(
                    (current_pos & (~sampled_pos)).sum().item()
                )
                step_diag["query_retained"] += int(
                    (current_pos & sampled_pos).sum().item()
                )

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

        if use_edge_score_structure_diag:
            self._finalize_edge_score_structure_diag(
                edge_score_structure_diag_parts,
                data.edge_score_diag_reference_edge_index,
                total_nodes,
                t_float,
                s_float,
            )

        if use_forward_equivariance_diag:
            self._log_forward_equivariance_diag(
                forward_equivariance_stats,
                t_float,
                s_float,
            )

        if use_global_exact_k:
            sampling_step = int(
                round(
                    float(t_float.mean().detach().cpu().item())
                    * float(getattr(self.cfg.model, "diffusion_steps", 100))
                )
            )
            selected_edges, selected_labels = (
                self._finalize_global_exact_k_candidates(
                    data,
                    global_exact_k_candidate_parts,
                    selection_mode,
                    sampling_step,
                    apply_connectivity_repair=use_exact_k_connectivity_repair,
                )
            )
            if use_ranking_intervention_diag:
                clean_selected_edges, _ = (
                    self._finalize_global_exact_k_candidates(
                        data,
                        clean_exact_k_candidate_parts,
                        selection_mode,
                        sampling_step,
                        apply_connectivity_repair=False,
                        log_selection=False,
                    )
                )
                self._log_exact_k_ranking_intervention(
                    clean_selected_edges=clean_selected_edges,
                    mixed_selected_edges=selected_edges,
                    current_edge_index=edge_index,
                    candidate_parts=clean_exact_k_candidate_parts,
                    total_nodes=total_nodes,
                    s_float=s_float,
                    t_float=t_float,
                )
            if use_exactk_equivariance_diag:
                paired_selected_edges, _ = (
                    self._finalize_global_exact_k_candidates(
                        data,
                        paired_exact_k_candidate_parts,
                        "deterministic_exact_k",
                        sampling_step,
                        apply_connectivity_repair=False,
                    )
                )
                self._log_exactk_equivariance_diag(
                    selected_edges,
                    paired_selected_edges,
                    diag_new_to_old.detach().cpu(),
                    total_nodes,
                    t_float,
                    s_float,
                )
            cur_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            cur_edge_attr_ids = torch.empty(
                (0,), dtype=torch.long, device=self.device
            )
            if selected_edges is not None and selected_labels is not None:
                cur_edge_index, cur_edge_attr_ids = self._replace_edges_with_labels(
                    cur_edge_index,
                    cur_edge_attr_ids,
                    selected_edges,
                    selected_labels,
                )
            keep = cur_edge_attr_ids != 0
            cur_edge_index = cur_edge_index[:, keep]
            cur_edge_attr_ids = cur_edge_attr_ids[keep].clamp(
                0, self.out_dims.E - 1
            )
            cur_edge_attr_onehot = F.one_hot(
                cur_edge_attr_ids, num_classes=self.out_dims.E
            ).float()
        elif use_connectivity_topk:
            selected_edges, selected_labels = self._finalize_connectivity_candidate_pools(
                data, connectivity_pools, connectivity_stats
            )
            cur_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            cur_edge_attr_ids = torch.empty((0,), dtype=torch.long, device=self.device)
            if selected_edges is not None and selected_labels is not None:
                cur_edge_index, cur_edge_attr_ids = self._replace_edges_with_labels(
                    cur_edge_index, cur_edge_attr_ids, selected_edges, selected_labels
                )
            keep = cur_edge_attr_ids != 0
            cur_edge_index = cur_edge_index[:, keep]
            cur_edge_attr_ids = cur_edge_attr_ids[keep].clamp(0, self.out_dims.E - 1)
            cur_edge_attr_onehot = F.one_hot(
                cur_edge_attr_ids, num_classes=self.out_dims.E
            ).float()
        elif use_topk_density_repair:
            selected_edges, selected_labels = self._finalize_topk_density_repair_candidates(
                data,
                repair_candidate_parts,
                cur_edge_index,
                cur_edge_attr_ids,
            )
            if selected_edges is not None and selected_labels is not None:
                cur_edge_index = selected_edges
                cur_edge_attr_ids = selected_labels.clamp(0, self.out_dims.E - 1)
                cur_edge_attr_onehot = F.one_hot(
                    cur_edge_attr_ids, num_classes=self.out_dims.E
                ).float()

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
        for attr_name in (
            "connectivity_init_edge_index",
            "connectivity_init_edge_labels",
            "connectivity_init_family",
            "connectivity_init_batch",
            "connectivity_block_of",
            "connectivity_family_targets",
            "edge_score_diag_reference_edge_index",
            "forward_equivariance_diag_node_type",
        ):
            if hasattr(data, attr_name):
                setattr(out, attr_name, getattr(data, attr_name))
        out = self._apply_block_family_budget_projection(out)
        if verbose_sampling and getattr(self, "local_rank", 0) == 0:
            before_edges = {
                (min(int(u), int(v)), max(int(u), int(v)))
                for u, v in zip(
                    edge_index[0].detach().cpu().tolist(),
                    edge_index[1].detach().cpu().tolist(),
                )
            }
            after_edges = {
                (min(int(u), int(v)), max(int(u), int(v)))
                for u, v in zip(
                    out.edge_index[0].detach().cpu().tolist(),
                    out.edge_index[1].detach().cpu().tolist(),
                )
            }
            retained_edges = len(before_edges & after_edges)
            added_edges = len(after_edges - before_edges)
            removed_edges = len(before_edges - after_edges)
            previous_overlap = (
                retained_edges / max(1, len(before_edges | after_edges))
            )
            query_count = max(1, int(step_diag["query_count"]))
            current_edge_p = (
                step_diag["current_edge_p_sum"]
                / max(1, int(step_diag["current_edge_p_count"]))
            )
            current_nonedge_p = (
                step_diag["current_nonedge_p_sum"]
                / max(1, int(step_diag["current_nonedge_p_count"]))
            )
            s_value = float(s_float.mean().detach().cpu())
            t_value = float(t_float.mean().detach().cpu())
            before_triangles = self._count_total_triangles_sparse(
                edge_index,
                edge_attr_ids,
                total_nodes,
            )
            out_edge_labels = (
                out.edge_attr.argmax(dim=-1).long()
                if out.edge_attr.dim() > 1
                else out.edge_attr.long()
            )
            after_triangles = self._count_total_triangles_sparse(
                out.edge_index,
                out_edge_labels,
                total_nodes,
            )
            print(
                "[采样-STEP] "
                f"s={s_value:.4f} "
                f"t={t_value:.4f} "
                f"delta_t={t_value - s_value:.4f} "
                f"edges={len(before_edges)}->{len(after_edges)} "
                f"triangles={before_triangles}->{after_triangles} "
                f"global_keep/add/del={retained_edges}/{added_edges}/{removed_edges} "
                f"prev_jaccard={previous_overlap:.6f} "
                f"clean_p={step_diag['raw_clean_p_sum'] / query_count:.6f}"
                f"->{step_diag['calibrated_clean_p_sum'] / query_count:.6f} "
                f"posterior_p={step_diag['posterior_p_sum'] / query_count:.6f} "
                f"posterior_current_edge/nonedge="
                f"{current_edge_p:.6f}/{current_nonedge_p:.6f} "
                f"query_pos={step_diag['current_query_pos']}/{query_count}"
                f"->{step_diag['sampled_query_pos']}/{query_count} "
                f"query_keep/add/del={step_diag['query_retained']}/"
                f"{step_diag['query_added']}/{step_diag['query_removed']}"
            )
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
        two_hop_scale_factor = self._two_hop_reliability_factor(t_float)
        endpoint_role_scale_factor = (
            self._endpoint_role_reliability_factor(t_float)
        )

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
            if self.sparse_hetero_y_dim > 0:
                y = torch.hstack(
                    [y, self._compute_sparse_hetero_y(sparse_noisy_data)]
                ).float()
            prepared = {
                "node_t": node,
                "edge_index_t": sparse_noisy_data["comp_edge_index_t"],
                "edge_attr_t": comp_edge_attr_t,
                "y_t": y,
                "batch": sparse_noisy_data["batch"],
                "charge_t": sparse_noisy_data["charge_t"],
            }
            if self.sparse_family_y_dim > 0:
                prepared["family_y_t"] = self._compute_sparse_family_y(
                    sparse_noisy_data
                )
            if bool(getattr(self.cfg.model, "use_edge_struct_features", False)):
                prepared["edge_struct_features_t"] = self._compute_edge_struct_features(
                    sparse_noisy_data
                )
            if "two_hop_structure_lookup" in sparse_noisy_data:
                prepared["two_hop_structure_lookup"] = sparse_noisy_data[
                    "two_hop_structure_lookup"
                ]
            if two_hop_scale_factor is not None:
                prepared["two_hop_scale_factor"] = two_hop_scale_factor
            if endpoint_role_scale_factor is not None:
                prepared["endpoint_role_scale_factor"] = (
                    endpoint_role_scale_factor
                )
            return prepared

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
        if self.sparse_hetero_y_dim > 0:
            y = torch.hstack(
                [y, self._compute_sparse_hetero_y(sparse_noisy_data)]
            ).float()

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
        if self.sparse_family_y_dim > 0:
            extra_sparse_noisy_data["family_y_t"] = self._compute_sparse_family_y(
                sparse_noisy_data
            )
        if bool(getattr(self.cfg.model, "use_edge_struct_features", False)):
            extra_sparse_noisy_data["edge_struct_features_t"] = (
                self._compute_edge_struct_features(sparse_noisy_data)
            )
        if "edge_input_residual_scale" in sparse_noisy_data:
            extra_sparse_noisy_data["edge_input_residual_scale"] = (
                sparse_noisy_data["edge_input_residual_scale"]
            )
        if "two_hop_structure_lookup" in sparse_noisy_data:
            extra_sparse_noisy_data["two_hop_structure_lookup"] = (
                sparse_noisy_data["two_hop_structure_lookup"]
            )
        if two_hop_scale_factor is not None:
            extra_sparse_noisy_data["two_hop_scale_factor"] = (
                two_hop_scale_factor
            )
        if endpoint_role_scale_factor is not None:
            extra_sparse_noisy_data["endpoint_role_scale_factor"] = (
                endpoint_role_scale_factor
            )

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
