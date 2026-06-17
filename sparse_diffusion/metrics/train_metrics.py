import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(
        self,
        lambda_train,
        edge_fraction,
        dataset_info=None,
        exist_pos_weight=None,
        exist_loss_type="bce",
        exist_focal_gamma=2.0,
        exist_focal_alpha=0.75,
        edge_neg_weight=1.0,
        relation_matrix_loss_weight=1.0,
        relation_matrix_loss_normalize=True,
        metapath2_loss_weight=1.0,
        metapath2_loss_normalize=True,
        metapath3_loss_weight=1.0,
        metapath3_loss_normalize=True,
        subtype_degree_loss_weight=1.0,
        subtype_degree_loss_normalize=True,
        subtype_degree_max=100,
        subtype_degree_use_full_graph_true=False,
        structure_loss_max_edges=0,
        edge_only_model=False,
        # 新结构指标：替换 relation_matrix/metapath/subtype_degree 用于 ideal
        structure_loss_type="legacy",  # "legacy" | "graph_metrics"
        degree_mmd_loss_weight=1.0,
        clustering_loss_weight=1.0,
        triangles_loss_weight=1.0,
        wedge_closure_loss_weight=0.0,
        edge_types_tv_loss_weight=1.0,
        structure_loss_max_nodes=3000,  # 节点数超过此值则跳过 A_node（避免 OOM）
        structure_triangles_normalize=True,  # 三角形数用相对误差归一化，避免主导梯度
    ):
        super().__init__()
        # 训练/验证指标与损失：仅保留 query 边上的「存在性 BCE + 正边子类别 CE」及准确率；结构统计仅用于 ideal 构造。
        # 其它（节点、结构L1、metapath 等）不再参与 forward；但结构相关 helper 仍保留以支持 ideal_at_t 构造。
        self.lambda_train = lambda_train
        self.lambda_train[0] = self.lambda_train[0] / edge_fraction

        # Loss weights (λ1, λ2)
        self.edge_exist_weight = 1.0
        self.edge_subtype_weight = 1.2
        # Existence loss: BCE by default; optional focal BCE for highly imbalanced query edges.
        self.exist_pos_weight = exist_pos_weight
        self.exist_loss_type = str(exist_loss_type).lower()
        self.exist_focal_gamma = float(exist_focal_gamma)
        self.exist_focal_alpha = float(exist_focal_alpha)

        # Epoch accumulators (weighted by number of relevant edges)
        self._epoch_exist_loss_sum = None
        self._epoch_exist_loss_count = None
        self._epoch_subtype_loss_sum = None
        self._epoch_subtype_loss_count = None
        self._epoch_tp = None
        self._epoch_pos = None
        self._epoch_tn = None
        self._epoch_neg = None
        self._epoch_subtype_correct_on_tp = None
        self._epoch_subtype_tp = None

        # (Legacy fields kept for compatibility with older configs / helper methods)
        self.pos_weight_per_edge_type = None
        self.edge_family_ranges = {}
        self.edge_family_multisub = set()
        self.edge_neg_weight = float(edge_neg_weight)
        self.edge_only_model = bool(edge_only_model)
        self.relation_matrix_loss_weight = float(relation_matrix_loss_weight)
        self.relation_matrix_loss_normalize = bool(relation_matrix_loss_normalize)
        self.relation_matrix_loss_total = None
        self.relation_matrix_loss_steps = None
        self.metapath2_loss_weight = float(metapath2_loss_weight)
        self.metapath2_loss_normalize = bool(metapath2_loss_normalize)
        self.metapath2_loss_total = None
        self.metapath2_loss_steps = None
        self.metapath3_loss_weight = float(metapath3_loss_weight)
        self.metapath3_loss_normalize = bool(metapath3_loss_normalize)
        self.metapath3_loss_total = None
        self.metapath3_loss_steps = None
        self.subtype_degree_loss_weight = float(subtype_degree_loss_weight)
        self.subtype_degree_loss_normalize = bool(subtype_degree_loss_normalize)
        self.subtype_degree_max = int(subtype_degree_max)
        self.subtype_degree_use_full_graph_true = bool(subtype_degree_use_full_graph_true)
        # 结构损失边数限制：<=0 不截断（全量），>0 时最多取该条边做结构统计；默认 0
        self.structure_loss_max_edges = int(structure_loss_max_edges)
        self.structure_loss_type = str(structure_loss_type)
        self.degree_mmd_loss_weight = float(degree_mmd_loss_weight)
        self.clustering_loss_weight = float(clustering_loss_weight)
        self.triangles_loss_weight = float(triangles_loss_weight)
        self.wedge_closure_loss_weight = float(wedge_closure_loss_weight)
        self.edge_types_tv_loss_weight = float(edge_types_tv_loss_weight)
        self.structure_loss_max_nodes = int(structure_loss_max_nodes)
        self.structure_triangles_normalize = bool(structure_triangles_normalize)
        self.subtype_degree_loss_total = None
        self.subtype_degree_loss_steps = None
        self.query_ideal_ce_total = None
        self.query_ideal_ce_steps = None
        self.type_offsets = {}
        self.type_sizes = {}
        self.node_type_names = []
        self.fam_endpoints = {}
        self.type_name2idx = {}
        self.num_edge_types = None

        if (
            dataset_info is not None
            and getattr(dataset_info, "edge_family_marginals", None)
            and getattr(dataset_info, "edge_family_offsets", None)
        ):
            edge_family_marginals = dataset_info.edge_family_marginals
            edge_family_offsets = dataset_info.edge_family_offsets
            num_edge_types = getattr(dataset_info, "output_dims", None)
            if num_edge_types is not None:
                num_edge_types = num_edge_types.E
            if not num_edge_types:
                num_edge_types = len(getattr(dataset_info, "bond_types", []))
            if num_edge_types:
                pos_weight = torch.ones(num_edge_types, dtype=torch.float)
                for fam_name, marginals in edge_family_marginals.items():
                    if not isinstance(marginals, torch.Tensor):
                        marginals = torch.tensor(marginals, dtype=torch.float)
                    u0 = float(marginals[0].item()) if marginals.numel() > 0 else 0.0
                    u1 = max(1.0 - u0, 1e-6)
                    w = (1.0 - u1) / u1
                    w = float(min(max(w, 1.0), 100.0))
                    offset = edge_family_offsets.get(fam_name, 0)
                    next_offset = num_edge_types
                    for _, o in edge_family_offsets.items():
                        if o > offset and o < next_offset:
                            next_offset = o
                    for gid in range(offset, next_offset):
                        if 0 <= gid < num_edge_types:
                            pos_weight[gid] = w
                pos_weight[0] = 1.0
                # Build family ranges for per-family metrics
                edge_family_offsets = dataset_info.edge_family_offsets
                fam_sorted = sorted(edge_family_offsets.items(), key=lambda x: x[1])
                for fam_name, offset in fam_sorted:
                    next_offset = num_edge_types
                    for _, off2 in fam_sorted:
                        if off2 > offset and off2 < next_offset:
                            next_offset = off2
                    self.edge_family_ranges[fam_name] = (offset, next_offset)
                    if (next_offset - offset) > 1:
                        self.edge_family_multisub.add(fam_name)
        if dataset_info is not None:
            self.type_offsets = getattr(dataset_info, "type_offsets", {}) or {}
            self.node_type_names = getattr(dataset_info, "node_type_names", []) or []
            self.fam_endpoints = getattr(dataset_info, "fam_endpoints", {}) or {}
            if self.type_offsets:
                sorted_types = sorted(self.type_offsets.items(), key=lambda x: x[1])
                self.type_name2idx = {t: i for i, (t, _) in enumerate(sorted_types)}
                num_subtypes_total = None
                out_dims = getattr(dataset_info, "output_dims", None)
                if out_dims is not None and hasattr(out_dims, "X"):
                    num_subtypes_total = int(out_dims.X)
                elif getattr(dataset_info, "input_dims", None) is not None and hasattr(dataset_info.input_dims, "X"):
                    num_subtypes_total = int(dataset_info.input_dims.X)
                else:
                    num_subtypes_total = 0
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        self.type_sizes[t_name] = int(sorted_types[i + 1][1] - off)
                    else:
                        self.type_sizes[t_name] = max(int(num_subtypes_total - off), 0)
            out_dims = getattr(dataset_info, "output_dims", None)
            if out_dims is not None and hasattr(out_dims, "E"):
                try:
                    self.num_edge_types = int(out_dims.E)
                except Exception:
                    self.num_edge_types = None

    def _subsample_structure_edges(self, pred, true_data):
        """Optionally subsample edges for structure losses to cap per-step cost.
        当 pred 与 true_data 边集不同（如 pred=comp 噪声+query、true=真实图）时，将 true 对齐到 pred 的边集：
        对 pred 的每条边在 true 中查找，存在则取该边属性，否则取 no-edge。
        """
        num_edges = pred.edge_attr.shape[0]
        num_classes = pred.edge_attr.shape[-1]
        device = pred.edge_attr.device
        dtype = pred.edge_attr.dtype
        pred_ei = pred.edge_index
        pred_ea = pred.edge_attr
        true_ei = true_data.edge_index
        true_ea = true_data.edge_attr
        if true_ea.dim() == 1:
            true_ea = torch.nn.functional.one_hot(
                true_ea.long().clamp(min=0), num_classes=num_classes
            ).to(device=device, dtype=dtype)
        if pred_ei.shape[1] != true_ei.shape[1] or not torch.equal(pred_ei, true_ei):
            true_aligned = torch.zeros(
                num_edges, num_classes, device=device, dtype=dtype
            )
            true_aligned[:, 0] = 1.0
            if true_ei.numel() > 0:
                batch_pred = getattr(pred, "batch", None)
                batch_true = getattr(true_data, "batch", None)
                n_max = max(
                    pred_ei[0].max().item(),
                    pred_ei[1].max().item(),
                    true_ei[0].max().item(),
                    true_ei[1].max().item(),
                ) + 1
                if batch_pred is not None and batch_true is not None:
                    key_pred = (
                        batch_pred[pred_ei[0]].long() * (n_max * n_max)
                        + pred_ei[0].long() * n_max
                        + pred_ei[1].long()
                    )
                    key_true = (
                        batch_true[true_ei[0]].long() * (n_max * n_max)
                        + true_ei[0].long() * n_max
                        + true_ei[1].long()
                    )
                else:
                    key_pred = pred_ei[0].long() * n_max + pred_ei[1].long()
                    key_true = true_ei[0].long() * n_max + true_ei[1].long()
                true_idx = {k: j for j, k in enumerate(key_true.tolist())}
                for i in range(num_edges):
                    k = key_pred[i].item()
                    if k in true_idx:
                        j = true_idx[k]
                        true_aligned[i] = true_ea[j]
            true_ea = true_aligned
        max_edges = self.structure_loss_max_edges
        if max_edges <= 0 or num_edges <= max_edges:
            return pred_ei, pred_ea, true_ea
        perm = torch.randperm(num_edges, device=device)[:max_edges]
        return pred_ei[:, perm], pred_ea[perm], true_ea[perm]

    def _compute_structure_stats(self, pred, true_data):
        """Compute all structure statistics in one pass."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return None
        if true_data.node.numel() == 0:
            return None

        edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            return None

        if true_data.node.dim() > 1:
            node_sub = true_data.node.argmax(dim=-1)
        else:
            node_sub = true_data.node.long()
        num_sub = pred.node.shape[-1]
        if pred_edge_attr.dim() > 1:
            num_edge_types = int(pred_edge_attr.shape[-1])
        elif true_edge_attr.dim() > 1:
            num_edge_types = int(true_edge_attr.shape[-1])
        elif getattr(self, "num_edge_types", None) is not None:
            num_edge_types = int(self.num_edge_types)
        else:
            pred_max = int(pred_edge_attr.max().item()) if pred_edge_attr.numel() > 0 else 0
            true_max = int(true_edge_attr.max().item()) if true_edge_attr.numel() > 0 else 0
            num_edge_types = max(pred_max, true_max) + 1
        num_edge_types = max(num_edge_types, 2)
        device = pred_edge_attr.device
        if pred_edge_attr.dtype.is_floating_point:
            dtype = pred_edge_attr.dtype
        else:
            dtype = torch.float32

        src_sub = node_sub[src]
        dst_sub = node_sub[dst]
        if true_edge_attr.dim() > 1:
            true_labels = true_edge_attr.argmax(dim=-1)
        else:
            true_labels = true_edge_attr.long()
        if pred_edge_attr.dtype in (torch.long, torch.int):
            if pred_edge_attr.dim() > 1:
                pred_prob = torch.softmax(pred_edge_attr.float(), dim=-1)
            else:
                pred_prob = F.one_hot(
                    pred_edge_attr.long().clamp(min=0),
                    num_classes=num_edge_types,
                ).float().to(device=device)
        else:
            pred_prob = torch.softmax(pred_edge_attr, dim=-1)
        pred_prob = pred_prob.to(dtype)

        def _scatter_add_base(size, dim, index, src):
            """Out-of-place scatter_add with base tied to pred for gradient flow."""
            base = torch.zeros(size, device=device, dtype=dtype) + 0.0 * pred_prob.sum()
            return torch.scatter_add(base, dim, index, src)

        # Family-isolated first-order relation statistics:
        # per family tensor over (src_subtype, dst_subtype, edge_subtype), support soft counts.
        family_relation_items = []
        if self.fam_endpoints and self.edge_family_ranges and self.type_offsets:
            node_type_ids = torch.full_like(node_sub, -1)
            sorted_types = sorted(self.type_offsets.items(), key=lambda x: x[1])
            for i, (t_name, off) in enumerate(sorted_types):
                if i + 1 < len(sorted_types):
                    next_off = sorted_types[i + 1][1]
                    t_mask = (node_sub >= off) & (node_sub < next_off)
                else:
                    t_mask = node_sub >= off
                node_type_ids[t_mask] = i

            src_type_ids = node_type_ids[src]
            dst_type_ids = node_type_ids[dst]
            for fam_name, (st, en) in self.edge_family_ranges.items():
                ep = self.fam_endpoints.get(fam_name, {})
                src_t = ep.get("src_type", None)
                dst_t = ep.get("dst_type", None)
                if src_t is None or dst_t is None:
                    continue
                if src_t not in self.type_name2idx or dst_t not in self.type_name2idx:
                    continue
                src_tid = self.type_name2idx[src_t]
                dst_tid = self.type_name2idx[dst_t]
                fam_mask = (src_type_ids == src_tid) & (dst_type_ids == dst_tid)
                if not fam_mask.any():
                    continue

                src_off = int(self.type_offsets.get(src_t, 0))
                dst_off = int(self.type_offsets.get(dst_t, 0))
                src_size = int(self.type_sizes.get(src_t, 0))
                dst_size = int(self.type_sizes.get(dst_t, 0))
                fam_subtypes = int(max(en - st, 0))
                if src_size <= 0 or dst_size <= 0 or fam_subtypes <= 0:
                    continue

                src_local = (src_sub[fam_mask] - src_off).long()
                dst_local = (dst_sub[fam_mask] - dst_off).long()
                valid_local = (
                    (src_local >= 0) & (src_local < src_size)
                    & (dst_local >= 0) & (dst_local < dst_size)
                )
                if not valid_local.any():
                    continue
                src_local = src_local[valid_local]
                dst_local = dst_local[valid_local]
                fam_true_labels = true_labels[fam_mask][valid_local]
                fam_pred_prob = pred_prob[fam_mask][valid_local]

                true_hist = torch.zeros(src_size * dst_size * fam_subtypes, device=device, dtype=dtype)
                # true: hard count only for this family's real subtype ids
                true_fam_mask = (fam_true_labels >= st) & (fam_true_labels < en)
                if true_fam_mask.any():
                    true_local_sub = (fam_true_labels[true_fam_mask] - st).long()
                    true_base = (src_local[true_fam_mask] * dst_size + dst_local[true_fam_mask]) * fam_subtypes
                    true_idx = true_base + true_local_sub
                    true_hist.scatter_add_(0, true_idx, torch.ones_like(true_idx, dtype=dtype))

                # pred: soft counts (base tied to pred_prob so gradient flows)
                fam_pred_slice = fam_pred_prob[:, st:en]
                fam_base = (src_local * dst_size + dst_local) * fam_subtypes
                fam_offsets = torch.arange(fam_subtypes, device=device).view(1, -1)
                fam_pred_idx = (fam_base.view(-1, 1) + fam_offsets).reshape(-1)
                pred_hist = _scatter_add_base(
                    src_size * dst_size * fam_subtypes, 0, fam_pred_idx, fam_pred_slice.reshape(-1)
                )
                family_relation_items.append((true_hist, pred_hist, src_size, dst_size, fam_subtypes))
        else:
            true_flat_idx = ((src_sub * num_sub + dst_sub) * num_edge_types + true_labels)
            true_rel_hist = torch.zeros(
                num_sub * num_sub * num_edge_types, device=device, dtype=dtype
            )
            true_rel_hist.scatter_add_(0, true_flat_idx, torch.ones_like(true_flat_idx, dtype=dtype))

            base_pair = (src_sub * num_sub + dst_sub) * num_edge_types
            type_offsets = torch.arange(num_edge_types, device=device).view(1, -1)
            pred_flat_idx = (base_pair.view(-1, 1) + type_offsets).reshape(-1)
            pred_rel_hist = _scatter_add_base(
                num_sub * num_sub * num_edge_types, 0, pred_flat_idx, pred_prob.reshape(-1)
            )
            family_relation_items.append((true_rel_hist, pred_rel_hist, num_sub, num_sub, num_edge_types))

        # Existence-based subtype adjacency
        pred_exist = 1.0 - pred_prob[:, 0]
        true_exist = (true_labels > 0).to(dtype=dtype)
        pair_idx = src_sub * num_sub + dst_sub
        A_true = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_true.scatter_add_(0, pair_idx, true_exist)
        A_true = A_true.view(num_sub, num_sub)
        A_pred = _scatter_add_base(num_sub * num_sub, 0, pair_idx, pred_exist).view(num_sub, num_sub)

        # Node degree histograms by subtype
        num_nodes = node_sub.shape[0]
        num_bins = self.subtype_degree_max + 2
        true_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        true_deg.scatter_add_(0, src, true_exist)
        true_deg.scatter_add_(0, dst, true_exist)
        pred_deg = _scatter_add_base(num_nodes, 0, src, pred_exist)
        pred_deg = torch.scatter_add(pred_deg, 0, dst, pred_exist)

        true_bin = true_deg.round().long().clamp(min=0, max=num_bins - 1)
        true_deg_hist = torch.zeros(num_sub * num_bins, device=device, dtype=dtype)
        true_deg_idx = node_sub * num_bins + true_bin
        true_deg_hist.scatter_add_(0, true_deg_idx, torch.ones_like(true_deg_idx, dtype=dtype))
        true_deg_hist = true_deg_hist.view(num_sub, num_bins)

        deg_cap = float(num_bins - 1)
        pred_deg_cap = pred_deg.clamp(min=0.0, max=deg_cap)
        low = torch.floor(pred_deg_cap).long()
        high = torch.clamp(low + 1, max=num_bins - 1)
        w_high = (pred_deg_cap - low.to(dtype=dtype)).clamp(min=0.0, max=1.0)
        w_low = 1.0 - w_high
        low_idx = node_sub * num_bins + low
        high_idx = node_sub * num_bins + high
        pred_deg_hist = _scatter_add_base(num_sub * num_bins, 0, low_idx, w_low)
        pred_deg_hist = torch.scatter_add(pred_deg_hist, 0, high_idx, w_high).view(num_sub, num_bins)

        # 全局度分布（用于 DegreeMMD）
        true_deg_global_bin = true_deg.round().long().clamp(min=0, max=num_bins - 1)
        true_deg_global_hist = torch.zeros(num_bins, device=device, dtype=dtype)
        true_deg_global_hist.scatter_add_(0, true_deg_global_bin, torch.ones_like(true_deg_global_bin, dtype=dtype))
        pred_deg_global_hist = _scatter_add_base(num_bins, 0, low, w_low)
        pred_deg_global_hist = torch.scatter_add(pred_deg_global_hist, 0, high, w_high)

        # 边类型分布（用于 EdgeTypesTV）
        true_type_onehot = F.one_hot(true_labels.clamp(min=0), num_classes=num_edge_types).float()
        true_type_dist = true_type_onehot.sum(dim=0)
        pred_type_dist = pred_prob.sum(dim=0)
        n_edges = pred_prob.shape[0]
        if n_edges > 0:
            true_type_dist = true_type_dist / true_type_dist.sum().clamp(min=1e-8)
            pred_type_dist = pred_type_dist / pred_type_dist.sum().clamp(min=1e-8)

        # 节点级邻接矩阵（用于 Clustering/Triangles/WedgeClosure）：
        # 默认保持稀疏边列表计算，只有在确实需要这些高阶结构项时才构建 dense 邻接。
        n_node = int(max(src.max().item(), dst.max().item())) + 1
        A_true_node = None
        A_pred_node = None
        need_node_adj = (
            float(getattr(self, "clustering_loss_weight", 0.0)) > 0.0
            or float(getattr(self, "triangles_loss_weight", 0.0)) > 0.0
            or float(getattr(self, "wedge_closure_loss_weight", 0.0)) > 0.0
        )
        if need_node_adj and n_node <= getattr(self, "structure_loss_max_nodes", 3000) and n_node > 0:
            A_true_flat = torch.zeros(n_node * n_node, device=device, dtype=dtype)
            A_true_flat.scatter_add_(0, src * n_node + dst, true_exist)
            A_true_flat.scatter_add_(0, dst * n_node + src, true_exist)
            A_true_node = A_true_flat.view(n_node, n_node)
            # pred 需保持梯度
            idx_ud = torch.cat([src * n_node + dst, dst * n_node + src], dim=0)
            val_ud = torch.cat([pred_exist, pred_exist], dim=0)
            A_pred_node = _scatter_add_base(n_node * n_node, 0, idx_ud, val_ud).view(n_node, n_node)

        return {
            "family_relation_items": family_relation_items,
            "A_true": A_true,
            "A_pred": A_pred,
            "true_deg_hist": true_deg_hist,
            "pred_deg_hist": pred_deg_hist,
            "true_deg_global_hist": true_deg_global_hist,
            "pred_deg_global_hist": pred_deg_global_hist,
            "true_type_dist": true_type_dist,
            "pred_type_dist": pred_type_dist,
            "A_true_node": A_true_node,
            "A_pred_node": A_pred_node,
            "n_node": n_node,
        }

    def _structure_losses_from_stats(self, stats, pred):
        if stats is None:
            zero = pred.edge_attr.sum() * 0.0
            return zero, zero, zero, zero

        family_relation_items = stats["family_relation_items"]
        A_true = stats["A_true"]
        A_pred = stats["A_pred"]
        true_deg_hist = stats["true_deg_hist"]
        pred_deg_hist = stats["pred_deg_hist"]

        rel_losses = []
        for true_hist, pred_hist, src_size, dst_size, fam_subtypes in family_relation_items:
            t = true_hist.view(src_size, dst_size, fam_subtypes)
            p = pred_hist.view(src_size, dst_size, fam_subtypes)
            if self.relation_matrix_loss_normalize:
                # Row-normalize by source subtype to remove scale mismatch across rows/families.
                t_row = t.sum(dim=(1, 2), keepdim=True)
                p_row = p.sum(dim=(1, 2), keepdim=True)
                active_rows = t_row.squeeze(-1).squeeze(-1) > 0
                if active_rows.any():
                    t = t / t_row.clamp(min=1.0)
                    p = p / p_row.clamp(min=1.0)
                    rel_losses.append(
                        torch.nn.functional.l1_loss(p[active_rows], t[active_rows], reduction="mean")
                    )
            else:
                rel_losses.append(torch.nn.functional.l1_loss(p, t, reduction="mean"))
        if rel_losses:
            relation_matrix_loss = torch.stack(rel_losses).mean()
        else:
            relation_matrix_loss = pred.edge_attr.sum() * 0.0

        # 二跳、三跳由一阶 A 递推：B = A@A，C = A@A@A；归一化后做 L1
        if self.metapath2_loss_normalize or self.metapath3_loss_normalize:
            a_scale_true = A_true.sum().clamp(min=1.0)
            a_scale_pred = A_pred.sum().clamp(min=1.0)
            A_true_n = A_true / a_scale_true
            A_pred_n = A_pred / a_scale_pred
        else:
            A_true_n = A_true
            A_pred_n = A_pred
        B_true = A_true_n @ A_true_n
        B_pred = A_pred_n @ A_pred_n
        if self.metapath2_loss_normalize:
            B_true = B_true / B_true.sum().clamp(min=1.0)
            B_pred = B_pred / B_pred.sum().clamp(min=1.0)
        metapath2_loss = torch.nn.functional.l1_loss(B_pred, B_true, reduction="mean")

        C_true = A_true_n @ A_true_n @ A_true_n
        C_pred = A_pred_n @ A_pred_n @ A_pred_n
        if self.metapath3_loss_normalize:
            C_true = C_true / C_true.sum().clamp(min=1.0)
            C_pred = C_pred / C_pred.sum().clamp(min=1.0)
        metapath3_loss = torch.nn.functional.l1_loss(C_pred, C_true, reduction="mean")

        active_rows = true_deg_hist.sum(dim=1) > 0
        if active_rows.any():
            if self.subtype_degree_loss_normalize:
                true_row_sum = true_deg_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
                pred_row_sum = pred_deg_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
                true_deg_hist = true_deg_hist / true_row_sum
                pred_deg_hist = pred_deg_hist / pred_row_sum
            subtype_degree_loss = torch.nn.functional.l1_loss(
                pred_deg_hist[active_rows], true_deg_hist[active_rows], reduction="mean"
            )
        else:
            subtype_degree_loss = pred.edge_attr.sum() * 0.0

        return relation_matrix_loss, metapath2_loss, metapath3_loss, subtype_degree_loss

    def _structure_losses_graph_metrics(self, stats, pred):
        """五项可微结构指标：DegreeMMD、Clustering、Triangles、WedgeClosure、EdgeTypesTV"""
        if stats is None:
            zero = pred.edge_attr.sum() * 0.0
            return zero, zero, zero, zero, zero

        device = pred.edge_attr.device
        dtype = pred.edge_attr.dtype if pred.edge_attr.dtype.is_floating_point else torch.float32
        zero = pred.edge_attr.sum() * 0.0

        # 1. DegreeMMD：全局度分布 L1（可微 MMD 代理）
        true_deg_hist = stats.get("true_deg_global_hist")
        pred_deg_hist = stats.get("pred_deg_global_hist")
        if true_deg_hist is not None and pred_deg_hist is not None:
            t_sum = true_deg_hist.sum().clamp(min=1e-8)
            p_sum = pred_deg_hist.sum().clamp(min=1e-8)
            t_norm = true_deg_hist / t_sum
            p_norm = pred_deg_hist / p_sum
            degree_mmd_loss = F.l1_loss(p_norm, t_norm, reduction="mean")
        else:
            degree_mmd_loss = zero

        # 2. Clustering：平均聚类系数 L1
        # 3. Triangles：三角形数 L1
        # 4. WedgeClosure：在真实图有 2-hop 楔形的位置，约束预测边闭合行为
        A_true_node = stats.get("A_true_node")
        A_pred_node = stats.get("A_pred_node")
        if A_true_node is not None and A_pred_node is not None:
            A3_true = A_true_node @ A_true_node @ A_true_node
            A3_pred = A_pred_node @ A_pred_node @ A_pred_node
            triangles_true = A3_true.diagonal().sum() / 6.0  # 无向图每三角形计 6 次
            triangles_pred = A3_pred.diagonal().sum() / 6.0
            triangles_raw = (triangles_pred - triangles_true).abs()
            if getattr(self, "structure_triangles_normalize", True):
                # 归一化：避免原始计数（10^4~10^6）主导梯度，使其与 DegreeMMD/Clustering/EdgeTypesTV（量级 ~1）相当
                triangles_scale = (triangles_true.abs() + triangles_pred.abs()).clamp(min=1.0) / 2.0
                triangles_loss = (triangles_raw / triangles_scale).clamp(max=10.0)
            else:
                triangles_loss = triangles_raw

            n_node = A_true_node.shape[0]
            deg_true = A_true_node.sum(dim=1).clamp(min=1e-8)
            deg_pred = A_pred_node.sum(dim=1).clamp(min=1e-8)
            tri_per_node_true = A3_true.diagonal() / 2.0  # 每节点参与三角形数
            tri_per_node_pred = A3_pred.diagonal() / 2.0
            denom_true = (deg_true * (deg_true - 1)).clamp(min=1e-8)
            denom_pred = (deg_pred * (deg_pred - 1)).clamp(min=1e-8)
            clust_true = (tri_per_node_true / denom_true).nan_to_num(0.0)
            clust_pred = (tri_per_node_pred / denom_pred).nan_to_num(0.0)
            clustering_loss = F.l1_loss(clust_pred, clust_true, reduction="mean")

            true_adj = (A_true_node > 0).to(dtype=dtype)
            pred_adj = A_pred_node.clamp(min=0.0, max=1.0)
            true_wedge = true_adj @ true_adj
            true_wedge = true_wedge.clone()
            true_wedge.fill_diagonal_(0.0)
            wedge_mask = true_wedge > 0
            if wedge_mask.any():
                # 按真实两跳路径数加权，优先学习“可闭包”的候选边。
                w = true_wedge[wedge_mask].detach()
                w = w / w.mean().clamp(min=1e-8)
                wedge_closure_loss = (w * (pred_adj[wedge_mask] - true_adj[wedge_mask]).abs()).mean()
            else:
                wedge_closure_loss = zero
        else:
            triangles_loss = zero
            clustering_loss = zero
            wedge_closure_loss = zero

        # 5. EdgeTypesTV：边类型分布 Total Variation
        true_type_dist = stats.get("true_type_dist")
        pred_type_dist = stats.get("pred_type_dist")
        if true_type_dist is not None and pred_type_dist is not None:
            edge_types_tv_loss = (0.5 * (pred_type_dist - true_type_dist).abs().sum())
        else:
            edge_types_tv_loss = zero

        return degree_mmd_loss, clustering_loss, triangles_loss, wedge_closure_loss, edge_types_tv_loss

    def _relation_matrix_loss(self, pred, true_data):
        """Subtype-level outgoing relation matrix loss on query edges."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        num_node_subtypes = pred.node.shape[-1]
        edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
        num_edge_types = pred_edge_attr.shape[-1]
        src = edge_index[0].long()
        dst = edge_index[1].long()

        src_sub = true_data.node[src].long()
        dst_sub = true_data.node[dst].long()

        # Build target matrix from true edge labels: M_true[src_sub, dst_sub, edge_type].
        true_labels = true_edge_attr.long()
        true_flat_idx = (
            (src_sub * num_node_subtypes + dst_sub) * num_edge_types + true_labels
        )
        true_hist = torch.zeros(
            num_node_subtypes * num_node_subtypes * num_edge_types,
            device=pred_edge_attr.device,
            dtype=pred_edge_attr.dtype,
        )
        true_hist.scatter_add_(
            0, true_flat_idx, torch.ones_like(true_flat_idx, dtype=pred_edge_attr.dtype)
        )

        # Build predicted matrix from edge-type probabilities.
        pred_prob = torch.softmax(pred_edge_attr, dim=-1)
        base_pair = (src_sub * num_node_subtypes + dst_sub) * num_edge_types
        type_offsets = torch.arange(num_edge_types, device=pred_edge_attr.device).view(1, -1)
        pred_flat_idx = (base_pair.view(-1, 1) + type_offsets).reshape(-1)
        pred_hist = torch.zeros_like(true_hist)
        pred_hist.scatter_add_(0, pred_flat_idx, pred_prob.reshape(-1))

        if self.relation_matrix_loss_normalize:
            true_hist = true_hist / true_hist.sum().clamp(min=1.0)
            pred_hist = pred_hist / pred_hist.sum().clamp(min=1.0)

        return torch.nn.functional.l1_loss(pred_hist, true_hist, reduction="mean")

    def _metapath2_subtype_loss(self, pred, true_data):
        """Two-hop subtype transition loss using subtype adjacency composition (A @ A)."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        node_sub = true_data.node.long()
        num_sub = pred.node.shape[-1]
        device = pred_edge_attr.device
        dtype = pred_edge_attr.dtype

        # Predicted edge existence probability P(edge exists) = 1 - P(no-edge).
        pred_prob = torch.softmax(pred_edge_attr, dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]

        # True edge existence indicator.
        true_labels = true_edge_attr.long()
        true_exist = (true_labels > 0).to(dtype=dtype)
        src_sub = node_sub[src]
        dst_sub = node_sub[dst]

        # Build subtype-level adjacency matrices A_true / A_pred with O(E) scatter.
        flat_idx = src_sub * num_sub + dst_sub
        A_true = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_pred = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_true.scatter_add_(0, flat_idx, true_exist)
        A_pred.scatter_add_(0, flat_idx, pred_exist)
        A_true = A_true.view(num_sub, num_sub)
        A_pred = A_pred.view(num_sub, num_sub)

        if self.metapath2_loss_normalize:
            A_true = A_true / A_true.sum().clamp(min=1.0)
            A_pred = A_pred / A_pred.sum().clamp(min=1.0)

        # Two-hop subtype transitions.
        B_true = A_true @ A_true
        B_pred = A_pred @ A_pred

        if self.metapath2_loss_normalize:
            B_true = B_true / B_true.sum().clamp(min=1.0)
            B_pred = B_pred / B_pred.sum().clamp(min=1.0)

        return torch.nn.functional.l1_loss(B_pred, B_true, reduction="mean")

    def _metapath3_subtype_loss(self, pred, true_data):
        """Three-hop subtype transition loss: C = A @ A @ A (from same A as 2-hop)."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        node_sub = true_data.node.long()
        num_sub = pred.node.shape[-1]
        device = pred_edge_attr.device
        dtype = pred_edge_attr.dtype

        pred_prob = torch.softmax(pred_edge_attr, dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]
        true_labels = true_edge_attr.long()
        true_exist = (true_labels > 0).to(dtype=dtype)
        src_sub = node_sub[src]
        dst_sub = node_sub[dst]

        flat_idx = src_sub * num_sub + dst_sub
        A_true = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_pred = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_true.scatter_add_(0, flat_idx, true_exist)
        A_pred.scatter_add_(0, flat_idx, pred_exist)
        A_true = A_true.view(num_sub, num_sub)
        A_pred = A_pred.view(num_sub, num_sub)

        if self.metapath3_loss_normalize:
            A_true = A_true / A_true.sum().clamp(min=1.0)
            A_pred = A_pred / A_pred.sum().clamp(min=1.0)

        C_true = A_true @ A_true @ A_true
        C_pred = A_pred @ A_pred @ A_pred
        if self.metapath3_loss_normalize:
            C_true = C_true / C_true.sum().clamp(min=1.0)
            C_pred = C_pred / C_pred.sum().clamp(min=1.0)

        return torch.nn.functional.l1_loss(C_pred, C_true, reduction="mean")

    def _subtype_degree_hist_loss(self, pred, true_data):
        """Subtype-level degree distribution loss (histogram over total degree).
        当 subtype_degree_use_full_graph_true=True 时，true_hist 用全图真实度（稳定目标），
        pred_hist 仍用当前 step 的 query 边子集上的预测度，二者按子类型归一化后比较。
        """
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        num_sub = pred.node.shape[-1]
        num_nodes = true_data.node.shape[0]
        device = pred.edge_attr.device
        dtype = pred.edge_attr.dtype
        num_bins = self.subtype_degree_max + 2  # 0..max_degree plus overflow
        node_sub = (
            true_data.node.argmax(dim=-1).long()
            if true_data.node.dim() > 1
            else true_data.node.long()
        )

        use_full_true = self.subtype_degree_use_full_graph_true
        if use_full_true:
            true_src = true_data.edge_index[0].long()
            true_dst = true_data.edge_index[1].long()
            if true_data.edge_attr.dim() > 1:
                true_exist_full = (true_data.edge_attr.argmax(dim=-1) > 0).to(dtype=dtype)
            else:
                true_exist_full = (true_data.edge_attr.long() > 0).to(dtype=dtype)
            true_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
            true_deg.scatter_add_(0, true_src, true_exist_full)
            true_deg.scatter_add_(0, true_dst, true_exist_full)
        else:
            edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
            src = edge_index[0].long()
            dst = edge_index[1].long()
            if src.numel() == 0:
                return pred.edge_attr.sum() * 0.0
            true_labels = true_edge_attr.long()
            if true_edge_attr.dim() > 1:
                true_labels = true_edge_attr.argmax(dim=-1)
            true_exist = (true_labels > 0).to(dtype=dtype)
            true_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
            true_deg.scatter_add_(0, src, true_exist)
            true_deg.scatter_add_(0, dst, true_exist)

        edge_index, pred_edge_attr, true_edge_attr = self._subsample_structure_edges(pred, true_data)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        pred_prob = torch.softmax(pred_edge_attr, dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]
        pred_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        pred_deg.scatter_add_(0, src, pred_exist)
        pred_deg.scatter_add_(0, dst, pred_exist)

        true_bin = true_deg.round().long().clamp(min=0, max=num_bins - 1)
        true_hist = torch.zeros(num_sub * num_bins, device=device, dtype=dtype)
        true_flat_idx = node_sub * num_bins + true_bin
        true_hist.scatter_add_(
            0, true_flat_idx, torch.ones_like(true_flat_idx, dtype=dtype)
        )
        true_hist = true_hist.view(num_sub, num_bins)

        pred_hist = torch.zeros(num_sub * num_bins, device=device, dtype=dtype)
        deg_cap = float(num_bins - 1)
        pred_deg_cap = pred_deg.clamp(min=0.0, max=deg_cap)
        low = torch.floor(pred_deg_cap).long()
        high = torch.clamp(low + 1, max=num_bins - 1)
        w_high = (pred_deg_cap - low.to(dtype=dtype)).clamp(min=0.0, max=1.0)
        w_low = 1.0 - w_high
        low_idx = node_sub * num_bins + low
        high_idx = node_sub * num_bins + high
        pred_hist.scatter_add_(0, low_idx, w_low)
        pred_hist.scatter_add_(0, high_idx, w_high)
        pred_hist = pred_hist.view(num_sub, num_bins)

        active_rows = true_hist.sum(dim=1) > 0
        if not active_rows.any():
            return pred.edge_attr.sum() * 0.0

        if self.subtype_degree_loss_normalize:
            true_row_sum = true_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
            pred_row_sum = pred_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
            true_hist = true_hist / true_row_sum
            pred_hist = pred_hist / pred_row_sum

        return torch.nn.functional.l1_loss(
            pred_hist[active_rows], true_hist[active_rows], reduction="mean"
        )

    def forward(
        self,
        pred,
        true_data,
        log: bool,
        structure_only_global: bool = False,
        query_mask=None,
    ):
        """训练/验证统一的边级损失（只在 query 边范围内计算）：
        - Existence NLL: BCEWithLogits on logit(exist) = log P(exist)/P(no-edge)
        - Subtype NLL: CE over subtype logits on true-positive edges
        - Total: λ1 * existence + λ2 * subtype，其中 λ2 默认按 1.2 自动放大
        """
        pred_edge = pred.edge_attr
        target = true_data.edge_attr
        if target.dim() > 1:
            target = target.argmax(dim=-1)

        if query_mask is not None:
            pred_edge = pred_edge[query_mask]
            target = target[query_mask]

        if pred_edge.numel() == 0 or target.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        pred_edge_flat = pred_edge.reshape(-1, pred_edge.shape[-1])
        true_labels = target.reshape(-1).long().clamp(min=0, max=pred_edge_flat.shape[-1] - 1)

        # Existence head (binary) derived from multi-class logits: logit = logsumexp(pos) - logit(no-edge)
        exist_logits = torch.logsumexp(pred_edge_flat[:, 1:], dim=-1) - pred_edge_flat[:, 0]
        labels_exist = (true_labels > 0).float()
        exist_pos_weight = getattr(self, "exist_pos_weight", None)
        pos_weight_t = None
        if exist_pos_weight is not None:
            if isinstance(exist_pos_weight, str) and exist_pos_weight.lower() == "auto":
                # auto: pos_weight = neg/pos (clamped), computed on current query edges
                pos = labels_exist.sum().clamp(min=1.0)
                neg = (1.0 - labels_exist).sum().clamp(min=1.0)
                pos_weight_t = (neg / pos).to(dtype=exist_logits.dtype, device=exist_logits.device)
            else:
                try:
                    pos_weight_t = torch.tensor(float(exist_pos_weight), device=exist_logits.device, dtype=exist_logits.dtype)
                except Exception:
                    pos_weight_t = None
        bce_exist = F.binary_cross_entropy_with_logits(
            exist_logits,
            labels_exist,
            reduction="none",
            pos_weight=pos_weight_t,
        )
        if getattr(self, "exist_loss_type", "bce") == "focal":
            prob_exist = torch.sigmoid(exist_logits)
            pt = torch.where(labels_exist > 0.5, prob_exist, 1.0 - prob_exist)
            alpha = float(getattr(self, "exist_focal_alpha", 0.75))
            alpha_t = torch.where(
                labels_exist > 0.5,
                torch.full_like(labels_exist, alpha),
                torch.full_like(labels_exist, 1.0 - alpha),
            )
            gamma = float(getattr(self, "exist_focal_gamma", 2.0))
            focal_weight = alpha_t * (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)
            loss_exist = (focal_weight * bce_exist).mean()
        else:
            loss_exist = bce_exist.mean()

        # Subtype head (multi-class on positives only)
        pos_mask = true_labels > 0
        if pos_mask.any():
            subtype_logits = pred_edge_flat[pos_mask][:, 1:]
            labels_subtype = (true_labels[pos_mask] - 1).clamp(min=0, max=subtype_logits.shape[-1] - 1)
            loss_subtype = F.cross_entropy(subtype_logits, labels_subtype, reduction="mean")
        else:
            loss_subtype = pred_edge_flat.sum() * 0.0

        eps = 1e-8
        pos_ratio = pos_mask.float().mean().clamp(min=eps).detach()
        total_loss = float(self.edge_exist_weight) * loss_exist + float(self.edge_subtype_weight) * loss_subtype
        # 与 val/epoch_NLL 同公式：λ1*exist_nll + λ2*subtype_nll，供每 epoch 监控/checkpoint 用（不跑完整验证时）
        step_nll = total_loss.detach()

        # Accuracies
        pred_labels = pred_edge_flat.argmax(dim=-1)
        pred_exist = pred_labels > 0
        true_exist = true_labels > 0
        tp = (pred_exist & true_exist).sum()
        tn = ((~pred_exist) & (~true_exist)).sum()
        n_pos = true_exist.sum()
        n_neg = (~true_exist).sum()
        pos_acc = (tp.float() / n_pos.float().clamp(min=1.0)) if n_pos > 0 else torch.tensor(-1.0, device=pred_edge_flat.device)
        neg_acc = (tn.float() / n_neg.float().clamp(min=1.0)) if n_neg > 0 else torch.tensor(-1.0, device=pred_edge_flat.device)
        tp_mask = pred_exist & true_exist
        subtype_acc_on_tp = (
            (pred_labels[tp_mask] == true_labels[tp_mask]).float().mean()
            if tp_mask.any()
            else torch.tensor(-1.0, device=pred_edge_flat.device)
        )

        # Update epoch accumulators
        # 使用按 step 的简单平均（每个 step 的 loss 已是 mean），避免 ideal 模式下 n_edges 与 pred/target
        # 规模不一致时导致 sum/count 爆炸（如 query 在 full 上映射错误时 n_edges 异常大）
        device = pred_edge_flat.device
        if self._epoch_exist_loss_sum is None:
            self._epoch_exist_loss_sum = torch.tensor(0.0, device=device)
            self._epoch_exist_loss_count = torch.tensor(0.0, device=device)
            self._epoch_subtype_loss_sum = torch.tensor(0.0, device=device)
            self._epoch_subtype_loss_count = torch.tensor(0.0, device=device)
            self._epoch_tp = torch.tensor(0.0, device=device)
            self._epoch_pos = torch.tensor(0.0, device=device)
            self._epoch_tn = torch.tensor(0.0, device=device)
            self._epoch_neg = torch.tensor(0.0, device=device)
            self._epoch_subtype_correct_on_tp = torch.tensor(0.0, device=device)
            self._epoch_subtype_tp = torch.tensor(0.0, device=device)
            self._epoch_nll_sum = torch.tensor(0.0, device=device)
            self._epoch_nll_count = torch.tensor(0.0, device=device)

        n_edges = torch.tensor(float(true_labels.numel()), device=device)
        # BCE/CE 与 NLL：按 step 平均（每 step 的 loss 已是 mean），避免加权累加时 n_edges 异常导致指标爆炸
        self._epoch_nll_sum += step_nll.detach()
        self._epoch_nll_count += 1.0
        self._epoch_exist_loss_sum += loss_exist.detach()
        self._epoch_exist_loss_count += 1.0
        if pos_mask.any():
            self._epoch_subtype_loss_sum += loss_subtype.detach()
            self._epoch_subtype_loss_count += 1.0
        self._epoch_tp += tp.detach().float()
        self._epoch_pos += n_pos.detach().float()
        self._epoch_tn += tn.detach().float()
        self._epoch_neg += n_neg.detach().float()
        if tp_mask.any():
            self._epoch_subtype_correct_on_tp += (pred_labels[tp_mask] == true_labels[tp_mask]).sum().detach().float()
            self._epoch_subtype_tp += tp_mask.sum().detach().float()

        # 训练期间不在每个 step 单独 log 到 wandb，而是只在 on_train_epoch_end 里一次性记录本 epoch 汇总指标
        return total_loss

    def compute_structure_loss_scalar(self, pred, true_data):
        """返回「pred 全图 vs true_data 全图」全局结构差值的标量。
        structure_loss_type='legacy'：relation_matrix、metapath2、metapath3、subtype_degree
        structure_loss_type='graph_metrics'：DegreeMMD、Clustering、Triangles、EdgeTypesTV
        用于 ideal_at_t 构造。"""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        old_max = self.structure_loss_max_edges
        self.structure_loss_max_edges = 0
        try:
            stats = self._compute_structure_stats(pred, true_data)
            if getattr(self, "structure_loss_type", "legacy") == "graph_metrics":
                dg, cl, tr, wc, et = self._structure_losses_graph_metrics(stats, pred)
                return (
                    self.degree_mmd_loss_weight * dg
                    + self.clustering_loss_weight * cl
                    + self.triangles_loss_weight * tr
                    + self.wedge_closure_loss_weight * wc
                    + self.edge_types_tv_loss_weight * et
                )
            r, m2, m3, d = self._structure_losses_from_stats(stats, pred)
            return (
                self.relation_matrix_loss_weight * r
                + self.metapath2_loss_weight * m2
                + self.metapath3_loss_weight * m3
                + self.subtype_degree_loss_weight * d
            )
        finally:
            self.structure_loss_max_edges = old_max

    def compute_structure_loss_triangles_only(self, pred, true_data, include_clustering=True):
        """仅返回三角形（及可选聚类）结构损失标量，用于训练时的三角形辅助损失。
        使模型除 CE(pred, ideal) 外，直接收到「预测图相对真实图三角形/聚类不足」的梯度。"""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        old_max = self.structure_loss_max_edges
        self.structure_loss_max_edges = 0
        try:
            stats = self._compute_structure_stats(pred, true_data)
            if stats is None:
                return pred.edge_attr.sum() * 0.0
            if getattr(self, "structure_loss_type", "legacy") != "graph_metrics":
                return pred.edge_attr.sum() * 0.0
            _, cl, tr, wc, _ = self._structure_losses_graph_metrics(stats, pred)
            out = self.triangles_loss_weight * tr
            if include_clustering:
                out = out + self.clustering_loss_weight * cl
            if getattr(self, "wedge_closure_loss_weight", 0.0) > 0:
                out = out + self.wedge_closure_loss_weight * wc
            return out
        finally:
            self.structure_loss_max_edges = old_max

    def compute_node_distribution_loss(
        self,
        pred_node,
        true_node,
        include_conditional: bool = True,
        conditional_weight: float = 0.5,
    ):
        """节点分布损失（无逐点对应约束）：
        - 全局子类别分布 TV
        - 可选：按 type 切片后的条件子类别分布 TV
        返回: (loss, stats_dict)
        """
        zero = pred_node.sum() * 0.0 if torch.is_tensor(pred_node) else torch.tensor(0.0)
        if pred_node is None or true_node is None:
            return zero, {"global_tv": zero, "conditional_tv": zero}
        if pred_node.numel() == 0 or true_node.numel() == 0:
            return zero, {"global_tv": zero, "conditional_tv": zero}
        if pred_node.dim() < 2:
            return zero, {"global_tv": zero, "conditional_tv": zero}

        device = pred_node.device
        dtype = pred_node.dtype if pred_node.dtype.is_floating_point else torch.float32
        num_sub = pred_node.shape[-1]

        # pred 视为 logits，转换为概率分布
        pred_prob = torch.softmax(pred_node, dim=-1).to(dtype=dtype)

        # true 转为 one-hot/概率
        if true_node.dim() == 1:
            true_prob = F.one_hot(
                true_node.long().clamp(min=0, max=num_sub - 1), num_classes=num_sub
            ).to(device=device, dtype=dtype)
        elif true_node.dim() == 2:
            if true_node.shape[-1] == num_sub:
                true_prob = true_node.to(device=device, dtype=dtype)
                row_sum = true_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                true_prob = true_prob / row_sum
            else:
                true_prob = F.one_hot(
                    true_node.argmax(dim=-1).long().clamp(min=0, max=num_sub - 1),
                    num_classes=num_sub,
                ).to(device=device, dtype=dtype)
        else:
            return zero, {"global_tv": zero, "conditional_tv": zero}

        p_pred = pred_prob.mean(dim=0)
        p_true = true_prob.mean(dim=0)
        p_pred = p_pred / p_pred.sum().clamp(min=1e-8)
        p_true = p_true / p_true.sum().clamp(min=1e-8)
        global_tv = 0.5 * (p_pred - p_true).abs().sum()

        conditional_tv = pred_prob.sum() * 0.0
        if include_conditional and self.type_offsets:
            sorted_types = sorted(self.type_offsets.items(), key=lambda x: x[1])
            tv_items = []
            for i, (_, off) in enumerate(sorted_types):
                off_i = int(off)
                if i + 1 < len(sorted_types):
                    next_off = int(sorted_types[i + 1][1])
                else:
                    next_off = int(num_sub)
                if off_i >= next_off or off_i >= num_sub:
                    continue
                next_off = min(next_off, num_sub)
                pred_slice = p_pred[off_i:next_off]
                true_slice = p_true[off_i:next_off]
                if pred_slice.numel() == 0:
                    continue
                pred_mass = pred_slice.sum()
                true_mass = true_slice.sum()
                if pred_mass <= 0 and true_mass <= 0:
                    continue
                pred_cond = pred_slice / pred_mass.clamp(min=1e-8)
                true_cond = true_slice / true_mass.clamp(min=1e-8)
                tv_items.append(0.5 * (pred_cond - true_cond).abs().sum())
            if len(tv_items) > 0:
                conditional_tv = torch.stack(tv_items).mean()

        total = global_tv + float(conditional_weight) * conditional_tv
        return total, {"global_tv": global_tv, "conditional_tv": conditional_tv}

    def reset(self):
        self._epoch_exist_loss_sum = None
        self._epoch_exist_loss_count = None
        self._epoch_subtype_loss_sum = None
        self._epoch_subtype_loss_count = None
        self._epoch_tp = None
        self._epoch_pos = None
        self._epoch_tn = None
        self._epoch_neg = None
        self._epoch_subtype_correct_on_tp = None
        self._epoch_subtype_tp = None
        self._epoch_nll_sum = None
        self._epoch_nll_count = None

    def log_epoch_metrics(self, log_step=None):
        """返回当前累计的 train_epoch 指标字典。

        log_step 参数保留以兼容旧接口，但实际 wandb.log 由 LightningModule.on_train_epoch_end 统一完成。
        """
        to_log = {}
        if self._epoch_exist_loss_sum is not None and self._epoch_exist_loss_count is not None and self._epoch_exist_loss_count.item() > 0:
            to_log["train_epoch/existence_BCE"] = (self._epoch_exist_loss_sum / self._epoch_exist_loss_count).item()
        if self._epoch_subtype_loss_sum is not None and self._epoch_subtype_loss_count is not None and self._epoch_subtype_loss_count.item() > 0:
            to_log["train_epoch/subtype_CE"] = (self._epoch_subtype_loss_sum / self._epoch_subtype_loss_count).item()
        if self._epoch_tp is not None and self._epoch_pos is not None and self._epoch_pos.item() > 0:
            to_log["train_epoch/existence_pos_acc"] = (self._epoch_tp / self._epoch_pos).item()
        if self._epoch_tn is not None and self._epoch_neg is not None and self._epoch_neg.item() > 0:
            to_log["train_epoch/existence_neg_acc"] = (self._epoch_tn / self._epoch_neg).item()
        if (
            self._epoch_subtype_correct_on_tp is not None
            and self._epoch_subtype_tp is not None
            and self._epoch_subtype_tp.item() > 0
        ):
            to_log["train_epoch/subtype_acc_on_tp"] = (
                self._epoch_subtype_correct_on_tp / self._epoch_subtype_tp
            ).item()
        # 与 val/epoch_NLL 同公式的 NLL（λ1*exist + λ2*subtype），每 epoch 都有，供 checkpoint 监控
        if self._epoch_nll_sum is not None and self._epoch_nll_count is not None and self._epoch_nll_count.item() > 0:
            to_log["train_epoch/NLL"] = (self._epoch_nll_sum / self._epoch_nll_count).item()

        return to_log
