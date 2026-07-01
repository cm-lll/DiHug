import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax

from sparse_diffusion import utils


def apply_dual_softmax(att_logits: Tensor, edge_index_i: Tensor, edge_family: Tensor, mu_gate: Tensor) -> Tensor:
    """Blend standard HGT target-node softmax with per-family softmax."""
    if edge_family is None or edge_family.numel() == 0:
        return softmax(att_logits, edge_index_i)
    max_fam = int(edge_family.max().item()) + 1 if edge_family.numel() else 1
    intra_index = edge_index_i.long() * max(max_fam, 1) + edge_family.long().clamp(min=0)
    att_global = softmax(att_logits, edge_index_i)
    att_intra = softmax(att_logits, intra_index)
    mu = torch.sigmoid(mu_gate).view(1, -1)
    return mu * att_global + (1.0 - mu) * att_intra


class HGTConv(MessagePassing):
    """pyHGT-style heterogeneous attention layer with optional edge phi fusion."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_types: int,
        num_relations: int,
        n_heads: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_edge_phi_fusion: bool = False,
        num_edge_families: int = 0,
        max_edge_phi: int = 0,
        use_dual_softmax: bool = False,
        use_query_context_gate: bool = False,
        query_context_gate_init: float = 0.2,
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="add", **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = max(int(num_types), 1)
        self.num_relations = max(int(num_relations), 1)
        self.n_heads = int(n_heads)
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_edge_phi_fusion = bool(use_edge_phi_fusion and num_edge_families > 0 and max_edge_phi > 0)
        self.use_dual_softmax = bool(use_dual_softmax)
        self.use_query_context_gate = bool(use_query_context_gate)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.empty(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.empty(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        glorot(self.relation_att)
        glorot(self.relation_msg)

        if self.use_edge_phi_fusion:
            self.phi_att = nn.Parameter(torch.zeros(num_edge_families, max_edge_phi, n_heads, self.d_k, self.d_k))
            self.phi_msg = nn.Parameter(torch.zeros(num_edge_families, max_edge_phi, n_heads, self.d_k, self.d_k))
        if self.use_dual_softmax:
            self.dual_softmax_mu = nn.Parameter(torch.full((n_heads,), 2.0))
        if self.use_query_context_gate:
            init = min(max(float(query_context_gate_init), 1e-4), 1.0 - 1e-4)
            self.query_context_gate_logit = nn.Parameter(
                torch.full((n_heads,), math.log(init / (1.0 - init)))
            )
        self.att = None

    def _channel_attention(
        self,
        logits: Tensor,
        edge_index_i: Tensor,
        edge_family: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        out = torch.zeros_like(logits)
        if not mask.any():
            return out
        if self.use_dual_softmax and edge_family is not None:
            out[mask] = apply_dual_softmax(
                logits[mask],
                edge_index_i[mask],
                edge_family[mask],
                self.dual_softmax_mu,
            )
        else:
            out[mask] = softmax(logits[mask], edge_index_i[mask])
        return out

    def _query_context_attention(
        self,
        logits: Tensor,
        edge_index_i: Tensor,
        edge_family: Optional[Tensor],
        edge_subtype: Tensor,
    ) -> Tensor:
        """Normalize visible G_t edges and no-edge query links separately."""
        query_mask = edge_subtype == 0
        context_mask = ~query_mask
        query_att = self._channel_attention(
            logits, edge_index_i, edge_family, query_mask
        )
        context_att = self._channel_attention(
            logits, edge_index_i, edge_family, context_mask
        )

        num_nodes = (
            int(edge_index_i.max().item()) + 1 if edge_index_i.numel() else 0
        )
        query_count = torch.bincount(
            edge_index_i[query_mask], minlength=num_nodes
        )
        context_count = torch.bincount(
            edge_index_i[context_mask], minlength=num_nodes
        )
        has_query = query_count[edge_index_i] > 0
        has_context = context_count[edge_index_i] > 0
        both = has_query & has_context

        query_gate = torch.sigmoid(self.query_context_gate_logit).view(1, -1)
        query_coeff = torch.where(
            both.view(-1, 1),
            query_gate,
            torch.ones_like(query_gate),
        )
        context_coeff = torch.where(
            both.view(-1, 1),
            1.0 - query_gate,
            torch.ones_like(query_gate),
        )
        return query_att * query_coeff + context_att * context_coeff

    def forward(
        self,
        node_inp: Tensor,
        node_type: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_subtype: Optional[Tensor] = None,
        edge_family: Optional[Tensor] = None,
    ) -> Tensor:
        # Cached for fused message: project Q/K/V once per node, then gather on edges.
        self._cached_node_inp = node_inp
        self._cached_node_type = node_type
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_subtype=edge_subtype,
            edge_family=edge_family,
        )

    def _project_nodes_by_type(self, node_inp: Tensor, node_type: Tensor, linears: nn.ModuleList) -> Tensor:
        """Apply type-specific linear maps to all nodes (num_types passes, each one batched matmul)."""
        out = torch.zeros(
            node_inp.size(0), self.n_heads, self.d_k, device=node_inp.device, dtype=node_inp.dtype
        )
        node_type = node_type.clamp(0, self.num_types - 1)
        for t in range(self.num_types):
            mask = node_type == t
            if mask.sum() == 0:
                continue
            out[mask] = linears[t](node_inp[mask]).view(-1, self.n_heads, self.d_k)
        return out

    def _relation_transform_by_type(
        self, edge_values: Tensor, edge_type: Tensor, relation_weight: Tensor
    ) -> Tensor:
        """Apply relation matrices without materializing one matrix per edge."""
        out = torch.zeros_like(edge_values)
        edge_type = edge_type.clamp(0, relation_weight.size(0) - 1)
        for relation_type in torch.unique(edge_type).tolist():
            idx = (edge_type == int(relation_type)).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            transformed = torch.einsum(
                "ehd,hdf->ehf",
                edge_values.index_select(0, idx),
                relation_weight[int(relation_type)],
            )
            out.index_copy_(0, idx, transformed)
        return out

    def _apply_edge_phi_fusion(
        self,
        k_mat: Tensor,
        v_mat: Tensor,
        edge_subtype: Tensor,
        edge_family: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ef = edge_family.clamp(0, self.phi_att.size(0) - 1)
        ep = edge_subtype.clamp(0, self.phi_att.size(1) - 1)
        k_out = torch.zeros_like(k_mat)
        v_out = torch.zeros_like(v_mat)
        chunk = 2048
        for start in range(0, ef.size(0), chunk):
            sl = slice(start, start + chunk)
            k_out[sl] = torch.einsum("ehd,ehdf->ehf", k_mat[sl], self.phi_att[ef[sl], ep[sl]])
            v_out[sl] = torch.einsum("ehd,ehdf->ehf", v_mat[sl], self.phi_msg[ef[sl], ep[sl]])
        return k_out, v_out

    def message(
        self,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        node_inp_i: Tensor,
        node_inp_j: Tensor,
        node_type_i: Tensor,
        node_type_j: Tensor,
        edge_type: Tensor,
        edge_subtype: Optional[Tensor] = None,
        edge_family: Optional[Tensor] = None,
    ) -> Tensor:
        node_inp = self._cached_node_inp
        node_type = self._cached_node_type
        edge_type = edge_type.clamp(0, self.num_relations - 1)

        q_all = self._project_nodes_by_type(node_inp, node_type, self.q_linears)
        k_all = self._project_nodes_by_type(node_inp, node_type, self.k_linears)
        v_all = self._project_nodes_by_type(node_inp, node_type, self.v_linears)

        qi = q_all[edge_index_i]
        kj = k_all[edge_index_j]
        vj = v_all[edge_index_j]

        k_transformed = self._relation_transform_by_type(kj, edge_type, self.relation_att)
        v_transformed = self._relation_transform_by_type(vj, edge_type, self.relation_msg)

        if self.use_edge_phi_fusion and edge_subtype is not None and edge_family is not None:
            phi_k, phi_v = self._apply_edge_phi_fusion(kj, vj, edge_subtype, edge_family)
            k_transformed = k_transformed + phi_k
            v_transformed = v_transformed + phi_v

        res_att = (qi * k_transformed).sum(dim=-1) * self.relation_pri[edge_type] / self.sqrt_dk
        if (
            self.use_query_context_gate
            and edge_subtype is not None
            and edge_subtype.numel() == res_att.shape[0]
        ):
            self.att = self._query_context_attention(
                res_att, edge_index_i, edge_family, edge_subtype
            )
        elif self.use_dual_softmax and edge_family is not None:
            self.att = apply_dual_softmax(res_att, edge_index_i, edge_family, self.dual_softmax_mu)
        else:
            self.att = softmax(res_att, edge_index_i)
        return (v_transformed * self.att.view(-1, self.n_heads, 1)).view(-1, self.out_dim)

    def message_legacy(
        self,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        node_inp_i: Tensor,
        node_inp_j: Tensor,
        node_type_i: Tensor,
        node_type_j: Tensor,
        edge_type: Tensor,
        edge_subtype: Optional[Tensor] = None,
        edge_family: Optional[Tensor] = None,
    ) -> Tensor:
        """pyHGT-style per-(src,dst,rel) loop; kept for numerical equivalence checks."""
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads, device=node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k, device=node_inp_i.device)
        edge_type = edge_type.clamp(0, self.num_relations - 1)

        for source_type in range(self.num_types):
            source_mask = node_type_j == int(source_type)
            if source_mask.sum() == 0:
                continue
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                type_mask = (node_type_i == int(target_type)) & source_mask
                if type_mask.sum() == 0:
                    continue
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & type_mask
                    if idx.sum() == 0:
                        continue
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_transformed = torch.bmm(
                        k_mat.transpose(1, 0), self.relation_att[relation_type]
                    ).transpose(1, 0)
                    v_transformed = torch.bmm(
                        v_mat.transpose(1, 0), self.relation_msg[relation_type]
                    ).transpose(1, 0)
                    if self.use_edge_phi_fusion and edge_subtype is not None and edge_family is not None:
                        e_idx = idx.nonzero(as_tuple=True)[0]
                        ef = edge_family[e_idx].clamp(0, self.phi_att.size(0) - 1)
                        ep = edge_subtype[e_idx].clamp(0, self.phi_att.size(1) - 1)
                        phi_k, phi_v = self._apply_edge_phi_fusion(k_mat, v_mat, ep, ef)
                        k_transformed = k_transformed + phi_k
                        v_transformed = v_transformed + phi_v
                    res_att[idx] = (
                        (q_mat * k_transformed).sum(dim=-1)
                        * self.relation_pri[relation_type]
                        / self.sqrt_dk
                    )
                    res_msg[idx] = v_transformed

        if self.use_dual_softmax and edge_family is not None:
            self.att = apply_dual_softmax(res_att, edge_index_i, edge_family, self.dual_softmax_mu)
        else:
            self.att = softmax(res_att, edge_index_i)
        return (res_msg * self.att.view(-1, self.n_heads, 1)).view(-1, self.out_dim)

    def update(self, aggr_out: Tensor, node_inp: Tensor, node_type: Tensor) -> Tensor:
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim, device=node_inp.device)
        node_type = node_type.clamp(0, self.num_types - 1)
        for target_type in range(self.num_types):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])
            mixed = trans_out * alpha + node_inp[idx] * (1 - alpha)
            res[idx] = self.norms[target_type](mixed) if self.use_norm else mixed
        return res


class SubtypeInputFusion(nn.Module):
    """Fuse diffusion node one-hot states with type-local subtype embeddings."""

    def __init__(self, in_dim: int, num_types: int, num_subtypes_per_type: list[int], subtype_dim: int = 32):
        super().__init__()
        self.num_types = max(int(num_types), 1)
        self.embeds = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for t in range(self.num_types):
            nsub = int(num_subtypes_per_type[t]) if t < len(num_subtypes_per_type) else 1
            self.embeds.append(nn.Embedding(max(nsub, 1), subtype_dim))
            self.fusions.append(nn.Linear(in_dim + subtype_dim, in_dim))

    def forward(self, node_feature: Tensor, node_type: Tensor, node_subtype: Tensor) -> Tensor:
        out = node_feature.clone()
        node_type = node_type.clamp(0, self.num_types - 1)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            sub = node_subtype[idx].clamp(min=0, max=self.embeds[t_id].num_embeddings - 1)
            h_sub = self.embeds[t_id](sub)
            out[idx] = self.fusions[t_id](torch.cat([node_feature[idx], h_sub], dim=-1))
        return out


class PyHGTDenoiser(nn.Module):
    """DiHuG denoiser: pyHGT node encoder plus query-edge diffusion heads."""

    def __init__(
        self,
        n_layers: int,
        input_dims: utils.PlaceHolder,
        hidden_dims: dict,
        output_dims: utils.PlaceHolder,
        dropout: float,
        sn_hidden_dim: int,
        output_y: bool = False,
        heterogeneous: bool = True,
        num_node_types: int = 0,
        num_node_subtypes: int = 0,
        num_relation_types: int = 0,
        edge_family_offsets: Optional[Dict[str, int]] = None,
        type_offsets: Optional[Dict[str, int]] = None,
        subtype_dim: int = 32,
        use_edge_phi_fusion: bool = True,
        use_dual_softmax: bool = True,
        use_query_context_gate: bool = False,
        query_context_gate_init: float = 0.2,
        use_two_hop_structure: bool = False,
        use_typed_two_hop_structure: bool = False,
        two_hop_structure_hidden_dim: int = 64,
        two_hop_structure_scale: float = 1.0,
        two_hop_structure_schedule: str = "fixed",
        use_endpoint_role_residual: bool = False,
        endpoint_role_hidden_dim: int = 64,
        endpoint_role_family_dim: int = 16,
        endpoint_role_scale: float = 1.0,
        use_family_edge_adapters: bool = False,
        family_edge_adapter_hidden_dim: int = 64,
        use_time_film: bool = False,
        use_edge_state_update: bool = False,
        edge_state_update_mode: str = "all",
        edge_input_residual_scale: float = 1.0,
        edge_only_model: bool = True,
        **_,
    ):
        super().__init__()
        self.heterogeneous = heterogeneous
        self.edge_only_model = edge_only_model
        self.output_y = output_y
        self.n_layers = int(n_layers)
        self.use_time_film = bool(use_time_film)
        self.use_edge_state_update = bool(use_edge_state_update)
        self.edge_input_residual_scale = float(edge_input_residual_scale)
        self.use_two_hop_structure = bool(use_two_hop_structure)
        self.use_typed_two_hop_structure = bool(
            use_typed_two_hop_structure
        )
        self.two_hop_structure_scale = float(two_hop_structure_scale)
        self.two_hop_structure_schedule = str(
            two_hop_structure_schedule or "fixed"
        ).lower()
        self.use_endpoint_role_residual = bool(
            use_endpoint_role_residual
        )
        self.use_family_edge_adapters = bool(use_family_edge_adapters)
        self.endpoint_role_scale = float(endpoint_role_scale)
        self.edge_state_update_mode = str(edge_state_update_mode or "all").lower()
        if self.edge_state_update_mode not in {"all", "last"}:
            self.edge_state_update_mode = "all"
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charge = output_dims.charge
        self.num_node_types = max(int(num_node_types), 1)
        self.num_relations = max(int(num_relation_types), 1)
        dx = hidden_dims["dx"]
        de = hidden_dims["de"]
        dy = hidden_dims["dy"]
        n_heads = hidden_dims["n_head"]

        self.lin_in_X = nn.Linear(input_dims.X + input_dims.charge + sn_hidden_dim, dx)
        self.lin_in_E = nn.Linear(input_dims.E, de)
        self.lin_in_y = nn.Linear(input_dims.y, dy)
        self.edge_context = nn.Linear(de + dy, de)
        if self.use_time_film:
            self.time_x_add = nn.Linear(dy, dx)
            self.time_x_mul = nn.Linear(dy, dx)
            self.time_e_add = nn.Linear(dy, de)
            self.time_e_mul = nn.Linear(dy, de)
            self.layer_time_x_add = nn.ModuleList([nn.Linear(dy, dx) for _ in range(n_layers)])
            self.layer_time_x_mul = nn.ModuleList([nn.Linear(dy, dx) for _ in range(n_layers)])
            self.layer_time_e_add = nn.ModuleList([nn.Linear(dy, de) for _ in range(n_layers)])
            self.layer_time_e_mul = nn.ModuleList([nn.Linear(dy, de) for _ in range(n_layers)])
        else:
            self.time_x_add = None
            self.time_x_mul = None
            self.time_e_add = None
            self.time_e_mul = None
            self.layer_time_x_add = nn.ModuleList()
            self.layer_time_x_mul = nn.ModuleList()
            self.layer_time_e_add = nn.ModuleList()
            self.layer_time_e_mul = nn.ModuleList()

        if self.use_edge_state_update:
            self.edge_state_updates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * dx + de + dy, hidden_dims["dim_ffE"]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims["dim_ffE"], de),
                )
                for _ in range(n_layers)
            ])
            self.edge_state_norms = nn.ModuleList([nn.LayerNorm(de) for _ in range(n_layers)])
        else:
            self.edge_state_updates = nn.ModuleList()
            self.edge_state_norms = nn.ModuleList()

        type_sizes = self._infer_type_sizes(type_offsets, num_node_subtypes)
        self.subtype_fusion = SubtypeInputFusion(dx, self.num_node_types, type_sizes, subtype_dim)
        num_edge_families = len(edge_family_offsets or {})
        self.edge_family_offsets = edge_family_offsets or {}
        self.register_buffer("family_starts", self._build_family_starts(self.edge_family_offsets), persistent=False)
        max_edge_phi = self._max_edge_phi(self.edge_family_offsets, self.out_dim_E)

        self.gcs = nn.ModuleList(
            [
                HGTConv(
                    dx,
                    dx,
                    self.num_node_types,
                    self.num_relations,
                    n_heads,
                    dropout=dropout,
                    use_norm=True,
                    use_edge_phi_fusion=use_edge_phi_fusion,
                    num_edge_families=num_edge_families,
                    max_edge_phi=max_edge_phi,
                    use_dual_softmax=use_dual_softmax,
                    use_query_context_gate=use_query_context_gate,
                    query_context_gate_init=query_context_gate_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.node_head = nn.Sequential(nn.LayerNorm(dx), nn.Linear(dx, output_dims.X + output_dims.charge))
        self.edge_head = self._make_edge_head(
            dx=dx,
            de=de,
            hidden_dim=hidden_dims["dim_ffE"],
            dropout=dropout,
            out_dim=output_dims.E,
        )
        # Low-rank bilinear: project src/dst to rank-16, elementwise product, MLP → scalar
        self.bilinear_rank = 16
        self.bilinear_proj_src = nn.Linear(dx, self.bilinear_rank, bias=False)
        self.bilinear_proj_dst = nn.Linear(dx, self.bilinear_rank, bias=False)
        self.bilinear_out = nn.Sequential(
            nn.Linear(self.bilinear_rank, self.bilinear_rank),
            nn.GELU(),
            nn.Linear(self.bilinear_rank, 1),
        )
        if self.use_family_edge_adapters and num_edge_families > 0:
            adapter_hidden = max(int(family_edge_adapter_hidden_dim), 8)
            self.family_edge_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(2 * dx + de),
                        nn.Linear(2 * dx + de, adapter_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(adapter_hidden, output_dims.E),
                    )
                    for _ in range(num_edge_families)
                ]
            )
            for adapter in self.family_edge_adapters:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        else:
            self.family_edge_adapters = nn.ModuleList()
        if self.use_two_hop_structure:
            structure_hidden = max(int(two_hop_structure_hidden_dim), 8)
            self.two_hop_structure_feature_dim = 4
            if self.use_typed_two_hop_structure:
                self.two_hop_structure_feature_dim += int(
                    max(num_node_types, 1)
                ) + 2 * int(max(num_edge_families, 1))
            self.two_hop_structure_head = nn.Sequential(
                nn.LayerNorm(self.two_hop_structure_feature_dim + dy),
                nn.Linear(self.two_hop_structure_feature_dim + dy, structure_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(structure_hidden, output_dims.E),
            )
            nn.init.zeros_(self.two_hop_structure_head[-1].weight)
            nn.init.zeros_(self.two_hop_structure_head[-1].bias)
        else:
            self.two_hop_structure_feature_dim = 4
            self.two_hop_structure_head = None
        if self.use_endpoint_role_residual:
            role_family_dim = max(int(endpoint_role_family_dim), 4)
            role_hidden = max(int(endpoint_role_hidden_dim), 8)
            self.endpoint_role_family_embedding = nn.Embedding(
                max(num_edge_families, 1), role_family_dim
            )
            # 5 symmetric + 4 directed features for total degree, repeated
            # for family degree, plus a same-type indicator.
            self.endpoint_role_head = nn.Sequential(
                nn.LayerNorm(19 + role_family_dim + dy),
                nn.Linear(19 + role_family_dim + dy, role_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(role_hidden, output_dims.E),
            )
            nn.init.zeros_(self.endpoint_role_head[-1].weight)
            nn.init.zeros_(self.endpoint_role_head[-1].bias)
        else:
            self.endpoint_role_family_embedding = None
            self.endpoint_role_head = None
        self.last_two_hop_base_residual_mean = 0.0
        self.last_two_hop_effective_residual_mean = 0.0
        self.last_two_hop_effective_scale_mean = 0.0
        # Backwards-compatible aliases used by older logging/checkpoints.
        self.last_two_hop_residual_mean = 0.0
        self.last_two_hop_scale_factor_mean = 1.0
        self.last_endpoint_role_base_residual_mean = 0.0
        self.last_endpoint_role_effective_residual_mean = 0.0
        self.last_endpoint_role_effective_scale_mean = 0.0
        self.y_head = nn.Linear(dy, output_dims.y) if output_y else None

    @staticmethod
    def _make_edge_head(dx: int, de: int, hidden_dim: int, dropout: float, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * dx + de, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _edge_head_logits(
        self,
        edge_features: Tensor,
        query_edge_stage_ids: Optional[Tensor] = None,
        query_edge_family_ids: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.edge_head(edge_features)
        if (
            self.use_family_edge_adapters
            and query_edge_family_ids is not None
            and len(self.family_edge_adapters) > 0
        ):
            fam = query_edge_family_ids.long().to(edge_features.device).reshape(-1)
            if fam.numel() == edge_features.shape[0]:
                for family_id in torch.unique(fam).tolist():
                    family_id = int(family_id)
                    if family_id < 0 or family_id >= len(self.family_edge_adapters):
                        continue
                    mask = fam == family_id
                    if mask.any():
                        out[mask] = out[mask] + self.family_edge_adapters[family_id](
                            edge_features[mask]
                        )
        return out

    @staticmethod
    def _infer_type_sizes(type_offsets: Optional[Dict[str, int]], num_node_subtypes: int) -> list[int]:
        if not type_offsets:
            return [max(int(num_node_subtypes), 1)]
        ordered = sorted((int(v), k) for k, v in type_offsets.items())
        starts = [v for v, _ in ordered] + [int(num_node_subtypes)]
        return [max(starts[i + 1] - starts[i], 1) for i in range(len(ordered))]

    @staticmethod
    def _build_family_starts(edge_family_offsets: Dict[str, int]) -> Tensor:
        if not edge_family_offsets:
            return torch.zeros(1, dtype=torch.long)
        starts = [0] * len(edge_family_offsets)
        for idx, (_, start) in enumerate(sorted(edge_family_offsets.items(), key=lambda kv: kv[1])):
            starts[idx] = int(start)
        return torch.tensor(starts, dtype=torch.long)

    @staticmethod
    def _max_edge_phi(edge_family_offsets: Dict[str, int], out_dim_e: int) -> int:
        if not edge_family_offsets:
            return max(int(out_dim_e), 1)
        starts = sorted(int(v) for v in edge_family_offsets.values()) + [int(out_dim_e)]
        sizes = [max(starts[i + 1] - starts[i], 1) for i in range(len(starts) - 1)]
        return max(max(sizes), 1)

    def _film_node_with_y(self, h: Tensor, y_h: Tensor, batch: Tensor, add_layer: nn.Linear, mul_layer: nn.Linear) -> Tensor:
        if y_h.numel() == 0 or h.numel() == 0:
            return h
        node_y = y_h[batch.long()]
        return add_layer(node_y) + (mul_layer(node_y) + 1.0) * h

    def _film_edge_with_y(self, e: Tensor, y_h: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        if y_h.numel() == 0 or e.numel() == 0 or self.time_e_add is None or self.time_e_mul is None:
            return e
        edge_y = y_h[batch[edge_index[0].long()].long()]
        return self.time_e_add(edge_y) + (self.time_e_mul(edge_y) + 1.0) * e

    def _film_edge_with_y_layers(self, e: Tensor, y_h: Tensor, edge_index: Tensor, batch: Tensor, add_layer: nn.Linear, mul_layer: nn.Linear) -> Tensor:
        if y_h.numel() == 0 or e.numel() == 0:
            return e
        edge_y = y_h[batch[edge_index[0].long()].long()]
        return add_layer(edge_y) + (mul_layer(edge_y) + 1.0) * e

    def _should_update_edge_state_layer(self, layer_idx: int) -> bool:
        if not self.use_edge_state_update:
            return False
        if self.edge_state_update_mode == "last":
            return int(layer_idx) == self.n_layers - 1
        return True

    def _update_edge_state_with_layer(
        self,
        e: Tensor,
        h: Tensor,
        y_h: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        layer_idx: int,
    ) -> Tensor:
        if (
            not self._should_update_edge_state_layer(layer_idx)
            or layer_idx >= len(self.edge_state_updates)
            or e.numel() == 0
        ):
            return e
        src_h = h[edge_index[0].long()]
        dst_h = h[edge_index[1].long()]
        edge_y = y_h[batch[edge_index[0].long()].long()] if y_h.numel() else e.new_zeros((e.size(0), 0))
        delta = self.edge_state_updates[layer_idx](torch.cat([src_h, dst_h, e, edge_y], dim=-1))
        e = self.edge_state_norms[layer_idx](e + delta)
        if self.use_time_film and y_h.numel() and layer_idx < len(self.layer_time_e_add):
            e = self._film_edge_with_y_layers(
                e, y_h, edge_index, batch, self.layer_time_e_add[layer_idx], self.layer_time_e_mul[layer_idx]
            )
        return e

    def _edge_subtype_from_attr(self, edge_attr: Tensor, edge_family_ids: Optional[Tensor]) -> Tensor:
        edge_global = edge_attr.argmax(dim=-1).long() if edge_attr.dim() == 2 else edge_attr.long()
        if edge_family_ids is None or self.family_starts.numel() == 0:
            return edge_global.clamp(min=0)
        starts = self.family_starts.to(edge_attr.device)
        fam = edge_family_ids.long().clamp(0, starts.numel() - 1)
        local = edge_global - starts[fam] + 1
        return local.clamp(min=0)

    @staticmethod
    def _sparse_values_for_keys(
        sparse_matrix: Tensor,
        row: Tensor,
        col: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Gather sparse matrix entries without materializing an NxN tensor."""
        if row.numel() == 0 or sparse_matrix._nnz() == 0:
            return torch.zeros(row.shape, device=row.device, dtype=torch.float)
        matrix = sparse_matrix.coalesce()
        matrix_keys = (
            matrix.indices()[0].long() * int(num_nodes)
            + matrix.indices()[1].long()
        )
        order = torch.argsort(matrix_keys)
        matrix_keys = matrix_keys[order]
        matrix_values = matrix.values()[order]
        query_keys = row.long() * int(num_nodes) + col.long()
        positions = torch.searchsorted(matrix_keys, query_keys)
        valid = positions < matrix_keys.numel()
        safe_positions = positions.clamp(max=max(matrix_keys.numel() - 1, 0))
        valid = valid & (matrix_keys[safe_positions] == query_keys)
        out = matrix_values.new_zeros(query_keys.shape)
        out[valid] = matrix_values[safe_positions[valid]]
        return out

    @staticmethod
    def _lookup_values_for_keys(
        lookup: dict,
        row: Tensor,
        col: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Gather values from a pre-sorted sparse key/value lookup."""
        keys = lookup["two_hop_keys"]
        values = lookup["two_hop_values"]
        if row.numel() == 0 or keys.numel() == 0:
            return values.new_zeros(row.shape)
        query_keys = row.long() * int(num_nodes) + col.long()
        positions = torch.searchsorted(keys, query_keys)
        valid = positions < keys.numel()
        safe_positions = positions.clamp(max=max(keys.numel() - 1, 0))
        valid = valid & (keys[safe_positions] == query_keys)
        out = values.new_zeros(query_keys.shape)
        out[valid] = values[safe_positions[valid]]
        return out

    @staticmethod
    def _lookup_matrix_values_for_keys(
        lookup: dict,
        value_name: str,
        row: Tensor,
        col: Tensor,
        num_nodes: int,
        feature_dim: int,
    ) -> Tensor:
        """Gather vector-valued sparse lookup entries by sorted two-hop keys."""
        keys = lookup.get("two_hop_keys")
        values = lookup.get(value_name)
        if (
            row.numel() == 0
            or keys is None
            or values is None
            or keys.numel() == 0
        ):
            return torch.zeros(
                (row.numel(), int(feature_dim)),
                device=row.device,
                dtype=torch.float,
            )
        query_keys = row.long() * int(num_nodes) + col.long()
        positions = torch.searchsorted(keys, query_keys)
        valid = positions < keys.numel()
        safe_positions = positions.clamp(max=max(keys.numel() - 1, 0))
        valid = valid & (keys[safe_positions] == query_keys)
        out = values.new_zeros((query_keys.numel(), values.shape[-1]))
        out[valid] = values[safe_positions[valid]]
        return out

    def _augment_typed_two_hop_lookup(
        self,
        lookup: dict,
        adjacency,
        adjacency_index: Tensor,
        adjacency_family: Optional[Tensor],
        node_type_ids: Optional[Tensor],
        num_nodes: int,
    ) -> None:
        """Attach typed two-hop path counts to an existing base lookup."""
        if not self.use_typed_two_hop_structure:
            return
        keys = lookup.get("two_hop_keys")
        device = adjacency_index.device
        num_types = int(max(self.num_node_types, 1))
        num_families = int(max(self.family_starts.numel(), 1))
        typed_dim = num_types + 2 * num_families
        if keys is None or keys.numel() == 0:
            lookup["typed_two_hop_values"] = torch.empty(
                (0, typed_dim),
                dtype=torch.float,
                device=device,
            )
            return

        node_type = (
            node_type_ids.long().to(device)
            if node_type_ids is not None
            else torch.zeros(num_nodes, dtype=torch.long, device=device)
        ).clamp(0, num_types - 1)
        family = (
            adjacency_family.long().to(device)
            if adjacency_family is not None
            else torch.zeros(adjacency_index.size(1), dtype=torch.long, device=device)
        ).clamp(0, num_families - 1)

        features = torch.zeros(
            (keys.numel(), typed_dim),
            dtype=torch.float,
            device=device,
        )

        def add_sparse_counts(mat, feature_offset: int):
            mat = mat.coalesce()
            if mat._nnz() == 0:
                return
            mat_keys = (
                mat.indices()[0].long() * int(num_nodes)
                + mat.indices()[1].long()
            )
            pos = torch.searchsorted(keys, mat_keys)
            valid = pos < keys.numel()
            safe = pos.clamp(max=max(keys.numel() - 1, 0))
            valid = valid & (keys[safe] == mat_keys)
            if valid.any():
                features[safe[valid], feature_offset] = mat.values()[valid]

        dst_type = node_type[adjacency_index[1].long()]
        for type_id in range(num_types):
            values = (dst_type == type_id).to(dtype=torch.float, device=device)
            typed_adjacency = torch.sparse_coo_tensor(
                adjacency_index,
                values,
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
            add_sparse_counts(
                torch.sparse.mm(typed_adjacency, adjacency),
                type_id,
            )

        for fam_id in range(num_families):
            values = (family == fam_id).to(dtype=torch.float, device=device)
            family_adjacency = torch.sparse_coo_tensor(
                adjacency_index,
                values,
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
            add_sparse_counts(
                torch.sparse.mm(family_adjacency, adjacency),
                num_types + fam_id,
            )
            add_sparse_counts(
                torch.sparse.mm(adjacency, family_adjacency),
                num_types + num_families + fam_id,
            )

        lookup["typed_two_hop_values"] = features.detach()

    def build_two_hop_structure_lookup(
        self,
        context_edge_index: Tensor,
        num_nodes: int,
        node_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> dict:
        """Build one detached current-G_t lookup reusable by all query blocks."""
        device = context_edge_index.device
        degree = torch.zeros(max(int(num_nodes), 0), dtype=torch.float, device=device)
        if context_edge_index.numel() == 0 or num_nodes <= 0:
            lookup = {
                "degree": degree,
                "two_hop_keys": torch.empty(0, dtype=torch.long, device=device),
                "two_hop_values": torch.empty(0, dtype=torch.float, device=device),
                "adjacency_nnz": 0,
                "two_hop_nnz": 0,
            }
            if self.use_typed_two_hop_structure:
                lookup["typed_two_hop_values"] = torch.empty(
                    (
                        0,
                        int(max(self.num_node_types, 1))
                        + 2 * int(max(self.family_starts.numel(), 1)),
                    ),
                    dtype=torch.float,
                    device=device,
                )
            if self.use_endpoint_role_residual:
                lookup.update(
                    self._build_endpoint_role_lookup(
                        context_edge_index,
                        num_nodes,
                        node_type_ids,
                        edge_family_ids,
                        batch,
                    )
                )
            return lookup

        with torch.no_grad():
            src = context_edge_index[0].long()
            dst = context_edge_index[1].long()
            non_loop = src != dst
            src = src[non_loop]
            dst = dst[non_loop]
            lo = torch.minimum(src, dst)
            hi = torch.maximum(src, dst)
            canonical = torch.unique(lo * int(num_nodes) + hi)
            raw_keys = lo * int(num_nodes) + hi
            raw_order = torch.argsort(raw_keys)
            raw_keys_sorted = raw_keys[raw_order]
            raw_first = torch.ones_like(raw_keys_sorted, dtype=torch.bool)
            raw_first[1:] = raw_keys_sorted[1:] != raw_keys_sorted[:-1]
            chosen = raw_order[raw_first]
            canonical = raw_keys_sorted[raw_first]
            lo = torch.div(canonical, int(num_nodes), rounding_mode="floor")
            hi = canonical.remainder(int(num_nodes))
            canonical_family = None
            if edge_family_ids is not None:
                canonical_family = edge_family_ids.long().to(device)[
                    non_loop
                ][chosen]
            adjacency_index = torch.stack(
                [torch.cat([lo, hi]), torch.cat([hi, lo])], dim=0
            )
            adjacency_family = None
            if canonical_family is not None:
                adjacency_family = torch.cat(
                    [canonical_family, canonical_family], dim=0
                )
            adjacency = torch.sparse_coo_tensor(
                adjacency_index,
                torch.ones(
                    adjacency_index.size(1),
                    device=device,
                    dtype=torch.float,
                ),
                (num_nodes, num_nodes),
                device=device,
            ).coalesce()
            degree = torch.sparse.sum(adjacency, dim=1).to_dense()
            two_hop = torch.sparse.mm(adjacency, adjacency).coalesce()
            keys = (
                two_hop.indices()[0].long() * int(num_nodes)
                + two_hop.indices()[1].long()
            )
            order = torch.argsort(keys)
            lookup = {
                "degree": degree.detach(),
                "two_hop_keys": keys[order].detach(),
                "two_hop_values": two_hop.values()[order].detach(),
                "adjacency_nnz": int(adjacency._nnz()),
                "two_hop_nnz": int(two_hop._nnz()),
            }
            self._augment_typed_two_hop_lookup(
                lookup,
                adjacency,
                adjacency_index,
                adjacency_family,
                node_type_ids,
                int(num_nodes),
            )
            if self.use_endpoint_role_residual:
                lookup.update(
                    self._build_endpoint_role_lookup(
                        context_edge_index,
                        num_nodes,
                        node_type_ids,
                        edge_family_ids,
                        batch,
                    )
                )
            return lookup

    def _build_endpoint_role_lookup(
        self,
        context_edge_index: Tensor,
        num_nodes: int,
        node_type_ids: Optional[Tensor],
        edge_family_ids: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> dict:
        device = context_edge_index.device
        node_type = (
            node_type_ids.long().to(device)
            if node_type_ids is not None
            else torch.zeros(num_nodes, dtype=torch.long, device=device)
        )
        graph_batch = (
            batch.long().to(device)
            if batch is not None
            else torch.zeros(num_nodes, dtype=torch.long, device=device)
        )
        num_families = max(
            int(self.endpoint_role_family_embedding.num_embeddings), 1
        )
        family_degree = torch.zeros(
            (num_families, num_nodes), dtype=torch.float, device=device
        )
        total_degree = torch.zeros(
            num_nodes, dtype=torch.float, device=device
        )
        if context_edge_index.numel():
            src = context_edge_index[0].long()
            dst = context_edge_index[1].long()
            lo = torch.minimum(src, dst)
            hi = torch.maximum(src, dst)
            keys = lo * int(num_nodes) + hi
            order = torch.argsort(keys)
            keys = keys[order]
            first = torch.ones_like(keys, dtype=torch.bool)
            first[1:] = keys[1:] != keys[:-1]
            chosen = order[first]
            src = lo[chosen]
            dst = hi[chosen]
            ones = torch.ones(src.numel(), dtype=torch.float, device=device)
            total_degree.scatter_add_(0, src, ones)
            total_degree.scatter_add_(0, dst, ones)
            if edge_family_ids is not None:
                family = edge_family_ids.long().to(device)[chosen].clamp(
                    0, num_families - 1
                )
                flat = family_degree.reshape(-1)
                flat.scatter_add_(0, family * num_nodes + src, ones)
                flat.scatter_add_(0, family * num_nodes + dst, ones)

        def normalize_by_type(values):
            normalized = torch.zeros_like(values)
            graph_count = (
                int(graph_batch.max().item()) + 1
                if graph_batch.numel()
                else 1
            )
            for graph_id in range(graph_count):
                for type_id in range(self.num_node_types):
                    mask = (graph_batch == graph_id) & (node_type == type_id)
                    if not mask.any():
                        continue
                    local = torch.log1p(values[..., mask])
                    mean = local.mean(dim=-1, keepdim=True)
                    std = local.std(dim=-1, keepdim=True, unbiased=False)
                    normalized[..., mask] = (local - mean) / (
                        std + 1e-6
                    )
            return normalized

        return {
            "endpoint_role_total_z": normalize_by_type(total_degree).detach(),
            "endpoint_role_family_z": normalize_by_type(
                family_degree
            ).detach(),
            "endpoint_role_node_type": node_type.detach(),
        }

    @staticmethod
    def _endpoint_pair_features(src_value, dst_value, same_type):
        symmetric = torch.stack(
            [
                src_value + dst_value,
                (src_value - dst_value).abs(),
                src_value * dst_value,
                torch.minimum(src_value, dst_value),
                torch.maximum(src_value, dst_value),
            ],
            dim=-1,
        )
        directed = torch.stack(
            [
                src_value,
                dst_value,
                src_value * dst_value,
                (src_value - dst_value).abs(),
            ],
            dim=-1,
        )
        same = same_type.float().unsqueeze(-1)
        return torch.cat(
            [symmetric * same, directed * (1.0 - same)], dim=-1
        )

    def _endpoint_role_residual(
        self,
        encoded: dict,
        query_edge_index: Tensor,
        edge_family_ids: Optional[Tensor],
        batch: Tensor,
    ) -> Tensor:
        if (
            not self.use_endpoint_role_residual
            or self.endpoint_role_head is None
            or query_edge_index.numel() == 0
        ):
            return encoded["h"].new_zeros(
                (query_edge_index.size(1), self.out_dim_E)
            )
        lookup = encoded.get("two_hop_structure_lookup") or {}
        total_z = lookup.get("endpoint_role_total_z")
        family_z = lookup.get("endpoint_role_family_z")
        node_type = lookup.get("endpoint_role_node_type")
        if total_z is None or family_z is None or node_type is None:
            return encoded["h"].new_zeros(
                (query_edge_index.size(1), self.out_dim_E)
            )
        src = query_edge_index[0].long()
        dst = query_edge_index[1].long()
        family = (
            edge_family_ids.long().to(src.device)
            if edge_family_ids is not None
            else torch.zeros_like(src)
        ).clamp(0, family_z.size(0) - 1)
        same_type = node_type[src] == node_type[dst]
        total_features = self._endpoint_pair_features(
            total_z[src], total_z[dst], same_type
        )
        family_features = self._endpoint_pair_features(
            family_z[family, src],
            family_z[family, dst],
            same_type,
        )
        role_features = torch.cat(
            [total_features, family_features, same_type.float().unsqueeze(-1)],
            dim=-1,
        )
        family_embedding = self.endpoint_role_family_embedding(family)
        y_h = encoded["y_h"]
        edge_y = (
            y_h[batch[src].long()]
            if y_h.numel()
            else role_features.new_zeros(
                (role_features.size(0), int(self.lin_in_y.out_features))
            )
        )
        base = self.endpoint_role_head(
            torch.cat([role_features, family_embedding, edge_y], dim=-1)
        )
        factor_by_graph = encoded.get("endpoint_role_scale_factor")
        factor = (
            factor_by_graph.reshape(-1, 1)[batch[src].long()]
            if factor_by_graph is not None
            else base.new_ones((base.size(0), 1))
        )
        effective_scale = self.endpoint_role_scale * factor
        residual = effective_scale * base
        self.last_endpoint_role_base_residual_mean = float(
            base.detach().abs().mean().cpu()
        )
        self.last_endpoint_role_effective_scale_mean = float(
            effective_scale.detach().mean().cpu()
        )
        self.last_endpoint_role_effective_residual_mean = float(
            residual.detach().abs().mean().cpu()
        )
        return residual

    def _two_hop_structure_features(
        self,
        query_edge_index: Tensor,
        context_edge_index: Tensor,
        num_nodes: int,
        lookup: Optional[dict] = None,
    ) -> Tensor:
        """Compute permutation-equivariant closure and endpoint-role features."""
        if query_edge_index.numel() == 0:
            return torch.zeros(
                (0, int(self.two_hop_structure_feature_dim)),
                device=query_edge_index.device,
                dtype=torch.float,
            )
        if num_nodes <= 0:
            return torch.zeros(
                (query_edge_index.size(1), int(self.two_hop_structure_feature_dim)),
                device=query_edge_index.device,
                dtype=torch.float,
            )

        if lookup is None:
            lookup = self.build_two_hop_structure_lookup(
                context_edge_index=context_edge_index,
                num_nodes=num_nodes,
            )
        degree = lookup["degree"]

        query_src = query_edge_index[0].long()
        query_dst = query_edge_index[1].long()
        common = self._lookup_values_for_keys(
            lookup, query_src, query_dst, num_nodes
        )
        degree_src = degree[query_src]
        degree_dst = degree[query_dst]
        degree_sum = degree_src + degree_dst
        union = (degree_sum - common).clamp_min(1.0)
        log_scale = math.log1p(max(int(num_nodes), 1))
        base = torch.stack(
            [
                torch.log1p(common) / log_scale,
                common / union,
                torch.log1p(degree_sum) / log_scale,
                (degree_src - degree_dst).abs() / degree_sum.clamp_min(1.0),
            ],
            dim=-1,
        )
        if not self.use_typed_two_hop_structure:
            return base
        typed_dim = int(self.two_hop_structure_feature_dim) - 4
        typed = self._lookup_matrix_values_for_keys(
            lookup,
            "typed_two_hop_values",
            query_src,
            query_dst,
            num_nodes,
            typed_dim,
        )
        typed = torch.log1p(typed) / log_scale
        if typed.shape[-1] != typed_dim:
            fixed = typed.new_zeros((typed.size(0), typed_dim))
            take = min(int(typed.shape[-1]), typed_dim)
            if take > 0:
                fixed[:, :take] = typed[:, :take]
            typed = fixed
        return torch.cat([base, typed], dim=-1)

    def _two_hop_structure_residual(
        self,
        encoded: dict,
        query_edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        if not self.use_two_hop_structure or self.two_hop_structure_head is None:
            return encoded["h"].new_zeros(
                (query_edge_index.size(1), self.out_dim_E)
            )
        structure = self._two_hop_structure_features(
            query_edge_index=query_edge_index,
            context_edge_index=encoded["structure_edge_index"],
            num_nodes=int(encoded["h"].size(0)),
            lookup=encoded.get("two_hop_structure_lookup"),
        )
        y_h = encoded["y_h"]
        if y_h.numel() and query_edge_index.numel():
            edge_y = y_h[batch[query_edge_index[0].long()].long()]
        else:
            edge_y = structure.new_zeros(
                (structure.size(0), int(self.lin_in_y.out_features))
            )
        factor_by_graph = encoded.get("two_hop_scale_factor")
        if factor_by_graph is None:
            edge_factor = structure.new_ones((structure.size(0), 1))
        elif query_edge_index.numel():
            graph_idx = batch[query_edge_index[0].long()].long()
            edge_factor = factor_by_graph.reshape(-1, 1)[graph_idx]
        else:
            edge_factor = structure.new_ones((structure.size(0), 1))
        base_residual = self.two_hop_structure_head(
            torch.cat([structure, edge_y], dim=-1)
        )
        effective_scale = self.two_hop_structure_scale * edge_factor
        residual = effective_scale * base_residual
        self.last_two_hop_scale_factor_mean = (
            float(edge_factor.detach().mean().cpu())
            if edge_factor.numel()
            else 1.0
        )
        self.last_two_hop_base_residual_mean = float(
            base_residual.detach().abs().mean().cpu()
        ) if base_residual.numel() else 0.0
        self.last_two_hop_effective_scale_mean = float(
            effective_scale.detach().mean().cpu()
        ) if effective_scale.numel() else 0.0
        self.last_two_hop_effective_residual_mean = float(
            residual.detach().abs().mean().cpu()
        ) if residual.numel() else 0.0
        self.last_two_hop_residual_mean = (
            self.last_two_hop_effective_residual_mean
        )
        return residual

    def forward(
        self,
        X: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        y: Tensor,
        batch: Tensor,
        node_type_ids: Optional[Tensor] = None,
        node_subtype_ids: Optional[Tensor] = None,
        relation_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        two_hop_structure_lookup: Optional[dict] = None,
        two_hop_scale_factor: Optional[Tensor] = None,
        endpoint_role_scale_factor: Optional[Tensor] = None,
        edge_input_residual_scale: Optional[float] = None,
    ) -> utils.SparsePlaceHolder:
        residual_scale = (
            self.edge_input_residual_scale
            if edge_input_residual_scale is None
            else float(edge_input_residual_scale)
        )
        encoded = self.encode_context(
            X=X,
            edge_attr=edge_attr,
            edge_index=edge_index,
            y=y,
            batch=batch,
            node_type_ids=node_type_ids,
            node_subtype_ids=node_subtype_ids,
            relation_type_ids=relation_type_ids,
            edge_family_ids=edge_family_ids,
            two_hop_structure_lookup=two_hop_structure_lookup,
            two_hop_scale_factor=two_hop_scale_factor,
            endpoint_role_scale_factor=endpoint_role_scale_factor,
        )
        src_h = encoded["h"][edge_index[0].long()]
        dst_h = encoded["h"][edge_index[1].long()]
        edge_logits = (
            self._edge_head_logits(
                torch.cat([src_h, dst_h, encoded["context_e"]], dim=-1),
                query_edge_family_ids=edge_family_ids,
            )
            + self._two_hop_structure_residual(
                encoded, edge_index, batch
            )
            + self._endpoint_role_residual(
                encoded, edge_index, edge_family_ids, batch
            )
            + residual_scale * edge_attr[:, : self.out_dim_E]
        )
        return utils.SparsePlaceHolder(
            node=encoded["node_logits"],
            edge_attr=edge_logits,
            edge_index=edge_index,
            y=encoded["y_out"],
            batch=batch,
            charge=encoded["charge_logits"],
        )

    def encode_context(
        self,
        X: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        y: Tensor,
        batch: Tensor,
        node_type_ids: Optional[Tensor] = None,
        node_subtype_ids: Optional[Tensor] = None,
        relation_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        two_hop_structure_lookup: Optional[dict] = None,
        two_hop_scale_factor: Optional[Tensor] = None,
        endpoint_role_scale_factor: Optional[Tensor] = None,
    ) -> dict:
        """Encode nodes using context edges only.

        Query candidates must never be passed through this method unless they
        are actual visible edges in G_t.
        """
        node_type_ids = (
            node_type_ids.long().to(X.device)
            if node_type_ids is not None
            else torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        ).clamp(0, self.num_node_types - 1)
        node_subtype_ids = (
            node_subtype_ids.long().to(X.device)
            if node_subtype_ids is not None
            else torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        )
        relation_type_ids = (
            relation_type_ids.long().to(X.device)
            if relation_type_ids is not None
            else torch.zeros(edge_index.size(1), dtype=torch.long, device=X.device)
        ).clamp(0, self.num_relations - 1)
        edge_family_ids = (
            edge_family_ids.long().to(X.device)
            if edge_family_ids is not None
            else torch.zeros(edge_index.size(1), dtype=torch.long, device=X.device)
        )

        x0 = X[:, : self.out_dim_X]
        charge0 = (
            X[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
            if self.out_dim_charge > 0 and X.size(-1) >= self.out_dim_X + self.out_dim_charge
            else X.new_zeros((X.size(0), 0))
        )
        h = self.lin_in_X(X)
        h = self.subtype_fusion(h, node_type_ids, node_subtype_ids)
        e = self.lin_in_E(edge_attr.float())
        y_h = self.lin_in_y(y)
        graph_y = y_h[batch[edge_index[0]].long()] if y_h.numel() else e.new_zeros((e.size(0), 0))
        if graph_y.numel():
            e = e + self.edge_context(torch.cat([e, graph_y], dim=-1))
        if self.use_time_film and y_h.numel():
            h = self._film_node_with_y(h, y_h, batch, self.time_x_add, self.time_x_mul)
            e = self._film_edge_with_y(e, y_h, edge_index, batch)
        edge_subtype = self._edge_subtype_from_attr(edge_attr, edge_family_ids)
        structure_edge_index = edge_index[:, edge_subtype > 0]

        for layer_idx, conv in enumerate(self.gcs):
            h = conv(h, node_type_ids, edge_index, relation_type_ids, edge_subtype, edge_family_ids)
            if self.use_time_film and y_h.numel() and layer_idx < len(self.layer_time_x_add):
                h = self._film_node_with_y(
                    h, y_h, batch, self.layer_time_x_add[layer_idx], self.layer_time_x_mul[layer_idx]
                )
            e = self._update_edge_state_with_layer(e, h, y_h, edge_index, batch, layer_idx)

        if self.edge_only_model:
            node_logits = x0
            charge_logits = charge0
        else:
            node_charge = self.node_head(h)
            node_logits = node_charge[:, : self.out_dim_X] + x0
            charge_logits = (
                node_charge[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge] + charge0
                if self.out_dim_charge > 0
                else charge0
            )
        y_out = self.y_head(y_h) + y[:, : self.out_dim_y] if self.y_head is not None else y
        return {
            "h": h,
            "y_h": y_h,
            "context_e": e,
            "node_logits": node_logits,
            "charge_logits": charge_logits,
            "y_out": y_out,
            "structure_edge_index": structure_edge_index,
            "two_hop_structure_lookup": two_hop_structure_lookup,
            "two_hop_scale_factor": two_hop_scale_factor,
            "endpoint_role_scale_factor": endpoint_role_scale_factor,
            "context_edge_family_ids": edge_family_ids,
        }

    def decode_queries(
        self,
        encoded: dict,
        query_edge_attr: Tensor,
        query_edge_index: Tensor,
        batch: Tensor,
        query_edge_family_ids: Optional[Tensor] = None,
        query_edge_stage_ids: Optional[Tensor] = None,
        edge_input_residual_scale: Optional[float] = None,
    ) -> Tensor:
        """Decode candidate edges without using them for message passing."""
        h = encoded["h"]
        y_h = encoded["y_h"]
        residual_scale = (
            self.edge_input_residual_scale
            if edge_input_residual_scale is None
            else float(edge_input_residual_scale)
        )
        query_edge_attr = query_edge_attr.float()
        expected = int(self.lin_in_E.in_features)
        if query_edge_attr.shape[-1] < expected:
            query_edge_attr = torch.cat(
                [
                    query_edge_attr,
                    query_edge_attr.new_zeros(
                        (query_edge_attr.shape[0], expected - query_edge_attr.shape[-1])
                    ),
                ],
                dim=-1,
            )
        elif query_edge_attr.shape[-1] > expected:
            query_edge_attr = query_edge_attr[:, :expected]
        e = self.lin_in_E(query_edge_attr)
        if y_h.numel() and query_edge_index.numel():
            edge_y = y_h[batch[query_edge_index[0].long()].long()]
            e = e + self.edge_context(torch.cat([e, edge_y], dim=-1))
            if self.use_time_film:
                e = self._film_edge_with_y(
                    e, y_h, query_edge_index, batch
                )
        if self.use_edge_state_update and query_edge_index.numel():
            # Query states are not propagated. They may still be refined from
            # final endpoint embeddings, preserving E_t^{ij} in the decoder.
            for layer_idx in range(self.n_layers):
                e = self._update_edge_state_with_layer(
                    e, h, y_h, query_edge_index, batch, layer_idx
                )
        src_h = h[query_edge_index[0].long()]
        dst_h = h[query_edge_index[1].long()]
        # Low-rank bilinear score: captures structural similarity
        bilinear_feat = self.bilinear_proj_src(src_h) * self.bilinear_proj_dst(dst_h)
        bilinear_score = self.bilinear_out(bilinear_feat)  # (E, 1)
        return (
            self._edge_head_logits(
                torch.cat([src_h, dst_h, e], dim=-1),
                query_edge_stage_ids=query_edge_stage_ids,
                query_edge_family_ids=query_edge_family_ids,
            )
            + torch.cat([bilinear_score, bilinear_score.new_zeros(bilinear_score.shape[0], self.out_dim_E - 1)], dim=-1)
            + self._two_hop_structure_residual(
                encoded, query_edge_index, batch
            )
            + self._endpoint_role_residual(
                encoded,
                query_edge_index,
                query_edge_family_ids,
                batch,
            )
            + residual_scale * query_edge_attr[:, : self.out_dim_E]
        )
