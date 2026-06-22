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
        two_hop_structure_hidden_dim: int = 64,
        use_time_film: bool = False,
        use_edge_state_update: bool = False,
        edge_state_update_mode: str = "all",
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
        self.use_two_hop_structure = bool(use_two_hop_structure)
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
        self.edge_head = nn.Sequential(
            nn.Linear(2 * dx + de, hidden_dims["dim_ffE"]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims["dim_ffE"], output_dims.E),
        )
        if self.use_two_hop_structure:
            structure_hidden = max(int(two_hop_structure_hidden_dim), 8)
            self.two_hop_structure_head = nn.Sequential(
                nn.LayerNorm(4 + dy),
                nn.Linear(4 + dy, structure_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(structure_hidden, output_dims.E),
            )
            nn.init.zeros_(self.two_hop_structure_head[-1].weight)
            nn.init.zeros_(self.two_hop_structure_head[-1].bias)
        else:
            self.two_hop_structure_head = None
        self.last_two_hop_residual_mean = 0.0
        self.y_head = nn.Linear(dy, output_dims.y) if output_y else None

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

    def _two_hop_structure_features(
        self,
        query_edge_index: Tensor,
        context_edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Compute permutation-equivariant closure and endpoint-role features."""
        if query_edge_index.numel() == 0:
            return torch.zeros(
                (0, 4), device=query_edge_index.device, dtype=torch.float
            )
        if context_edge_index.numel() == 0 or num_nodes <= 0:
            return torch.zeros(
                (query_edge_index.size(1), 4),
                device=query_edge_index.device,
                dtype=torch.float,
            )

        src = context_edge_index[0].long()
        dst = context_edge_index[1].long()
        non_loop = src != dst
        src = src[non_loop]
        dst = dst[non_loop]
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        canonical = torch.unique(lo * int(num_nodes) + hi)
        lo = torch.div(canonical, int(num_nodes), rounding_mode="floor")
        hi = canonical.remainder(int(num_nodes))
        adjacency_index = torch.stack(
            [torch.cat([lo, hi]), torch.cat([hi, lo])], dim=0
        )
        adjacency = torch.sparse_coo_tensor(
            adjacency_index,
            torch.ones(
                adjacency_index.size(1),
                device=query_edge_index.device,
                dtype=torch.float,
            ),
            (num_nodes, num_nodes),
            device=query_edge_index.device,
        ).coalesce()
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        two_hop = torch.sparse.mm(adjacency, adjacency).coalesce()

        query_src = query_edge_index[0].long()
        query_dst = query_edge_index[1].long()
        common = self._sparse_values_for_keys(
            two_hop, query_src, query_dst, num_nodes
        )
        degree_src = degree[query_src]
        degree_dst = degree[query_dst]
        degree_sum = degree_src + degree_dst
        union = (degree_sum - common).clamp_min(1.0)
        log_scale = math.log1p(max(int(num_nodes), 1))
        return torch.stack(
            [
                torch.log1p(common) / log_scale,
                common / union,
                torch.log1p(degree_sum) / log_scale,
                (degree_src - degree_dst).abs() / degree_sum.clamp_min(1.0),
            ],
            dim=-1,
        )

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
        )
        y_h = encoded["y_h"]
        if y_h.numel() and query_edge_index.numel():
            edge_y = y_h[batch[query_edge_index[0].long()].long()]
        else:
            edge_y = structure.new_zeros(
                (structure.size(0), int(self.lin_in_y.out_features))
            )
        residual = self.two_hop_structure_head(
            torch.cat([structure, edge_y], dim=-1)
        )
        self.last_two_hop_residual_mean = float(
            residual.detach().abs().mean().cpu()
        ) if residual.numel() else 0.0
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
    ) -> utils.SparsePlaceHolder:
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
        )
        src_h = encoded["h"][edge_index[0].long()]
        dst_h = encoded["h"][edge_index[1].long()]
        edge_logits = (
            self.edge_head(
                torch.cat([src_h, dst_h, encoded["context_e"]], dim=-1)
            )
            + self._two_hop_structure_residual(
                encoded, edge_index, batch
            )
            + edge_attr[:, : self.out_dim_E]
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
        }

    def decode_queries(
        self,
        encoded: dict,
        query_edge_attr: Tensor,
        query_edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Decode candidate edges without using them for message passing."""
        h = encoded["h"]
        y_h = encoded["y_h"]
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
        return (
            self.edge_head(torch.cat([src_h, dst_h, e], dim=-1))
            + self._two_hop_structure_residual(
                encoded, query_edge_index, batch
            )
            + query_edge_attr[:, : self.out_dim_E]
        )
