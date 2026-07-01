import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm


import torch_geometric.nn.pool as pool
from torch_geometric.utils import softmax, sort_edge_index

from sparse_diffusion import utils
from sparse_diffusion.models.transconv_layer import TransformerConv
from sparse_diffusion.models.heterogeneous_transconv_layer import HeterogeneousTransformerConv
from sparse_diffusion.models.layers import SparseXtoy, SparseEtoy


def _build_self_attn_block(
    heterogeneous: bool,
    dx: int,
    de: int,
    dy: int,
    n_head: int,
    dropout: float,
    last_layer: bool,
    num_node_types: int,
    num_node_subtypes: int,
    num_relation_types: int,
    type_embed_dim: int,
    subtype_embed_dim: int,
    relation_embed_dim: int,
    edge_family_offsets: Optional[dict],
    use_type_modulation: bool,
    edge_only: bool,
    use_family_film: bool,
    use_family_edge_update: bool,
    use_relation_attention_matrix: bool,
    use_relation_message_matrix: bool,
    use_family_y_film: bool,
    use_family_y_in_attention: bool,
    use_family_y_in_edge_film: bool,
    family_y_dim: int,
    family_edge_update_hidden_dim: int,
):
    if heterogeneous:
        return HeterogeneousTransformerConv(
            dx=dx,
            de=de,
            dy=dy,
            heads=n_head,
            concat=True,
            dropout=dropout,
            bias=True,
            last_layer=last_layer,
            heterogeneous=True,
            num_node_types=num_node_types,
            num_node_subtypes=num_node_subtypes,
            num_relation_types=num_relation_types,
            type_embed_dim=type_embed_dim,
            subtype_embed_dim=subtype_embed_dim,
            relation_embed_dim=relation_embed_dim,
            edge_family_offsets=edge_family_offsets,
            use_type_modulation=use_type_modulation,
            edge_only=edge_only,
            use_family_film=use_family_film,
            use_family_edge_update=use_family_edge_update,
            use_relation_attention_matrix=use_relation_attention_matrix,
            use_relation_message_matrix=use_relation_message_matrix,
            use_family_y_film=use_family_y_film,
            use_family_y_in_attention=use_family_y_in_attention,
            use_family_y_in_edge_film=use_family_y_in_edge_film,
            family_y_dim=family_y_dim,
            family_edge_update_hidden_dim=family_edge_update_hidden_dim,
        )
    return TransformerConv(
        dx=dx,
        de=de,
        dy=dy,
        heads=n_head,
        concat=True,
        dropout=dropout,
        bias=True,
        last_layer=last_layer,
        edge_only=edge_only,
    )


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        last_layer: bool = True,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
        # 异质图相关参数
        heterogeneous: bool = False,
        num_node_types: int = 0,
        num_node_subtypes: int = 0,
        num_relation_types: int = 0,
        type_embed_dim: int = 64,
        subtype_embed_dim: int = 64,
        relation_embed_dim: int = 64,
        edge_family_offsets: Optional[dict] = None,
        use_type_modulation: bool = True,  # 是否使用类别调制子类别
        edge_only: bool = False,  # 若 True：只更新边 E=f(QK,E)，不更新节点 X（用于最后几层「E=QK+E」重复）
        use_family_film: bool = False,
        use_family_edge_update: bool = False,
        use_relation_attention_matrix: bool = False,
        use_relation_message_matrix: bool = False,
        use_family_y_film: bool = False,
        use_family_y_in_attention: bool = False,
        use_family_y_in_edge_film: bool = False,
        family_y_dim: int = 0,
        family_edge_update_hidden_dim: int = 128,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.last_layer = last_layer
        self.heterogeneous = heterogeneous
        self.edge_only = edge_only

        self.self_attn = _build_self_attn_block(
            heterogeneous=heterogeneous,
            dx=dx,
            de=de,
            dy=dy,
            n_head=n_head,
            dropout=dropout,
            last_layer=last_layer,
            num_node_types=num_node_types,
            num_node_subtypes=num_node_subtypes,
            num_relation_types=num_relation_types,
            type_embed_dim=type_embed_dim,
            subtype_embed_dim=subtype_embed_dim,
            relation_embed_dim=relation_embed_dim,
            edge_family_offsets=edge_family_offsets,
            use_type_modulation=use_type_modulation,
            edge_only=edge_only,
            use_family_film=use_family_film,
            use_family_edge_update=use_family_edge_update,
            use_relation_attention_matrix=use_relation_attention_matrix,
            use_relation_message_matrix=use_relation_message_matrix,
            use_family_y_film=use_family_y_film,
            use_family_y_in_attention=use_family_y_in_attention,
            use_family_y_in_edge_film=use_family_y_in_edge_film,
            family_y_dim=family_y_dim,
            family_edge_update_hidden_dim=family_edge_update_hidden_dim,
        )

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)  # TODO: set norm
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)  # TODO: set norm
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)

        if self.last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)  # TODO: set norm
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)

        self.activation = F.relu

    def forward(
        self, 
        X: Tensor, 
        edge_index: Tensor, 
        edge_attr: Tensor, 
        y: Tensor, 
        batch: Tensor,
        # 异质图元数据（可选）
        node_type_ids: Optional[Tensor] = None,
        node_subtype_ids: Optional[Tensor] = None,
        relation_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        family_y_hidden: Optional[Tensor] = None,
        type_offsets: Optional[dict] = None,
    ):
        """Pass the input through the encoder layer.
        X: (N, d)
        edge_index: (M, 2)
        edge_attr: (M, d)
        batch: (n)
        y: (n)
        """
        if self.heterogeneous:
            new_x, new_edge_attr, new_y = self.self_attn(
                X, edge_index, edge_attr, y, batch,
                node_type_ids=node_type_ids,
                node_subtype_ids=node_subtype_ids,
                relation_type_ids=relation_type_ids,
                edge_family_ids=edge_family_ids,
                family_y_hidden=family_y_hidden,
                type_offsets=type_offsets,
            )
        else:
            new_x, new_edge_attr, new_y = self.self_attn(X, edge_index, edge_attr, y, batch)

        if self.edge_only:
            # Edge-only mode: keep node states fixed, only update edges.
            edge_attr = self.normE1(edge_attr + new_edge_attr)
            if self.last_layer:
                y = self.norm_y1(y + new_y)
            else:
                y = new_y
            ff_outputE = self.linE2(self.activation(self.linE1(edge_attr)))
            edge_attr = self.normE2(edge_attr + ff_outputE)
            if self.last_layer:
                ff_output_y = self.lin_y2(self.activation(self.lin_y1(y)))
                y = self.norm_y2(y + ff_output_y)
            return X, edge_attr, y

        X = self.normX1(X + new_x)
        edge_attr = self.normE1(edge_attr + new_edge_attr)

        if self.last_layer:
            y = self.norm_y1(y + new_y)
        else:
            y = new_y

        ff_outputX = self.linX2(self.activation(self.linX1(X)))
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.activation(self.linE1(edge_attr)))
        edge_attr = self.normE2(edge_attr + ff_outputE)

        if self.last_layer:
            ff_output_y = self.lin_y2(self.activation(self.lin_y1(y)))
            y = self.norm_y2(y + ff_output_y)

        return X, edge_attr, y


class EdgeOnlyTransformerLayer(nn.Module):
    """Edge-only transformer layer: keep X fixed, update only E (and optional y)."""

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        last_layer: bool = True,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
        heterogeneous: bool = False,
        num_node_types: int = 0,
        num_node_subtypes: int = 0,
        num_relation_types: int = 0,
        type_embed_dim: int = 64,
        subtype_embed_dim: int = 64,
        relation_embed_dim: int = 64,
        edge_family_offsets: Optional[dict] = None,
        use_type_modulation: bool = True,
        use_family_film: bool = False,
        use_family_edge_update: bool = False,
        use_relation_attention_matrix: bool = False,
        use_relation_message_matrix: bool = False,
        use_family_y_film: bool = False,
        use_family_y_in_attention: bool = False,
        use_family_y_in_edge_film: bool = False,
        family_y_dim: int = 0,
        family_edge_update_hidden_dim: int = 128,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.last_layer = last_layer
        self.heterogeneous = heterogeneous
        self.self_attn = _build_self_attn_block(
            heterogeneous=heterogeneous,
            dx=dx,
            de=de,
            dy=dy,
            n_head=n_head,
            dropout=dropout,
            last_layer=last_layer,
            num_node_types=num_node_types,
            num_node_subtypes=num_node_subtypes,
            num_relation_types=num_relation_types,
            type_embed_dim=type_embed_dim,
            subtype_embed_dim=subtype_embed_dim,
            relation_embed_dim=relation_embed_dim,
            edge_family_offsets=edge_family_offsets,
            use_type_modulation=use_type_modulation,
            edge_only=True,
            use_family_film=use_family_film,
            use_family_edge_update=use_family_edge_update,
            use_relation_attention_matrix=use_relation_attention_matrix,
            use_relation_message_matrix=use_relation_message_matrix,
            use_family_y_film=use_family_y_film,
            use_family_y_in_attention=use_family_y_in_attention,
            use_family_y_in_edge_film=use_family_y_in_edge_film,
            family_y_dim=family_y_dim,
            family_edge_update_hidden_dim=family_edge_update_hidden_dim,
        )

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        if self.last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.activation = F.relu

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        y: Tensor,
        batch: Tensor,
        node_type_ids: Optional[Tensor] = None,
        node_subtype_ids: Optional[Tensor] = None,
        relation_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        family_y_hidden: Optional[Tensor] = None,
        type_offsets: Optional[dict] = None,
    ):
        if self.heterogeneous:
            _, new_edge_attr, new_y = self.self_attn(
                X,
                edge_index,
                edge_attr,
                y,
                batch,
                node_type_ids=node_type_ids,
                node_subtype_ids=node_subtype_ids,
                relation_type_ids=relation_type_ids,
                edge_family_ids=edge_family_ids,
                family_y_hidden=family_y_hidden,
                type_offsets=type_offsets,
            )
        else:
            _, new_edge_attr, new_y = self.self_attn(X, edge_index, edge_attr, y, batch)

        edge_attr = self.normE1(edge_attr + new_edge_attr)
        if self.last_layer:
            y = self.norm_y1(y + new_y)
        else:
            y = new_y

        ff_outputE = self.linE2(self.activation(self.linE1(edge_attr)))
        edge_attr = self.normE2(edge_attr + ff_outputE)
        if self.last_layer:
            ff_output_y = self.lin_y2(self.activation(self.lin_y1(y)))
            y = self.norm_y2(y + ff_output_y)
        return X, edge_attr, y


class GraphTransformerConv(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dims: utils.PlaceHolder,
        hidden_dims: dict,
        output_dims: utils.PlaceHolder,
        dropout: 0.1,
        sn_hidden_dim: int,
        output_y: bool = False,
        # 异质图相关参数
        heterogeneous: bool = False,
        num_node_types: int = 0,
        num_node_subtypes: int = 0,
        num_relation_types: int = 0,
        type_embed_dim: int = 64,
        subtype_embed_dim: int = 64,
        relation_embed_dim: int = 64,
        edge_family_offsets: Optional[dict] = None,
        type_offsets: Optional[dict] = None,
        use_type_modulation: bool = True,
        edge_only_model: bool = False,  # True：所有层仅更新边，节点仅作条件输入（固定节点、只生成边）
        use_family_film: bool = False,
        use_family_edge_update: bool = False,
        use_relation_attention_matrix: bool = False,
        use_relation_message_matrix: bool = False,
        use_family_y_film: bool = False,
        use_family_y_in_attention: bool = False,
        use_family_y_in_edge_film: bool = False,
        family_y_dim: int = 0,
        family_edge_update_hidden_dim: int = 128,
        use_edge_struct_features: bool = False,
        edge_struct_feature_dim: int = 8,
        edge_struct_hidden_dim: int = 64,
        edge_struct_residual_scale: float = 1.0,
        edge_struct_use_family_y: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.edge_only_model = edge_only_model
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charge = output_dims.charge
        self.output_y = output_y
        self.dropout = dropout
        self.heterogeneous = heterogeneous
        self.type_offsets = type_offsets
        self.family_y_dim = max(int(family_y_dim or 0), 0)
        self.use_edge_struct_features = bool(use_edge_struct_features)
        self.edge_struct_feature_dim = max(int(edge_struct_feature_dim or 0), 0)
        self.edge_struct_residual_scale = float(edge_struct_residual_scale or 0.0)
        self.edge_struct_use_family_y = bool(edge_struct_use_family_y)

        # 调试信息：记录模型初始化时的input_dims
        print(f"[DEBUG] GraphTransformerConv.__init__: input_dims.X = {input_dims.X}, input_dims.E = {input_dims.E}, input_dims.charge = {input_dims.charge}, sn_hidden_dim = {sn_hidden_dim}")
        print(f"[DEBUG] GraphTransformerConv.__init__: lin_in_X输入维度 = {input_dims.X + input_dims.charge + sn_hidden_dim}, hidden_dims['dx'] = {hidden_dims['dx']}")
        print(f"[DEBUG] GraphTransformerConv.__init__: lin_in_E输入维度 = {input_dims.E}, hidden_dims['de'] = {hidden_dims['de']}")
        if self.edge_only_model:
            print("[DEBUG] GraphTransformerConv: edge_only_model=True (all layers update E only, X fixed as conditioning)")
        
        self.lin_in_X = nn.Linear(
            input_dims.X + input_dims.charge + sn_hidden_dim, hidden_dims["dx"]
        )
        self.lin_in_E = nn.Linear(input_dims.E, hidden_dims["de"])
        self.lin_in_y = nn.Linear(input_dims.y, hidden_dims["dy"])
        if self.family_y_dim > 0:
            self.lin_in_family_y = nn.Sequential(
                nn.LayerNorm(self.family_y_dim),
                nn.Linear(self.family_y_dim, hidden_dims["dy"]),
                nn.GELU(),
                nn.Linear(hidden_dims["dy"], hidden_dims["dy"]),
            )
        else:
            self.lin_in_family_y = None

        # last layer is True when we keep the last output layers of y
        self.tf_layers = nn.ModuleList([])
        for i in range(n_layers):
            last_layer_flag = True if output_y else (i < n_layers - 1)
            common_kwargs = dict(
                dx=hidden_dims["dx"],
                de=hidden_dims["de"],
                dy=hidden_dims["dy"],
                n_head=hidden_dims["n_head"],
                dim_ffE=hidden_dims["dim_ffE"],
                last_layer=last_layer_flag,
                heterogeneous=heterogeneous,
                num_node_types=num_node_types,
                num_node_subtypes=num_node_subtypes,
                num_relation_types=num_relation_types,
                type_embed_dim=type_embed_dim,
                subtype_embed_dim=subtype_embed_dim,
                relation_embed_dim=relation_embed_dim,
                edge_family_offsets=edge_family_offsets,
                use_type_modulation=use_type_modulation,
                use_family_film=use_family_film,
                use_family_edge_update=use_family_edge_update,
                use_relation_attention_matrix=use_relation_attention_matrix,
                use_relation_message_matrix=use_relation_message_matrix,
                use_family_y_film=use_family_y_film,
                use_family_y_in_attention=use_family_y_in_attention,
                use_family_y_in_edge_film=use_family_y_in_edge_film,
                family_y_dim=hidden_dims["dy"] if self.family_y_dim > 0 else 0,
                family_edge_update_hidden_dim=family_edge_update_hidden_dim,
            )
            if self.edge_only_model:
                layer = EdgeOnlyTransformerLayer(**common_kwargs)
            else:
                layer = XEyTransformerLayer(
                    dim_ffX=hidden_dims["dim_ffX"],
                    edge_only=False,
                    **common_kwargs,
                )
            self.tf_layers.append(layer)
        self.out_ln_X = nn.LayerNorm(hidden_dims["dx"])
        self.out_ln_E = nn.LayerNorm(hidden_dims["de"])
        self.lin_out_X = nn.Linear(
            hidden_dims["dx"], output_dims.X + output_dims.charge
        )
        self.lin_out_E = nn.Linear(hidden_dims["de"], output_dims.E)

        if self.output_y:
            self.out_ln_y = nn.LayerNorm(hidden_dims["dy"])
            self.lin_out_y = nn.Linear(hidden_dims["dy"], output_dims.y)
        if self.use_edge_struct_features and self.edge_struct_feature_dim > 0:
            edge_struct_in = self.edge_struct_feature_dim
            if self.edge_struct_use_family_y:
                edge_struct_in += hidden_dims["dy"]
            self.edge_struct_residual_head = nn.Sequential(
                nn.LayerNorm(edge_struct_in),
                nn.Linear(edge_struct_in, edge_struct_hidden_dim),
                nn.GELU(),
                nn.Linear(edge_struct_hidden_dim, output_dims.E),
            )
            init.zeros_(self.edge_struct_residual_head[-1].weight)
            init.zeros_(self.edge_struct_residual_head[-1].bias)
        else:
            self.edge_struct_residual_head = None

    def forward(
        self, 
        X, 
        edge_attr, 
        edge_index, 
        y, 
        batch,
        # 异质图元数据（可选）
        node_type_ids: Optional[Tensor] = None,
        node_subtype_ids: Optional[Tensor] = None,
        relation_type_ids: Optional[Tensor] = None,
        edge_family_ids: Optional[Tensor] = None,
        family_y_table: Optional[Tensor] = None,
        edge_struct_features: Optional[Tensor] = None,
        **_,
    ):
        # Save for residual connection
        X0 = X.clone()
        edge_attr0 = edge_attr.clone()
        y0 = y.clone()

        # In edge-only mode, nodes are fixed prediction targets and only used
        # as conditioning states to produce Q/K for edge updates.
        X_fixed = X0[:, : self.out_dim_X]
        if self.out_dim_charge > 0:
            X_charge_fixed = X0[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
        else:
            X_charge_fixed = X0.new_zeros((X0.size(0), 0))

        # Input block
        X = self.lin_in_X(X)
        edge_attr = self.lin_in_E(edge_attr)
        y = self.lin_in_y(y)
        family_y_hidden = None
        if (
            self.lin_in_family_y is not None
            and family_y_table is not None
            and family_y_table.numel() > 0
        ):
            family_y_hidden = self.lin_in_family_y(family_y_table.float())

        # Transformer layers
        for layer in self.tf_layers:
            if self.heterogeneous:
                X, edge_attr, y = layer(
                    X, edge_index, edge_attr, y, batch,
                    node_type_ids=node_type_ids,
                    node_subtype_ids=node_subtype_ids,
                    relation_type_ids=relation_type_ids,
                    edge_family_ids=edge_family_ids,
                    family_y_hidden=family_y_hidden,
                    type_offsets=self.type_offsets,
                )
            else:
                X, edge_attr, y = layer(X, edge_index, edge_attr, y, batch)

        # Output block
        if self.edge_only_model:
            X = X_fixed
            charges = X_charge_fixed
        else:
            X = self.lin_out_X(self.out_ln_X(X))
            charges = (
                X[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
                + X0[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
            )
            X = X[:, : self.out_dim_X] + X0[:, : self.out_dim_X]
        edge_attr = self.lin_out_E(self.out_ln_E(edge_attr))
        if self.output_y:
            y = self.lin_out_y(self.out_ln_y(y))

        # For heterogeneous directed relations, keep directional edge outputs.
        # Symmetrizing (u->v) and (v->u) can blur relation direction semantics.
        if self.heterogeneous:
            edge_index_out = edge_index
            edge_attr_out = edge_attr
        else:
            # Homogeneous/undirected setup: keep the original symmetric merge behavior.
            edge_index_out, top_edge_attr = sort_edge_index(edge_index, edge_attr)
            _, bot_edge_attr = sort_edge_index(edge_index[[1, 0]], edge_attr)
            edge_attr_out = top_edge_attr + bot_edge_attr

        edge_attr = edge_attr_out + edge_attr0[:, : self.out_dim_E]
        if (
            self.edge_struct_residual_head is not None
            and edge_struct_features is not None
            and edge_struct_features.numel() > 0
            and edge_struct_features.shape[0] == edge_attr.shape[0]
        ):
            struct_input = edge_struct_features.to(
                device=edge_attr.device, dtype=edge_attr.dtype
            )
            if struct_input.size(-1) != self.edge_struct_feature_dim:
                if struct_input.size(-1) > self.edge_struct_feature_dim:
                    struct_input = struct_input[:, : self.edge_struct_feature_dim]
                else:
                    pad = struct_input.new_zeros(
                        struct_input.size(0),
                        self.edge_struct_feature_dim - struct_input.size(-1),
                    )
                    struct_input = torch.cat([struct_input, pad], dim=-1)
            if self.edge_struct_use_family_y:
                fam_cond = struct_input.new_zeros((struct_input.size(0), y.size(-1)))
                if (
                    family_y_hidden is not None
                    and edge_family_ids is not None
                    and edge_family_ids.numel() == struct_input.size(0)
                ):
                    edge_batch = batch[edge_index_out[0].long()].long()
                    fam_ids = edge_family_ids.to(edge_attr.device).long()
                    valid = (
                        (edge_batch >= 0)
                        & (edge_batch < family_y_hidden.size(0))
                        & (fam_ids >= 0)
                        & (fam_ids < family_y_hidden.size(1))
                    )
                    if valid.any():
                        fam_cond[valid] = family_y_hidden[
                            edge_batch[valid], fam_ids[valid]
                        ].to(dtype=edge_attr.dtype)
                struct_input = torch.cat([struct_input, fam_cond], dim=-1)
            edge_attr = edge_attr + self.edge_struct_residual_scale * self.edge_struct_residual_head(
                struct_input
            )

        if self.output_y:
            y = y + y0[:, : self.out_dim_y]

        return utils.SparsePlaceHolder(
            node=X,
            edge_attr=edge_attr,
            edge_index=edge_index_out,
            y=y,
            batch=batch,
            charge=charges,
        )
