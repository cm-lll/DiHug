"""Matched joint-readout framework: optional CN INPUT, never a scalar addition.

This architecture requires fresh native weights. Original v9/CN v2 checkpoints
are not compatible and are never imported by engineering commands.
"""
import itertools
import numpy as np
import torch
from torch import nn
from hsgd.holders import PlaceHolder
from hsgd.vendor.conv_transformer_model import GraphTransformerConv


class TypedSparseDenoiser(nn.Module):
    architecture_version = "joint_local_global_cn_v1"

    def __init__(self,n_types,n_families,layers=3,width=128,edge_width=64,heads=4,dropout=0.,
                 family_pairs=None,prior=None,global_width=32,ff_node=None,ff_edge=None,
                 ff_global=None,cn_hidden=0,joint_width=128,max_wedges=2000000,max_query_paths=2000000):
        super().__init__()
        if width%heads or edge_width%heads:raise ValueError("hidden dimensions must divide heads")
        self.n_types,self.n_families=n_types,n_families
        self.family_pairs=tuple(family_pairs or itertools.combinations_with_replacement(range(n_types),2))
        if len(self.family_pairs)!=n_families or len(set(self.family_pairs))!=n_families:raise ValueError("one relation state per type pair required")
        if any(a>b or a<0 or b>=n_types for a,b in self.family_pairs):raise ValueError("invalid schema")
        prior=np.full(n_families,.5) if prior is None else np.asarray(prior,dtype=float)
        if prior.shape!=(n_families,) or not np.isfinite(prior).all() or ((prior<0)|(prior>1)).any():raise ValueError("invalid prior")
        self.register_buffer("prior",torch.tensor(prior,dtype=torch.float64))
        self.backbone=GraphTransformerConv(layers,PlaceHolder(X=n_types,E=2*n_families+1,y=1,charge=0),
            dict(dx=width,de=edge_width,dy=global_width,n_head=heads,dim_ffX=2*width if ff_node is None else ff_node,
                 dim_ffE=2*edge_width if ff_edge is None else ff_edge),PlaceHolder(X=n_types,E=n_families+1,y=0,charge=0),
            dropout=dropout,sn_hidden_dim=0)
        if ff_global is not None:
            for layer in self.backbone.tf_layers:
                if layer.last_layer:
                    layer.lin_y1=nn.Linear(global_width,ff_global)
                    layer.lin_y2=nn.Linear(ff_global,global_width)
        if cn_hidden not in (0,32):raise ValueError("baseline0 or CN32 only")
        self.max_wedges=int(max_wedges);self.max_query_paths=int(max_query_paths)
        if min(self.max_wedges,self.max_query_paths)<1:raise ValueError("resource guard must be positive")
        self.cn_feature_width=32+n_types+1
        self.common_neighbor=None
        # Every field enters ONE nonlinear network before the two logits.
        # Previous backbone logits are input information, not an output residual.
        fields=[("edge",edge_width),("endpoint_mean",width),("endpoint_absdiff",width),
                ("backbone_global",global_width),("time",1),("Xt_statistics",2*n_families),
                ("family",n_families),("current_state",2),("backbone_logits",2),("CN",self.cn_feature_width)]
        self.joint_slices={};offset=0
        for name,size in fields:self.joint_slices[name]=slice(offset,offset+size);offset+=size
        self.joint_head=nn.Sequential(nn.Linear(offset,joint_width),nn.SiLU(),nn.Linear(joint_width,2))
        # Initialize shared parameters before optional parameters, so the same
        # seed gives A/B identical common backbone/head initialization.
        if cn_hidden:
            from hsgd.common_neighbor import CommonNeighborFeatures
            self.common_neighbor=CommonNeighborFeatures(width,n_types,n_families,32)

    def forward(self,types,existing_edges,existing_families,query_edges,query_families,time):
        return self.forward_upstream(types,existing_edges,existing_families,query_edges,query_families,time)

    def forward_upstream(self,types,existing_edges,existing_families,query_edges,query_families,time):
        from hsgd.computational import FrozenTypedContext
        token=("one-forward",float(time))
        context=FrozenTypedContext(self,types,existing_edges,existing_families,token)
        try:return context.predict(query_edges,query_families,time,token)
        finally:context.close()

    def joint_predict(self,ctx,graph,time,mask_cn=False,return_features=False):
        result,hidden=self.backbone(ctx.x,graph.attrs,graph.edges,
            ctx.x.new_tensor([[float(time)]]),ctx.batch,return_hidden=True)
        index=graph.query_index;family=graph.query_families
        query=graph.edges[:,index];left,right=hidden["node"][query[0]],hidden["node"][query[1]]
        old_logits=torch.stack([result.edge_attr[index,0],result.edge_attr[index,family+1]],-1)
        cn=ctx.x.new_zeros((len(index),self.cn_feature_width))
        if self.common_neighbor is not None and not mask_cn:
            cn=self.common_neighbor.aggregate(ctx,query,family,time)
        state=(graph.states[index]>0).long()
        fields=[hidden["edge"][index],(left+right)*.5,(left-right).abs(),
            hidden["global_state"].expand(len(index),-1),ctx.x.new_full((len(index),1),float(time)),
            ctx.global_statistics.expand(len(index),-1),
            torch.nn.functional.one_hot(family,self.n_families).float(),
            torch.nn.functional.one_hot(state,2).float(),old_logits,cn]
        features=torch.cat(fields,-1)
        logits=self.joint_head(features)
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():raise FloatingPointError("nonfinite fusion features/logits")
        if return_features:return logits,features
        return logits
