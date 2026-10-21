"""One candidate change: relocate role context to the nonlinear joint readout.

This is an untrained new architecture. Old C weights are NOT compatible native
weights. No training, guidance, quota allocation, or sampling adjustments here.
Degree residual, role token encoder, attention, CN and diffusion are unchanged.
"""
import math
import time
import numpy as np
import torch
from torch import nn
from role_condition.model import ConditionalDenoiser, NodeCondition, capacities, degrees, features


class LateNodeCondition(NodeCondition):
    def __init__(self,R,K,width):
        super().__init__(R,K,width,True)
        del self.role_out

    def forward(self,ctx,t,target):
        start=time.perf_counter();types=ctx.types;K=ctx.model.n_types;R=ctx.model.n_families
        cap=capacities(types,ctx.model.family_pairs)
        d=degrees(types,ctx.old,ctx.of,R)
        xf=features(d,cap[types])
        x=torch.as_tensor(np.c_[xf,np.eye(K,dtype=np.float32)[types],
                               np.full((len(types),1),t,dtype=np.float32)],device=ctx.device)
        stats_seconds=time.perf_counter()-start;start=time.perf_counter()
        degree_delta=self.degree_out(self.degree_encoder(x))
        degree_seconds=time.perf_counter()-start;start=time.perf_counter()
        if target is None:raise ValueError("explicit anonymous target required")
        if target.type_counts!=tuple(map(int,np.bincount(types))) or target.schema!=ctx.model.family_pairs:
            raise ValueError("condition inventory/schema mismatch")
        context=x.new_zeros((len(types),32))
        for typ,rows in enumerate(target.roles):
            ix=torch.as_tensor(np.flatnonzero(types==typ),device=ctx.device)
            vec=np.array([v for v,n in rows]);counts=np.array([n for v,n in rows])
            freq=counts/target.type_counts[typ]
            encoded=self.role_encoder(torch.as_tensor(np.c_[features(vec,cap[typ]),freq,
                np.log1p(counts)/np.log1p(target.type_counts[typ])].astype(np.float32),device=ctx.device))
            q=self.query(x[ix]);k=self.key(encoded);v=self.value(encoded)
            att=(q@k.T/math.sqrt(32)+torch.as_tensor(np.log(freq),device=ctx.device,dtype=x.dtype)).softmax(-1)
            context=context.index_copy(0,ix,att@v)
        if not torch.isfinite(context).all() or not torch.isfinite(degree_delta).all():
            raise FloatingPointError("nonfinite condition")
        # Same degree residual as B. Role context goes to readout, NOT initial nodes.
        return degree_delta,dict(degree=d.copy(),role_context=context,statistics_seconds=stats_seconds,
                                  degree_network_seconds=degree_seconds,
                                  role_attention_seconds=time.perf_counter()-start)


class LateConditionalDenoiser(ConditionalDenoiser):
    architecture_version="role_condition_late_joint_v1"

    def __init__(self,*args,**kwargs):
        kwargs.pop("mode",None)
        super().__init__(*args,mode="C",**kwargs)
        original=self.branch
        branch=LateNodeCondition(self.n_families,self.n_types,self.backbone.lin_in_X.out_features)
        branch.load_state_dict({k:v for k,v in original.state_dict().items() if not k.startswith("role_out.")},strict=True)
        self.branch=branch
        self.role_fusion=nn.Linear(64,self.joint_head[0].out_features,bias=False)
        nn.init.zeros_(self.role_fusion.weight)

    @staticmethod
    def endpoint_roles(context,query):
        left,right=context[query[0]],context[query[1]]
        return torch.cat([(left+right)*.5,(left-right).abs()],-1)

    def readout(self,features,role_pairs):
        # One nonlinear head, not independent scalar additions to final logits.
        hidden=self.joint_head[0](features)+self.role_fusion(role_pairs)
        return self.joint_head[2](self.joint_head[1](hidden))

    def joint_predict(self,ctx,graph,time,mask_cn=False,return_features=False):
        self.node_condition(ctx,time)
        result,hidden=self.backbone(ctx.x,graph.attrs,graph.edges,
            ctx.x.new_tensor([[float(time)]]),ctx.batch,return_hidden=True,
            node_residual=ctx.node_residual)
        index=graph.query_index;family=graph.query_families;query=graph.edges[:,index]
        left,right=hidden["node"][query[0]],hidden["node"][query[1]]
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
        pairs=self.endpoint_roles(ctx.condition_diagnostic["role_context"],query)
        logits=self.readout(features,pairs)
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite late fusion")
        return (logits,features) if return_features else logits
