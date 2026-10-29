"""Single early CN injection; topology always comes from the frozen real Xt.

Metadata canonicalization is not learned. Learned CN and the projection retain
autograd. No persistent cache or fallback is introduced here.
"""
import numpy as np
import torch
from torch import nn

def canonical_pairs(edges,families,n):
    e=edges.detach().cpu().numpy();f=families.detach().cpu().numpy()
    if e.shape!=(2,len(f)):raise ValueError('edge/family rows')
    if e.size and ((e<0).any() or (e>=n).any() or (e[0]==e[1]).any()):raise ValueError('invalid endpoints')
    lo=np.minimum(e[0],e[1]);hi=np.maximum(e[0],e[1])
    keys,first,inverse=np.unique(lo*n+hi,return_index=True,return_inverse=True)
    if not np.array_equal(f,f[first][inverse]):raise ValueError('same pair has conflicting families')
    pairs=np.stack([keys//n,keys%n])
    return (torch.as_tensor(pairs,dtype=torch.long,device=edges.device),
            torch.as_tensor(f[first],dtype=torch.long,device=edges.device),
            torch.as_tensor(inverse,dtype=torch.long,device=edges.device))

def install(model):
    if model.arm!='E' or model.common_neighbor is None:raise ValueError('E only')
    first=model.joint_head[0];s=model.joint_slices['CN']
    if (first.in_features,first.out_features,s.start,s.stop,model.cn_feature_width)!=(776,128,739,776,37):
        raise ValueError('reviewed head layout differs')
    new=nn.Linear(739,128,device=first.weight.device,dtype=first.weight.dtype)
    with torch.no_grad():new.weight.copy_(first.weight[:,:739]);new.bias.copy_(first.bias)
    model.joint_head[0]=new
    model.cn_edge_projection=nn.Linear(37,64,bias=False,device=first.weight.device,dtype=first.weight.dtype)
    nn.init.zeros_(model.cn_edge_projection.weight)
    model.cn_feature_width=0;model.joint_slices['CN']=slice(739,739)
    model.architecture_version='E_earlyCN_v1'
    return model

def edge_cn_residual(model,ctx,graph,time,mask_cn=False):
    pairs,families,inverse=canonical_pairs(graph.edges,graph.families,len(ctx.types))
    if mask_cn or pairs.shape[1]==0:return ctx.x.new_zeros((graph.edges.shape[1],64))
    # contextual_nodes inside aggregate calls lin_in_E + shared first layer
    # on ctx.directed REAL edges, never the main path or this residual.
    cn=model.common_neighbor.aggregate(ctx,pairs,families,time)
    if cn.shape!=(pairs.shape[1],37):raise ValueError('CN feature layout')
    return model.cn_edge_projection(cn)[inverse]
