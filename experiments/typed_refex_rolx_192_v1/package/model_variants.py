"""A: original CN; B: typed ReFeX, no CN; C: B plus frozen-basis coefficients."""
import time
import numpy as np
import torch
from torch import nn
from hsgd.model import TypedSparseDenoiser
from features import extract, encode, digest

class DegreeBranch(nn.Module):
    """Only E07 BASE's active degree branch; no anonymous-role parameters."""
    def __init__(self,R,K,width):
        super().__init__()
        self.degree_encoder=nn.Sequential(nn.Linear(2*R+K+1,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.degree_out=nn.Linear(32,width)
        nn.init.zeros_(self.degree_out.weight);nn.init.zeros_(self.degree_out.bias)
    def input(self,ctx,t):
        sizes=np.bincount(ctx.types,minlength=ctx.model.n_types);cap=np.zeros((len(sizes),ctx.model.n_families))
        for r,(a,b) in enumerate(ctx.pairs):cap[a,r]=sizes[b]-int(a==b);cap[b,r]=sizes[a]-int(a==b)
        d=np.zeros((len(ctx.types),ctx.model.n_families))
        np.add.at(d,(ctx.old[0],ctx.of),1);np.add.at(d,(ctx.old[1],ctx.of),1)
        safe=np.maximum(cap[ctx.types],1)
        normalized=np.c_[d/safe,np.log1p(d)/np.log1p(safe)].astype(np.float32)
        return torch.as_tensor(np.c_[normalized,np.eye(ctx.model.n_types,dtype=np.float32)[ctx.types],
                    np.full((len(ctx.types),1),t,dtype=np.float32)],device=ctx.device)
    def forward(self,x):return self.degree_out(self.degree_encoder(x))

class StructuralDenoiser(TypedSparseDenoiser):
    architecture_version='typed_refex_rolx_192_v1'
    def __init__(self,case,cfg,arm,basis=None):
        if arm not in ('A','B','C'):raise ValueError('arm')
        args=dict(cfg['model']);args['cn_hidden']=32 if arm=='A' else 0
        super().__init__(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],**args)
        self.branch=DegreeBranch(case['n_families'],case['n_types'],cfg['model']['width'])
        self.arm=arm;self.feature_calls=0;self.feature_seconds=0.;self.encoding_seconds=0.;self.transfer_seconds=0.
        if arm!='A':
            # CN is last in the original feature layout: remove every CN input
            # and its learned readout columns, not merely mask a CN score.
            start=self.joint_slices['CN'].start
            first=self.joint_head[0];new=nn.Linear(start,first.out_features)
            with torch.no_grad():new.weight.copy_(first.weight[:,:start]);new.bias.copy_(first.bias)
            self.joint_head[0]=new;self.cn_feature_width=0;self.joint_slices['CN']=slice(start,start)
            self.refex_projection=nn.Linear(21*case['n_families'],cfg['model']['width'],bias=False)
            nn.init.zeros_(self.refex_projection.weight)
        if arm=='C':
            if basis is None:raise ValueError('frozen basis required')
            self.register_buffer('role_basis',torch.tensor(basis,dtype=torch.float64))
            self.role_projection=nn.Linear(len(basis),cfg['model']['width'],bias=False)
            nn.init.zeros_(self.role_projection.weight)
            self._basis_cpu=np.array(basis,copy=True);self._basis_cpu.flags.writeable=False
            self._basis_hash=digest(self._basis_cpu)
            self._basis_tensor_identity=(id(self.role_basis),self.role_basis._version)

    def joint_predict(self,ctx,graph,time,mask_cn=False,return_features=False):
        if not hasattr(ctx,'degree_input'):ctx.degree_input=self.branch.input(ctx,time)
        ctx.node_residual=self.branch(ctx.degree_input)
        if self.arm!='A':
            key=(ctx.state_version,float(time),self.arm)
            if hasattr(ctx,'structural_key') and ctx.structural_key!=key:raise ValueError('stale structural cache')
            if self.arm=='C':
                # Check both state_dict replacement/in-place writes and private CPU source.
                identity=(id(self.role_basis),self.role_basis._version)
                if identity!=self._basis_tensor_identity:raise ValueError('basis replaced; explicit checkpoint reload required')
                if hasattr(ctx,'basis_identity') and ctx.basis_identity!=identity:raise ValueError('basis changed')
                if digest(self._basis_cpu)!=self._basis_hash:raise ValueError('basis CPU changed')
                ctx.basis_identity=identity
            if not hasattr(ctx,'structural_features'):
                begin=__import__('time').perf_counter();f=extract(ctx.types,ctx.old,ctx.of,ctx.pairs)
                self.feature_seconds+=__import__('time').perf_counter()-begin;self.feature_calls+=1
                begin=__import__('time').perf_counter()
                g=None
                if self.arm=='C':g,ctx.encoding_receipt=encode(f,self._basis_cpu)
                self.encoding_seconds+=__import__('time').perf_counter()-begin
                begin=__import__('time').perf_counter()
                ctx.structural_features=torch.tensor(f,dtype=torch.float32,device=ctx.device)
                ctx.structural_roles=None if g is None else torch.tensor(g,dtype=torch.float32,device=ctx.device)
                if ctx.device.type=='cuda':torch.cuda.synchronize(ctx.device)
                self.transfer_seconds+=__import__('time').perf_counter()-begin
                ctx.structural_key=key
            # Cache only fixed input arrays, not learned outputs across optimizer steps.
            ctx.node_residual=ctx.node_residual+self.refex_projection(ctx.structural_features)
            if self.arm=='C':ctx.node_residual=ctx.node_residual+self.role_projection(ctx.structural_roles)
        return super().joint_predict(ctx,graph,time,mask_cn,return_features)

    def load_state_dict(self,state_dict,strict=True):
        result=super().load_state_dict(state_dict,strict=strict)
        if self.arm=='C':
            self._basis_cpu=self.role_basis.detach().cpu().numpy().copy();self._basis_cpu.flags.writeable=False
            self._basis_hash=digest(self._basis_cpu)
            self._basis_tensor_identity=(id(self.role_basis),self.role_basis._version)
        return result

    def _apply(self,fn):
        result=super()._apply(fn)
        if hasattr(self,'role_basis'):
            self._basis_tensor_identity=(id(self.role_basis),self.role_basis._version)
        return result

def copy_common(source,target):
    """Actually copy tensors; differing CN readout columns explicitly mapped."""
    src=source.state_dict();dst=target.state_dict();copied=[]
    for name,value in dst.items():
        if name in src and src[name].shape==value.shape:
            value.copy_(src[name]);copied.append(name)
        elif name=='joint_head.0.weight':
            value.copy_(src[name][:,:value.shape[1]]);copied.append(name+'[:,:shared]')
    target.load_state_dict(dst)
    for name in copied:
        if name.endswith('[:,:shared]'):
            assert torch.equal(target.joint_head[0].weight,src['joint_head.0.weight'][:,:target.joint_head[0].in_features])
        else:assert torch.equal(target.state_dict()[name],src[name])
    return copied
