"""Same complete model, target contribution enabled/disabled. CPU-only Newton."""
import time
import numpy as np
import torch
from capacity_readout import CapacityDenoiser
import capacity_readout
import newton_solver
from role_condition.model import capacities,degrees,features

class Reader:
    def __init__(self,check=lambda:None,failure_folder=None,max_history=None,verify_context=False):
        self.check=check;self.rows=[];self.pending=[];self.failure_folder=failure_folder;self.max_history=max_history
        self.total_calls=0;self.total_CPU_seconds=0.;self.maxima={'row':0.,'column':0.,'context_tail':0.};self.contract_counts={}
        self.verify_context=verify_context;self._context_parts=[]
    def __call__(self,L,counts):
        start=time.perf_counter();P,LP,r=newton_solver.solve(L,counts,self.check)
        if not r.get('new_contract_pass'):
            if self.failure_folder is not None:
                import json
                from pathlib import Path
                folder=Path(self.failure_folder);folder.mkdir(parents=True,exist_ok=True);name='read_'+str(time.time_ns())
                np.savez_compressed(folder/(name+'.npz'),L=L,counts=counts)
                (folder/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
            raise newton_solver.ReadoutFailure(str(r))
        row=float(abs(P.sum(1)-1).max());col=float((abs(P.sum(0)-counts)/np.maximum(1,counts)).max())
        if row>1e-8 or col>1e-8:raise FloatingPointError('capacity residual')
        from scipy.special import logsumexp
        if abs(logsumexp(LP,axis=1)).max()>2e-12:raise FloatingPointError('log row normalization')
        finite=np.isfinite(LP);rounded=finite&(P==0)
        if np.any(LP[rounded]>=-700) or np.any(P[~finite]!=0):raise FloatingPointError('support/rounding')
        np.testing.assert_array_equal(P[finite&~rounded],np.exp(LP[finite&~rounded]))
        if finite.all():
            D=LP-L
            if abs(D-D[:,:1]-D[:1,:]+D[0,0]).max()>2e-10:raise FloatingPointError('KKT')
        mass=rounded.sum(1)*np.exp(-700.)
        if mass.max(initial=0)>1e-280 or mass.sum()>1e-276:raise FloatingPointError('tail mass')
        r=dict(r,independent_row=row,independent_column=col,tail_row_mass=mass.tolist(),CPU_wall_seconds=time.perf_counter()-start)
        self.total_calls+=1;self.total_CPU_seconds+=r['CPU_wall_seconds']
        self.maxima['row']=max(self.maxima['row'],row);self.maxima['column']=max(self.maxima['column'],col)
        self.contract_counts[r['status']]=self.contract_counts.get(r['status'],0)+1
        self.rows.append(r);self.pending.append((r,P) if self.verify_context else (r,None))
        if self.max_history is not None:self.rows=self.rows[-self.max_history:]
        return P,r
    def install(self,model):
        self.previous=capacity_readout.allocation_marginals;capacity_readout.allocation_marginals=self
        def value(module,args,output):
            r,P=self.pending.pop(0);v=output.detach().cpu().numpy()
            # Small role-value copy audits the declared context error; never
            # moves the full node/edge network to CPU.
            b=float((np.asarray(r['tail_row_mass'])*abs(v).max()).max(initial=0))
            r['context_tail_bound']=b
            if not np.isfinite(v).all() or b>1e-250:raise FloatingPointError('role context tail')
            self.maxima['context_tail']=max(self.maxima['context_tail'],b)
            if self.verify_context:self._context_parts.append((r,P,v.copy()))
        self.h=model.branch.value.register_forward_hook(value)
        def context_check(module,args,output):
            ctx=args[0];observed=output[1]['role_context'].detach().cpu().numpy()
            if len(self._context_parts)!=model.n_types:raise AssertionError('role context coverage')
            for typ,(r,P,v) in enumerate(self._context_parts):
                f32=P.astype(np.float32).astype(float);k=P.shape[1];gamma=k*2**-24/(1-k*2**-24)
                conversion=(abs(f32-P)@abs(v.astype(float))).max(1)
                dot=(gamma*(abs(f32)@abs(v.astype(float)))).max(1)+k*np.finfo(np.float32).tiny
                err=abs(observed[ctx.types==typ]-P@v.astype(float)).max(1)
                if not np.all(err<=conversion+dot+1e-12):raise FloatingPointError('cross-device role context arithmetic bound')
                r.update(context_FP32_conversion_bound=float(conversion.max()),context_FP32_dot_bound=float(dot.max()),context_observed_error=float(err.max()))
            self._context_parts=[]
        self.context_hook=model.branch.register_forward_hook(context_check) if self.verify_context and model.arm=='role' else None
        return self
    def close(self):
        self.h.remove()
        if self.context_hook is not None:self.context_hook.remove()
        capacity_readout.allocation_marginals=self.previous
    def summary(self):return {'calls':self.total_calls,'CPU_seconds':self.total_CPU_seconds,'maxima':dict(self.maxima),'contracts':dict(self.contract_counts),'retained_receipts':len(self.rows)}

class PairDenoiser(CapacityDenoiser):
    architecture_version='imdb_role_pair_same_architecture_v1'
    def __init__(self,*args,arm,**kwargs):
        if arm not in ('base','role'):raise ValueError('base/role only')
        super().__init__(*args,**kwargs);self.arm=arm
    def set_target(self,target):
        if self.arm=='base' and target is not None:raise ValueError('base may not consume target')
        return super().set_target(target)
    def node_condition(self,ctx,t):
        if self.arm=='role':return super().node_condition(ctx,t)
        ctx.check(ctx.state_version)
        key=(float(t),None,self.training,torch.is_grad_enabled())
        if hasattr(ctx,'condition_key'):
            if ctx.condition_key!=key:raise ValueError('base context mode/time changed')
            return ctx.node_residual
        types=ctx.types;cap=capacities(types,self.family_pairs)
        d=degrees(types,ctx.old,ctx.of,self.n_families)
        x=torch.as_tensor(np.c_[features(d,cap[types]),np.eye(self.n_types,dtype=np.float32)[types],
            np.full((len(types),1),t,dtype=np.float32)],device=ctx.device)
        delta=self.branch.degree_out(self.branch.degree_encoder(x))
        ctx.condition_key=key;ctx.node_residual=delta
        ctx.condition_diagnostic={'degree':d,'role_context':x.new_zeros((len(types),32))}
        return delta

def create(case,cfg,seed,arm,device):
    # CPU seed affects initialization only; never seeds every visible GPU.
    import random
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
    m=PairDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,
        prior=case['prior'],T=cfg['T'],arm=arm,**cfg['model'])
    return m.to(device)

def paired_initial(case,cfg,seed):
    a=create(case,cfg,seed,'base','cpu');b=create(case,cfg,seed+1,'role','cpu')
    b.load_state_dict({k:v.clone() for k,v in a.state_dict().items()},strict=True)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
    return a,b
