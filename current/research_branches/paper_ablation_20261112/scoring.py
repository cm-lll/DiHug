"""Two frozen reward definitions. Generic is a type-blind ablation, NOT official GDPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as s
base,w,np,torch=s.base,s.w,s.np,s.torch
from heat_solver_candidate import heat
base.heat.heat=heat
TypedScore=base.Score

class GenericScore:
    """Same joint-kernel construction without node/relation labels; reference-only scales.

    One pooled node population: nine return channels, constant untyped arrival,
    total log degree; global count penalty. No type-specific reward feedback.
    The MODEL and LEGAL SUPPORT remain heterogeneous and unchanged.
    """
    def __init__(self,ds,case,query,fam,labels):
        self.q=query;self.n=len(case['space'].types);self.types=np.zeros(self.n,dtype=int)
        self.target=float(labels.sum())
        a=base.heat.adj(self.n,query[:,labels.astype(bool)].T)
        self.scale=float(np.log1p(max(1.,a.sum(1).mean())))
        self.h,self.d=self.features(labels)
        self.hb=base.joint.median_bandwidth(self.h);self.db=base.joint.median_bandwidth(self.d)
        self.s=dict(log_total_degree_scale=self.scale,heat_bandwidth_squared=self.hb,
                    degree_bandwidth_squared=self.db,definition='type-blind joint heat/degree plus total edge count')
    def features(self,b):
        a=base.heat.adj(self.n,self.q[:,b.astype(bool)].T)
        return heat(a,self.types)['feat'],np.log1p(a.sum(1))[:,None]/self.scale
    def __call__(self,b):
        h,d=self.features(b)
        def k(ha,da,hb,db):
            return base.joint.kernel(ha,hb,self.hb)*base.joint.kernel(da,db,self.db)
        v=base.joint.mmd(k(h,d,h,d),k(self.h,self.d,self.h,self.d),k(h,d,self.h,self.d))
        count=float((float(b.sum())-self.target)**2/max(1.,self.target)**2)
        return float(v),count,[float(v)]

def case_from(path):
    z=np.load(path);edges=z['edges'];edges=edges.T if edges.shape[0]!=2 else edges
    return w.common.make_case(z['types'],edges,tuple(map(tuple,z['schema'])))
