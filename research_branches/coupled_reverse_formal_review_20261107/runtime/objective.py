"""One frozen objective: CE outside + PB + calibrated coupled triangle risk."""
import json
import torch
from pb_degree import RelationDegreeCDF
from triangle_risk import CoupledTriangle
from sensitivity_scale import evaluate, VERSION as SCALE_VERSION
from .science import MarginalKernel,common
VERSION='PB_plus_coupled_reverse_sensitivity_v1'
PROPOSED_WEIGHT=1.0
RISK_WEIGHT=0.013715673231930771

class Objective:
    def __init__(self,case,cfg,dataset):
        args=(case['space'].types,case['space'].pairs,case['prior'],case['clean'])
        kw=dict(data_sha256=cfg['datasets'][dataset]['sha256'])
        self.pb=RelationDegreeCDF(*args,**kw)
        self.triangle=CoupledTriangle(*args,**kw)
        self.kernel=MarginalKernel(case['prior'],common.schedule(1000))
        self.last={}
    def objective_binding(self):
        return dict(version=VERSION,pb=self.pb.objective_binding(),risk_weight=RISK_WEIGHT,
                    scale_version=SCALE_VERSION,denominator_floor=1.,T=1000,R=1,
                    formula='CE + PB + lambda * sum_c E[(T_c(Bq)-T_c(Bk))^2] / max(A,1)',
                    A='sum_e,c (K1-K0)_e E_k[C_ec^2]')
    def assert_binding(self,binding):
        if json.dumps(binding,sort_keys=True)!=json.dumps(self.objective_binding(),sort_keys=True):
            raise ValueError('full coupled objective binding mismatch')
    def __call__(self,p,query,families,inp):
        pb,detail=self.pb(p,query,families)
        ctx=self.triangle.context(inp['edges'],int(inp['t']),self.kernel)
        risk,parts,scale=evaluate(ctx,p,query,families,accumulation_dtype=torch.float64)
        self.risk_weighted=RISK_WEIGHT*risk
        self.last=dict(PB_loss=float(pb.detach()),triangle_risk_normalized=float(risk.detach()),
            weighted_triangle_risk=float(self.risk_weighted.detach()),
            sensitivity_denominator=scale['denominator'],raw_sensitivity=scale['raw_sensitivity'],
            sensitivity_floor_applied=scale['floor_applied'],frozen_Xt_sha256=ctx.state_sha256,
            typed_triangle_risk=[dict(types=list(x['types']),risk=float(x['risk'].detach())) for x in parts])
        return pb+self.risk_weighted,detail

def make_objective(case,cfg,dataset):
    return Objective(case,cfg,dataset)

