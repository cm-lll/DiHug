"""No auxiliary objective: same-environment ordinary CE control."""
import json
import torch
from pb_degree import RelationDegreeCDF
from joint_kernel import JointKernel
VERSION='same_environment_unweighted_CE_control_v1'
PROPOSED_WEIGHT=0.0
class Objective:
    def __init__(self,case,cfg,dataset,kind):
        if kind!='CE':raise ValueError('only registered pure CE controls')
        self.kind=kind
        self.binding=dict(version=VERSION,kind=kind,weight=0.,CE='unchanged sumCE/S',R=1,T=1000,
                          data_sha256=cfg['datasets'][dataset]['sha256'])
        self.last={}
    def objective_binding(self):
        return self.binding
    def assert_binding(self,b):
        if json.dumps(b,sort_keys=True)!=json.dumps(self.objective_binding(),sort_keys=True):raise ValueError('objective identity')
    def weights(self,query,families,inp,*,device,dtype):
        self.last=dict(weight_min=1.,weight_max=1.,weight_mean_error=0.,objective_kind=self.kind)
        return torch.ones(len(families),device=device,dtype=dtype)
    def __call__(self,p,query,families,inp,*,logits):
        raise RuntimeError('CE control delegates to original production update; no auxiliary call')
def make_objective(case,cfg,dataset,kind):return Objective(case,cfg,dataset,kind)
