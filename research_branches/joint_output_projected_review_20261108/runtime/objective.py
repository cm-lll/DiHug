"""Single frozen choice per task: historical PB or replacing full joint kernel."""
import json
import torch
from pb_degree import RelationDegreeCDF
from joint_kernel import JointKernel
from projected_degree import ProjectedDegreeCDF
VERSION='registered_identity_pair_rotation_CDF_v1'
PROPOSED_WEIGHT=1.0
class Objective:
    def __init__(self,case,cfg,dataset,kind):
        if kind!='PS':raise ValueError('unregistered objective')
        self.kind=kind
        cls=ProjectedDegreeCDF
        self.pb=cls(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=cfg['datasets'][dataset]['sha256'])
        self.last={}
    def objective_binding(self):
        return dict(version=VERSION,kind=self.kind,target=self.pb.objective_binding(),weight=1.,CE='unchanged sumCE/S',R=1,T=1000,extra_PB=False)
    def assert_binding(self,b):
        if json.dumps(b,sort_keys=True)!=json.dumps(self.objective_binding(),sort_keys=True):raise ValueError('objective identity')
    def weights(self,query,families,inp,*,device,dtype):
        self.last=dict(weight_min=1.,weight_max=1.,weight_mean_error=0.,objective_kind=self.kind)
        return torch.ones(len(families),device=device,dtype=dtype)
    def __call__(self,p,query,families,inp,*,logits):
        aux,detail=self.pb(p,query,families)
        self.last['objective_value']=float(aux.detach())
        return aux,detail
def make_objective(case,cfg,dataset,kind):return Objective(case,cfg,dataset,kind)
