"""Frozen CE-external PB + one-count-shifted typed triangle mean deviance."""
import json
import numpy as np
import torch
from pb_degree import RelationDegreeCDF
from triangle_risk import CoupledTriangle
from shifted_count_deviance import evaluate, reverse_log_probability, VERSION as SCORE_VERSION
from .science import MarginalKernel, common

VERSION='PB_plus_typed_triangle_mean_one_count_v1'
PROPOSED_WEIGHT=1.0
STRUCTURE_WEIGHT=1.0

class Objective:
    def __init__(self,case,cfg,dataset):
        args=(case['space'].types,case['space'].pairs,case['prior'],case['clean'])
        kw=dict(data_sha256=cfg['datasets'][dataset]['sha256'])
        self.pb=RelationDegreeCDF(*args,**kw)
        self.triangle=CoupledTriangle(*args,**kw)
        self.kernel=MarginalKernel(case['prior'],common.schedule(1000))
        self.last={};self.device_groups={}

    def objective_binding(self):
        return dict(version=VERSION,pb=self.pb.objective_binding(),structure_weight=STRUCTURE_WEIGHT,
            score_version=SCORE_VERSION,count_shift=1.,T=1000,R=1,
            formula='CE + PB + mean_c[mu-m+(m+1)*log((m+1)/(mu+1))]',
            mean='independent one-step reverse edge marginal triangle expectation',
            teacher='same frozen Xt,t,X0 and unchanged reverse kernel',
            input='float64 logit gap; no probability inverse; fixed support enforced',
            averaging='nonempty declared typed triangle groups; impossible-at-time groups zero')

    def assert_binding(self,binding):
        if json.dumps(binding,sort_keys=True)!=json.dumps(self.objective_binding(),sort_keys=True):
            raise ValueError('full shifted triangle objective binding mismatch')

    def __call__(self,p,query,families,inp,*,logits):
        pb,detail=self.pb(p,query,families)
        owner=self.triangle;query=np.asarray(query);families=np.asarray(families)
        keys=np.minimum(*query)*owner.n+np.maximum(*query);order=np.argsort(keys)
        if not np.array_equal(keys[order],owner.query_keys):raise ValueError('full mutable triangle query support')
        if not np.array_equal(families[order],owner.f[owner.mutable]):raise ValueError('ordered relation mapping')
        if logits.shape!=(len(keys),2):raise ValueError('two-state logits required')
        context=owner.context(inp['edges'],int(inp['t']),self.kernel)
        device=logits.device
        a=torch.as_tensor(context.a,device=device);delta=torch.as_tensor(context.b-context.a,device=device)
        k=torch.as_tensor(context.k,device=device)
        gap=logits[:,1].double()-logits[:,0].double()
        full=torch.zeros(len(owner.y),device=device,dtype=torch.float64).index_copy(0,
            torch.as_tensor(np.flatnonzero(owner.mutable),device=device),gap[torch.as_tensor(order,device=device)])
        if str(device) not in self.device_groups:
            self.device_groups[str(device)]=[dict(types=x['types'],triangles=torch.as_tensor(x['triangles'],device=device)) for x in owner.groups]
        loss,parts=evaluate(reverse_log_probability(full,a,delta),k,self.device_groups[str(device)])
        self.structure_weighted=STRUCTURE_WEIGHT*loss
        self.last=dict(PB_loss=float(pb.detach()),shifted_triangle_mean_deviance=float(loss.detach()),
            weighted_shifted_triangle_mean_deviance=float(self.structure_weighted.detach()),
            count_shift=1.,structure_weight=STRUCTURE_WEIGHT,frozen_Xt_sha256=context.state_sha256,
            typed_triangle_mean_deviance=parts)
        return pb+self.structure_weighted,detail

def make_objective(case,cfg,dataset):return Objective(case,cfg,dataset)
