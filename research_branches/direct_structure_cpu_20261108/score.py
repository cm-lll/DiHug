"""Frozen direct hard-graph loss; CPU only. No model, CE or optimizer changes."""
from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'research_branches/hard_structure_edge_direction_cpu_20261108'))
from metrics import Metrics

class DirectScore:
    def __init__(self,types,query,families,reference,bits,relations):
        self.metrics=Metrics(types,query,families,reference,bits,relations)
        self.degree_scale=max(1.,2*float(np.sum(reference))/len(types))
        self.triangle_scale=max(1.,float(self.metrics.ref_triangles))
        self.base=self.combine(dict(total_degree_W1=self.metrics.base_w1,
                                    joint_degree_TV=self.metrics.base_tv,
                                    triangle_abs_error=abs(self.metrics.triangles-self.metrics.ref_triangles)))
    def combine(self,values):
        return (values['total_degree_W1']/self.degree_scale+values['joint_degree_TV']+
                values['triangle_abs_error']/self.triangle_scale)/3
    def differences(self):
        delta=self.metrics.changes()
        return self.combine(delta),delta
    def logit_estimate(self,probabilities):
        p=np.asarray(probabilities,dtype=np.float64)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        change,delta=self.differences()
        # change is flip(B)-B, convert to L(B_-e,1)-L(B_-e,0).
        return p*(1-p)*change*delta['flip_sign']
