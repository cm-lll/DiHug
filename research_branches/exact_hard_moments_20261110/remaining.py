"""Unchanged sampled joint-degree/component costs, omitting unused triangle work."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def remaining_costs(core,bits):
    result=[]
    for b in bits:
        active=np.asarray(b,dtype=bool);i,j=core.q[:,active];f=core.f[active]
        degree=np.zeros_like(core.ref['degree']);np.add.at(degree,(i,f),1);np.add.at(degree,(j,f),1)
        joint=[]
        for ix in core.nodes:
            cost=abs(degree[ix,None,:]-core.ref['degree'][ix][None,:,:]).sum(-1);r,c=linear_sum_assignment(cost)
            joint.append(float(cost[r,c].mean())/max(1,core.n-1))
        graph=csr_matrix((np.ones(len(i)*2,dtype=np.uint8),(np.r_[i,j],np.r_[j,i])),shape=(core.n,core.n))
        nc=connected_components(graph,directed=False,return_labels=False)
        result.append([np.mean(joint),((nc-core.ref['components'])/max(1,core.n-1))**2])
    return np.asarray(result,dtype=np.float64)
