import itertools
import numpy as np
import torch
from experiments.structure_output_v1.role_loss import RoleDistributionLoss

torch.set_num_threads(1)

def objective(types,edges):
    nt=max(types)+1;pairs=list(itertools.combinations_with_replacement(range(nt),2))
    return RoleDistributionLoss(types,pairs,[.5]*len(pairs),np.asarray(edges,dtype=np.int64).reshape(-1,2).T,dtype=torch.float64)

def test_binary_counts_independent_enumeration():
    types=[0,0,1,1,2];slots=list(itertools.combinations(range(5),2));o=objective(types,[])
    for bits in range(1024):
        edges=[e for j,e in enumerate(slots) if bits&(1<<j)];P=torch.zeros(5,5,dtype=torch.float64)
        for u,v in edges:P[u,v]=P[v,u]=1
        counts=o.counts(P);expected=np.zeros(tuple(counts.shape))
        for u,v in edges:
            f=o.pairs.index(tuple(sorted((types[u],types[v]))));expected[u,f]+=1;expected[v,f]+=1
        for i,j,k in itertools.combinations(range(5),3):
            if P[i,j]*P[j,k]*P[k,i]:
                for c,a,b in ((i,j,k),(j,i,k),(k,i,j)):expected[c,o.nf+o.channels.index(tuple(sorted((types[a],types[b]))))]+=1
        assert np.array_equal(counts.numpy(),expected)

def test_probabilistic_triangle_expectation_and_gradient():
    o=objective([0,0,0],[]);q=np.array([[0,0,1],[1,2,2]]);p=torch.tensor([.2,.4,.7],dtype=torch.float64,requires_grad=True)
    c=o.counts(o.matrix(p,q));assert torch.allclose(c[:,1],torch.ones(3,dtype=torch.float64)*.056)
    grad=torch.autograd.grad(c[0,1],p)[0];assert torch.allclose(grad,torch.tensor([.28,.14,.08],dtype=torch.float64))
    brute=torch.zeros_like(c)
    for b in itertools.product([0,1],repeat=3):
        weight=np.prod([float(p[i]) if x else 1-float(p[i]) for i,x in enumerate(b)])
        brute+=weight*o.counts(o.matrix(torch.tensor(b,dtype=torch.float64),q))
    assert torch.allclose(c,brute)

def test_zero_at_reference_and_type_permutation():
    types=np.array([0,0,1,1,2]);edges=[(0,2),(0,3),(2,3),(1,4)];o=objective(types,edges)
    q=np.array(list(itertools.combinations(range(5),2))).T;p=torch.tensor([float(tuple(e) in edges) for e in q.T],dtype=torch.float64)
    assert float(o(p,q)[0])==0
    permutation=np.array([1,0,3,2,4]);P=o.matrix(p,q);reordered=P[permutation][:,permutation]
    assert float(o(reordered[q[0],q[1]],q)[0])<1e-25

def test_gradcheck_and_no_global_rng_use():
    before=np.random.get_state();tbefore=torch.get_rng_state().clone()
    o=objective([0,0,1,1],[(0,2),(1,3)]);q=np.array(list(itertools.combinations(range(4),2))).T
    p=torch.tensor([.13,.27,.39,.44,.62,.81],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:o(x,q)[0],(p,),eps=1e-6,atol=1e-5,rtol=1e-3)
    after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:];assert torch.equal(tbefore,torch.get_rng_state())

def test_fixed_support_and_zero_weight_ce():
    o=RoleDistributionLoss([0,0,1],[(0,0),(0,1),(1,1)],[0,.5,0],np.array([[0],[2]]),dtype=torch.float64)
    q=np.array([[0,1],[2,2]]);logits=torch.tensor([[.2,.5],[.7,.1]],dtype=torch.float64,requires_grad=True)
    p=logits.softmax(-1)[:,1];P=o.matrix(p,q);assert P[0,1]==0 and torch.count_nonzero(P.diag())==0
    ce=torch.nn.functional.cross_entropy(logits,torch.tensor([1,0]));aux=o(p,q)[0]
    a=torch.autograd.grad(ce+0*aux,logits,retain_graph=True)[0];b=torch.autograd.grad(ce,logits)[0];assert torch.equal(a,b)
