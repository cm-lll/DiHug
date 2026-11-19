import importlib.util
import itertools
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from triangle_risk import triangle_risk, CoupledTriangle

ROOT = Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
math = load('prior_math',ROOT/'results/coupled_reverse_structure_theory_20261107/check_math.py')
kernel = load('prior_kernel',ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/kernel.py')
torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def test_exhaustive_anova(self):
        rng=np.random.default_rng(2026110704)
        for rep in range(16):
            q,k=rng.uniform(.01,.99,(2,6))
            if rep%4==0:q[0]=k[0]=0.;q[1]=k[1]=1.
            if rep%4==1:q=k.copy()
            if rep%4==2:k=(k>.5).astype(float)
            w,bq,bk=math.enumerate_pair(q,k)
            for mask in [np.ones(4,bool)]+math.MASKS:
                tri=math.TRI[mask]
                exact=w@((bq[:,tri].all(2).sum(1)-bk[:,tri].all(2).sum(1))**2)
                val,parts=triangle_risk(torch.tensor(q),torch.tensor(k),torch.tensor(tri))
                self.assertAlmostEqual(float(val),float(exact),places=11)
                self.assertTrue(all(float(parts[x])>=0 for x in ['bias_squared','order1','order2','order3']))
    def test_gradcheck(self):
        q=torch.tensor([.1,.7,.5,.2,.8,.3],dtype=torch.float64,requires_grad=True)
        k=torch.tensor([.4,.6,.2,.9,.3,.5],dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda q:triangle_risk(q,k,torch.tensor(math.TRI))[0],(q,)))
    def test_zero_cusp(self):
        q=torch.tensor([0.,.7,.3,.2,1.,.5],dtype=torch.float64,requires_grad=True)
        loss,_=triangle_risk(q,q.detach(),torch.tensor(math.TRI))
        self.assertEqual(float(loss),0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss,q)[0],torch.zeros_like(q)))
    def test_near_equality(self):
        k=np.array([.2,.4,.5,.7,.8,.3])
        for delta in [1e-4,1e-8,1e-12]:
            q=k+delta*np.array([1,-1,1,-1,1,-1]);w,bq,bk=math.enumerate_pair(q,k)
            ex=w@((bq[:,math.TRI].all(2).sum(1)-bk[:,math.TRI].all(2).sum(1))**2)
            v,_=triangle_risk(torch.tensor(q),torch.tensor(k),torch.tensor(math.TRI))
            self.assertLess(abs(float(v)-ex),max(1e-25,abs(ex)*1e-10))
    def test_t1_reference_and_permutation(self):
        types=np.array([0,0,1,1]);schema=[(0,0),(0,1),(1,1)]
        clean=np.array([[0,0,1],[1,2,2]]);prior=np.array([1.,.5,0.])
        obj=CoupledTriangle(types,schema,prior,clean,data_sha256='a'*64)
        ker=kernel.MarginalKernel(prior,np.array([1.,.9,.5,0.]))
        ctx=obj.context(clean,1,ker)
        query=np.array([[0,0,1,1],[2,3,2,3]]);fam=np.ones(4,int)
        p=torch.tensor([.7,.2,.8,.1],dtype=torch.float64,requires_grad=True)
        val,rows=ctx.evaluate(p,query,fam)
        perm=np.array([2,0,3,1]);v2,_=ctx.evaluate(p[perm],query[:,perm],fam[perm])
        self.assertEqual(float(val),float(v2))
        y=torch.tensor([1.,0.,1.,0.],dtype=torch.float64,requires_grad=True)
        zero,_=ctx.evaluate(y,query,fam);self.assertEqual(float(zero),0.)
        self.assertTrue(torch.equal(torch.autograd.grad(zero,y)[0],torch.zeros_like(y)))
        with self.assertRaises(ValueError):ctx.evaluate(p[:-1],query[:,:-1],fam[:-1])
        with self.assertRaises(ValueError):ctx.evaluate(p,query,np.zeros(4,int))
        with self.assertRaises(ValueError):obj.context(np.array([[0,0,1,2],[1,2,2,3]]),1,ker)
    def test_node_relabel(self):
        q=torch.tensor([.1,.7,.5,.2,.8,.3],dtype=torch.float64)
        k=torch.tensor([.4,.6,.2,.9,.3,.5],dtype=torch.float64)
        perm=np.array([2,0,3,1]);eid={e:i for i,e in enumerate(math.EDGES)}
        order=np.array([eid[tuple(sorted((perm[u],perm[v])))] for u,v in math.EDGES])
        self.assertAlmostEqual(float(triangle_risk(q,k,torch.tensor(math.TRI))[0]),
                               float(triangle_risk(q[order],k[order],torch.tensor(math.TRI))[0]),places=13)
    def test_empty_group(self):
        q=torch.tensor([.2],requires_grad=True)
        v,_=triangle_risk(q,q.detach(),torch.empty((0,3),dtype=torch.long))
        self.assertEqual(float(v),0.);self.assertEqual(float(torch.autograd.grad(v,q)[0]),0.)

if __name__=='__main__':unittest.main(verbosity=2)
