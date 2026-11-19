import copy, itertools, json, unittest
import numpy as np
import torch
from pb_degree import RelationDegreeCDF, combine_normal1
torch.set_num_threads(1)

def small():
    types=np.array([0,0,1,1]);schema=np.array([[0,0],[0,1],[1,1]])
    edges=np.array([[0,0,2],[1,2,3]])
    obj=RelationDegreeCDF(types,schema,[1,.25,1],edges,data_sha256='0'*64)
    q=np.array([[0,0,1,1],[2,3,2,3]]);f=np.ones(4,dtype=int)
    return obj,q,f,types,schema,edges

class Tests(unittest.TestCase):
    def test_enumeration(self):
        obj,q,f,*_=small();p=np.array([.2,.4,.6,.8]);H=np.zeros((4,3))
        for bits in itertools.product([0,1],repeat=4):
            b=np.array(bits);mass=np.prod(np.where(b,p,1-p));d=np.zeros(4,dtype=int)
            np.add.at(d,q[0],b);np.add.at(d,q[1],b)
            H[np.arange(4),d]+=mass
        per=[]
        for i,(lo,hi) in enumerate(obj.bounds):
            nodeidx=np.flatnonzero(obj.types==obj.groups[i]['node_type'])
            c=H[nodeidx].mean(0).cumsum()[:2];per.append(.5*np.mean((c-obj.target_cdf[i,:2])**2))
        loss,gg=obj(torch.tensor(p),q,f)
        np.testing.assert_allclose(gg.detach().numpy(),per,atol=1e-14)
        self.assertAlmostEqual(loss.item(),np.mean(per),14)

    def test_gradcheck(self):
        obj,q,f,*_=small();x=torch.tensor([-.7,.3,1.1,-.9],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:obj(x.sigmoid(),q,f)[0],(x,),eps=1e-6,atol=1e-7,rtol=1e-5))

    def test_query_shuffle_orientation(self):
        obj,q,f,*_=small();p=torch.tensor([.2,.4,.6,.8],requires_grad=True,dtype=torch.float64);a=obj(p,q,f)[0];ga=torch.autograd.grad(a,p)[0]
        perm=np.array([2,0,3,1]);b=obj(p[perm],q[::-1,perm],f[perm])[0];gb=torch.autograd.grad(b,p)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ga,gb,rtol=0,atol=0)

    def test_support_rejections(self):
        obj,q,f,*_=small();p=torch.ones(4)*.5
        for qq,ff,pp in [(q[:,:3],f[:3],p[:3]),(q[:,[0,0,2,3]],f,p),(q,np.zeros(4,dtype=int),p)]:
            with self.assertRaises(ValueError):obj(pp,qq,ff)
        for v in [float('nan'),-0.01,1.01]:
            bad=p.clone();bad[0]=v
            with self.assertRaises(ValueError):obj(bad,q,f)

    def test_target_zero_boundaries(self):
        obj,q,f,*_=small();p=torch.tensor([1.,0.,0.,0.],dtype=torch.float64,requires_grad=True)
        loss=obj(p,q,f)[0];self.assertEqual(loss.item(),0);g=torch.autograd.grad(loss,p)[0];self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        for vals in [[0.,0.,0.,0.],[1.,1.,1.,1.]]:
            p=torch.tensor(vals,dtype=torch.float64,requires_grad=True);v=obj(p,q,f)[0];self.assertTrue(torch.isfinite(torch.autograd.grad(v,p)[0]).all())

    def test_no_mutable_support(self):
        obj=RelationDegreeCDF(np.array([0,0]),[[0,0]],[0],np.empty((2,0),dtype=int),data_sha256='0'*64)
        p=torch.empty(0,requires_grad=True);loss,_=obj(p,np.empty((2,0),dtype=int),np.empty(0,dtype=int));loss.backward();self.assertEqual(loss.item(),0)

    def test_identity_and_restore(self):
        obj,q,f,types,schema,edges=small();binding=json.loads(json.dumps(obj.objective_binding()));fresh=RelationDegreeCDF(types,schema,[1,.25,1],edges,data_sha256='0'*64);fresh.assert_binding(binding)
        x=torch.tensor([.2,.4,.6,.8],requires_grad=True,dtype=torch.float64);state=torch.get_rng_state().clone()
        a=obj(x,q,f)[0];b=fresh(x,q,f)[0];torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));wrong=copy.deepcopy(binding);wrong['proposed_weight']=2
        with self.assertRaises(ValueError):fresh.assert_binding(wrong)
        wrong=copy.deepcopy(binding);wrong['ordered_schema']=[[1,1],[0,1],[0,0]]
        with self.assertRaises(ValueError):fresh.assert_binding(wrong)
        wrong=RelationDegreeCDF(types,schema,[1,.25,1],edges,data_sha256='1'*64)
        with self.assertRaises(ValueError):wrong.assert_binding(binding)

    def test_relabel_invariance(self):
        obj,q,f,types,schema,edges=small();mapping=np.array([1,0,3,2]);other=RelationDegreeCDF(types,schema,[1,.25,1],mapping[edges],data_sha256='2'*64)
        x=torch.tensor([.2,.4,.6,.8],requires_grad=True,dtype=torch.float64);a=obj(x,q,f)[0];b=other(x,mapping[q],f)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=1e-16);torch.testing.assert_close(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,x)[0],rtol=0,atol=1e-16)

    def test_ce_full_weight(self):
        obj,q,f,*_=small();x=torch.tensor([.2,.4,.6,.8],dtype=torch.float64,requires_grad=True);ce=torch.tensor(3.,requires_grad=True)
        total,detail=combine_normal1(ce,x,q,f,obj,schema_slots=6)
        self.assertEqual(detail['unweighted_CE'].item(),.5);self.assertAlmostEqual(torch.autograd.grad(total,ce)[0].item(),1/6,7)
        with self.assertRaises(ValueError):combine_normal1(ce,x,q,f,obj,schema_slots=4)

if __name__=='__main__':unittest.main()
