import itertools,unittest
import numpy as np
import torch
from joint_degree import JointRelationDegreeCDF,_NodeCDF
from pb_degree import RelationDegreeCDF
torch.set_num_threads(1)

def tiny(edges=None):
    types=np.array([0,0,0,1]);schema=[(0,0),(0,1),(1,1)]
    if edges is None:edges=np.array([[0,0],[1,3]])
    obj=JointRelationDegreeCDF(types,schema,[1/3,1/3,0],edges,data_sha256='0'*64)
    q=np.array([[0,0,0,1,1,2],[1,2,3,2,3,3]])
    f=np.array([0,0,1,0,1,1])
    return obj,q,f

class JointTests(unittest.TestCase):
    def test_enumeration(self):
        obj,q,f=tiny();p=np.array([.2,.3,.4,.5,.6,.7]);H=np.zeros((2,1))
        for bits in itertools.product([0,1],repeat=6):
            b=np.array(bits);mass=np.prod(np.where(b,p,1-p));d=np.zeros((4,3),int)
            for j in range(6):d[q[0,j],f[j]]+=b[j];d[q[1,j],f[j]]+=b[j]
            for k in range(2):H[k,0]+=mass*np.mean((d[:3,0]<=k)&(d[:3,1]<=0))
        expected=.5*np.mean((H-obj.joint_targets[0])**2)
        loss,details=obj(torch.tensor(p),q,f)
        self.assertAlmostEqual(details[-1].item(),expected,14)
        self.assertAlmostEqual(loss.item(),details[:3].mean().item()+expected,14)

    def test_gradcheck(self):
        obj,q,f=tiny();x=torch.linspace(-1.,1.,6,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:obj(x.sigmoid(),q,f)[0],(x,),eps=1e-6,atol=1e-7,rtol=1e-5))

    def test_captures_same_marginal_different_combination(self):
        obj,q,f=tiny()
        correct=torch.tensor([1.,0.,1.,0.,0.,0.],dtype=torch.float64)
        swapped=torch.tensor([1.,0.,0.,0.,0.,1.],dtype=torch.float64)
        self.assertEqual(obj(correct,q,f)[0].item(),0.)
        total,parts=obj(swapped,q,f)
        self.assertTrue(torch.equal(parts[:3],torch.zeros(3,dtype=torch.float64)))
        self.assertGreater(parts[-1].item(),0.)

    def test_hard_gradient_and_relabel(self):
        obj,q,f=tiny();p=torch.tensor([1.,0.,1.,0.,0.,0.],dtype=torch.float64,requires_grad=True)
        v=obj(p,q,f)[0];torch.testing.assert_close(torch.autograd.grad(v,p)[0],torch.zeros_like(p),atol=0,rtol=0)
        perm=np.array([2,0,1,3]);other=JointRelationDegreeCDF(obj.types,obj.schema,obj.prior,perm[np.array([[0,0],[1,3]])],data_sha256='1'*64)
        x=torch.linspace(.1,.9,6,dtype=torch.float64,requires_grad=True);a=obj(x,q,f)[0];b=other(x,perm[q],f)[0]
        torch.testing.assert_close(a,b,atol=1e-15,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,x)[0],atol=1e-15,rtol=0)

    def test_query_invariance_and_binding(self):
        obj,q,f=tiny();p=torch.linspace(.1,.9,6,dtype=torch.float64,requires_grad=True);order=[5,2,0,3,1,4]
        a=obj(p,q,f)[0];b=obj(p[order],q[::-1,order],f[order])[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(a,p)[0],torch.autograd.grad(b,p)[0],atol=0,rtol=0)
        fresh,_,_=tiny();fresh.assert_binding(obj.objective_binding())

    def test_marginal_exact_unchanged_and_rng(self):
        obj,q,f=tiny();base=RelationDegreeCDF(obj.types,obj.schema,obj.prior,np.array([[0,0],[1,3]]),data_sha256='0'*64)
        p=torch.linspace(.1,.9,6,dtype=torch.float64,requires_grad=True);rng=torch.get_rng_state().clone()
        _,a=obj(p,q,f);_,b=base(p,q,f)
        torch.testing.assert_close(a[:3],b,atol=0,rtol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
