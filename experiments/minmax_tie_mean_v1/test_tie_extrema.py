"""Small CPU regressions. No optimizer updates or dictionary fitting."""
import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'package'))
import torch
from unittest.mock import patch
from hsgd.vendor.tie_extrema import equal_tie_max_pool as pool
from hsgd.vendor.layers import SparseXtoy,SparseEtoy

torch.set_num_threads(1)
class TieRule(unittest.TestCase):
    def test_negative_empty_and_zero_fill_not_tied(self):
        x=torch.tensor([[-2.,0.],[-2.,-1.],[-4.,0.]],requires_grad=True);b=torch.tensor([0,0,2])
        y=pool(x,b,4);y.sum().backward()
        self.assertTrue(torch.equal(y,torch.tensor([[-2.,0.],[0.,0.],[-4.,0.],[0.,0.]])))
        self.assertTrue(torch.equal(x.grad,torch.tensor([[.5,1.],[.5,0.],[1.,1.]])))
    def test_min_sign(self):
        x=torch.tensor([[1.],[1.],[3.]],requires_grad=True);y=-pool(-x,torch.zeros(3,dtype=torch.long));y.backward(torch.tensor([[-.6]]))
        self.assertTrue(torch.equal(x.grad,torch.tensor([[-.3],[-.3],[0.]])))
    def test_permutation_and_duplicate_pullback(self):
        a=torch.tensor([[2.,4.],[2.,1.],[0.,4.]],requires_grad=True);b=torch.zeros(3,dtype=torch.long);g=torch.tensor([[.8,-.5]])
        pool(a,b).backward(g);expected=a.grad.clone();perm=torch.tensor([2,0,1]);z=a.detach()[perm].requires_grad_();pool(z,b).backward(g)
        self.assertTrue(torch.equal(z.grad,expected[perm]))
        z=a.detach().requires_grad_();pool(z.repeat(2,1),b.repeat(2)).backward(g);self.assertTrue(torch.equal(z.grad,expected))
    def test_nonfinite_upstream_rejected(self):
        x=torch.ones((2,1),requires_grad=True)
        with self.assertRaises(FloatingPointError):pool(x,torch.zeros(2,dtype=torch.long)).backward(torch.tensor([[float('nan')]]))
    def test_shared_modules_route_all_extrema(self):
        import hsgd.vendor.layers as layers
        calls=[]
        def observe(*args,**kwargs):calls.append(args[0].shape);return pool(*args,**kwargs)
        with patch.object(layers,'equal_tie_max_pool',observe):
            SparseXtoy(2,2)(torch.ones(3,2),torch.zeros(3,dtype=torch.long))
            e=torch.tensor([[0,1],[1,2]])
            for triu in [False,True]:SparseEtoy(2,2)(e,torch.ones(2,2),torch.zeros(3,dtype=torch.long),triu,num_graphs=1)
        self.assertEqual(len(calls),6);self.assertEqual(calls[-1],torch.Size([4,2]))
    def test_smooth_double_gradcheck(self):
        x=torch.tensor([[.1,3.],[2.,-.6],[-2.,5.]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v:pool(v,torch.zeros(3,dtype=torch.long)),(x,)))

if __name__=='__main__':unittest.main()
