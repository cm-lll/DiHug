import unittest
import numpy as np
from feedback import loo, count_terms, feedback_record, eager_loss

class FeedbackTests(unittest.TestCase):
    def test_loo_sign_and_additivity(self):
        h=np.array([.1,.3,.4,.2]); n=np.array([.3,.1,.2,.5])
        np.testing.assert_allclose(loo(h+n),loo(h)+loo(n),atol=1e-15)
        self.assertGreater(loo([1.,2.])[1],0)
        np.testing.assert_array_equal(loo([1.,1.]),[0.,0.])

    def test_no_K2_standardization(self):
        with self.assertRaises(ValueError): loo([1.,1.001],True)
        np.testing.assert_array_equal(loo([1.,1.,1.,1.],True),np.zeros(4))

    def test_support_and_boundaries(self):
        for mode in ('legacy','bounded_variance'):
            np.testing.assert_array_equal(count_terms([0,3,10],[0,3,10],[4,100,10],mode),np.zeros(3))
            with self.assertRaises(ValueError): count_terms([1,3],[0,3],[10,100],mode)
        x=count_terms([999],[3],[1000],'bounded_variance')
        self.assertTrue(0<=x[0]<=1)

    def test_equal_standardized_deviation(self):
        S=np.array([100.,1000.]);ref=np.array([3.,267.]);v=ref*(1-ref/S)
        x=count_terms(ref+np.sqrt(v),ref,S,'bounded_variance')
        self.assertAlmostEqual(x[0],x[1],places=14)

    def test_order_and_cardinality(self):
        ref=np.array([3.,267.,17.]);S=np.array([100.,1000.,300.]);m=ref+1
        a=count_terms(m,ref,S,'bounded_variance')
        np.testing.assert_allclose(a[::-1],count_terms(m[::-1],ref[::-1],S[::-1],'bounded_variance'))

    def test_mean_loss_and_gradient(self):
        import torch
        x=torch.tensor([-1.,-2.],requires_grad=True)
        a=eager_loss(x,2.,trajectory_count=2,time_count=8)
        b=eager_loss(x.repeat(3),2.,trajectory_count=2,time_count=8)
        self.assertEqual(float(a),float(b))
        ga,=torch.autograd.grad(a,x);gb,=torch.autograd.grad(b,x)
        torch.testing.assert_close(ga,gb)

    def test_record_is_explicit(self):
        r=feedback_record([.1,.2],[.5,.1],100)
        self.assertTrue(r['K2_components_collinear'])
        self.assertEqual(r['logscore_reduction'],'mean over active slots')

if __name__=='__main__': unittest.main()
