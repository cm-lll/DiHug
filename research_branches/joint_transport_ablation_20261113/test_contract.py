"""CPU algebra tests, no imported production model."""
import unittest
import numpy as np
from transport import assignment
class ContractTests(unittest.TestCase):
    def test_four_chain_loo(self):
        c=np.array([.2,.1,.8,.5]);a=(4*c-c.sum())/3
        np.testing.assert_allclose(a,[c[i]-np.delete(c,i).mean() for i in range(4)])
        self.assertAlmostEqual(a.sum(),0)
        self.assertGreater(a[2],0);self.assertLess(a[1],0)
    def test_squared_detects_placement(self):
        ref=np.ones((8,1))*2
        spread=ref.copy();spread[:4]+=1
        concentrated=ref.copy();concentrated[0]+=2;concentrated[1:3]+=1
        self.assertEqual(spread.sum(),concentrated.sum())
        self.assertGreater(assignment(concentrated,ref)[0],assignment(spread,ref)[0])
    def test_permutation_and_mean_bound(self):
        rng=np.random.default_rng(2026111308);x=rng.normal(size=(12,5));y=rng.normal(size=(12,5));c,_=assignment(x,y)
        self.assertAlmostEqual(c,assignment(x[rng.permutation(12)],y)[0],places=12)
        self.assertGreaterEqual(c+1e-12,float(np.sum((x.mean(0)-y.mean(0))**2)))
    def test_feedback_denominator(self):
        self.assertEqual(sum(1/(4*8) for _ in range(4) for _ in range(8)),1)
if __name__=='__main__':unittest.main()
