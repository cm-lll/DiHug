"""Predeclared80-digit reference cases; explicit machine-readable values."""
import numpy as np
import mpmath as mp
from solver import solve
from prepare import OUT,save,check
import scaling as scaling_reference
def main():
    mp.mp.dps=80;rows=[]
    for name,L in [('two_by_two_4to1',np.array([[np.log(4.),0],[0,np.log(4.)]])),
                   ('three_by_three_moderate',np.array([[2.,1,0],[0,1,3.],[1.,2,.5]])),
                   ('two_by_two_concentrated_80',np.array([[80.,0],[0,80.]]))]:
        check();n,k=L.shape;c=np.ones(k,dtype=int)
        def probs(b):
            b=list(b)+[mp.mpf(0)];result=[]
            for row in L:
                v=[mp.exp(mp.mpf(float(x))+y) for x,y in zip(row,b)];result.append([x/sum(v) for x in v])
            return result
        def gradient(*b):
            p=probs(b);return tuple(sum(p[i][j] for i in range(n))-int(c[j]) for j in range(k-1))
        bias=mp.findroot(gradient,tuple(mp.mpf(0) for _ in range(k-1)),tol=mp.mpf('1e-65'),maxsteps=100)
        if k==2:bias=[bias[0] if hasattr(bias,'rows') else bias]
        ref=probs(bias);floatref=np.array([[float(x) for x in row] for row in ref])
        p,lp,r=solve(L,c,check);q,s=scaling_reference.scale(L,c,check)
        assert r['new_contract_pass'];np.testing.assert_allclose(p,floatref,atol=2e-8,rtol=0)
        rows.append(dict(case=name,dps=80,reference_probabilities=[[mp.nstr(x,80) for x in row] for row in ref],
            newton_max_abs=float(abs(p-floatref).max()),scaling_max_abs=float(abs(q-floatref).max()),
            newton=r,scaling=s))
    save(OUT/'high_precision.json',dict(status='passed',cases=rows))
if __name__=='__main__':main()
