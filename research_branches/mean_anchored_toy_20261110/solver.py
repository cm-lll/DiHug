"""Exact one-active-objective certificate before generic constrained solver.

For each i the maximum of g_i*d on the ball is attained at
gbar+c*||gbar||*g_i/||g_i||. If i is also worst there, this is
a primal/dual optimal solution for max min_j g_j*d.
"""
import numpy as np
def solve(G,generic,c=.5):
    m=G.mean(0);n=np.linalg.norm(m)
    if n==0:return generic(G,c)
    for i,g in enumerate(G):
        gn=np.linalg.norm(g)
        if gn==0:continue
        d=m+c*n*g/gn;upper=float(g@m+c*n*gn);lower=float(np.min(G@d));gap=(upper-lower)/(n*n)
        if -1e-12<=gap<=1e-10:
            w=np.zeros(len(G));w[i]=1
            return d,dict(first_status='analytic_single_active',status='certified optimal',success=True,rank=int(np.linalg.matrix_rank(G)),iterations=0,primal=lower/(n*n),dual=upper/(n*n),gap=gap,tolerance=2e-6*max(1,abs(upper/(n*n))),ball_violation=float(max(0,np.linalg.norm(d-m)**2/(n*n)-c*c)),kkt_residual=0.,multipliers=w.tolist(),norm_ratio=float(np.linalg.norm(d)/n),mean_descent_ratio=float(m@d/(n*n)))
    return generic(G,c)
