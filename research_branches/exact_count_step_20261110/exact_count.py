"""Pure NumPy arithmetic; no device/environment changes on import."""
import itertools
import numpy as np

def exact(p,f,targets):
    R=len(targets);gradient=np.zeros_like(p);risks=[]
    for r,m in enumerate(targets):
        ix=f==r;pp=p[ix];mu=pp.sum();variance=(pp*(1-pp)).sum();den=R*max(1,m)**2
        risks.append(((mu-m)**2+variance)/den)
        gradient[ix]=pp*(1-pp)*(2*(mu-m)+1-2*pp)/den
    return np.array(risks),gradient

def tests():
    p=np.array([.13,.27,.45,.61,.78,.91]);f=np.array([0,0,0,1,1,2]);target=np.array([1,1,0])
    bits=np.array(list(itertools.product([0,1],repeat=6)),float)
    costs=np.column_stack([((bits[:,f==r].sum(1)-m)/max(1,m))**2/len(target) for r,m in enumerate(target)])
    mass=np.prod(np.where(bits,p,1-p),1);risk,g=exact(p,f,target)
    np.testing.assert_allclose(mass@costs,risk,atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose((mass*costs.sum(1))@(bits-p),g,atol=1e-14,rtol=1e-13)
    return dict(passed=True,graphs=64)
