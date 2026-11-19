"""Diagonal reparameterization of the identical constrained inverse CE objective."""
import round26_nocache as original
from round01 import np,time
from scipy.optimize import minimize
REG=original.REG;objective=original.objective;pg=original.pg

def diagonal(phi,weight):
    centered=phi-phi.mean(axis=1,keepdims=True)
    return np.einsum('i,ijk,ijk->k',weight,centered,centered,optimize=True)/phi.shape[1]+REG

def solve(phi,y,weight):
    scale=1/np.sqrt(diagonal(phi,weight));calls=0;history=[];tic=time.monotonic()
    def fun(v):
        nonlocal calls
        assert time.monotonic()-tic<600
        calls+=1;w=scale*v;f,g=objective(w,phi,y,weight)
        history.append(dict(call=calls,objective=f,projected_gradient_max=pg(w,g)))
        return f,g*scale
    width=phi.shape[-1]
    result=minimize(fun,np.zeros(width),method='L-BFGS-B',jac=True,
        bounds=[(None,None)]*(width//2)+[(0,None)]*(width//2),
        options=dict(maxiter=4000,maxfun=8000,ftol=0.,gtol=1e-9))
    w=scale*result.x;f,g=objective(w,phi,y,weight)
    return w,dict(success=bool(result.success),status=int(result.status),message=str(result.message),
        iterations=int(result.nit),function_gradient_calls=calls,objective=f,
        projected_gradient_max=pg(w,g),minimum_quadratic=float(w[width//2:].min()),
        seconds=time.monotonic()-tic,scale=scale.tolist(),history=history,
        optimization_coordinates='v, w=scale*v; unchanged original-space objective and gate')
