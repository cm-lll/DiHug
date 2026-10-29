"""Unit-entropy reduced role-bias dual, damped Newton. No alternate solver."""
import time
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,connected_components

def reduce_support(s,c):
    n,k=s.shape
    if s.all():return s.copy(),[(np.arange(n),np.arange(k))],dict(feasible=True,forced_zero=0,certificate='full support')
    i,j=np.nonzero(s);source=n+k;sink=source+1
    row=np.r_[np.full(n,source),i,n+np.arange(k)];col=np.r_[np.arange(n),n+j,np.full(k,sink)]
    val=np.r_[np.ones(n,dtype=np.int64),np.ones(len(i),dtype=np.int64),c]
    flow=maximum_flow(coo_matrix((val,(row,col)),shape=(sink+1,sink+1),dtype=np.int64).tocsr(),source,sink)
    if flow.flow_value!=n:return None,None,dict(feasible=False,certificate='integer maxflow',flow=int(flow.flow_value))
    used=flow.flow[:n,n:n+k].toarray()>0
    ui,uj=np.nonzero(used);vi,vj=np.nonzero(s&~used)
    residual=coo_matrix((np.ones(len(ui)+len(vi)),(np.r_[n+uj,vi],np.r_[ui,n+vj])),shape=(n+k,n+k)).tocsr()
    _,lab=connected_components(residual,directed=True,connection='strong')
    keep=used|(s&(lab[:n,None]==lab[n:][None,:]))
    ii,jj=np.nonzero(keep)
    graph=coo_matrix((np.ones(2*len(ii)),(np.r_[ii,n+jj],np.r_[n+jj,ii])),shape=(n+k,n+k)).tocsr()
    nc,lab=connected_components(graph,directed=False)
    components=[(np.flatnonzero(lab[:n]==a),np.flatnonzero(lab[n:]==a)) for a in range(nc)]
    assert all(len(ix)==int(c[jx].sum()) and len(ix) and len(jx) for ix,jx in components)
    return keep,components,dict(feasible=True,forced_zero=int((s&~keep).sum()),certificate='integer flow + residual SCC',assignment=np.argmax(used,axis=1).tolist())

def dual(L,c,b,hessian=True):
    logits=L+b;z=logsumexp(logits,axis=1);lp=logits-z[:,None]
    p=np.zeros_like(lp);use=lp>=-700;p[use]=np.exp(lp[use])
    g=p.sum(0)-c
    H=np.diag(p.sum(0))-p.T@p if hessian else None
    return float(z.sum()-c@b),g,H,lp,p

def objective_difference(lp,p,c,d):
    # Stable log E exp(d), subtract linear part before summing at small d.
    if np.max(abs(d))<.5:
        m=p@d;z=d[None,:]-m[:,None]
        rem=np.expm1(z)-z
        row=m+np.log1p(np.sum(p*z,axis=1)+np.sum(p*rem,axis=1))
    else:row=logsumexp(lp+d,axis=1)
    return float(row.sum()-c@d),float(np.abs(row).sum()+abs(c@d))

def solve(ll,counts,check=lambda:None):
    started=time.perf_counter();r=dict(status='failed',iterations=0,function_evaluations=0,components=[])
    L0=np.asarray(ll,dtype=np.float64);c0=np.asarray(counts)
    if L0.ndim!=2 or c0.ndim!=1 or L0.shape[1]!=len(c0) or not len(L0) or not len(c0):raise ValueError('matrix/count shape')
    if not np.issubdtype(c0.dtype,np.integer) or (c0<0).any() or int(c0.sum())!=len(L0):raise ValueError('integer capacities')
    if np.isnan(L0).any() or np.isposinf(L0).any():raise ValueError('likelihood NaN/+inf')
    if L0.size>2000000 or L0.size*128+L0.shape[1]**2*32>256*1024**2:raise MemoryError('resource gate')
    def gate():
        check()
        if time.perf_counter()-started>30:raise TimeoutError('single matrix30s')
    try:
        gate();active=c0>0;L=L0[:,active].copy();c=c0[active];support=np.isfinite(L)
        keep,components,proof=reduce_support(support,c);r['support']=proof;r['support_seconds']=time.perf_counter()-started
        if keep is None:r['status']='support_infeasible';return None,None,r
        L[~keep]=-np.inf;P=np.zeros_like(L);LP=np.full_like(L,-np.inf);bias=np.zeros(len(c))
        for ix,jx in components:
            gate();B=L[np.ix_(ix,jx)].copy();cc=c[jx]
            row=B.max(1);B-=row[:,None];col=B.max(0);B-=col
            if not np.isfinite(B[np.isfinite(L[np.ix_(ix,jx)])]).all():raise FloatingPointError('gauge range')
            b=np.zeros(len(jx));damping=1e-6;accepted=False;iters=evals=0;backtracks=0
            for it in range(201):
                gate();f,g,H,lp,p=dual(B,cc,b);evals+=1;r['function_evaluations']+=1
                if r['function_evaluations']>4000:raise RuntimeError('function evaluation limit')
                if not np.isfinite(f) or not np.isfinite(g).all():raise FloatingPointError('dual nonfinite')
                if np.max(abs(g)/np.maximum(1,cc))<=1e-8:accepted=True;iters=it;break
                if it==200:break
                moved=False
                for attempt in range(8):
                    if len(jx)==1:break
                    A=H[:-1,:-1]+damping*np.eye(len(jx)-1)
                    try:direction=np.r_[cho_solve(cho_factor(A,lower=True,check_finite=False),-g[:-1],check_finite=False),0.]
                    except np.linalg.LinAlgError:damping=min(1e12,damping*10);continue
                    slope=float(g@direction)
                    if not np.isfinite(direction).all() or slope>=0:damping=min(1e12,damping*10);continue
                    step=min(1.,1000/max(abs(direction).max(),1.))
                    for bt in range(32):
                        gate();change,scale=objective_difference(lp,p,cc,step*direction);evals+=1;r['function_evaluations']+=1
                        if r['function_evaluations']>4000:raise RuntimeError('function evaluation limit')
                        if np.isfinite(change) and change<=1e-4*step*slope+64*np.finfo(float).eps*max(1.,scale):
                            b+=step*direction;damping=max(1e-12,damping/3);moved=True;backtracks+=bt;break
                        step*=.5
                    if moved:break
                    damping=min(1e12,damping*10)
                if not moved:r['status']='line_search_failed';break
                iters=it+1
            r['iterations']+=iters
            r['components'].append(dict(n=len(ix),k=len(jx),iterations=iters,evaluations=evals,backtracks=backtracks,damping=damping,converged=accepted))
            if not accepted:
                r['status']=r['status'] if r['status']!='failed' else 'iteration_limit'
                return None,None,r
            P[np.ix_(ix,jx)]=p;LP[np.ix_(ix,jx)]=lp;bias[jx]=b-col
        gate();fullp=np.zeros_like(L0);fullLP=np.full_like(L0,-np.inf);fullp[:,active]=P;fullLP[:,active]=LP
        residual=abs(fullp.sum(0)-c0)/np.maximum(1,c0)
        rowres=float(abs(fullp.sum(1)-1).max());colres=float(residual.max())
        rounded=np.isfinite(LP)&(LP<-700);mass=rounded.sum(1)*np.exp(-700.)
        # Independently reconstruct full log-softmax from original L and role biases.
        Z=logsumexp(L+bias,axis=1);reference=L+bias-Z[:,None]
        station=float(abs(reference[keep]-LP[keep]).max(initial=0))
        original = int(rounded.sum())==0
        valid=rowres<=1e-8 and colres<=1e-8 and station<=2e-10 and mass.max(initial=0)<=1e-280 and mass.sum()<=1e-276
        r.update(status=('converged_original' if original else 'converged_bounded_tail') if valid else 'independent_validation_failed',
            original_contract_pass=valid and original,new_contract_pass=valid,row_residual=rowres,column_residual=colres,stationarity_residual=station,
            rounded_positive_count=int(rounded.sum()),row_omitted_mass_upper=mass.tolist(),total_omitted_mass_upper=float(mass.sum()),
            input_mathematical_zeros=int(np.isneginf(L0).sum()),constraint_forced_zeros=proof['forced_zero'],zero_frequency_columns=int((~active).sum()))
        good=P>0
        r['primal_objective']=float(np.sum(P[good]*(L[good]-LP[good])))
        return (fullp,fullLP,r) if valid else (None,None,r)
    except (TimeoutError,InterruptedError,RuntimeError,FloatingPointError) as exc:
        r.update(status='timeout' if isinstance(exc,TimeoutError) else 'solver_failed',reason=repr(exc));return None,None,r
    finally:r['seconds']=time.perf_counter()-started;r['solver_seconds']=r['seconds']-r.get('support_seconds',0)

class ReadoutFailure(RuntimeError):pass
def required(L,c,check=lambda:None):
    p,lp,r=solve(L,c,check)
    if not r.get('new_contract_pass',False):raise ReadoutFailure(str(r))
    return p,r
