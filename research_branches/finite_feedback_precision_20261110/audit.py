"""Independent hard costs, LOO formula, pair expectation and projection checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,csv,hashlib,itertools,resource
import numpy as np
from scipy.optimize import nnls
from scipy.special import expit
import run as q
r=q.r;OUT=q.OUT
def read(n):return json.loads((OUT/n).read_text())
def main():
    start=time.time();assert read('receipt.json')['status']=='complete';a=np.load(OUT/'arrays.npz');meta=read('conditions.json');protocol=read('protocol.json')
    f,C,tri,cfg,kernel=r.setup();B=a['bits'];ind=[]
    def raw(bits):
        A=np.zeros((4,4),int);degrees=np.zeros((4,3));counts=np.zeros(3);tr=np.zeros(2)
        for e,(i,j) in enumerate(f.edges):
            if bits[e]:A[i,j]=A[j,i]=1;degrees[i,f.f[e]]+=1;degrees[j,f.f[e]]+=1;counts[f.f[e]]+=1
        for i,j,k in itertools.combinations(range(4),3):
            if A[i,j] and A[j,k] and A[k,i]:tr[int(f.types[[i,j,k]].sum())-1]+=1
        reach=A.astype(bool)|np.eye(4,dtype=bool)
        for k in range(4):reach|=reach[:,k,None]&reach[k,None,:]
        nc=len({tuple(row) for row in reach})
        return counts,degrees,tr,nc
    rc,rd,rt,rn=raw(f.ref)
    for b in B:
        cnt,d,t,n=raw(b);joint=[]
        for typ in (0,1):
            ix=np.flatnonzero(f.types==typ);joint.append(min(np.abs(d[ix]-rd[list(perm)]).sum()/len(ix)/3 for perm in itertools.permutations(ix)))
        ind.append([np.mean(((cnt-rc)/np.maximum(1,rc))**2),np.mean(joint),*((t-rt)**2),((n-rn)/3)**2])
    cost_error=float(abs(np.array(ind)-a['costs']).max());assert cost_error<1e-12
    algebra=0.;exact_error=0.;pair_error=0.;proj_error=0.;all_rows=[]
    for ci,m in enumerate(meta):
        p=a['probabilities'][ci];F=a['features'][ci];bits=B[a['ids'][ci]]
        rng=np.random.default_rng(np.random.SeedSequence([2026111054,m['seed'],0 if m['arm']=='common' else 1,m['update'],m['t']]))
        np.testing.assert_array_equal((rng.random((q.REPS,q.BATCH,64,6))<p[None,:,None,:]).astype(np.uint8),bits)
        cp=np.prod(np.where(B[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2)
        score=np.einsum('bse,bed->bsd',B[None,:,:]-p[:,None,:],F)
        G=np.einsum('bs,sj,bsd->jd',cp,a['costs'],score)/q.BATCH
        exact_error=max(exact_error,float(abs(G-a['exact'][ci]).max()));assert exact_error<1e-12
        if ci in (0,1,2):
            # Exhaustively average K2 leave-one-out estimator over both draws.
            for b in range(q.BATCH):
                delta=a['costs'][:,None,:]-a['costs'][None,:,:]
                scdiff=score[b,:,None,:]-score[b,None,:,:]
                pair=np.einsum('st,stj,std->jd',cp[b,:,None]*cp[b,None,:],delta,scdiff)/2
                ref=np.einsum('s,sj,sd->jd',cp[b],a['costs'],score[b]);pair_error=max(pair_error,float(abs(pair-ref).max()))
        for ki,k in enumerate(q.KS):
            costs=a['costs'][a['ids'][ci,:,:,:k]];adv=costs-(costs.sum(2,keepdims=True)-costs)/(k-1)
            scores=np.einsum('rbke,bed->rbkd',bits[:,:,:k]-p[None,:,None,:],F)
            estimated=np.einsum('rbkj,rbkd->rjd',adv,scores)/(q.BATCH*k)
            algebra=max(algebra,float(abs(estimated-a['estimates'][ci,ki]).max()));assert algebra<1e-12
            # Independent NNLS dual on representative replicate, not the active-set solver.
            test=estimated[0];lam,_=nnls(test.T,-test.mean(0),maxiter=1000);check=test.mean(0)+test.T@lam
            proj_error=max(proj_error,float(abs(check-a['directions'][ci,ki,0,1]).max()));assert proj_error<1e-8
            for di,name in enumerate(('mean','common')):
                gs=a['directions'][ci,ki,:,di];exactdir=np.array(m['exact_directions'][name]);norms=np.linalg.norm(gs,axis=1);tn=np.linalg.norm(exactdir)
                active=norms>1e-14;units=np.zeros_like(gs);units[active]=-gs[active]/norms[active,None];slopes=units@G.T
                estimslopes=np.einsum('rjd,rd->rj',estimated,units);cos=np.full(q.REPS,np.nan)
                if tn>1e-14:cos[active]=gs[active]@exactdir/(norms[active]*tn)
                z=dict(condition=ci,seed=m['seed'],state_arm=m['arm'],update=m['update'],t=m['t'],K=k,direction=name,replicates=q.REPS,
                    mean_cosine=float(np.nanmean(cos)),relative_RMSE=float(np.sqrt(np.mean(np.sum((gs-exactdir)**2,axis=1)))/tn),
                    relative_empirical_mean_error=float(np.linalg.norm(gs.mean(0)-exactdir)/tn),zero_rate=float((~active).mean()),
                    true_mean_improves_rate=float((slopes.mean(1)<-1e-12).mean()),true_any_harm_rate=float((slopes.max(1)>1e-12).mean()),
                    estimated_any_harm_rate=float((estimslopes.max(1)>1e-12).mean()),
                    mean_max_positive_slope=float(np.maximum(slopes.max(1),0).mean()),p95_max_positive_slope=float(np.quantile(np.maximum(slopes.max(1),0),.95)),
                    mean_exact_average_slope=float(slopes.mean()),mean_norm=float(norms.mean()),true_norm=float(tn))
                for j in range(5):
                    z[f'true_harm_rate_{j}']=float((slopes[:,j]>1e-12).mean());z[f'mean_slope_{j}']=float(slopes[:,j].mean())
                    z[f'zero_estimate_rate_{j}']=float((np.linalg.norm(estimated[:,j],axis=1)<=1e-14).mean())
                    z[f'gradient_relative_RMSE_{j}']=float(np.sqrt(np.mean(np.sum((estimated[:,j]-G[j])**2,axis=1)))/np.linalg.norm(G[j])) if np.linalg.norm(G[j])>1e-14 else None
                all_rows.append(z)
    assert pair_error<1e-12
    agg=[];timeagg=[]
    numeric=[k for k,v in all_rows[0].items() if isinstance(v,(float,np.floating))]
    for k in q.KS:
        for name in ('mean','common'):
            rr=[x for x in all_rows if x['K']==k and x['direction']==name]
            agg.append(dict(K=k,direction=name,conditions=len(rr),**{key:float(np.mean([x[key] for x in rr])) for key in numeric}))
            for t in q.TIMES:
                ss=[x for x in rr if x['t']==t];timeagg.append(dict(K=k,direction=name,t=t,conditions=len(ss),**{key:float(np.mean([x[key] for x in ss])) for key in numeric}))
    for path,h in protocol['bindings'].items():assert hashlib.sha256(q.Path(path).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==read('freeze.json')['protocol_SHA256']
    r.table('condition_summary.csv',all_rows);r.table('aggregate.csv',agg);r.table('time_summary.csv',timeagg)
    r.dump('summary.json',dict(aggregate=agg,by_time=timeagg,condition_count=len(meta),warning='descriptive repeated Monte Carlo trials on shared toy models, not independent training replication; normalized-direction slopes are local first-order diagnostics'))
    result=dict(passed=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,independent_cost_max_error=cost_error,
        independent_exact_gradient_error=exact_error,LOO_algebra_max_error=algebra,exact_K2_expectation_error=pair_error,NNLS_projection_max_error=proj_error,
        hard_samples_reconstructed=True,bindings_unchanged=True,GPU_hours=0,parameter_updates=0)
    r.dump('independent_audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
