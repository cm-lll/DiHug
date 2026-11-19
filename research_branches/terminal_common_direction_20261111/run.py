"""Stored terminal gradients only: cone projection and strict descent certificates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,importlib.util,itertools,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import nnls,minimize
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_common_direction_20261111'
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def simplex_min_norm(U):
    H=U@U.T;m=len(U);solutions=[]
    for k in range(1,m+1):
        for inds in itertools.combinations(range(m),k):
            ix=list(inds);A=np.block([[H[np.ix_(ix,ix)],np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(k),1.];sol=np.linalg.lstsq(A,rhs,rcond=1e-12)[0]
            if max(abs(A@sol-rhs))>1e-9 or min(sol[:k])<-1e-10:continue
            a=np.zeros(m);a[ix]=np.maximum(sol[:k],0);a/=sum(a);v=a@U;sq=v@v
            residual=max(0.,float(sq-min(U@v)),float(max(abs((U@v-sq)[a>1e-10]))))
            if residual<1e-9:solutions.append((float(sq),a,v,residual))
    if not solutions:raise ArithmeticError('no certified simplex solution')
    sq,a,v,res=min(solutions,key=lambda x:x[0]);margin=np.sqrt(max(0,sq))
    d=-v/margin if margin>1e-8 else np.zeros(U.shape[1])
    return dict(weights=a.tolist(),margin=float(margin),direction=d.tolist(),residual=res,status='strict_descent' if margin>1e-8 else 'not_numerically_certified')
def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s limit')));signal.alarm(60)
    rows=[];tests=[];receipt=dict(status='running',GPU_hours=0,model_calls=0,training_updates=0,new_graphs=0)
    try:
        source=ROOT/'results/toy_gradient_alignment_20261111/conditions.json';solverpath=ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py'
        spec=importlib.util.spec_from_file_location('cone',solverpath);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        paths=[Path(__file__),source,solverpath,ROOT/'results/toy_gradient_alignment_20261111/protocol.json',ROOT/'results/toy_gradient_alignment_20261111/checks.json']
        binds={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        protocol=dict(conditions=24,source=str(source),input='saved exact terminal five-risk gradients; no model or reverse-chain evaluation',cone='min ||v-b||^2/2 s.t. G v>=0, b=mean rows of G; descent displacement=-v',strict='max over ||d||<=1 min_i -U_i d, U_i=G_i/||G_i||; row scaling for geometric diagnosis only',strict_dual='min simplex a ||sum a_i U_i||',tolerances=dict(KKT=1e-9,rank_rcond=1e-12,strict_margin=1e-8),zero_rows='report; strict all-objective decrease impossible when any row exactly zero; no artificial ridge',comparisons='mean-gradient direction, nonincrease cone, max-min unit-gradient direction',limits=dict(seconds=60,threads=1,RSS_MiB=2048,GPU=0),bindings=binds)
        dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        # Predetermined algebra cases cover opposition, duplicated rows, and zero rows.
        for name,U,expected in [('orthogonal',np.eye(2),1/np.sqrt(2)),('opposite',np.array([[1.,0],[-1,0]]),0),('duplicate',np.array([[1.,0],[1,0]]),1),('zero',np.array([[0.,0],[1,0]]),0)]:
            s=simplex_min_norm(U);assert abs(s['margin']-expected)<1e-8;tests.append(dict(name=name,passed=True,margin=s['margin']))
        for rec in json.loads(source.read_text()):
            G=np.asarray(rec['Gterm']);b=G.mean(0);bn=np.linalg.norm(b);gn=np.linalg.norm(G,axis=1)
            if np.any(gn==0):raise ValueError('actual zero gradient requires explicit non-strict analysis')
            U=G/gn[:,None];coeff,cert=mod.project_mean(G@G.T);v=coeff@G;vn=np.linalg.norm(v)
            # Independent NNLS dual: min ||G.T lambda + b||, lambda>=0.
            lam,_=nnls(U.T,-b,maxiter=1000);v2=b+U.T@lam
            rel=float(np.linalg.norm(v-v2)/max(bn,1e-30));assert rel<1e-7
            assert min(U@v)>-1e-8*max(bn,1e-30)
            identity=float(abs(b@v-v@v)/max(bn*bn,1e-30));assert identity<1e-7
            strict=simplex_min_norm(U);a=np.array(strict['weights']);direction=np.array(strict['direction'])
            alt=minimize(lambda x:.5*np.linalg.norm(x@U)**2,np.ones(5)/5,jac=lambda x:U@(x@U),bounds=[(0,1)]*5,constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)}],method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
            if not alt.success:raise ArithmeticError(alt.message)
            gap=float(abs(np.linalg.norm(alt.x@U)**2-strict['margin']**2));assert gap<1e-8
            if strict['status']=='strict_descent':assert min(-U@direction)>=strict['margin']-1e-8
            coneunit=-v/vn if vn>1e-14 else np.zeros_like(v)
            train=np.mean(rec['Gtrain'],axis=0);train=-train/np.linalg.norm(train)
            rows.append(dict(seed=rec['seed'],arm=rec['arm'],update=rec['update'],mean_norm=float(bn),row_norms=gn.tolist(),cone_direction=(-v).tolist(),cone_unit_direction=coneunit.tolist(),cone_norm_retained=float(vn/bn),cone_mean_progress_original_step=float((b@v)/(bn*bn)),cone_mean_progress_equal_norm=float(-b@coneunit/bn),cone_risk_unit_slopes=(G@coneunit).tolist(),cone_cosine_progress=(-U@coneunit).tolist(),active_indices=np.flatnonzero(abs(U@coneunit)<1e-8).tolist(),strict=strict,strict_risk_unit_slopes=(G@direction).tolist(),strict_mean_progress_equal_norm=float(-b@direction/bn),mean_risk_unit_slopes=(-G@b/bn).tolist(),training_mean_progress_equal_norm=float(-b@train/bn),training_cosine_progress=(-U@train).tolist(),certificate=cert,nnls_relative_error=rel,projection_identity_relative_error=identity,simplex_independent_objective_gap=gap))
        for p,h in binds.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=len(rows),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        dump('conditions.json',rows);dump('tests.json',tests);receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
