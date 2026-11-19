"""Finite hard-trajectory terminal score estimator; frozen toy weights, no updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,hashlib,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/terminal_mc_reliability_20261111'
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
m=load('train',ROOT/'research_branches/terminal_common_train_20261111/run.py');r=m.r
def cos(x,y):
    den=np.linalg.norm(x)*np.linalg.norm(y);return None if den==0 else float(x@y/den)
def score(f,w,t,x,y):
    F=f.features[t][x];p=expit(F@w);dq=(f.k[t][1][x]-f.k[t][0][x])*p*(1-p);q=f.k[t][0][x]+p*(f.k[t][1][x]-f.k[t][0][x])
    return (((f.B[y]-q)*dq/(q*(1-q)))@F)
def logp(f,w,t,x,y):
    p=expit(f.features[t][x]@w);q=f.k[t][0][x]+p*(f.k[t][1][x]-f.k[t][0][x])
    return float(np.sum(f.B[y]*np.log(q)+(1-f.B[y])*np.log1p(-q)))
def gates(f,w):
    errors=[];zero=[]
    for t in (1,25,100):
        for x in (0,17,63):
            p=expit(f.features[t][x]@w);q=f.k[t][0][x]+p*(f.k[t][1][x]-f.k[t][0][x]);prob=np.prod(np.where(f.B>0,q,1-q),axis=1)
            S=np.array([score(f,w,t,x,y) for y in range(64)]);zero.append(float(abs(prob@S).max()))
            for y in (0,21,63):
                for j in range(len(w)):
                    e=np.eye(len(w))[j]*1e-5;fd=(logp(f,w+e,t,x,y)-logp(f,w-e,t,x,y))/2e-5
                    errors.append(abs(fd-S[y,j]))
    assert max(errors)<1e-7 and max(zero)<1e-12
    return dict(finite_difference_count=len(errors),max_fd=float(max(errors)),conditional_zero_score=float(max(zero)))
def sample(f,w,rng,n=512):
    x=np.searchsorted(np.cumsum(f.prior),rng.random(n));S=np.zeros((n,len(w)));states=[x.astype(np.uint8)]
    powers=2**np.arange(5,-1,-1)
    for t in range(100,0,-1):
        F=f.features[t][x];p=expit(F@w);difference=f.k[t][1][x]-f.k[t][0][x];q=f.k[t][0][x]+p*difference;dq=difference*p*(1-p)
        B=(rng.random((n,6))<q).astype(float)
        S+=np.einsum('ie,ied->id',(B-q)*dq/(q*(1-q)),F)
        x=(B@powers).astype(int);states.append(x.astype(np.uint8))
    return x,S,np.array(states).T
def main():
    start=time.time();OUT.mkdir(exist_ok=False);r.OUT=OUT
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 second bound')));signal.alarm(180)
    rows=[];truth=[];arrays={};tests=[];receipt=dict(status='running',GPU_hours=0,training_updates=0,production_calls=0,hard_toy_paths=0)
    try:
        f,C,tri,cfg,_=r.setup();f=m.a.m.augment(f)
        source=ROOT/'results/terminal_common_train_20261111';hist=np.load(source/'histories.npz')
        paths=[Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'research_branches/terminal_common_train_20261111/run.py',ROOT/'research_branches/terminal_common_direction_20261111/run.py',ROOT/'research_branches/toy_typed_endpoint_20261110/run.py',source/'histories.npz',source/'audit.json']
        for p in paths:r.bind(p)
        assert json.loads((source/'audit.json').read_text())['passed']
        protocol=dict(track='CORE finite trajectory gradient reliability',seeds=r.SEEDS,arm='common',updates=[0,20],conditions=6,K=[32,128,512],repeats=32,random_seed_base=2026111111,nested_prefix=True,estimator='mean_n (C_n - mean_{j!=n} C_j) times sum_t grad log actual P_theta(Xt-1|Xt); no clipping/SD normalization',common='same fixed row-normalized max-min solver; report zero/uncertified estimates; no fallback masquerading as success',scoring='exact terminal gradients assess first-order slopes only; no weight update or finite candidate evaluation',harm_tolerance='normalized exact directional slope >1e-8; raw magnitudes also retained',budgets=dict(seconds=180,threads=1,RSS_MiB=2048,GPU=0),bindings=r.BIND)
        r.dump('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();r.dump('freeze.json',dict(epoch=time.time(),protocol_sha256=ph))
        for seed in r.SEEDS:
            for step in (0,20):
                key=f'{seed}_{step}';w=hist[f'{seed}_common_parameters'][step];test=gates(f,w);tests.append(dict(seed=seed,update=step,**test))
                cost,G,dist=m.terminal(f,C,w);gn=np.linalg.norm(G,axis=1);U=G/gn[:,None];cert=m.s.simplex_min_norm(U);dtrue=np.array(cert['direction'])
                truth.append(dict(seed=seed,update=step,gradient=G.tolist(),terminal_costs=cost.tolist(),common_certificate=cert));allS=[];allY=[];allX=[]
                for repeat in range(32):
                    rng=np.random.default_rng(np.random.SeedSequence([2026111111,seed,step,repeat]));y,S,X=sample(f,w,rng);allS.append(S);allY.append(y);allX.append(X);receipt['hard_toy_paths']+=512
                    for K in (32,128,512):
                        c=C[y[:K]];center=c-c[0];adv=(K*center-center.sum(0))/(K-1);est=adv.T@S[:K]/K
                        assert np.isfinite(est).all();en=np.linalg.norm(est,axis=1);relative=float(np.linalg.norm(est-G)/np.linalg.norm(G));status='zero_gradient_row';dc=np.zeros_like(w);ec={}
                        if min(en)>0:
                            ec=m.s.simplex_min_norm(est/en[:,None]);status=ec['status'];dc=np.array(ec['direction'])
                        normslopes=U@dc;estslopes=est@dc
                        if status=='strict_descent':assert max(estslopes)<=1e-9
                        bm=est.mean(0);dm=-bm/np.linalg.norm(bm) if np.linalg.norm(bm)>0 else np.zeros_like(w)
                        rows.append(dict(seed=seed,update=step,repeat=repeat,K=K,gradient=est.tolist(),relative_gradient_error=relative,mean_gradient_cos=cos(bm,G.mean(0)),row_cos=[cos(x,z) for x,z in zip(est,G)],status=status,certificate=ec,direction=dc.tolist(),cos_to_exact_common=cos(dc,dtrue),true_normalized_slopes=normslopes.tolist(),true_slopes=(G@dc).tolist(),estimated_slopes=estslopes.tolist(),true_any_harm=bool(max(normslopes)>1e-8),true_all_strict_improve=bool(max(normslopes)<-1e-8),true_mean_slope=float(G.mean(0)@dc),sample_mean_direction_true_slopes=(G@dm).tolist()))
                arrays[key+'_scores']=np.array(allS);arrays[key+'_endpoints']=np.array(allY);arrays[key+'_states']=np.array(allX)
                r.dump('progress.json',dict(conditions=len(truth),paths=receipt['hard_toy_paths'],seconds=time.time()-start))
        for p,h in r.BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
        receipt.update(status='complete',conditions=len(truth),estimates=len(rows),bindings_unchanged=True)
    except BaseException as e:receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.dump('estimates.json',rows);r.dump('truth.json',truth);r.dump('tests.json',tests);np.savez_compressed(OUT/'sample_evidence.npz',**arrays)
        receipt.update(seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);r.dump('receipt.json',receipt);print(json.dumps(receipt))
    return 0 if receipt['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
