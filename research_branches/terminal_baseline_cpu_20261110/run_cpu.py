"""Exact baseline moments and read-only saved trajectory diagnostics; no model."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, gc, hashlib, importlib.util, json, resource, signal, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/terminal_structure_real_20261109T233600'
OUT=ROOT/'results/terminal_baseline_cpu_20261110'
START=time.monotonic()
BIND={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()
def bind(p):
    p=Path(p); BIND[str(p)]=sha(p); return p
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,bind(p)); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def check():
    if time.monotonic()-START>590: raise TimeoutError('CPU stop reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2: raise MemoryError('2GiB RSS cap')
def exact():
    m=load('finite_baseline',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py')
    m.kernel_module=load('finite_bound_kernel',m.KERNEL)
    cfg=json.loads(bind(m.PROTOCOL).read_text()); f=m.FiniteDiffusion(cfg); rows=[]
    for seed in cfg['initializations']:
        theta=np.random.default_rng(seed).normal(0,.15,f.D);theta[:3]+=logit(f.pi)
        r=f.evaluate(theta);P,DP=r['P'],r['DP']; oracle={0:f.cost}
        for t in range(1,f.T+1):oracle[t]=P[t]@oracle[t-1]
        for method in ['zero','population_constant','prestate_cost','conditional_value_oracle']:
            prob=np.eye(f.S);mean=np.zeros((f.S,f.S,f.D));second=np.zeros((f.S,f.S))
            for t in range(1,f.T+1):
                b={'zero':np.zeros(f.S),'population_constant':np.full(f.S,r['metrics']['terminal_cost']),
                   'prestate_cost':f.cost,'conditional_value_oracle':oracle[t]}[method]
                score=DP[t]/P[t][:,:,None]
                # x=current, z=next, y=terminal; all paths summed, none sampled.
                a=(f.cost[None,None,:,None]-b[:,None,None,None])*score[:,:,None,:]
                nxtmean=np.einsum('xz,zyd->xyd',P[t],mean)+np.einsum('xz,xzyd,zy->xyd',P[t],a,prob)
                nxtsecond=P[t]@second+2*np.einsum('xz,xzyd,zyd->xy',P[t],a,mean)+np.einsum('xz,xzyd,xzyd,zy->xy',P[t],a,a,prob)
                prob=P[t]@prob;mean,second=nxtmean,nxtsecond
            g=np.einsum('x,xyd->d',f.prior,mean)
            error=float(np.max(abs(g-r['g']))); assert error<1e-9
            moment=float(f.prior@second.sum(1)); variance=moment-float(g@g)
            assert variance>=-1e-10 and abs(float(f.prior@prob@f.cost)-r['metrics']['terminal_cost'])<1e-12
            rows.append(dict(seed=seed,method=method,gradient_max_error=error,mean_norm=float(np.linalg.norm(g)),second_moment=moment,variance_trace=variance))
            check()
        # Independent zero-cost-score martingale identity: E||sum s||²=sum E||s||².
        marg=f.prior.copy();mart=0.
        for t in range(f.T,0,-1):
            mart+=float(np.einsum('x,xzd,xzd,xz->',marg,DP[t]/P[t][:,:,None],DP[t]/P[t][:,:,None],P[t]));marg=marg@P[t]
        u=np.zeros((f.S,f.D));v=np.zeros(f.S)
        for t in range(1,f.T+1):
            s=DP[t]/P[t][:,:,None]
            v=P[t]@v+2*np.einsum('xz,xzd,zd->x',P[t],s,u)+np.einsum('xz,xzd,xzd->x',P[t],s,s)
            u=P[t]@u+DP[t].sum(1)
        assert abs(float(f.prior@v)-mart)<1e-10
        save('exact_moments',rows)
    return rows
def real():
    import torch
    torch.set_num_threads(1)
    p=json.loads(bind(SRC/'protocol.json').read_text());S=load('frozen_score_baseline',SRC/'frozen_score.py')
    slotmod=load('bound_slots_baseline',ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/hsgd/slots.py')
    costs=json.loads(bind(SRC/'terminal_costs.json').read_text());old=json.loads(bind(SRC/'gradient_diagnostics.json').read_text())
    rows=[];states=[];checks=[]
    for ds,model in p['models'].items():
        types,schema,ref=S.load(bind(model['data']));rf=S.features(ref,types,schema);rx=S.descriptor(types,S.bfs(ref))
        space=slotmod.TypedSlots(types,tuple(map(tuple,schema)))
        endpoints=np.concatenate([space.decode(i,np.arange(n)) for i,n in enumerate(space.counts)],axis=1)
        def score(a):
            c=S.terms(S.features(a,types,schema),rf,types,schema);c['component_distance']=S.dist(S.descriptor(types,S.bfs(a)),rx,types)
            return np.array([c[k] for k in p['score_terms']])
        assert np.max(abs(score(ref)))<1e-12
        C=[];B=[];files=[];times=[]
        for ss in p['sample_seeds']:
            folder=SRC/ds/str(ss);file=bind(folder/'score_gradients.pt');files.append(file)
            payload=torch.load(file,map_location='cpu',weights_only=False);tt=list(map(int,payload['times']));times.append(tt)
            assert len(tt)==8 and payload['gradients'].shape[0]==8 and torch.isfinite(payload['gradients']).all()
            del payload;gc.collect()
            _,_,a=S.load(bind(folder/'graph.npz')); c=score(a);previous=next(x for x in costs if x['dataset']==ds and x['sample_seed']==ss)
            assert max(abs(c[i]-previous[k]) for i,k in enumerate(p['score_terms']))<1e-12
            C.append(c);bb=[]
            for t in tt:
                z=np.load(bind(folder/f'step_{t:04d}.npz'));xt=z['xt'];assert len(xt)==space.total and np.isin(xt,[0,1]).all()
                a=np.zeros_like(ref);e=endpoints[:,xt.astype(bool)];a[e[0],e[1]]=True;a|=a.T
                b=score(a);bb.append(b)
                states.append(dict(dataset=ds,sample_seed=ss,t=t,edges=int(a.sum()//2),**dict(zip(p['score_terms'],b.tolist()))));check()
            B.append(bb)
        save('state_costs',states);C=np.array(C);B=np.array(B)
        # Ten block products; load at most two 203MiB tensors and 8x65536 double chunks.
        gram=np.zeros((32,32));tic=time.monotonic()
        for i in range(4):
            gi=torch.load(files[i],map_location='cpu',weights_only=False)['gradients'].numpy()
            for j in range(i,4):
                gj=gi if j==i else torch.load(files[j],map_location='cpu',weights_only=False)['gradients'].numpy()
                block=np.zeros((8,8))
                for start in range(0,gi.shape[1],65536):
                    x=gi[:,start:start+65536].astype(np.float64);y=gj[:,start:start+65536].astype(np.float64);block+=x@y.T;check()
                gram[i*8:(i+1)*8,j*8:(j+1)*8]=block;gram[j*8:(j+1)*8,i*8:(i+1)*8]=block.T
                del gj;gc.collect()
            del gi;gc.collect();print(json.dumps(dict(dataset=ds,gram_blocks_completed_row=i,seconds=time.monotonic()-tic)),flush=True)
        assert np.linalg.eigvalsh(gram).min()>-1e-8*max(1,np.trace(gram))
        np.savez_compressed(OUT/(ds+'_sufficient_statistics.npz'),gram=gram,terminal_costs=C,prestate_costs=B,times=times,terms=p['score_terms'])
        def norm(w):return float(np.sqrt(max(0,w@gram@w)))
        def cosine(a,b):
            den=norm(a)*norm(b);return float(a@gram@b/den) if den else None
        for k,term in enumerate(p['score_terms']):
            c=C[:,k];loo=(c.sum()-c)/3
            for method in ['uncentered','LOO','prestate_cost']:
                coeff=np.repeat(c[:,None],8,axis=1)
                if method=='LOO':coeff-=loo[:,None]
                elif method=='prestate_cost':coeff-=B[:,:,k]
                per=np.zeros((4,32))
                for i in range(4):per[i,i*8:(i+1)*8]=coeff[i]*125
                estimate=per.mean(0);gn=norm(estimate);spread=float(np.sqrt(np.mean([norm(v-estimate)**2 for v in per])))
                if method=='LOO':
                    h=[]
                    for i,j in [(0,1),(2,3)]:
                        q=np.zeros(32);q[i*8:(i+1)*8]=(c[i]-c[j])*125/2;q[j*8:(j+1)*8]=-(c[i]-c[j])*125/2;h.append(q)
                else:h=[per[:2].mean(0),per[2:].mean(0)]
                row=dict(dataset=ds,term=term,method=method,gradient_norm=gn,contribution_rms_spread=spread,spread_over_norm=spread/gn if gn else None,disjoint_pairs_cosine=cosine(*h),mean_abs_feedback=float(abs(coeff).mean()),max_abs_feedback=float(abs(coeff).max()))
                if method=='LOO':
                    o=next(v for v in old if v['dataset']==ds and v['term']==term)
                    err=abs(gn-o['gradient_norm'])/max(gn,1e-15);assert err<1e-9
                    checks.append(dict(dataset=ds,term=term,old_LOO_norm_relative_error=err))
                rows.append(row)
        save('real_diagnostics',rows);save('reproduction_checks',checks)
    return rows
def main():
    OUT.mkdir(exist_ok=False,parents=True);signal.alarm(600)
    protocol=Path(__file__).with_name('protocol.json');save('protocol',json.loads(bind(protocol).read_text()));bind(Path(__file__))
    save('start',dict(UTC=datetime.now(timezone.utc).isoformat(),GPU=0,CPU_threads=1))
    exact();real();check()
    for path,digest in BIND.items():assert sha(path)==digest,path
    save('bindings',BIND)
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,CPU_threads=1,GPU_hours=0,model_calls=0,new_graphs=0,parameter_updates=0,source_assets_unchanged=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
