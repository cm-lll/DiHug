"""Bounded saved-probability CPU audit; gradients are w.r.t. independent logits only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import hashlib,importlib.util,itertools,json,time,traceback,resource,csv
from pathlib import Path
import numpy as np
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_feedback_audit_20261111'
OLD=ROOT/'results/real192_update_audit_20261111'


def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


heat=module('feedback_heat','research_branches/typed_heat_audit_20261111/run.py')
joint=module('feedback_joint','research_branches/typed_heat_degree_audit_20261111/run.py')
REPEATS=8
K=16
RANDOM_ROOT=2026111201


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def table(name,rows):
    save(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def loo(p,bits,cost):
    b=bits.astype(np.float64)
    advantage=(len(cost)*cost-cost.sum())/(len(cost)-1)
    return np.mean(advantage[:,None]*(b-p),axis=0)


def count_exact(p,f,reference,r):
    expected=np.bincount(f,weights=p,minlength=r)
    target=np.bincount(f,weights=reference,minlength=r)
    variance=np.bincount(f,weights=p*(1-p),minlength=r)
    denom=np.maximum(1,target)**2
    risk=float(np.mean(((expected-target)**2+variance)/denom))
    gradient=(2*(expected[f]-target[f])+1-2*p)*p*(1-p)/(r*denom[f])
    return risk,gradient


def neutralize(g,p,f):
    """Diagnostic projection only: tangent to fixed expected count per relation."""
    out=np.array(g,copy=True);v=p*(1-p)
    for rel in np.unique(f):
        mask=f==rel;vr=v[mask];den=vr@vr
        if den>0:out[...,mask]-=((out[...,mask]@vr)/den)[...,None]*vr
    return out


def cosine(x,y):
    d=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(x@y/d) if d>1e-30 else None


def stats(g):
    mean=g.mean(0);norm=float(np.linalg.norm(mean));var=float(np.sum((g-mean)**2)/(len(g)-1))
    return dict(mean_norm=norm,variance_trace=var,
                SE_over_mean_norm=float(np.sqrt(var/len(g))/norm) if norm>1e-30 else None,
                split_half_cosine=cosine(g[:len(g)//2].mean(0),g[len(g)//2:].mean(0)),
                zero_repeats=int(np.sum(np.linalg.norm(g,axis=1)==0)))


def checks():
    p=np.array([.03,.2,.45,.62,.81,.94]);f=np.array([0,0,0,1,1,1]);ref=np.array([0,1,0,1,0,1])
    bits=np.array(list(itertools.product([0,1],repeat=6)),dtype=float)
    counts=np.column_stack([bits[:,f==r].sum(1) for r in [0,1]])
    target=np.bincount(f,weights=ref);cost=np.mean((counts-target)**2/np.maximum(1,target)**2,axis=1)
    probs=np.prod(np.where(bits,p,1-p),axis=1)
    exact=(probs*cost)@(bits-p)
    risk,analytic=count_exact(p,f,ref,2)
    np.testing.assert_allclose(analytic,exact,atol=1e-13);assert abs(risk-probs@cost)<1e-13
    # Arbitrary nonlinear fixed hard-graph cost, checked without model calls.
    arbitrary=np.log1p((counts[:,0]-counts[:,1])**2)+.3*np.prod(bits[:,:3],axis=1)
    gradient=(probs*arbitrary)@(bits-p)
    def expectation(l):
        q=expit(l);return np.prod(np.where(bits,q,1-q),axis=1)@arbitrary
    l=logit(p);fd=[]
    for j in range(6):
        h=np.zeros(6);h[j]=1e-5;fd.append((expectation(l+h)-expectation(l-h))/2e-5)
    np.testing.assert_allclose(gradient,fd,atol=1e-9,rtol=1e-7)
    pair=np.zeros(6)
    for i in range(64):
        for j in range(64):pair+=probs[i]*probs[j]*loo(p,bits[[i,j]],arbitrary[[i,j]])
    np.testing.assert_allclose(pair,gradient,atol=1e-12)
    projected=neutralize(gradient,p,f)
    for r in [0,1]:assert abs(projected[f==r]@(p*(1-p))[f==r])<1e-12
    return dict(analytic_count_exact_pass=True,score_function_finite_difference_max_error=float(np.max(abs(gradient-fd))),
                exhaustive_K2_LOO_pass=True,count_neutral_projection_pass=True)


def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((OLD/'protocol.json').read_text())['tasks'];panel=[];bindings={}
    for task in tasks:
        for t in [1,250,1000]:
            p=OLD/task['id']/f'validation_u2000_raw_t{t}.npz'
            panel.append(dict(task=task['id'],dataset=task['parent']['dataset'],train_seed=task['parent']['seed'],update=2000,weights='raw',t=t,path=str(p),checkpoint=task['parent']['checkpoint']))
            bindings[str(p)]=sha(p)
    for p in [Path(__file__).resolve(),Path(heat.__file__),Path(joint.__file__),Path(heat.old.__file__),Path(heat.old.lib.__file__),ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json',OLD/'protocol.json']:
        bindings[str(p)]=sha(p)
    for task in tasks:
        assert sha(task['parent']['data'])==task['parent']['data_sha256']
        bindings[task['parent']['data']]=sha(task['parent']['data'])
    save('protocol.json',dict(frozen_before_new_samples=True,panel=panel,K=K,repeats=REPEATS,random_root=RANDOM_ROOT,
        scope='12 cached raw A/u2000 conditions x 128 independent conditional clean hard-graph draws = 1536; no model calls/parameter updates/reverse chains',
        objective='E[C_joint(G)]+E[C_count(G)]; coefficient 1 each only as an untuned diagnostic combination; report components separately, not a final training coefficient approval',
        joint='Unchanged joint heat-degree Gaussian per-type MMD2/2 with old reference scales/bandwidths',
        count='Mean over declared R relations of (N_r-m_r)^2/max(1,m_r)^2; exact Poisson-binomial first two moments and derivative; fixed-zero relations retained in denominator',
        estimator='K16 leave-one-out score-function gradient wrt Bernoulli logits, no std normalization, clipping or straight-through; independent repeat random streams',
        interpretation='No exact real-graph joint gradient. Split halves are independent Monte Carlo draws from same p, not independent noisy inputs or training seeds; do not call split cosine unbiased SNR or proof of descent.',
        count_neutral='Diagnostic Euclidean projection off v_r=p(1-p) per relation; first-order expected-count-neutral LOGIT directions, not an optimizer or change to gradient estimator',
        decision='Do not train if feedback remains unreliable; no weight/sample-count search based on output. Positive score sensitivity is not reliable model gradient evidence.',
        budget='single CPU thread; timeout600s; GPU0',bindings=bindings))
    tests=checks();save('checks.json',tests)
    scaling=json.loads((ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json').read_text())
    meta={};rows=[];relations=[];timings=[]
    for task in tasks:
        ds=task['parent']['dataset'];z=np.load(OLD/task['id']/'validation_u2000_raw_t1.npz')
        data=np.load(task['parent']['data']);types=data['types'];r=len(data['schema'])
        core=heat.old.FastScore(types,z['query'],z['families'],z['labels'],r)
        reference=core.reference;h=heat.heat(heat.adj(core.n,core.q[:,reference.astype(bool)].T),types)['feat']
        rd=joint.degrees(reference,core.q,core.f,core.n,r);df=np.zeros_like(rd,dtype=float)
        for t in np.unique(types):
            sel=types==t;df[sel]=np.log1p(rd[sel])/scaling[ds]['log_degree_scales'][str(t)]/np.sqrt(r)
        refkernels={}
        for t in np.unique(types):
            sel=types==t;s=scaling[ds]
            refkernels[int(t)]=joint.kernel(h[sel],h[sel],s['heat_bandwidth_squared'][str(t)])*joint.kernel(df[sel],df[sel],s['degree_bandwidth_squared'][str(t)])
        meta[task['id']]=dict(core=core,h=h,d=df,k=refkernels,ds=ds,fixed={k:z[k].copy() for k in ['query','families','ids','labels']})

    def score(bits,m):
        core=m['core'];ds=m['ds'];tick=time.perf_counter()
        h=heat.heat(heat.adj(core.n,core.q[:,bits.astype(bool)].T),core.types)['feat']
        d=joint.degrees(bits,core.q,core.f,core.n,core.r);terms=[]
        for t in np.unique(core.types):
            sel=core.types==t;s=scaling[ds]
            df=np.log1p(d[sel])/s['log_degree_scales'][str(t)]/np.sqrt(core.r)
            xx=joint.kernel(h[sel],h[sel],s['heat_bandwidth_squared'][str(t)])*joint.kernel(df,df,s['degree_bandwidth_squared'][str(t)])
            xy=joint.kernel(h[sel],m['h'][sel],s['heat_bandwidth_squared'][str(t)])*joint.kernel(df,m['d'][sel],s['degree_bandwidth_squared'][str(t)])
            terms.append(joint.mmd(xx,m['k'][int(t)],xy))
        timings.append(time.perf_counter()-tick)
        return float(np.mean(terms)),terms

    for index,item in enumerate(panel):
        if time.time()-start>570:raise TimeoutError('CPU budget approaching')
        m=meta[item['task']];c=m['core'];z=np.load(item['path']);p=z['probability']
        for k,v in m['fixed'].items():np.testing.assert_array_equal(z[k],v)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        count_risk,gc=count_exact(p,c.f,c.reference,c.r)
        gradients=[];allcost=[];typecost=[];counts=[];unique=[];mc_count_gradients=[]
        target=np.bincount(c.f,weights=c.reference,minlength=c.r)
        for rep in range(REPEATS):
            rng=np.random.default_rng(np.random.SeedSequence([RANDOM_ROOT,item['train_seed'],item['t'],rep]))
            bits=(rng.random((K,len(p)))<p).astype(np.uint8)
            costs=[];cc=[]
            for b in bits:
                if time.time()-start>580:raise TimeoutError('CPU budget approaching')
                cost,types=score(b,m);costs.append(cost);typecost.append(types)
                nc=np.bincount(c.f,weights=b,minlength=c.r);cc.append(np.mean((nc-target)**2/np.maximum(1,target)**2));counts.append(int(b.sum()))
            gradients.append(loo(p,bits,np.asarray(costs)));mc_count_gradients.append(loo(p,bits,np.asarray(cc)))
            allcost.extend(costs);unique.append(len(np.unique(bits,axis=0)))
        gg=np.asarray(gradients);neutral=neutralize(gg,p,c.f);combined=gg+gc
        rec={k:v for k,v in item.items() if k not in ['path','checkpoint']}
        rec.update(expected_edges=float(p.sum()),sample_edges_mean=float(np.mean(counts)),
                   joint_cost_mean=float(np.mean(allcost)),joint_cost_sd=float(np.std(allcost,ddof=1)),
                   count_expected_cost=count_risk,count_exact_gradient_norm=float(np.linalg.norm(gc)),
                   mean_unique_per16=float(np.mean(unique)),uncertain_slots=int(np.sum((p>0)&(p<1))),
                   joint_count_cosine=cosine(gg.mean(0),gc),
                   joint_count_neutral_fraction=float(np.linalg.norm(neutral.mean(0))/np.linalg.norm(gg.mean(0))) if np.linalg.norm(gg.mean(0))>0 else None)
        for name,array in [('joint',gg),('joint_count_neutral',neutral),('combined',combined),('count_MC',np.asarray(mc_count_gradients))]:
            for k,v in stats(array).items():rec[name+'_'+k]=v
        for rel in range(c.r):
            sel=c.f==rel
            rr=dict(task=item['task'],dataset=item['dataset'],train_seed=item['train_seed'],t=item['t'],relation=rel,
                slots=int(sel.sum()),reference_edges=int(target[rel]),expected_edges=float(p[sel].sum()),
                count_gradient_norm=float(np.linalg.norm(gc[sel])),joint_gradient_norm=float(np.linalg.norm(gg.mean(0)[sel])),
                joint_split_cosine=cosine(gg[:4].mean(0)[sel],gg[4:].mean(0)[sel]))
            relations.append(rr)
        np.savez_compressed(OUT/f'panel_{index:02d}.npz',probability=p,joint_costs=np.asarray(allcost),type_costs=np.asarray(typecost),
            joint_logit_gradients=gg,count_exact_logit_gradient=gc,count_MC_logit_gradients=np.asarray(mc_count_gradients),sample_edge_counts=np.asarray(counts))
        rows.append(rec);table('panel_summary',rows);table('per_relation',relations)
        print(json.dumps(dict(completed=index+1,total=len(panel),elapsed=time.time()-start,task=item['task'],t=item['t'],joint_split=rec['joint_split_half_cosine'])),flush=True)
    for p,h in bindings.items():assert sha(p)==h
    save('receipt.json',dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        threads=1,conditions=len(rows),conditional_hard_graph_draws=len(rows)*REPEATS*K,
        median_score_seconds=float(np.median(timings)),model_calls=0,GPU_hours=0,training_updates=0,reverse_chains=0,source_assets_unchanged=True))


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure.json',{'traceback':traceback.format_exc()})
        raise
