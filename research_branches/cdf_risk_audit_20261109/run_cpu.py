"""Saved assets and probability arithmetic only: no Torch/model/RNG graph draws."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, itertools, json, math, resource, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cdf_risk_audit_20261109/attempt2'
START=time.monotonic(); HASHES={}
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(600,610))
def check():
    if time.monotonic()-START>600:raise TimeoutError('600 second scientific limit')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p);HASHES[str(p)]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def load(p):
    with np.load(bind(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(name, obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in obj for k in r)));w.writeheader();w.writerows(obj)
def ratio(a,b):return float(a/b) if b else None
def pmf(p):
    h=np.ones((len(p),1))
    for j in range(p.shape[1]):
        v=p[:,j:j+1];out=np.zeros((len(p),h.shape[1]+1));out[:,:-1]+=(1-v)*h;out[:,1:]+=v*h;h=out
    return h
def remove_factor(h,p):
    """Stable-direction Bernoulli polynomial division; no clipping/renormalization."""
    cap=h.shape[1]-1;u=np.zeros((len(p),cap));left=p<=.5;right=~left
    if left.any():
        a=h[left];v=p[left];prev=np.zeros(len(v))
        for k in range(cap):u[left,k]=(a[:,k]-v*prev)/(1-v);prev=u[left,k]
    if right.any():
        a=h[right];v=p[right];nxt=np.zeros(len(v))
        for k in range(cap-1,-1,-1):u[right,k]=(a[:,k+1]-(1-v)*nxt)/v;nxt=u[right,k]
    reconstructed=np.zeros_like(h);reconstructed[:,:-1]+=(1-p[:,None])*u;reconstructed[:,1:]+=p[:,None]*u
    assert np.max(np.abs(reconstructed-h))<2e-11
    assert np.min(u)>-2e-11 and np.max(np.abs(u.sum(1)-1))<2e-10
    return u

DATA=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets'
def asset(ds):
    z=load(DATA/(ds+'.npz'));types=z['types'];schema=z['schema'];e=z['edges'];e=e.T if e.shape[0]!=2 else e
    ref=np.zeros((len(types),len(types)),dtype=int);ref[e[0],e[1]]=ref[e[1],e[0]]=1
    return types,schema,ref
def make_groups(types,schema,ref,query,families):
    keys=query[0]*len(types)+query[1];assert len(set(np.minimum(*query)*len(types)+np.maximum(*query)))==len(keys)
    groups=[]
    for r,pair in enumerate(schema):
        # Match original PB: skip fully fixed relations.
        u,v=np.triu_indices(len(types),1)
        allowed=np.all(np.sort(types[np.array([u,v])].T,axis=1)==pair,axis=1)
        count=ref[u[allowed],v[allowed]].sum();capacity=int(allowed.sum())
        if count==0 or count==capacity:continue
        ids=np.flatnonzero(families==r)
        assert len(ids)==capacity
        assert np.array_equal(np.sort(types[query[:,ids]].T,axis=1),np.tile(pair,(len(ids),1)))
        for typ in sorted(set(pair)):
            nodes=np.flatnonzero(types==typ);ix=np.array([ids[np.any(query[:,ids]==i,axis=0)] for i in nodes]);cap=ix.shape[1]
            deg=np.array([ref[query[0,j],query[1,j]].sum() for j in ix],int)
            target=np.cumsum(np.bincount(deg,minlength=cap+1)/len(nodes))[:cap]
            rec=dict(relation=int(r),node_type=int(typ),nodes=nodes,indices=ix,capacity=cap,target=target)
            if pair[0]==pair[1]:
                look={int(x):i for i,x in enumerate(nodes)}
                rec['shared']=(np.array([look[int(x)] for x in query[0,ids]]),np.array([look[int(x)] for x in query[1,ids]]),ids)
            groups.append(rec)
    return groups
def evaluate(p,groups):
    result=[]
    for g in groups:
        h=pmf(p[g['indices']]);n=len(h);a=h.cumsum(1)[:,:-1]
        bias=.5*np.mean((a.mean(0)-g['target'])**2)
        diagonal=(a*(1-a)).sum(0)/n**2;shared=np.zeros(g['capacity'])
        if 'shared' in g:
            i,j,ids=g['shared'];prob=p[ids]
            ui=remove_factor(h[i],prob);uj=remove_factor(h[j],prob)
            shared=2*(prob[:,None]*(1-prob[:,None])*ui*uj).sum(0)/n**2
        diag=.5*float(diagonal.mean());cov=.5*float(shared.mean())
        assert diag>=-1e-12 and cov>=-1e-12
        result.append(dict(relation=g['relation'],node_type=g['node_type'],nodes=n,capacity=g['capacity'],bias=float(bias),variance_diagonal=diag,variance_shared=cov,variance=diag+cov,risk=float(bias+diag+cov)))
    values=np.array([[r[k] for k in ('bias','variance','risk')] for r in result]).mean(0)
    return values,result
def tests():
    # Exhaustive finite distributions, including same-type shared-edge covariance.
    rows=[]
    for types in (np.array([0,0,0]),np.array([0,1,1])):
        schema=np.array([[0,0]]) if max(types)==0 else np.array([[0,1]])
        u,v=np.triu_indices(3,1);mask=np.array([tuple(sorted((types[a],types[b])))==tuple(schema[0]) for a,b in zip(u,v)])
        q=np.array([u[mask],v[mask]]);ref=np.zeros((3,3),int);ref[q[0,0],q[1,0]]=ref[q[1,0],q[0,0]]=1
        gs=make_groups(types,schema,ref,q,np.zeros(q.shape[1],int))
        for vals in itertools.product([0.,.2,.5,.8,1.],repeat=q.shape[1]):
            p=np.array(vals);ana,_=evaluate(p,gs);exact=0.
            for bits in itertools.product((0,1),repeat=len(p)):
                b=np.array(bits);weight=np.prod(np.where(b,p,1-p));loss=[]
                for g in gs:
                    d=b[g['indices']].sum(1);cdf=np.mean(d[:,None]<=np.arange(g['capacity']),axis=0)
                    loss.append(.5*np.mean((cdf-g['target'])**2))
                exact+=weight*np.mean(loss)
            assert abs(ana[2]-exact)<1e-12
            rows.append(dict(homogeneous=bool(max(types)==0),p=list(vals),risk=float(ana[2]),exact=float(exact)))
    save('tests',rows)

def graphs():
    base=ROOT/'results/joint_output_formal_projected_20261108T043000'
    inv=read(base/'graph_inventory.json');jobs=[]
    for r in inv:
        metric_dir=Path(r['metric_dir']) if 'metric_dir' in r else base/'evaluation'/r['task_id']/'u3000'/str(r['sample_seed'])
        jobs.append(dict(r,panel='matched_four_grid',file=r['graph_file'],expected_sha=r['graph_sha256'],metric=str(metric_dir/'original.json')))
    confirm=ROOT/'results/pb_confirmation_formal_20261107T005911'
    for r in read(confirm/'evaluation/inventory.json'):
        metric=confirm/'evaluation'/r['task_id']/'u3000'/str(r['sample_seed'])/'original.json';m=read(metric)
        jobs.append(dict(r,panel='new_seed_confirmation',metric=str(metric),expected_sha=m['graph_binding']['graph_sha256']))
    assert len(jobs)==144 and len({r['file'] for r in jobs})==144
    save('graph_inventory',jobs)
    per=[];pool={};scalars=[];allmetrics=[];cfgs={ds:asset(ds) for ds in ('IMDB','PubMed')}
    for job in jobs:
        check();ds=job['dataset'];types,schema,ref=cfgs[ds]
        z=load(job['file']);assert HASHES[str(Path(job['file']))]==job['expected_sha'];assert np.array_equal(z['types'],types) and np.array_equal(z['schema'],schema)
        edges=z['edges'];edges=edges if edges.shape[0]==2 else edges.T
        a=np.zeros_like(ref);a[edges[0],edges[1]]=a[edges[1],edges[0]]=1
        assert a.trace()==0 and a.sum()==2*edges.shape[1]
        u,v=np.triu_indices(len(types),1);q=np.array([u,v]);look={tuple(x):i for i,x in enumerate(schema)};fam=np.array([look[tuple(sorted((types[x],types[y])))] for x,y in zip(u,v)])
        gs=make_groups(types,schema,ref,q,fam);cdfs=[];target=[]
        for g in gs:
            ids=g['indices'];d=a[q[0,ids],q[1,ids]].sum(1);cdfs.append(np.mean(d[:,None]<=np.arange(g['capacity']),axis=0));target.append(g['target'])
        ident={k:job[k] for k in ('panel','dataset','train_seed','arm','sample_seed')};key=tuple(ident[k] for k in ('panel','dataset','train_seed','arm'))
        pool.setdefault(key,[]).append((cdfs,target))
        pb=float(np.mean([.5*np.mean((h-t)**2) for h,t in zip(cdfs,target)]))
        m=read(job['metric']);b=m['graph_binding'];assert b['graph_sha256']==job['expected_sha'] and b['updates']==3000 and b['readout']=='EMA0.99' and b['R']==1 and b['T']==1000
        for k in ('dataset','train_seed','sample_seed'):assert m[k]==job[k]
        st=m['statistics'];assert st['edges']==edges.shape[1]
        per.append(dict(ident,PB=pb,edges=st['edges'],triangles=st['triangles'],lcc=st['lcc'],degree_W1=m['distance']['main_incident4_W1'],joint_TV=m['distance']['main_joint_degree_TV']))
        for section in ('distance','absolute_error'):
            for k,value in m[section].items():
                if value is None or isinstance(value,(int,float)):
                    allmetrics.append(dict(ident,section=section,metric=k,value=value))
        rref=read(confirm/'evaluation'/f'reference_{ds}.json')['statistics']
        for metric in ('edges','triangles','lcc','components','isolates','maximum_degree'):
            scalars.append(dict(ident,metric=metric,value=st[metric],reference=rref[metric]))
    summaries=[]
    for key,items in pool.items():
        assert len(items)==4
        errors=[];bias=[];var=[]
        for j in range(len(items[0][0])):
            h=np.array([x[0][j] for x in items]);t=items[0][1][j]
            errors.append(.5*np.mean((h-t)**2));bias.append(.5*np.mean((h.mean(0)-t)**2));var.append(.5*np.mean((h-h.mean(0))**2))
        risk=float(np.mean(errors));b=float(np.mean(bias));v=float(np.mean(var));assert abs(risk-b-v)<1e-12
        summaries.append(dict(zip(('panel','dataset','train_seed','arm'),key),samples=4,mean_graph_PB=risk,pooled_bias=b,sample_dispersion=v,dispersion_fraction=ratio(v,risk)))
    scalar_summary=[]
    for key in sorted(set(tuple(r[k] for k in ('panel','dataset','train_seed','arm','metric')) for r in scalars)):
        rr=[r for r in scalars if tuple(r[k] for k in ('panel','dataset','train_seed','arm','metric'))==key];x=np.array([r['value'] for r in rr]);ref=rr[0]['reference'];mean=float(x.mean());var=float(x.var());mse=float(np.mean((x-ref)**2));bias=(mean-ref)**2
        assert abs(mse-var-bias)<1e-6*max(1,mse)
        scalar_summary.append(dict(zip(('panel','dataset','train_seed','arm','metric'),key),samples=4,values=x.tolist(),reference=ref,mean=mean,sample_SD=float(x.std(ddof=1)),MAE=float(np.mean(abs(x-ref))),bias_squared=bias,variance_population=var,MSE=mse,variance_fraction=ratio(var,mse),below=int((x<ref).sum()),above=int((x>ref).sum())))
    # Reuse and verify previously computed PB values rather than inventing a new metric.
    prior=read(base/'hard_graph_PB_PS_model_summary.json')
    for old in prior:
        row=next(r for r in summaries if r['panel']=='matched_four_grid' and all(r[k]==old[k] for k in ('dataset','train_seed','arm')))
        assert abs(row['mean_graph_PB']-old['hard_graph_PB_mean'])<1e-12
    save('per_graph',per);save('graph_CDF_decomposition',summaries);save('scalar_decomposition',scalar_summary);save('existing_all_scalar_metrics',allmetrics)
    oldps=read(base/'hard_graph_PS_bias_dispersion.json');save('existing_PS_decomposition',oldps)

def probabilities():
    base=ROOT/'results/input_mask_screen_20261106';bindings=read(base/'bindings.json');prior=read(ROOT/'results/degree_uncertainty_cpu_20261106/receipt.json')['source_hashes'];previous=read(ROOT/'results/pb_degree_preparation_20261106/saved_probability_checks.json')
    assert len(bindings)==24
    rows=[];details=[];directions=[];edges=[];binding=[]
    for b in bindings:
        check();ds=b['task'].split('_')[0];pfile=base/f'{b["task"]}_{b["kind"]}_{b["t"]}_FULL.npz';z=load(pfile);assert HASHES[str(pfile)]==prior[str(pfile)]
        types,schema,ref=asset(ds);assert HASHES[str(DATA/(ds+'.npz'))]==b['data_sha256']
        p=z['probabilities'].astype(float);q=z['query'];fam=z['family'];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        gs=make_groups(types,schema,ref,q,fam);v,rr=evaluate(p,gs)
        old=next(r for r in previous if r['task']==b['task'] and r['kind']==b['kind'] and r['t']==b['t'] and r['dtype']=='torch.float64');assert abs(v[0]-old['loss'])<1e-12
        ident=dict(dataset=ds,task=b['task'],model_family='STRUCTURE_late: A plus triangle/component input; FULL channels',readout='EMA0.99',updates=3000,train_seed=int(b['task'].split('_')[1]),kind=b['kind'],t=b['t'])
        rows.append(dict(ident,bias=float(v[0]),variance=float(v[1]),risk=float(v[2]),variance_fraction=ratio(v[1],v[2]),variance_to_bias=ratio(v[1],v[0])))
        details.extend(dict(ident,**r) for r in rr);binding.append(dict(b,probability_file=str(pfile),sha256=HASHES[str(pfile)]))
        # No new samples. Derivatives of arithmetic at archived probabilities only.
        def shifted(direction,h):
            e=np.exp(h*direction);return p*e/(1-p+p*e)
        d=2*p-1;centered=d.copy()
        for rel in np.unique(fam):
            mask=fam==rel;weights=p[mask]*(1-p[mask]);den=weights.sum()
            if den:centered[mask]-=np.sum(weights*d[mask])/den
        for name,direction in [('common_logit_increase',np.ones_like(p)),('sharpen_logits',d),('relation_mass_neutral_sharpen',centered)]:
            derivatives=[]
            for h in (1e-3,5e-4):derivatives.append((evaluate(shifted(direction,h),gs)[0]-evaluate(shifted(direction,-h),gs)[0])/(2*h))
            assert np.max(abs(derivatives[0]-derivatives[1]))<2e-6
            directions.append(dict(ident,direction=name,bias_derivative=float(derivatives[1][0]),variance_derivative=float(derivatives[1][1]),risk_derivative=float(derivatives[1][2]),h_consistency=float(np.max(abs(derivatives[0]-derivatives[1])))))
        # Predeclared small per-edge panel, not outcome-selected. One reference positive
        # and one negative per mutable relation in the four forward t500 inputs.
        if b['kind']=='forward' and b['t']==500:
            labels=ref[q[0],q[1]]
            for rel in np.unique(fam):
                for label in (0,1):
                    candidates=np.flatnonzero((fam==rel)&(labels==label))
                    if not len(candidates):continue
                    chosen=min(candidates,key=lambda i:hashlib.sha256(f'20261109-cdf-risk:{b["task"]}:{rel}:{label}:{int(q[0,i])}:{int(q[1,i])}'.encode()).hexdigest())
                    direction=np.zeros_like(p);direction[chosen]=1
                    der=(evaluate(shifted(direction,1e-3),gs)[0]-evaluate(shifted(direction,-1e-3),gs)[0])/.002
                    edges.append(dict(ident,relation=int(rel),label=int(label),slot=int(chosen),u=int(q[0,chosen]),v=int(q[1,chosen]),p=float(p[chosen]),bias_derivative=float(der[0]),variance_derivative=float(der[1]),risk_derivative=float(der[2]),variance_pushes_hardening=bool(der[1]*(2*p[chosen]-1)<0)))
        save('progress',dict(completed_probabilities=len(rows),seconds=time.monotonic()-START))
    save('probability_binding',binding);save('conditional_summary',rows);save('conditional_groups',details);save('directional_derivatives',directions);save('selected_edge_derivatives',edges)

def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(scope='CPU-only existing assets; no model imports/updates/new graph draws',probabilities='all 24 archived STRUCTURE_late FULL inputs from input_mask_screen_20261106: A+triangle/component inputs EMA3000; task names misleadingly end A_late; forward100/500/800 and archived generated99/499/799 separately',graphs='128 matched O/A/PB/JK/PS graphs plus 16 confirmation graphs; four samples/model; panels kept separate',group_weights='original PB equal mutable relation-endpoint groups; 0.5 squared CDF / capacity',covariance='exact shared-edge correction; polynomial division in stable direction; no clipping',directions=['common_logit_increase','sharpen_logits','relation_mass_neutral_sharpen'],selected_edges='four forward t500 states; one positive and negative per relation by fixed SHA ranking, not score; derivative wrt logit',limits=dict(CPU_threads=1,worker_seconds=600,memory_GiB=2,GPU_hours=0),interpretation='finite four-graph decomposition, not unbiased population variance or independent training repeats; conditional STRUCTURE inputs not ordinary A/PB/JK/PS endpoint predictions; no causal attribution'))
    tests();graphs();probabilities()
    assert all(sha(p)==s for p,s in HASHES.items())
    HASHES[str(Path(__file__))]=sha(__file__)
    save('binding',HASHES)
    save('receipt',dict(status='complete',wall_seconds=time.monotonic()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,updates=0,new_graphs=0))
    print(json.dumps(read(OUT/'receipt.json')))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT,help='Must be a new directory; old evidence is never overwritten.')
    OUT=parser.parse_args().output.resolve()
    try:main()
    except Exception:
        import traceback
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
