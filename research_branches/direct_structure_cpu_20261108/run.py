import sys,time,json,csv,hashlib,resource,itertools,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
torch.set_num_threads(1)
from score import DirectScore
from metrics import brute
R=Path(__file__).resolve().parents[2];O=R/'results/direct_structure_cpu_20261108';E=R/'results/hard_structure_edge_direction_cpu_20261108';PARENT=R/'results/hard_structure_response_20261108T182154'
read=lambda p:json.loads(Path(p).read_text());P=read(O/'protocol.json');start=time.time();bindings={};rows=[];rels=[];costs=[]
status=dict(status='running',graphs=0,flips=0,independent_real_checks=0,GPU=0,model_calls=0,updates=0,new_real_draws=0)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):bindings[str(p)]=sha(p);return p
def save(n,x):(O/n).write_text(json.dumps(x,indent=2))
def write(n,x):
    if not x:return
    with (O/n).open('w') as f:w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def check():
    if time.time()>=P['science_stop_epoch'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>P['max_RSS_bytes']:raise TimeoutError('budget')
def independent(types,q,f,y,b,r):
    # Independent full adjacency, sorted degree W1 and tuple-count TV.
    n=len(types);a=np.zeros((n,n),dtype=np.int64);ar=np.zeros_like(a);a[q[0],q[1]]=b;a[q[1],q[0]]=b;ar[q[0],q[1]]=y;ar[q[1],q[0]]=y
    tr=int(np.trace(a@a@a)//6);target=int(np.trace(ar@ar@ar)//6)
    v=np.zeros((n,r),dtype=int);vr=np.zeros_like(v)
    for i,(u,w) in enumerate(q.T):v[u,f[i]]+=int(b[i]);v[w,f[i]]+=int(b[i]);vr[u,f[i]]+=int(y[i]);vr[w,f[i]]+=int(y[i])
    wd=[];tv=[]
    for t in np.unique(types):
        ix=types==t;wd.append(np.mean(abs(np.sort(a.sum(1)[ix])-np.sort(ar.sum(1)[ix]))));c=Counter(map(tuple,v[ix]));cr=Counter(map(tuple,vr[ix]));tv.append(.5*sum(abs(c.get(k,0)-cr.get(k,0)) for k in c.keys()|cr.keys())/int(ix.sum()))
    return dict(total_degree_W1=float(np.mean(wd)),joint_degree_TV=float(np.mean(tv)),triangle_abs_error=float(abs(tr-target)))
def tiny():
    types=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=types[q[0]]+types[q[1]];states=np.array(list(itertools.product([0,1],repeat=6)))
    refs=[np.zeros(6,dtype=int),np.ones(6,dtype=int),np.array([1,0,1,1,0,0]),np.array([0,1,0,1,0,1])];errs=[];delta_error=0.;checks=0
    for ri,y in enumerate(refs):
        scores=[];est=[]
        for b in states:
            obj=DirectScore(types,q,f,y,b,3);base=brute(types,q,f,y,b,3);assert abs(obj.base-obj.combine(base))<1e-12
            scores.append(obj.combine(base));change,dd=obj.differences();est.append(change*dd['flip_sign'])
            for e in range(6):
                flipped=b.copy();flipped[e]=1-flipped[e];expected=obj.combine(brute(types,q,f,y,flipped,3))-obj.combine(base)
                delta_error=max(delta_error,abs(change[e]-expected));assert abs(change[e]-expected)<1e-12;checks+=1
        scores=np.array(scores);est=np.array(est)
        for pi,p in enumerate([np.full(6,.5),np.array([.01,.1,.3,.7,.9,.99]),np.array([.8,.2,.6,.4,.1,.9])]):
            mass=np.prod(np.where(states,p,1-p),axis=1);g=np.sum(mass[:,None]*est,axis=0)*p*(1-p);sf=np.sum(mass[:,None]*scores[:,None]*(states-p),axis=0)
            z=np.log(p/(1-p));fd=[]
            for e in range(6):
                z1=z.copy();z0=z.copy();z1[e]+=1e-5;z0[e]-=1e-5;p1=1/(1+np.exp(-z1));p0=1/(1+np.exp(-z0));fd.append(np.dot(np.prod(np.where(states,p1,1-p1),axis=1)-np.prod(np.where(states,p0,1-p0),axis=1),scores)/2e-5)
            err=float(np.max(abs(g-fd)));assert err<1e-8;assert np.max(abs(g-sf))<1e-12
            errs.append(dict(reference=ri,probability_set=pi,finite_difference_max_error=err,score_function_max_error=float(np.max(abs(g-sf)))))
        p=np.array([0.,1.,.25,.75,0.,1.]);mass=np.prod(np.where(states,p,1-p),axis=1);g=np.sum(mass[:,None]*est,axis=0)*p*(1-p);assert (g[[0,1,4,5]]==0).all()
    save('tiny_checks.json',dict(hard_states=256,flip_checks=checks,max_delta_error=delta_error,exact_expectations=errs,zero_one_probability_boundary_passed=True))
def stats(score,d,prob):
    good=score < -1e-12;mass=abs(score)*prob*(1-prob);den=float(mass[good].sum());neg=d < -1e-12;pos=d >1e-12;common=~pos.any(1)&neg.any(1);damage=~neg.any(1)&pos.any(1)
    out=dict(slots=len(score),improving=int(good.sum()),mass=den,common_improvements=int(common.sum()),common_missed=int((common&~good).sum()),nonbetter_damage_rewarded=int((damage&good).sum()),tradeoffs_rewarded=int((good&pos.any(1)&neg.any(1)).sum()))
    for j,k in enumerate(['triangle_abs_error','total_degree_W1','joint_degree_TV']):
        bad=good&pos[:,j];out[k+'_bad']=int(bad.sum());out[k+'_bad_mass']=float(mass[bad].sum());out[k+'_bad_fraction']=float(bad.sum()/good.sum()) if good.any() else None;out[k+'_bad_mass_fraction']=float(mass[bad].sum()/den) if den else None
    return out
def main():
    for p,h in read(O/'source_binding.json').items():assert sha(p)==h
    tiny();source=list(csv.DictReader(bind(E/'graph_summary.csv').open()));tasks=read(bind(R/'results/output_boundary_replay_20261108T081654/tasks.json'));data={};maxerr=0.;scales=[]
    for ds in ['IMDB','PubMed']:
        cfg=next(x for x in tasks if x['dataset']==ds)['row']['config']['datasets'][ds];path=R/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/cfg['file'];assert sha(path)==cfg['sha256']
        with np.load(bind(path)) as z:types=z['types'].copy()
        data[ds]=(types,{int(t):v for t,_,v in torch.load(bind(PARENT/f'{ds}_fixed_inputs.pt'),map_location='cpu')['inputs']})
    for row in source:
        check();ds=row['condition'].split('_')[0];types,inputs=data[ds];inp=inputs[int(row['t'])];q=np.asarray(inp['query']);f=np.asarray(inp['f']);y=np.asarray(inp['labels']);r=int(f.max())+1
        name=f"{row['condition']}_{row['arm']}_t{row['t']}_draw{row['draw']}"
        with np.load(bind(E/f'{name}_flips.npz')) as z:old={k:z[k].copy() for k in z.files}
        with np.load(bind(PARENT/f"{row['condition']}_{row['arm']}_t{row['t']}.npz")) as z:prob=z['probability'].copy()
        b=((1-old['flip_sign'])/2).astype(int);before=time.perf_counter();obj=DirectScore(types,q,f,y,b,r);score,d=obj.differences();elapsed=time.perf_counter()-before
        keys=['triangle_abs_error','total_degree_W1','joint_degree_TV'];delta=np.stack([d[k] for k in keys],axis=1)
        for k in keys:assert np.max(abs(d[k]-old[k+'_flip']))<1e-10
        assert abs(obj.metrics.base_w1-float(row['degree_W1']))<1e-10;assert abs(obj.metrics.base_tv-float(row['joint_TV']))<1e-10
        assert obj.metrics.triangles==int(row['triangles']);assert obj.metrics.ref_triangles==int(row['reference_triangles'])
        for e in [0,len(b)-1]:
            flipped=b.copy();flipped[e]=1-flipped[e];val=obj.combine(independent(types,q,f,y,flipped,r));err=abs(val-obj.base-score[e]);maxerr=max(maxerr,err);assert err<1e-10;status['independent_real_checks']+=1
        ids={k:row[k] for k in ['condition','arm','t','draw','triangle_state']};ids['dataset']=ds
        for method,s in [('MMD',old['score_flip']),('DIRECT',score)]:
            ss=stats(s,delta,prob);rows.append(dict(**ids,method=method,**ss))
            if method=='DIRECT':assert ss['common_missed']==0 and ss['nonbetter_damage_rewarded']==0
            for family in np.unique(f):
                ix=f==family;rels.append(dict(**ids,method=method,relation=int(family),**stats(s[ix],delta[ix],prob[ix])))
        costs.append(dict(**ids,slots=len(b),seconds=elapsed,degree_scale=obj.degree_scale,triangle_scale=obj.triangle_scale,base_score=obj.base))
        grad=prob*(1-prob)*score*d['flip_sign'];assert np.isfinite(grad).all()
        np.savez_compressed(O/f'{name}_direct.npz',score_flip=score,logit_conditional_estimate=grad)
        status['graphs']+=1;status['flips']+=len(b);save('progress.json',status)
    write('graph_summary.csv',rows);write('relations.csv',rels);write('costs.csv',costs);save('asset_binding.json',bindings)
    assert all(sha(p)==h for p,h in bindings.items())
    production=read(R/'results/joint_output_formal_projected_20261108T043000/source_binding.json');assert all(sha(p)==h for p,h in production.items());assert not torch.cuda.is_initialized()
    status.update(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,max_independent_score_error=maxerr,assets_unchanged=True,production_files_unchanged=len(production));save('receipt.json',status);print(json.dumps(status),flush=True)
try:main()
except Exception as e:
    status.update(status='failed',error=repr(e),seconds=time.time()-start);save('receipt.json',status);write('graph_summary_partial.csv',rows);traceback.print_exc();sys.exit(1)
