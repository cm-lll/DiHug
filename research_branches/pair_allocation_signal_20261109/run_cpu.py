"""Frozen paired-position CPU diagnostic; no denoiser, updates, or new graph draws."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,resource,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('joint_degree_frozen',ROOT/'research_branches/joint_degree_distance_20261109/run.py')
jdmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(jdmod)
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
from score import RootedScore
sys.path.insert(0,str(ROOT/'research_branches/explicit_structure_cpu_20261109'))
from explicit_score import ExplicitScore,connectivity
OUT=ROOT/'results/pair_allocation_signal_20261109'
FULL=ROOT/'results/joint_degree_full_signal_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
START=time.monotonic();BIND={};EPS=1e-12

def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x and isinstance(x[0],dict):
        with (OUT/(n+'.csv')).open('w',newline='') as h:
            w=csv.DictWriter(h,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    if time.monotonic()-START>540:raise TimeoutError('540s science budget')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:raise MemoryError('2GiB RSS cap')
def cos(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/d) if d else None
def sigmoid(x):return 1/(1+np.exp(-x))
def adjacency(q,b,n):
    a=np.zeros((n,n),bool);a[q[0],q[1]]=b.astype(bool);return a|a.T
def flip(raw,a,q,types,channel,slot,rel):
    u,v=map(int,q[:,slot]);s=-1 if a[u,v] else 1
    raw[u,rel]+=s;raw[v,rel]+=s
    for w in np.flatnonzero(a[u]&a[v]):
        raw[u,channel[types[v],types[w]]]+=s
        raw[v,channel[types[u],types[w]]]+=s
        raw[w,channel[types[u],types[v]]]+=s
    a[u,v]=a[v,u]=not a[u,v]
    assert raw.min()>=0
def describe(sc,jd,raw,a):
    val=sc.terms(raw);jj,_=jd.terms(raw[:,:10]);tri=float(sc.triangle_counts(raw).sum())
    result={k:float(v[0]) for k,v in val.items()}
    result.update(joint_distance=jj['joint'],score=result['total']+(jj['joint']-result['joint_TV'])/8,
                  triangle_count=tri,triangle_error=abs(tri-float(sc.reftri.sum())))
    result.update(connectivity(a));return result

def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600)
    save('protocol',dict(track='CORE training-objective feasibility support',
        scope='PubMed0422 two historical raw2040 parents CE/PS; t1000; two archived hard draws each. Not independent training seeds.',
        pairs='16 unique Cartesian real/nonedge pairs per eligible relation, uniform without replacement, SHA256 fixed seed per state/relation. No label/score selection.',
        target='Fixed background, L=(1-q)F10+qF01; q=sigmoid(z_f-z_e). Score unchanged. Relation-equal, pair-equal frozen empirical objective.',
        comparator='Same backgrounds/pairs independent Bernoulli risk over00/10/01/11; archived full-coordinate signal cosine only, no matching-scale efficacy claim.',
        limits=dict(threads=1,RSS_MiB=2048,science_seconds=540,hard_seconds=600,GPU=0),
        decision='No automatic model calls/training; report position differentiation, costs, scale and estimand changes. Never equate zero logit-gradient sum with conserved expected edges.',
        initial_read_error='Before this worker: requested nonexistent binding.json from old full-signal output; actual completed audit uses completion.output_hashes and verified_prior_bindings. No science computation repeated.'))
    completion=read(FULL/'completion.json');oldbinding=read(FULL/'verified_prior_bindings.json')
    def checked(p,table):
        p=bind(p);assert table[str(p)]==BIND[str(p)],str(p);return p
    fixture=checked(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz',oldbinding)
    with np.load(fixture) as z:types=z['types'].copy();fq=z['query'].copy();ff=z['families'].copy();fy=z['reference'].copy()
    inpfile=checked(OLD/'PubMed_fixed_inputs.pt',oldbinding)
    inp=next(x for t,seed,x in torch.load(inpfile,map_location='cpu',weights_only=False)['inputs'] if t==1000)
    q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));n=len(types)
    i=np.argsort(q[0]*n+q[1]);j=np.argsort(fq[0]*n+fq[1])
    assert np.array_equal(q[:,i],fq[:,j]) and np.array_equal(f[i],ff[j]) and np.array_equal(y[i],fy[j])
    core=RootedScore(types,q,f,y,10);sc=ExplicitScore(core)
    jd=jdmod.JointDegree(dict(types=types,query=q,families=f,reference=y))
    channel=np.full((4,4),-1,int)
    for k,(ta,tb) in enumerate(core.triangle_pairs):channel[ta,tb]=channel[tb,ta]=10+k
    refconn=connectivity(adjacency(q,y,n));states=[];plan=[]
    for parent in ['CE','PS']:
        for draw in [0,1]:
            tag=f'PubMed_{parent}_CE_t1000_draw{draw}'
            with np.load(checked(FULL/(tag+'.npz'),completion['output_hashes'])) as z:data={k:z[k].copy() for k in z.files}
            p,b=data['probability'],data['bits']
            assert np.array_equal(data['relation'],f) and np.array_equal(data['reference'],y)
            assert np.all((p>0)&(p<1)) and np.isin(b,[0,1]).all()
            oldp=checked(OLD/f'PubMed_{parent}_CE_t1000.npz',oldbinding)
            with np.load(oldp) as z:assert np.array_equal(p,z['probability'].astype(float))
            pairs=[];empty=[]
            for r in range(10):
                live=np.flatnonzero((f==r)&(b==1));absent=np.flatnonzero((f==r)&(b==0))
                if not len(live) or not len(absent):empty.append(r);continue
                seed=int.from_bytes(hashlib.sha256(f'paired_position_v1|{parent}|{draw}|{r}'.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed);count=min(16,len(live)*len(absent))
                ids=rng.choice(len(live)*len(absent),count,replace=False)
                for ix in ids:pairs.append(dict(relation=r,e=int(live[ix//len(absent)]),f=int(absent[ix%len(absent)]),seed=seed,within_relation_count=count))
            states.append((tag,data,pairs));plan.append(dict(state=tag,empty_relations=empty,pairs=pairs))
    save('frozen_pairs',plan) # frozen before scores of any candidate are read/computed
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py'):bind(path)
    bind(Path(__file__));save('initial_binding',BIND)
    records=[];summaries=[];tests=[];vectors={};labelrows=[];relrows=[]
    features=['joint_distance','relation_W1','joint_TV','low','hub','typed_triangle_L1','node_triangle_W1','triangle_error','total_degree_W1']
    for state_index,(tag,data,pairs) in enumerate(states):
        limits();tick=time.monotonic();p,b=data['probability'],data['bits'];z=np.log(p)-np.log1p(-p)
        with torch.no_grad():raw=core.features(torch.from_numpy(b)).numpy()[:,:core.base_width]
        a=adjacency(q,b,n);base=describe(sc,jd,raw,a)
        pg=np.zeros(len(p));ug=np.zeros(len(p));active=len({x['relation'] for x in pairs});local=[]
        for pi,pair in enumerate(pairs):
            limits();e,add,r=pair['e'],pair['f'],pair['relation'];weight=1/(active*pair['within_relation_count'])
            rr=raw.copy();aa=a.copy();flip(rr,aa,q,types,channel,e,r);flip(rr,aa,q,types,channel,add,r)
            assert aa.sum()==a.sum() and np.array_equal(rr[:,:10].sum(0),raw[:,:10].sum(0))
            after=describe(sc,jd,rr,aa);delta=after['score']-base['score']
            F10=base['score'];F01=after['score'];F00=F10+data['new_delta'][e];F11=F10+data['new_delta'][add]
            cond=float(sigmoid(z[add]-z[e]));h=delta*cond*(1-cond)
            ge=p[e]*(1-p[e])*((1-p[add])*(F10-F00)+p[add]*(F11-F01))
            gf=p[add]*(1-p[add])*((1-p[e])*(F01-F00)+p[e]*(F11-F10))
            pg[e]-=weight*h;pg[add]+=weight*h;ug[e]+=weight*ge;ug[add]+=weight*gf
            fd=delta*(sigmoid(z[add]+1e-5-z[e])-sigmoid(z[add]-1e-5-z[e]))/2e-5
            assert abs(fd-h)<1e-10
            if pi%16==0:
                bb=b.copy();bb[e]=0;bb[add]=1
                with torch.no_grad():full=core.features(torch.from_numpy(bb)).numpy()[:,:core.base_width]
                err=float(np.max(abs(rr-full)));assert err==0
                # Independently verify stored single-flip scores used in four-state risk.
                for slot,target in [(e,F00),(add,F11)]:
                    sr=raw.copy();sa=a.copy();flip(sr,sa,q,types,channel,slot,r)
                    assert abs(describe(sc,jd,sr,sa)['score']-target)<1e-10
                tests.append(dict(state=tag,pair=pi,full_feature_max_difference=err,finite_difference_error=float(abs(fd-h))))
            row=dict(state=tag,pair=pi,relation=r,remove=e,add=add,label_remove=int(y[e]),label_add=int(y[add]),
                p_remove=float(p[e]),p_add=float(p[add]),q=cond,score_change=float(delta),pair_g_remove=float(-h),pair_g_add=float(h),
                independent_g_remove=float(ge),independent_g_add=float(gf),
                independent_common=float((ge+gf)/2),independent_position=float((gf-ge)/2),
                conditional_sensitivity=cond*(1-cond),
                **{k+'_change':float(after[k]-base[k]) for k in features},
                **{k+'_error_change':float(abs(after[k]-refconn[k])-abs(base[k]-refconn[k])) for k in refconn})
            records.append(row);local.append(row)
        ce=(p-y)/(n*(n-1)/2);fullg=data['new_g'];res=fullg.copy()
        for r in range(10):res[f==r]-=res[f==r].mean()
        norm=lambda v:float(np.linalg.norm(v))
        def stat(rows):
            good=[x for x in rows if x['score_change']<-EPS];mass=sum(abs(x['pair_g_add']) for x in good)
            return dict(pairs=len(rows),rewarded=len(good),neutral=sum(abs(x['score_change'])<=EPS for x in rows),
                triangle_cost=sum(x['triangle_error_change']>EPS for x in good),
                typed_triangle_cost=sum(x['typed_triangle_L1_change']>EPS for x in good),
                hub_cost=sum(x['hub_change']>EPS for x in good),low_cost=sum(x['low_change']>EPS for x in good),
                isolates_cost=sum(x['isolates_error_change']>EPS for x in good),
                joint_cost=sum(x['joint_distance_change']>EPS for x in good),
                triangle_cost_feedback_share=sum(abs(x['pair_g_add']) for x in good if x['triangle_error_change']>EPS)/mass if mass else None)
        summary=dict(state=tag,**stat(local),seconds=time.monotonic()-tick,active_relations=active,
            pair_norm=norm(pg),independent_same_pairs_norm=norm(ug),archived_full_norm=norm(fullg),
            cosine_pair_independent=cos(pg,ug),cosine_pair_full=cos(pg,fullg),cosine_pair_residual=cos(pg,res),cosine_pair_CE=cos(pg,ce),
            max_abs_relation_gradient_sum=max(abs(float(pg[f==r].sum())) for r in range(10)),
            expected_edges_directional_derivative=float(-(p*(1-p))@pg),
            base_triangles=base['triangle_count'],reference_triangles=float(sc.reftri.sum()))
        summaries.append(summary);vectors[tag]=pg
        for r in range(10):
            rows=[x for x in local if x['relation']==r]
            relrows.append(dict(state=tag,relation=r,**stat(rows),gradient_sum=float(pg[f==r].sum()),norm=norm(pg[f==r])))
            for label in [0,1]:
                mask=(f==r)&(y==label)
                labelrows.append(dict(state=tag,relation=r,label=label,support=int(mask.sum()),selected_support=int(np.count_nonzero(pg[mask])),
                    gradient_sum=float(pg[mask].sum()),gradient_norm=norm(pg[mask]),CE_cosine=cos(pg[mask],ce[mask]),
                    infinitesimal_probability_mass_change=float(-(p[mask]*(1-p[mask]))@pg[mask])))
        np.savez_compressed(OUT/(tag+'_vectors.npz'),pair=pg,independent_same_pairs=ug,archived_full=fullg,archived_residual=res,probability=p,reference=y,relation=f)
        save('pairs',records);save('summary',summaries);save('relations',relrows);save('relation_labels',labelrows);save('tests',tests)
        print(json.dumps(summary),flush=True)
        if state_index==0:
            estimated=(time.monotonic()-tick)*len(states)
            save('calibration',dict(first_state_seconds=time.monotonic()-tick,linear_panel_seconds=estimated,safety_factor=2,remaining_seconds=540-(time.monotonic()-START)))
            if 2*estimated>540-(time.monotonic()-START):raise TimeoutError('full-panel estimate insufficient, preserve no scope change')
    paired=[]
    for parent in ['CE','PS']:
        x=vectors[f'PubMed_{parent}_CE_t1000_draw0'];yy=vectors[f'PubMed_{parent}_CE_t1000_draw1']
        paired.append(dict(parent=parent,cosine=cos(x,yy),support_intersection=int(((x!=0)&(yy!=0)).sum()),
            limitation='Different fixed pairs and backgrounds; two draws do not estimate complete gradient variance.'))
    save('draw_comparisons',paired)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    assert not torch.cuda.is_initialized();limits();save('binding',BIND)
    save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        pairs=len(records),states=4,tests=len(tests),models=0,GPU_hours=0,updates=0,new_graph_samples=0,
        counterfactual_edits=len(records),assets_unchanged=True))
    print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise
