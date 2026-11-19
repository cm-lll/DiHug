"""Frozen-score audit on archived graphs/probabilities; never calls a model."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, json, resource, time, traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from score import RootedScore, RootedIncremental, kernel
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/rooted_connection_20261109T015142'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
EDGE=ROOT/'results/hard_structure_edge_direction_cpu_20261108'
OUT=ROOT/'results/rooted_feedback_audit_20261109'
BIND={}
START=time.time()
def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    assert time.time()-START<600,'CPU wall limit'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048,'RSS limit'

def metrics(core,raw,ref):
    """Descriptive metric vector; no new loss is constructed."""
    raw=raw[None] if raw.ndim==2 else raw
    values={k:np.zeros(len(raw)) for k in ('total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error')}
    for t,ix in core.nodes.items():
        ix=ix.numpy();d=raw[:,ix,:core.r];rd=ref[ix,:core.r]
        td=d.sum(-1);rtd=rd.sum(-1)
        values['total_degree_W1']+=abs(np.sort(td,axis=1)-np.sort(rtd)).mean(1)
        active=core.masks[t][:core.r].numpy()
        values['relation_degree_W1']+=abs(np.sort(d[:,:,active],axis=1)-np.sort(rd[:,active],axis=0)).mean((1,2))
        target=Counter(map(tuple,rd))
        for j in range(len(raw)):
            cur=Counter(map(tuple,d[j]));values['joint_degree_TV'][j]+=.5*sum(abs(cur.get(k,0)-target.get(k,0)) for k in cur.keys()|target.keys())/len(ix)
        for key,sl in [('node_triangle_W1',slice(core.r,core.base_width)),('root_path_W1',slice(core.base_width,None))]:
            act=core.masks[t][sl].numpy()
            if act.any():values[key]+=abs(np.sort(raw[:,ix,sl][:,:,act],axis=1)-np.sort(ref[ix,sl][:,act],axis=0)).mean((1,2))
        values['hub_error']+=abs(np.sort(td,axis=1)[:,-2:]-np.sort(rtd)[-2:]).mean(1)
    values={k:v/len(core.nodes) for k,v in values.items()}
    tri=raw[:,:,core.r:core.base_width].sum((1,2))/3;rt=ref[:,core.r:core.base_width].sum()/3
    values['triangle_error']=abs(tri-rt)
    values['edge_error']=abs(raw[:,:,:core.r].sum((1,2))/2-ref[:,:core.r].sum()/2)
    return values

def main():
    OUT.mkdir(exist_ok=False)
    save('protocol',dict(scope='CPU saved assets only; frozen RootedScore, CE and lambda unchanged',
        final_graphs=64,final_slots='exact same 4096 preselected slots from previous audit',
        conditional_panel='IMDB/PubMed x historical CE/PS parent, CE continuation x t1/500/1000 x archived draws0/1;24 graphs',
        edge_subset_repeats=256,edge_subset_seed=2026110912,edge_subset_size=1024,
        limits=dict(wall_seconds=600,RSS_MiB=2048,threads=1,GPU=0),
        model_calls=0,updates=0,new_graph_samples=0,sign_tolerance=1e-12,
        limitations='Archived conditional predictions predate RC; free logits not network parameters; two hard draws do not establish Monte Carlo convergence; final balanced slots not population rates; no candidate tuning'))
    sources=[Path(__file__),Path(__file__).parent/'score.py',Path(__file__).parent/'runtime/objective.py',Path(__file__).parent/'runtime/slot_update.py',ROOT/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py',ROOT/'research_branches/hard_structure_rebar_cpu_20261108/objective.py']
    for p in sources:bind(p)
    cores={}
    for ds in ['IMDB','PubMed']:
        with np.load(bind(RUN/f'{ds}_cpu_cost_panel.npz')) as z:cores[ds]=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
    previous=read(ROOT/'results/rooted_connection_postmortem_20261109/edge_directions.json')
    edges=[];summary=[]
    for job in read(RUN/'evaluation_jobs.json'):
        limits();core=cores[job['dataset']];ident={k:job[k] for k in ('dataset','train_seed','arm','sample_seed')}
        path=Path(job['folder'])/'graph.npz';b=read(path.parent/'binding.json');bind(path);assert BIND[str(path)]==b['graph_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['types'],core.types)
            a=np.zeros((core.n,core.n));a[z['edges'][0],z['edges'][1]]=1;a+=a.T
        q=core.query.numpy();bits=a[q[0],q[1]];inc=RootedIncremental(core,bits)
        chosen=[v for v in previous if all(v[k]==x for k,x in ident.items())]
        ix=np.array([v['slot'] for v in chosen]);d=inc.differences(ix)*(1-2*bits[ix])
        assert np.max(abs(d-np.array([v['score_change'] for v in chosen])))<1e-11
        raw,_=inc.raw_flips(ix);ref=core.features(core.reference).numpy();old=metrics(core,inc.raw,ref);new=metrics(core,raw,ref)
        counts=dict(improving=0,triangle_worse=0,three_worse=0,no_gain_in_checked=0)
        for j,e in enumerate(ix):
            changes={k:float(new[k][j]-old[k][0]) for k in old};good=d[j]<-1e-12
            counts['improving']+=int(good);counts['triangle_worse']+=int(good and changes['triangle_error']>1e-12)
            counts['three_worse']+=int(good and all(changes[k]>1e-12 for k in ('triangle_error','total_degree_W1','joint_degree_TV')))
            counts['no_gain_in_checked']+=int(good and all(v>=-1e-12 for v in changes.values()) and any(v>1e-12 for v in changes.values()))
            edges.append(dict(**ident,slot=int(e),present=int(bits[e]),relation=int(core.families[e]),score_change=float(d[j]),**{k+'_change':v for k,v in changes.items()}))
        summary.append(dict(**ident,slots=len(ix),**counts))
    save('final_edge_metrics',edges);save('final_direction_summary',summary)
    print('final graph checks complete',flush=True)
    # No auxiliary graphs sampled here: exact archived bits and saved p.
    panel=[];repeats=[];relrows=[];vectors={};rng=np.random.default_rng(2026110912)
    prior_binding=read(ROOT/'results/identity_structure_audit_20261109/asset_source_binding.json')
    def checked(p):
        p=bind(p);assert str(p) in prior_binding and BIND[str(p)]==prior_binding[str(p)],str(p);return p
    for ds in ['IMDB','PubMed']:
        fixed=torch.load(checked(OLD/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        for t,seed,inp in fixed:
            assert int(t) in [1,500,1000]
            q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));core=RootedScore(cores[ds].types,q,f,y,10)
            for parent in ['CE','PS']:
                tag=f'{ds}_{parent}_CE_t{t}'
                with np.load(checked(OLD/f'{tag}.npz')) as z:p=z['probability'].astype(float)
                assert p.shape==y.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                for draw in [0,1]:
                    limits();started=time.time()
                    with np.load(checked(EDGE/f'{tag}_draw{draw}_flips.npz')) as z:
                        bits=(1-z['flip_sign'])/2
                        changes={k:z[k+'_flip'].copy() for k in ('total_degree_W1','joint_degree_TV','triangle_abs_error')}
                    inc=RootedIncremental(core,bits);delta=inc.differences();seconds=time.time()-started
                    # Independent full-score checks on fixed coordinates, not selected for disagreement.
                    for e in np.linspace(0,len(p)-1,3,dtype=int):
                        bb=bits.copy();bb[e]=1-bb[e]
                        assert abs(float(core(torch.from_numpy(bb)))-inc.base_score-(1-2*bits[e])*delta[e])<1e-10
                    g=p*(1-p)*delta;ce=(p-y)/(core.n*(core.n-1)/2);M=len(p);m=min(1024,M);factor=M/m
                    g2=float(g@g);ce2=float(ce@ce);dot=float(g@ce);flip=(1-2*bits)*delta;good=flip<-1e-12
                    values=[]
                    for rep in range(256):
                        ix=rng.choice(M,m,replace=False);mass=float(g[ix]@g[ix]);hatdot=float(factor*(g[ix]@ce[ix]));hat2=factor**2*mass
                        row=dict(panel=tag,draw=draw,repeat=rep,relative_error_sq=(g2+(factor**2-2*factor)*mass)/g2 if g2 else None,
                            cosine_with_full=np.sqrt(mass/g2) if g2 and mass else None,
                            cosine_with_CE=hatdot/np.sqrt(hat2*ce2) if hat2 and ce2 else None,
                            dot_with_CE=hatdot,dot_sign_reversed=bool(hatdot*dot<0))
                        values.append(row);repeats.append(row)
                    panel.append(dict(dataset=ds,parent_history=parent,t=int(t),draw=draw,slots=M,score=inc.base_score,full_seconds=seconds,
                        full_gradient_norm=np.sqrt(g2),CE_gradient_norm=np.sqrt(ce2),weighted_full_to_CE=.1*np.sqrt(g2/ce2) if ce2 else None,
                        expected_weighted_HT_rms_to_CE=.1*np.sqrt(factor*g2/ce2) if ce2 else None,
                        cosine_full_CE=dot/np.sqrt(g2*ce2) if g2*ce2 else None,
                        predicted_relative_MSE=factor-1,empirical_relative_MSE=float(np.mean([v['relative_error_sq'] for v in values])) if g2 else None,
                        HT_cosine_full_median=float(np.median([v['cosine_with_full'] for v in values])) if g2 and all(v['cosine_with_full'] is not None for v in values) else None,
                        CE_dot_sign_reversal_count=sum(v['dot_sign_reversed'] for v in values),
                        score_improving_slots=int(good.sum()),triangle_worse=int((good&(changes['triangle_abs_error']>1e-12)).sum()),
                        three_worse=int((good&np.logical_and.reduce([v>1e-12 for v in changes.values()])).sum())))
                    for rel in np.unique(f):
                        mask=f==rel;sel=mask&good
                        relrows.append(dict(dataset=ds,parent_history=parent,t=int(t),draw=draw,relation=int(rel),slots=int(mask.sum()),improving=int(sel.sum()),
                            triangle_worse=int((sel&(changes['triangle_abs_error']>1e-12)).sum()),gradient_norm=float(np.linalg.norm(g[mask])),
                            positive_conflict_mass=float(abs(g[mask&(y==1)&(g>0)]).sum()),negative_conflict_mass=float(abs(g[mask&(y==0)&(g<0)]).sum())))
                    vectors.setdefault(tag,[]).append(g)
                    np.savez_compressed(OUT/f'{tag}_draw{draw}_full.npz',logit_gradient=g,CE_logit_gradient=ce,conditional_difference=delta)
                    save('conditional_panel',panel);save('conditional_relations',relrows)
                    print(tag,draw,round(seconds,3),flush=True)
    save('edge_subset_repeats',repeats)
    pair=[]
    for tag,v in vectors.items():
        x,y=v;den=np.linalg.norm(x)*np.linalg.norm(y)
        pair.append(dict(panel=tag,two_draw_cosine=float(x@y/den) if den else None,
            relative_difference=float(np.linalg.norm(x-y)/(0.5*(np.linalg.norm(x)+np.linalg.norm(y)))) if den else None,
            limitation='two hard samples only; not converged expectation/variance'))
    save('hard_draw_comparison',pair)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('bindings',BIND)
    save('receipt',dict(status='complete',wall_seconds=time.time()-START,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_graphs=len(summary),final_flips=len(edges),conditional_graphs=len(panel),subset_repeats=len(repeats),GPU_hours=0,model_calls=0,updates=0,new_graph_samples=0,sources_unchanged=True))
    print((OUT/'receipt.json').read_text(),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():save('failure',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
