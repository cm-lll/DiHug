"""Archived-only stratified accounting. No objective search or model computation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,time,hashlib,resource,signal,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/rooted_connection_20261109'))
sys.path.insert(0,str(ROOT/'research_branches/explicit_structure_cpu_20261109'))
from score import RootedScore,RootedIncremental
from explicit_score import ExplicitScore,connectivity
SRC=ROOT/'results/explicit_structure_cpu_20261109'
RUN=ROOT/'results/rooted_connection_20261109T015142'
OUT=ROOT/'results/structure_allocation_audit_20261109/complete'
EPS=1e-12;START=time.monotonic();BIND={}
BLOCKS=['joint','closure','low','hub']
PARTS=['joint_TV','relation_W1','low','hub','typed_triangle_L1','node_triangle_W1']
COEF=np.array([.125,.125,.25,.25,.125,.125])
METRICS=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error']

def bind(p):
    p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(name,x):
    (OUT/(name+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
    if isinstance(x,list) and x:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def limits():
    if time.monotonic()-START>600:raise TimeoutError('600s CPU wall cap')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>2048:raise MemoryError('2GiB RSS cap')
def summarize(rr,weighted=False):
    w=np.array([r['inventory_weight'] if weighted else 1. for r in rr]);den=float(w.sum())
    good=np.array([r['explicit_change']<-EPS for r in rr]);g=float(w[good].sum())
    out=dict(selected=len(rr),estimated_slots=den,rewarded=g,
        rewarded_fraction=g/den if den else None,
        mixed_rewarded=float(sum(a for a,r,b in zip(w,rr,good) if b and max(r['explicit_'+k+'_change'] for k in BLOCKS)>EPS)))
    for k in METRICS:
        v=np.array([r[k+'_change'] for r in rr]);out[k+'_rewarded_worse']=float(w[good&(v>EPS)].sum())
    for k in BLOCKS:
        v=np.array([r['explicit_'+k+'_change'] for r in rr]);out[k+'_mean_abs']=float(w@abs(v)/den)
    return out
def tables(rows,keys,name):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    save(name,[dict(zip(keys,key),weighting=tag,**summarize(rr,weight)) for key,rr in sorted(groups.items()) for tag,weight in [('balanced_selected',False),('inventory_HT',True)]])
def ledger(r):
    v=np.array([r['explicit_'+k+'_change'] for k in PARTS])*COEF
    assert abs(v.sum()-r['explicit_change'])<1e-10
    r.update({k+'_contribution':float(z) for k,z in zip(PARTS,v)})
    r['degree_contribution']=float(v[:4].sum());r['closure_contribution']=float(v[4:].sum())
    r['degree_improved_terms']=int((v[:4]<-EPS).sum())
    r['degree_cost_terms']=int((v[:4]>EPS).sum())
    r['closure_cost_degree_paid']=bool(r['explicit_change']<-EPS and v[4:].sum()>EPS and v[:4].sum()<-EPS)
    # Accounting necessity only, not new candidate scoring or proof of redundancy.
    costs=float(np.maximum(v,0).sum());benefits=np.maximum(-v[:4],0)
    r['multiple_degree_benefits_needed']=bool(r['closure_cost_degree_paid'] and max(benefits)<=costs+EPS)
    return r

def main():
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} <= {'protocol.json','failure.json'}, 'Refuse overwrite of completed/partial scientific output'
        (OUT/'protocol_first_attempt.json').write_bytes((OUT/'protocol.json').read_bytes())
    OUT.mkdir(exist_ok=True,parents=True);signal.alarm(600)
    save('protocol',dict(scope='same64 old graphs4096 selected flips; same172 old exchanges; no new candidate',
        allocation='original uniform32 absent+32 present per graph; HT weight=N_present_or_absent/32; not a training gradient estimate',
        node_groups='current hard graph, per-type total degree bottom floor(n/2), top2; all tied cutoff nodes included, hub precedence',
        opportunities='number of common neighbors of endpoints in frozen current graph; zero/nonzero; no query edges inserted',
        double_count='exact six contributions and within-relation/direction sign association; correlation is not proof of redundant semantics',
        limits=dict(seconds=600,threads=1,RSS_MiB=2048,GPU=0),
        caveat='historical development assets, overlapping edits, not independent trials or training outcomes'))
    for p in [Path(__file__),ROOT/'research_branches/explicit_structure_cpu_20261109/explicit_score.py',ROOT/'research_branches/rooted_connection_20261109/score.py',ROOT/'research_branches/rooted_connection_20261109/postmortem_cpu.py']:bind(p)
    prior=read(SRC/'binding.json');rows=read(SRC/'final_edges.json');oldex=read(SRC/'exchanges.json');jobs=read(RUN/'evaluation_jobs.json')
    def historical(p):
        p=bind(p);assert prior[str(p)]==BIND[str(p)],str(p);return p
    cores={};scorers={};inventory=[];reference=[];allrows=[];exchange=[]
    for ds in ['IMDB','PubMed']:
        with np.load(historical(RUN/f'{ds}_cpu_cost_panel.npz')) as z:core=RootedScore(z['types'],z['query'],z['families'],z['reference'],10)
        cores[ds]=core;scorers[ds]=ExplicitScore(core)
        q=np.asarray(core.query);f=np.asarray(core.families);ty=np.asarray(core.types);y=np.asarray(core.reference)
        for rel in np.unique(f):
            ix=np.flatnonzero(f==rel);u,v=q[:,ix[0]]
            reference.append(dict(dataset=ds,relation=int(rel),endpoint_types=f'{ty[u]}-{ty[v]}',slots=len(ix),edges=int(y[ix].sum()),density=float(y[ix].mean())))
    save('reference_inventory',reference)
    for job in jobs:
        limits();ident={k:job[k] for k in ['dataset','train_seed','arm','sample_seed']};ds=ident['dataset'];core=cores[ds];q=core.query.numpy();f=core.families.numpy()
        path=historical(Path(job['folder'])/'graph.npz');binding=read(path.parent/'binding.json');assert BIND[str(path)]==binding['graph_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['types'],core.types);a=np.zeros((core.n,core.n),dtype=bool);a[z['edges'][0],z['edges'][1]]=True;a|=a.T
        bits=a[q[0],q[1]].astype(float);degree=a.sum(1);nodegroup=np.full(core.n,'middle',dtype=object)
        for t,ii in core.nodes.items():
            ii=ii.numpy();sd=np.sort(degree[ii]);nodegroup[ii[degree[ii]<=sd[max(1,len(ii)//2)-1]]]='low';nodegroup[ii[degree[ii]>=sd[-min(2,len(ii))]]]='hub'
        selected=[dict(r) for r in rows if all(r[k]==v for k,v in ident.items())]
        for present in [0,1]:
            assert sum(r['present']==present for r in selected)==32
            for rel in np.unique(f):
                pool=(bits==present)&(f==rel);n=int(pool.sum());chosen=sum(r['present']==present and r['relation']==rel for r in selected)
                inventory.append(dict(**ident,relation=int(rel),operation='delete' if present else 'add',population=n,selected=chosen,
                    probability=32/int((bits==present).sum()),zero_selected_nonempty=bool(n and not chosen)))
        for r in selected:
            e=r['slot'];u,v=q[:,e];assert bits[e]==r['present'] and f[e]==r['relation']
            cn=int((a[u]&a[v]).sum());r.update(operation='delete' if r['present'] else 'add',inventory_weight=float((bits==r['present']).sum()/32),
                u=int(u),v=int(v),degree_u=int(degree[u]),degree_v=int(degree[v]),endpoint_groups='-'.join(sorted([nodegroup[u],nodegroup[v]])),
                common_neighbors=cn,closure_opportunity='nonzero' if cn else 'zero',graph_edges=int(bits.sum()),audit_slots=len(bits))
            allrows.append(ledger(r))
        if ident==dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232):
            sc=scorers[ds];inc=RootedIncremental(core,bits);before=sc.terms(inc.raw);add=3489
            assert bits[add]==0
            b=bits.copy();b[add]=1;ai=RootedIncremental(core,b)
            for start in range(0,len(oldex),16):
                batch=oldex[start:start+16];ix=[r['removed_slot'] for r in batch]
                assert all(bits[e]==1 and f[e]==f[add] for e in ix)
                raw,_=ai.raw_flips(ix);after=sc.terms(raw)
                for j,old in enumerate(batch):
                    r=dict(old,**ident,added_slot=add,relation=int(f[add]))
                    for k in BLOCKS+PARTS:r['explicit_'+k+'_change']=float(after[k][j]-before[k][0])
                    assert abs(float(after['total'][j]-before['total'][0])-r['explicit_change'])<1e-10
                    assert abs(raw[j,:,:core.r].sum()-inc.raw[:,:core.r].sum())<1e-10
                    r['relation_inventory_unchanged']=True;exchange.append(ledger(r))
    save('final_edges',allrows);save('inventory',inventory);save('exchanges',exchange)
    for keys,name in [(['dataset'],'dataset'),(['dataset','operation'],'operation'),(['dataset','train_seed','operation'],'seed_operation'),
        (['dataset','operation','relation'],'relation_operation'),(['dataset','operation','endpoint_groups'],'endpoint_operation'),
        (['dataset','operation','closure_opportunity'],'opportunity_operation')]:tables(allrows,keys,name)
    breakdown=[]
    for ds in ['IMDB','PubMed']:
        for seed in sorted({r['train_seed'] for r in allrows if r['dataset']==ds}):
            rr=[r for r in allrows if r['dataset']==ds and r['train_seed']==seed];good=[r for r in rr if r['explicit_change']<-EPS]
            breakdown.append(dict(dataset=ds,train_seed=seed,rewarded=len(good),closure_cost_degree_paid=sum(r['closure_cost_degree_paid'] for r in rr),
                multiple_degree_benefits_needed=sum(r['multiple_degree_benefits_needed'] for r in rr),
                multi_degree_gain_with_closure_cost=sum(r['closure_cost_degree_paid'] and r['degree_improved_terms']>=2 for r in rr),
                all_degree_terms_improve=sum(r['degree_improved_terms']==4 for r in good)))
    save('contribution_summary',breakdown)
    association=[]
    for ds in ['IMDB','PubMed']:
        for op in ['add','delete']:
            for rel in sorted({r['relation'] for r in allrows if r['dataset']==ds}):
                rr=[r for r in allrows if r['dataset']==ds and r['operation']==op and r['relation']==rel]
                for x,y in itertools.combinations(PARTS[:4],2):
                    v=np.array([r[x+'_contribution'] for r in rr]);w=np.array([r[y+'_contribution'] for r in rr]);mask=(abs(v)>EPS)&(abs(w)>EPS)
                    association.append(dict(dataset=ds,operation=op,relation=rel,x=x,y=y,selected=len(rr),both_nonzero=int(mask.sum()),
                        same_sign=int(((v*w>0)&mask).sum()),opposite_sign=int(((v*w<0)&mask).sum()),
                        pearson=float(np.corrcoef(v,w)[0,1]) if len(rr)>1 and np.std(v)>EPS and np.std(w)>EPS else None))
    save('association',association)
    exsum=[]
    for method,col in [('old_DIRECT','old_DIRECT_change'),('explicit','explicit_change')]:
        good=[r for r in exchange if r[col]<-EPS]
        exsum.append(dict(method=method,exchanges=len(exchange),rewarded=len(good),
            **{k+'_worse':sum(r[k+'_change']>EPS for r in good) for k in METRICS},
            closure_cost_degree_paid=sum(r['closure_cost_degree_paid'] for r in good),
            all_four_nonworse=sum(max(r['explicit_'+k+'_change'] for k in BLOCKS)<=EPS for r in good)))
    save('exchange_summary',exsum)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    save('binding',BIND);save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        final_flips=len(allrows),exchanges=len(exchange),model_calls=0,GPU_hours=0,updates=0,new_graph_samples=0,assets_unchanged=True))
    print((OUT/'operation.json').read_text());print((OUT/'exchange_summary.json').read_text());print((OUT/'receipt.json').read_text())
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
