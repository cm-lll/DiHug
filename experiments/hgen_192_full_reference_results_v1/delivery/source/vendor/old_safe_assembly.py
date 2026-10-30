"""Explicit totalization of undefined author assembly paths; preserve defined proposals."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,signal,copy,random
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import core as c
from prepare_check import same

def sample(score,target,patterns,groups,seed,max_attempts=None):
    c.seed(seed);env=c.load();n=score.shape[0]
    if len(score.nonzero()[0])<target:
        a=env['symmetric'](score)>0
        return a.toarray().astype(np.uint8),[],True,'upstream_low_support_return'
    a=np.zeros((n,n),dtype=np.uint8);s=score.toarray().copy();np.fill_diagonal(s,0)
    degree=s.sum(0);bytype=[dict(zip(groups[t],[degree[i] for i in groups[t]])) for t in range(len(groups))]
    masks={t:env['array_subtraction'](range(n),groups[t]) for t in groups}
    if not patterns:return a,[],False,'empty_pattern_support'
    if not any(len(p)>=2 for p in patterns):return a,[],False,'no_transition_pattern'
    limit=max_attempts if max_attempts is not None else 20*max(1,target//2)
    logs=[];directed=0;started=time.time()
    for attempt in range(limit):
        if time.time()-started>600:return a,logs,False,'wall_watchdog'
        pattern=env['sample_from_dict'](patterns,1)[0];seq=[];reason=None
        if sum(bytype[pattern[0]].values())<=0:reason='zero_root_mass'
        else:
            root=env['sample_from_dict'](bytype[pattern[0]],1)[0];seq=[int(root)]
            for typ in pattern[1:]:
                row=s[seq[-1],:].copy();row[masks[typ]]=0
                if row.sum()<=0:reason='zero_transition_mass';break
                seq.append(int(np.random.choice(n,p=row/row.sum())))
        added=0
        if reason is None:
            for u,v in zip(seq[:-1],seq[1:]):
                if a[u,v]==0:a[u,v]=a[v,u]=1;added+=1;directed+=2
        logs.append(dict(attempt=attempt,pattern=list(map(int,pattern)),nodes=seq,status=reason or 'accepted',added=added,directed_edges=directed))
        if directed>target:return a,logs,True,'target_exceeded_as_upstream'
    return a,logs,False,'proposal_limit'

def gate():
    env=c.load();groups={0:list(range(4)),1:list(range(4,8))};p={(0,1,0):5,(1,0,1):3}
    s=sp.csr_matrix(np.ones((8,8))-np.eye(8))
    for seed in range(10):
        c.seed(seed);a=env['heterogeneous_graph_assemble'](s,12,p,groups);r1=c.rng()
        b,logs,ok,_=sample(s,12,p,groups,seed);r2=c.rng()
        assert ok and np.array_equal(a,b) and same(r1,r2)
    # Strong score only within each type: cross-type proposals cannot complete.
    x=np.zeros((8,8));x[:4,:4]=1;x[4:,4:]=1;np.fill_diagonal(x,0)
    a,l,ok,why=sample(sp.csr_matrix(x),12,p,groups,12,max_attempts=3)
    assert not ok and len(l)==3 and a.sum()==0 and all(z['status']=='zero_transition_mass' for z in l)
    a,l,ok,why=sample(sp.csr_matrix(x),12,{},groups,12);assert not ok and why=='empty_pattern_support'
    a,l,ok,why=sample(sp.csr_matrix(x),100,p,groups,12);assert ok and np.array_equal(a,x)
    return dict(passed=True,positive_support_exact_replays=10,boundary_cases=3,training=0,model_calls=0)

def run(parent,out):
    out.mkdir(exist_ok=False,parents=True);checks=gate();c.write(out/'engineering_gate.json',checks)
    source={str(Path(__file__).resolve()):c.sha(__file__),str(c.ROOT/'docs/HGEN_OFFICIAL_COMPATIBILITY_20261016.md'):c.sha(c.ROOT/'docs/HGEN_OFFICIAL_COMPATIBILITY_20261016.md')}
    for p in [Path(c.__file__),c.UP/'utils.py',c.ROOT/'paper_revision_lab/hgen_motif_r2/review.py',c.ROOT/'paper_revision_lab/hgen_motif_r2/review_endpoints.py',c.ROOT/'paper_revision_lab/scripts/typed_triangle_supplement.py']:
        source[str(p)]=c.sha(p)
    c.write(out/'sources_at_launch.json',source)
    c.write(out/'status.json',dict(stage='waiting_for_strict_branch_acceptance'))
    while not (parent/'controller_exit.json').exists():
        if (parent/'failure_recover_None_None.json').exists():raise RuntimeError('parent recovery failed; compatible branch not started')
        time.sleep(30)
    assert json.loads((parent/'independent_acceptance.json').read_text())['engineering_passed']
    import accept as review
    public=json.loads((c.ROOT/'results/paper_revision_pr3b/pr3b_20261008_stage2_attribution_v1/public_motif_counts.json').read_text())
    records=[];tic=time.time();logchecks=0
    for ds in ['imdb','pubmed']:
        ref=json.loads((parent/'data'/ds/'reference.json').read_text());n=len(ref['node_types']);groups={t:[i for i,x in enumerate(ref['node_types']) if x==t] for t in range(4)}
        for seed in [310,311,312]:
            origin=parent/ds/f'seed{seed}'
            for name in ['checkpoint.pt','scores.npz','patterns.json','generated_walks.npz']:source[str(origin/name)]=c.sha(origin/name)
            score=sp.load_npz(origin/'scores.npz');patterns={tuple(x['types']):x['count'] for x in json.loads((origin/'patterns.json').read_text())}
            for rep in range(10):
                dest=out/ds/f'seed{seed}'/f'graph{rep:02d}';dest.mkdir(parents=True);ss=seed*10000+rep+1000000
                a,ledger,ok,reason=sample(score,sum(len(f['edges']) for f in ref['families'])*2,patterns,groups,ss)
                np.savez_compressed(dest/'raw_adjacency.npz',adjacency=a)
                with (dest/'actions.jsonl').open('w') as f:
                    for row in ledger:f.write(json.dumps(row)+'\n')
                # Independent edge-set reconstruction, all proposals and accounting.
                edges=set();cnt=0
                for row in ledger:
                    before=len(edges)
                    if row['status']=='accepted':
                        assert len(row['nodes'])==len(row['pattern'])
                        assert [ref['node_types'][i] for i in row['nodes']]==row['pattern']
                        for u,v in zip(row['nodes'][:-1],row['nodes'][1:]):assert u!=v and score[u,v]>0;edges.add(tuple(sorted((u,v))))
                    assert row['added']==len(edges)-before and row['directed_edges']==2*len(edges);cnt+=1
                if ledger:assert edges==set(zip(*np.triu(a,k=1).nonzero()))
                logchecks+=cnt;g,valid=c.graph_payload(ref,a);c.write(dest/'graph.json',g)
                row=dict(dataset=ds,seed=seed,repeat=rep,complete=ok,reason=reason,validity=valid,proposals=len(ledger),rejected=sum(x['status']!='accepted' for x in ledger),path=str(dest/'graph.json'),sha256=c.sha(dest/'graph.json'))
                if ok and valid['simple']:
                    extended=copy.deepcopy(ref);names={f['name'] for f in ref['families']}
                    for f in g['families']:
                        if f['name'] not in names:q=copy.deepcopy(f);q['edges']=[];extended['families'].append(q)
                    row['metrics'],_,_=review.metric(g,extended,public[ds]['templates'])
                    obs,_=review.tt.measure(g);rr,_=review.tt.measure(extended)
                    row['typed_triangles']=review.tt.compare(obs,rr,sorted(set(review.tt.schema_classes(extended))|set(obs['counts'])|set(rr['counts'])))
                c.write(dest/'request.json',row);records.append(row)
                c.write(out/'status.json',dict(stage='assembly_and_independent_accounting',requests=len(records),total=60))
    assert len(records)==60 and all(c.sha(p)==h for p,h in source.items())
    c.write(out/'sources.json',source);c.write(out/'per_graph.json',records)
    summary={}
    for ds in ['imdb','pubmed']:
        rr=[r for r in records if r['dataset']==ds];valid=[r for r in rr if 'metrics' in r]
        summary[ds]=dict(requests=30,completed=sum(r['complete'] for r in rr),schema_valid_completed=sum(r['complete'] and r['validity']['schema_valid'] for r in rr),statistics={k:review.moments([r['metrics'][k] for r in valid]) for k in ['triangles','motif_mean_relative_error','clustering','degree_wasserstein','LCC']},typed_tv=review.moments([r['typed_triangles']['composition_tv_when_defined'] for r in valid if r['typed_triangles']['composition_tv_when_defined'] is not None]))
    c.write(out/'independent_acceptance.json',dict(engineering_passed=True,requests=60,complete=sum(r['complete'] for r in records),full_action_accounting=logchecks,summary=summary,wall_seconds=time.time()-tic,cpu_seconds=time.process_time(),training=0,new_model_walks=0,limitations=['Standalone minimal compatibility extension, not the original undefined branch','Whole-path rejection changes the law on histories containing zero support','No quality-based filtering; complete/incomplete requests retained; source model reused']))
    c.write(out/'status.json',dict(stage='completed_and_audited',requests=60))

if __name__=='__main__':
    try:run(Path(sys.argv[1]),Path(sys.argv[2]))
    except BaseException:
        import traceback
        c.write(Path(sys.argv[2])/'failure.json',dict(traceback=traceback.format_exc()));raise
