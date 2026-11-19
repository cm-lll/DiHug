"""Bounded CPU hard-counterfactual audit. No model or production changes."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, csv, time, signal, resource, hashlib, itertools, traceback, importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('switch_helpers', ROOT/'research_branches/pair_allocation_signal_20261109/run_cpu.py')
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
OUT = ROOT/'results/degree_preserving_switch_20261109'
START = time.monotonic(); BIND = {}; EPS = 1e-12

def save(name, obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        with (OUT/(name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, list(obj[0])); w.writeheader(); w.writerows(obj)

def bind(path, expected=None):
    path = Path(path); value = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None: assert value == expected[str(path)], str(path)
    BIND[str(path)] = value
    return path

def read(path): return json.loads(bind(path).read_text())

def limit():
    if time.monotonic()-START > 540: raise TimeoutError('540-second scientific stop')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 2048: raise MemoryError('2 GiB RSS stop')

def population(query, families, bits, relation, lookup):
    """Uniform population of legal distinct two-edge switches, not score selected."""
    result = []
    for e, f in itertools.combinations(np.flatnonzero((families == relation) & (bits == 1)), 2):
        a,b = map(int,query[:,e]); c,d = map(int,query[:,f])
        if len({a,b,c,d}) != 4: continue
        for edges in [((a,c),(b,d)), ((a,d),(b,c))]:
            new = [lookup.get(tuple(sorted(x)), -1) for x in edges]
            if min(new) < 0: continue
            if all(families[x] == relation and bits[x] == 0 for x in new):
                result.append((int(e),int(f),*sorted(map(int,new))))
    assert len(result) == len(set(result))
    return result

def independent_raw(query, bits, types, relations, channels):
    n=len(types); a=H.adjacency(query,bits,n); raw=np.zeros((n,relations+10))
    for k in np.flatnonzero(bits):
        u,v=query[:,k]; r=FAMILIES[k]; raw[u,r]+=1;raw[v,r]+=1
    for i in range(n):
        for j in np.flatnonzero(a[i] & (np.arange(n)>i)):
            for k in np.flatnonzero(a[i]&a[j]&(np.arange(n)>j)):
                raw[i,channels[types[j],types[k]]]+=1
                raw[j,channels[types[i],types[k]]]+=1
                raw[k,channels[types[i],types[j]]]+=1
    return raw

def algebra_test():
    rng=np.random.default_rng(2026110901); max_q=0.; max_fd=0.
    for _ in range(100):
        z=rng.uniform(-8,8,4); p=H.sigmoid(z); costs=rng.normal(size=2)
        w0=p[0]*p[1]*(1-p[2])*(1-p[3]);w1=(1-p[0])*(1-p[1])*p[2]*p[3]
        q=H.sigmoid(z[2]+z[3]-z[0]-z[1]);max_q=max(max_q,abs(q-w1/(w0+w1)))
        g=(costs[1]-costs[0])*q*(1-q)*np.array([-1.,-1.,1.,1.])
        for j in range(4):
            def loss(x): return costs[0]+(costs[1]-costs[0])*H.sigmoid(x[2]+x[3]-x[0]-x[1])
            plus=z.copy();minus=z.copy();plus[j]+=1e-5;minus[j]-=1e-5
            max_fd=max(max_fd,abs((loss(plus)-loss(minus))/2e-5-g[j]))
    assert max_q<1e-12 and max_fd<1e-9
    # Independent finite graph check: all 1024 undirected five-node graphs.
    q=np.array(list(itertools.combinations(range(5),2))).T;f=np.zeros(10,int)
    lookup={tuple(x):i for i,x in enumerate(q.T)};switches=0
    for code in range(1024):
        b=np.array([(code>>i)&1 for i in range(10)])
        for e,f0,g,h in population(q,f,b,0,lookup):
            b2=b.copy();b2[[e,f0]]=0;b2[[g,h]]=1
            a=H.adjacency(q,b,5);a2=H.adjacency(q,b2,5)
            assert np.array_equal(a.sum(1),a2.sum(1))
            switches+=1
    return dict(probability_cases=100,finite_difference_coordinates=400,max_conditional_error=max_q,max_fd_error=max_fd,
                exhaustive_graphs=1024,legal_switches=switches)

def stats(rows):
    good=[r for r in rows if r['score_change'] < -EPS]
    result=dict(switches=len(rows),rewarded=len(good),neutral=sum(abs(r['score_change'])<=EPS for r in rows),
                penalized=sum(r['score_change']>EPS for r in rows),feedback_abs_sum=sum(abs(r['h']) for r in rows))
    for metric in ['triangle_error','typed_triangle_L1','node_triangle_W1','triangle_composition_TV','LCC_error','components_error']:
        result[metric+'_rewarded_cost']=sum(r[metric+'_change']>EPS for r in good)
    result['rewarded_add_triangles']=sum(r['triangle_count_change']>EPS for r in good)
    result['rewarded_remove_triangles']=sum(r['triangle_count_change']<-EPS for r in good)
    return result

def main():
    global FAMILIES
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(600)
    save('protocol',dict(scope='four archived PubMed0422 A raw2040 t1000 conditional hard draws, CE/PS historical parents; one training seed',
        samples='three independent SHA256 seeds per background/relation; uniform without replacement from every legal distinct same-relation two-edge switch; 16 or census',
        objective='unchanged explicit score with joint transport replacement; no scoring used in candidate selection; no repeated graph editing',
        invariants='each node each relation degree, all edges/type inventories, no self loops or duplicate edges; connectivity is evaluated, NOT constrained',
        risk='two-pattern conditional risk, not unbiased full graph risk; background/candidate selection derivatives omitted',
        limits=dict(threads=1,memory_MiB=2048,science_seconds=540,hard_seconds=600,GPU=0),
        decision='feasibility/cost/tradeoffs only; no model calls or formal loss changes; do not select best replicate'))
    save('algebra_tests',algebra_test())
    oldbind=read(H.OUT/'binding.json');complete=read(H.FULL/'completion.json')
    with np.load(bind(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz',oldbind)) as z:
        types=z['types'].copy();fq=z['query'].copy();ff=z['families'].copy();fy=z['reference'].copy()
    saved=torch.load(bind(H.OLD/'PubMed_fixed_inputs.pt',oldbind),map_location='cpu',weights_only=False)
    inp=next(x for t,s,x in saved['inputs'] if t==1000)
    query,families,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));FAMILIES=families
    n=len(types);i=np.argsort(query[0]*n+query[1]);j=np.argsort(fq[0]*n+fq[1])
    assert np.array_equal(query[:,i],fq[:,j]) and np.array_equal(families[i],ff[j]) and np.array_equal(y[i],fy[j])
    assert n==192 and int(y.sum())==926 and len(y)==18336
    lookup={tuple(sorted(map(int,uv))):i for i,uv in enumerate(query.T)}
    assert len(lookup)==len(y)
    states=[];plans=[];tick=time.monotonic()
    for parent in ['CE','PS']:
        for draw in [0,1]:
            tag=f'PubMed_{parent}_CE_t1000_draw{draw}'
            with np.load(bind(H.FULL/(tag+'.npz'),complete['output_hashes'])) as z:
                p=z['probability'].copy();bits=z['bits'].copy();assert np.array_equal(z['reference'],y) and np.array_equal(z['relation'],families)
            assert np.all((p>0)&(p<1)) and np.isin(bits,[0,1]).all()
            parts=[]
            for r in range(10):
                limit();pop=population(query,families,bits,r,lookup)
                for rep in range(3):
                    seed=int.from_bytes(hashlib.sha256(f'switch_v1|{tag}|{r}|{rep}'.encode()).digest()[:8],'little')
                    ids=np.random.default_rng(seed).choice(len(pop),min(16,len(pop)),replace=False)
                    parts.append(dict(relation=r,replicate=rep,seed=seed,population=len(pop),moves=[pop[k] for k in ids]))
            plans.append(dict(state=tag,parts=parts));states.append((tag,p,bits,parts))
    save('frozen_candidates',plans);plan_seconds=time.monotonic()-tick
    # All candidate IDs frozen before any candidate structural scores.
    core=H.RootedScore(types,query,families,y,10);sc=H.ExplicitScore(core)
    jd=H.jdmod.JointDegree(dict(types=types,query=query,families=families,reference=y))
    channels=np.full((4,4),-1,int)
    for k,(a,b) in enumerate(core.triangle_pairs):channels[a,b]=channels[b,a]=10+k
    refconn=H.connectivity(H.adjacency(query,y,n));refcomp=sc.reftri/sc.reftri.sum()
    def describe(raw,a):
        x=H.describe(sc,jd,raw,a);tc=sc.triangle_counts(raw)[0]
        x['triangle_composition_TV']=float(abs(tc/tc.sum()-refcomp).sum()/2) if tc.sum() else None
        return x
    for m in list(sys.modules.values()):
        path=getattr(m,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py'):bind(path)
    bind(__file__);save('initial_binding',BIND)
    rows=[];summaries=[];relations=[];tests=[];bases=[];durations=[];cosines=[];unique=0
    for si,(tag,p,bits,parts) in enumerate(states):
        limit();tick=time.monotonic();z=np.log(p)-np.log1p(-p)
        with torch.no_grad():raw=core.features(torch.from_numpy(bits)).numpy()[:,:core.base_width]
        a=H.adjacency(query,bits,n);base=describe(raw,a);bases.append(dict(state=tag,**base))
        cache={};vectors=[];active=sum(x['population']>0 for x in parts if x['replicate']==0)
        for rep in range(3):
            v=np.zeros(len(bits));local=[]
            for part in [x for x in parts if x['replicate']==rep]:
                rr=[];r=part['relation'];m=len(part['moves'])
                for pos,move in enumerate(part['moves']):
                    limit();key=tuple(move);e,f,g,h=key
                    if key not in cache:
                        rrraw=raw.copy();aa=a.copy()
                        for slot in key:H.flip(rrraw,aa,query,types,channels,slot,r)
                        assert np.array_equal(rrraw[:,:10],raw[:,:10]) and aa.sum()==a.sum()
                        after=describe(rrraw,aa)
                        for name in ['relation_W1','joint_TV','joint_distance','low','hub','total_degree_W1']:
                            assert abs(after[name]-base[name])<1e-12
                        assert after['isolates']==base['isolates']
                        d=after['score']-base['score'];closure=(after['typed_triangle_L1']-base['typed_triangle_L1']+after['node_triangle_W1']-base['node_triangle_W1'])/8
                        assert abs(d-closure)<1e-12
                        cond=float(H.sigmoid(z[g]+z[h]-z[e]-z[f]));grad=d*cond*(1-cond)
                        row=dict(remove0=e,remove1=f,add0=g,add1=h,q=cond,h=float(grad),score_change=float(d),
                            **{name+'_change':float(after[name]-base[name]) for name in ['triangle_error','triangle_count','typed_triangle_L1','node_triangle_W1']},
                            triangle_composition_TV_change=(after['triangle_composition_TV']-base['triangle_composition_TV']) if after['triangle_composition_TV'] is not None and base['triangle_composition_TV'] is not None else None,
                            LCC_error_change=abs(after['LCC']-refconn['LCC'])-abs(base['LCC']-refconn['LCC']),
                            components_error_change=abs(after['components']-refconn['components'])-abs(base['components']-refconn['components']),
                            LCC_change=after['LCC']-base['LCC'],components_change=after['components']-base['components'])
                        assert row['triangle_composition_TV_change'] is not None # these four backgrounds have nonzero triangles
                        cache[key]=row;unique+=1
                    row=cache[key]
                    if rep==0 and pos==0:
                        bb=bits.copy();bb[[e,f]]=0;bb[[g,h]]=1
                        independent=independent_raw(query,bb,types,10,channels)
                        ar=raw.copy();at=a.copy()
                        for slot in key:H.flip(ar,at,query,types,channels,slot,r)
                        assert np.array_equal(independent,ar)
                        tests.append(dict(state=tag,relation=r,max_feature_error=float(abs(independent-ar).max())))
                    if m:v[list(key)]+=row['h']*np.array([-1.,-1.,1.,1.])/(m*active)
                    record=dict(state=tag,replicate=rep,relation=r,position=pos,**row)
                    rows.append(record);rr.append(record);local.append(record)
                relations.append(dict(state=tag,replicate=rep,relation=r,population=part['population'],empty=not m,**stats(rr)))
            node_gradient=np.zeros((n,10));np.add.at(node_gradient,(query[0],families),v);np.add.at(node_gradient,(query[1],families),v)
            assert abs(node_gradient).max()<1e-16
            dp=-p*(1-p)*v;node_dp=np.zeros((n,10));np.add.at(node_dp,(query[0],families),dp);np.add.at(node_dp,(query[1],families),dp)
            summaries.append(dict(state=tag,replicate=rep,active_relations=active,logit_gradient_norm=float(np.linalg.norm(v)),
                max_node_relation_logit_sum=float(abs(node_gradient).max()),max_node_relation_expected_degree_derivative=float(abs(node_dp).max()),
                **stats(local)))
            vectors.append(v)
        for aidx,bidx in itertools.combinations(range(3),2):
            cosines.append(dict(state=tag,replicate0=aidx,replicate1=bidx,cosine=H.cos(vectors[aidx],vectors[bidx])))
        np.savez_compressed(OUT/(tag+'_vectors.npz'),vectors=np.array(vectors))
        durations.append(dict(state=tag,seconds=time.monotonic()-tick,unique_moves=len(cache)))
        for name,obj in [('moves',rows),('summary',summaries),('relations',relations),('tests',tests),('bases',bases),('durations',durations),('replicate_cosines',cosines)]:save(name,obj)
        print(json.dumps(durations[-1]),flush=True)
        if si==0:
            estimate=durations[-1]['seconds']*3*2
            save('calibration',dict(plan_seconds=plan_seconds,first_state_seconds=durations[-1]['seconds'],remaining_estimate_with_factor2=estimate,remaining_budget=540-(time.monotonic()-START)))
            if estimate>540-(time.monotonic()-START):raise TimeoutError('full frozen panel estimated over budget')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in BIND.items())
    assert not torch.cuda.is_initialized();limit();save('binding',BIND)
    save('receipt',dict(status='complete',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        move_occurrences=len(rows),unique_scored_moves=unique,expected_occurrences=sum(len(p['moves']) for s in plans for p in s['parts']),
        states=4,replicates=3,independent_graph_feature_checks=len(tests),models=0,GPU_hours=0,updates=0,new_random_graphs=0,
        counterfactual_graphs=unique,assets_unchanged=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print((OUT/'receipt.json').read_text())

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise
