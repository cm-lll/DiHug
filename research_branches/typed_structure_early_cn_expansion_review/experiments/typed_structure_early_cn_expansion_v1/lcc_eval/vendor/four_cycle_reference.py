"""Bounded read-only cross-reference/collection audit. No model imports/calls."""
import bisect
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import statistics
import time

import networkx as nx
import numpy as np

from audit_legacy_roles_20261007 import independent_metrics
from sparse_diffusion.structure_energy.graph import load_typed_graph, TypedGraph
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import ExchangeableRoleDegreePrior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/legacy_collection_audit_20261007'
BASE=ROOT/'results/legacy_roles_audit_20261007'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name,d):
    tmp=OUT/(name+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False))
    tmp.replace(OUT/name)


def encode(payload):
    g=nx.Graph()
    g.add_nodes_from((i,{'type':payload['type_names'][t]}) for i,t in enumerate(payload['node_types']))
    edge_set=set()
    for family,v in payload['families'].items():
        for u,w in v['edges']:
            g.add_edge(u,w,family=family)
            edge_set.add((family,min(u,w),max(u,w)))
    families=sorted(payload['families'])
    fd={i:Counter(g[i][j]['family'] for j in g[i]) for i in g}
    hist=Counter((g.nodes[i]['type'],tuple(fd[i][f] for f in families)) for i in g)
    signature=(tuple(sorted(Counter(payload['node_types']).items())),
               tuple((f,len(payload['families'][f]['edges'])) for f in families),tuple(sorted(hist.items())))
    return {'g':g,'signature':signature,'edges':edge_set,'wl':None,
            'family_counts':tuple((f,len(payload['families'][f]['edges'])) for f in families)}


def timeout_handler(*_):
    raise TimeoutError('isomorphism deadline')


def iso(a,b):
    if a['signature']!=b['signature']:
        return 'different_invariant'
    if a['edges']==b['edges'] and dict(a['g'].nodes(data='type'))==dict(b['g'].nodes(data='type')):
        return 'same_labeled'
    for x in [a,b]:
        if x['wl'] is None:
            x['wl']=nx.weisfeiler_lehman_graph_hash(x['g'],edge_attr='family',node_attr='type')
    if a['wl']!=b['wl']:
        return 'different_wl'
    signal.setitimer(signal.ITIMER_REAL,0.5)
    try:
        same=nx.is_isomorphic(a['g'],b['g'],node_match=nx.algorithms.isomorphism.categorical_node_match('type',None),
                              edge_match=nx.algorithms.isomorphism.categorical_edge_match('family',None))
        return 'same_vf2' if same else 'different_vf2'
    except TimeoutError:
        return 'unknown'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


def four_cycles(enc):
    # Each simple (not necessarily induced) 4-cycle has two opposite pairs.
    g=enc['g']; counts=Counter()
    for u in g:
        for v in range(u+1,len(g)):
            common=sorted(set(g[u]) & set(g[v]))
            for a,b in itertools.combinations(common,2):
                cycle=[u,a,v,b]
                reps=[]
                for order in [cycle,cycle[::-1]]:
                    for s in range(4):
                        z=order[s:]+order[:s]
                        reps.append(tuple((g.nodes[z[k]]['type'],g[z[k]][z[(k+1)%4]]['family']) for k in range(4)))
                counts[repr(min(reps))]+=1
    assert all(c%2==0 for c in counts.values())
    return {k:v//2 for k,v in counts.items()}


def graphicality(folder,payload):
    pp=folder/'exchangeable_role_degree_prior.json';sp=folder/'sampling_state.json'
    if not pp.exists() or not sp.exists():
        return {'status':'missing_prior_or_assignment_state'}
    state=json.loads(sp.read_text());prior=ExchangeableRoleDegreePrior.load(pp)
    graph=TypedGraph.from_json_dict(payload)
    target=prior.sample_assignment_for_graph(graph,state['role_assignment_seed'])
    h=hashlib.sha256('|'.join(prior.family_names).encode()+np.ascontiguousarray(target).tobytes()).hexdigest()
    if h!=state['role_assignment_hash']:
        return {'status':'assignment_hash_mismatch','computed':h,'saved':state['role_assignment_hash']}
    valid={}
    for j,f in enumerate(prior.family_names):
        spec=graph.family_specs[f];a=sorted(map(int,target[list(spec.src_nodes),j]),reverse=True)
        if spec.bipartite:
            b=sorted(map(int,target[list(spec.dst_nodes),j]));prefix=[0]+list(itertools.accumulate(b))
            left=0; ok=sum(a)==sum(b)
            for k,d in enumerate(a,1):
                left+=d;t=bisect.bisect_right(b,k)
                ok=ok and left<=prefix[t]+k*(len(b)-t)
            valid[f]=ok
        else:
            valid[f]=nx.is_graphical(a,method='eg')
    return {'status':'checked','by_family':valid,'all_graphical':all(valid.values())}


def partition_summary(items):
    n=len(items);parent=list(range(n));pair_results=[];different=set();unknown=[]
    def find(i):
        while parent[i]!=i:
            i=parent[i]
        return i
    for i,j in itertools.combinations(range(n),2):
        v=iso(items[i]['enc'],items[j]['enc']);pair_results.append([i,j,v])
        if v.startswith('same'):
            parent[find(j)]=find(i)
        elif v.startswith('different'):
            different.add((i,j))
        else:
            unknown.append([i,j])
    components=defaultdict(list)
    for i in range(n):components[find(i)].append(i)
    # A clique of proven distinct representative classes is a valid lower bound.
    clique=[]
    for comp in components.values():
        if all(any(tuple(sorted((i,j))) in different for i in comp for j in old) for old in clique):
            clique.append(comp)
    exact=not unknown
    sizes=[len(v) for v in components.values()]
    e8=sum(1-(math.comb(n-s,8)/math.comb(n,8) if n-s>=8 else 0) for s in sizes) if exact and n>=8 else None
    distances=[]
    for i,j in itertools.combinations(range(n),2):
        a,b=items[i]['enc']['edges'],items[j]['enc']['edges']
        distances.append(1-len(a&b)/max(1,len(a|b)))
    return dict(n=n,classes_lower=len(clique),classes_upper=len(sizes),exact=exact,
                E8=e8,maximum_class_fraction=max(sizes)/n if exact else None,
                family_count_vectors=len(set(x['enc']['family_counts'] for x in items)),
                joint_degree_signatures=len(set(x['enc']['signature'] for x in items)),
                raw_id_edge_jaccard_distance_mean=statistics.mean(distances) if distances else None,
                jaccard_caveat='Descriptive only: anonymous role reassignments can inflate labeled-edge differences.',
                pair_results=pair_results,unknown_pairs=unknown)


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,timeout_handler)
    cache={}
    old=json.loads((BASE/'saved_audit.json').read_text())
    for row in old['rows']:
        p=ROOT/'results'/row['batch']/row['dataset']/f"seed{row['seed']}"/row['arm']/'final_graph.json'
        if digest(p)==old['source_sha256'][str(p.relative_to(ROOT))]:cache[digest(p)]=row['metrics']
    cross=json.loads((BASE/'cross_dataset_inventory.json').read_text())
    for row in cross['rows']:
        if row.get('independent_metrics_status')=='passed':cache[row['graph_sha256']]=row['recomputed_metrics']
    paths=sorted(set([ROOT/p for p in old['source_sha256'] if p.endswith('gate_report.json')]+[ROOT/r['path'] for r in cross['rows']]))
    paths+=sorted((BASE/'replay').glob('*/seed*/gate_report.json'))
    refs={};reference_records=[]
    for ds,folder in [('imdb','IMDB_movie_subgraphs'),('pubmed','PubMed_baseline_subgraphs'),('syn100','HGEN_syn_100_subgraphs'),('syn200','HGEN_syn_200_subgraphs'),('syn500','HGEN_syn_500_subgraphs'),('dblp','DBLP_four_area_subgraphs')]:
        frozen=ROOT/f'paper_tables/frozen_main16/targets/{ds}.json'
        source=ROOT/f'data/{folder}/subgraph_000'
        if frozen.exists():
            payload=json.loads(frozen.read_text());hashes={str(frozen.relative_to(ROOT)):digest(frozen)};origin='archived_main16_target'
        else:
            payload=load_typed_graph(source,active_only=True).to_json_dict()
            hashes={str(p.relative_to(ROOT)):digest(p) for p in [source/'nodes.pt',source/'edges.pt',source/'meta.json']};origin='current_dataset_not_certified_historical_snapshot'
        enc=encode(payload);metrics=independent_metrics(payload)
        ref=dict(payload=payload,enc=enc,metrics=metrics,origin=origin,hashes=hashes)
        ref['four_cycles']=four_cycles(enc) if len(enc['g'])<=600 else None
        refs[ds]=ref;reference_records.append({k:v for k,v in ref.items() if k not in ['payload','enc']}|{'dataset':ds})
    rows=[];groups=defaultdict(list);nongeneration=[];fresh=0
    for index,p in enumerate(paths):
        d=json.loads(p.read_text());gp=p.with_name('final_graph.json')
        if 'final' not in d or not gp.exists():
            nongeneration.append(dict(path=str(p.relative_to(ROOT)),reason='no_final_graph_or_generation_metrics',report=d));continue
        payload=json.loads(gp.read_text());h=digest(gp)
        metrics=cache.get(h)
        if metrics is None:metrics=independent_metrics(payload);cache[h]=metrics;fresh+=1
        mismatch={k:metrics[k]-d['final'][k] for k in metrics if not np.isclose(metrics[k],d['final'][k],atol=1e-8,rtol=1e-10)}
        batch=p.parts[len(ROOT.parts)+1]
        if batch=='legacy_roles_audit_20261007':batch='replay_20261007';ds=p.parts[-3];arm='context'
        elif batch=='dblp_role_completion_150k_0717':ds='dblp';arm='completion'
        else:
            ds=p.parts[len(ROOT.parts)+2].split('_s')[0]
            arm=p.parts[-3] if p.parts[-2]=='sample' else p.parts[-2]
        enc=encode(payload);ref=refs[ds]
        compatible=all(np.isclose(ref['metrics'][k],v,atol=1e-8,rtol=1e-10) for k,v in d['real'].items() if k in ref['metrics']) and payload['node_types']==ref['payload']['node_types'] and sorted(payload['families'])==sorted(ref['payload']['families'])
        copy=iso(enc,ref['enc']) if compatible else 'unknown_reference_mismatch'
        four=four_cycles(enc) if len(enc['g'])<=600 else None
        ferror=None
        if four is not None and compatible:
            keys=set(four)|set(ref['four_cycles'])
            ferror=sum(abs(four.get(k,0)-ref['four_cycles'].get(k,0)) for k in keys)/max(1,sum(ref['four_cycles'].values()))
        trierr=abs(metrics['triangles']-d['real']['triangles'])/max(1,d['real']['triangles'])
        clerr=abs(metrics['clustering']-d['real']['clustering'])/max(1e-12,d['real']['clustering'])
        trace=d.get('trace',[]);last=trace[-1] if trace else {}
        row=dict(path=str(p.relative_to(ROOT)),graph_sha256=h,report_sha256=digest(p),dataset=ds,batch=batch,arm=arm,seed=d.get('seed'),
                 steps=d.get('steps'),policy=d.get('phase_policy','unrecorded'),initial_source=d.get('initial_graph_source'),
                 context_scale=d.get('context_scale'),last_phase=last.get('phase'),transition=last.get('role_phase_transition_step'),
                 role_error_per_node=last.get('relation_degree_error_per_node'),role_threshold=d.get('role_error_per_node_threshold'),
                 metrics=metrics,metric_mismatches=mismatch,reference_metrics=d['real'],reference_compatible=compatible,
                 reference_origin=ref['origin'],copy_check=copy,typed_four_cycles=four,typed_four_error=ferror,
                 triangle_relative_error=trierr,clustering_relative_error=clerr,target_feasibility=graphicality(p.parent,payload))
        rows.append(row);groups[(batch,ds,arm)].append(dict(row=row,enc=enc))
        if index%10==0:print('audited',index+1,'/',len(paths),flush=True)
    summary=[]
    for key,items in sorted(groups.items()):
        ss=partition_summary(items)
        sensitivity=[]
        for tau in [.05,.10,.20]:
            if any(x['row']['reference_metrics']['triangles']==0 for x in items):
                sensitivity.append(dict(tolerance=tau,status='not_applicable_zero_reference_triangles',count=None,denominator=len(items),collection=None))
                continue
            chosen=[x for x in items if x['row']['reference_metrics']['triangles']>0 and x['row']['triangle_relative_error']<=tau and x['row']['clustering_relative_error']<=tau and x['row']['copy_check'].startswith('different')]
            sensitivity.append(dict(tolerance=tau,status='descriptive_only',count=len(chosen),denominator=len(items),collection=partition_summary(chosen) if chosen else None))
        summary.append(dict(batch=key[0],dataset=key[1],arm=key[2],collection=ss,
                            final_phases=dict(Counter(x['row']['last_phase'] for x in items)),
                            reference_copy_status=dict(Counter(x['row']['copy_check'] for x in items)),
                            tri_clustering_sensitivity=sensitivity))
    result=dict(status='completed',rows=rows,groups=summary,reference_records=reference_records,nongeneration_reports=nongeneration,
                graph_records=len(rows),fresh_metric_computations=fresh,wall_seconds=time.perf_counter()-started,
                limits=['No training, new sampling or model calls.','Groups with n<8 have no E8; generated seeds are not training replicates.',
                        '5/10/20% sensitivity is retrospective descriptive, not a frozen success gate. It omits behavioral fidelity and is not qualified generation utility.',
                        'Current reference compatibility by metrics does not prove original adjacency identity.',
                        'Typed four-cycles only computed for <=600 nodes. Large-graph typed motifs and behavioral utility remain unassessed.',
                        'WL equality is never accepted as isomorphism; deadline unknowns remain unknown.',
                        'Continuation runs and paired arms are not independent new references.'])
    save('report.json',result)
    save('decision.json',dict(status='audit_complete_not_joint_utility_pass',graph_records=len(rows),
         metric_mismatches=sum(bool(r['metric_mismatches']) for r in rows),sf0='issuance_only_no_completion_found',
         new_training=0,new_scientific_samples=0,report='report.json'))
    print(json.dumps({k:v for k,v in result.items() if k in ['status','graph_records','fresh_metric_computations','wall_seconds']}),flush=True)


if __name__=='__main__':main()
