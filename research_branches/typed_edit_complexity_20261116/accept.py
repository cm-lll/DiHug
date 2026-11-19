"""Independent saved-output acceptance and crossed-seed summaries."""
from common import *
from metrics import compare
import argparse,csv,concurrent.futures

def training():
    records=[]
    for ds in ['imdb','pubmed','freebase','acm']:
        for seed in range(810,818):
            orders=[]
            for arm in ['M','F','W']:
                p=RUN/'fits'/ds/f'seed{seed}'/arm;r=read(p/'report.json')
                assert r['completed'] and r['updates']==4700 and len(r['history'])==100
                assert sha(p/'model.pt')==r['checkpoint_sha256'];assert sha(RUN/'cache'/ds/'data.npz')==r['cache_sha256']
                z=np.load(p/'validation.npz');logits=z['logits'].astype(np.float64);y=z['targets'];shift=logits-logits.max(1,keepdims=True)
                ce=float(np.mean(np.log(np.exp(shift).sum(1))-shift[np.arange(len(y)),y]));prob=np.exp(shift)/np.exp(shift).sum(1,keepdims=True)
                assert abs(ce-r['validation']['ce'])<2e-5 and np.max(abs(prob-z['probabilities']))<1e-5
                assert abs(float((logits.argmax(1)==y).mean())-r['validation']['top1'])<1e-6
                cache=np.load(RUN/'cache'/ds/'data.npz');val=cache['ids']>=30
                assert np.array_equal(y,cache['y'][val])
                from dihug.core.mixed_edit_diffusion import load_mixed_edit_checkpoint
                m,_,_=load_mixed_edit_checkpoint(p/'model.pt','cpu')
                _,_,independent_prob=c.eval_model(m,cache['x'][val],y,'cpu')
                assert np.max(abs(independent_prob-z['probabilities']))<2e-5
                records.append(r);orders.append(r['batch_order_sha256'])
            assert len(set(orders))==1,(ds,seed)
    put(RUN/'training_acceptance.json',{'passed':True,'fits':len(records),'formal_updates':sum(r['updates'] for r in records),'records':records})

def one(p):
    b=read(p/'binding.json');r=read(p/'report.json');g=restore_graph(read(p/'graph.json'));start=Path(b['start_path']);initial=restore_graph(read(start/'initial.json'))
    assert r['completed'] and sha(p/'graph.json')==r['graph_sha256']
    assert sha(start/'initial.json')==b['initial_sha256'] and sha(start/'target.json')==b['target_sha256']
    if b['arm']!='N':assert sha(b['checkpoint'])==b['checkpoint_sha256']
    else:assert b['training_seed'] is None and b['N_calls']==0
    g.validate();g.validate_disjoint_family_domains();assert np.array_equal(g.node_types,initial.node_types)
    degrees=[]
    for f in sorted(initial.family_specs):
        assert len(g.family_edges[f])==len(initial.family_edges[f])
        degrees.append(g.family_degrees(f))
        if b['kind']=='suffix':assert np.array_equal(g.family_degrees(f),initial.family_degrees(f))
    d=np.column_stack(degrees);target=np.array(read(start/'target.json'));residual=float(np.square(d-target).sum()/g.num_nodes)
    assert abs(residual-r['trace'][-1]['relation_degree_error_per_node'])<1e-6
    assert r['proposed']==b['config']['steps']*16 and sum(r['action_counts'].values())==b['config']['steps']
    v=compare(g,graph(b['dataset']),read(RUN/'data'/b['dataset']/'reference_metrics.json'))
    last=r['trace'][-1]
    for key in ['triangles','triangle_count']:
        if key in last:assert abs(last[key]-v['triangles'])<1e-4
    v.update({k:b[k] for k in ['dataset','kind','arm','training_seed','prefix_seed']})
    v.update(path=str(p),target_residual_per_node=residual,stage2_reached=last['phase']=='switch_refinement',
        transition_step=last['role_phase_transition_step'],wall_seconds=r['wall_seconds'],applied=r['applied'],
        noop_fraction=r['action_counts'].get('noop',0)/b['config']['steps'],graph_sha256=r['graph_sha256'])
    return v

def summarize(records):
    rng=np.random.default_rng(2026111608);groups=[];contrasts=[];seedrows=[]
    keys=['primary_error','triangle_vector_relative_l1','cycle_vector_relative_l1','triangles','typed_triangle_tv',
          'degree_w1','target_residual_per_node','stage2_reached','connected','lcc_fraction','wall_seconds','noop_fraction']
    for ds in ['imdb','pubmed','freebase','acm']:
        for kind in ['full','suffix']:
            rr=[r for r in records if r['dataset']==ds and r['kind']==kind]
            prefixes=sorted(set(r['prefix_seed'] for r in rr));matrices={}
            for arm in ['M','F','W','N']:
                ar=[r for r in rr if r['arm']==arm]
                if not ar:continue
                seeds=[None] if arm=='N' else list(range(810,818))
                sm=[]
                for seed in seeds:
                    sr=[r for r in ar if r['training_seed']==seed];assert len(sr)==len(prefixes)
                    row={'dataset':ds,'kind':kind,'arm':arm,'training_seed':seed,'n_starts':len(sr),
                         **{k:float(np.mean([r[k] for r in sr])) for k in keys}}
                    seedrows.append(row);sm.append(row)
                groups.append({'dataset':ds,'kind':kind,'arm':arm,'unique_outputs':len(ar),'training_repeats':0 if arm=='N' else len(seeds),
                    'available_starts':prefixes,'start_denominator':8,
                    'metrics':{k:{'mean':float(np.mean([r[k] for r in ar])),
                        'output_sd':float(np.std([r[k] for r in ar],ddof=1)) if len(ar)>1 else None,
                        'training_seed_mean_sd':float(np.std([r[k] for r in sm],ddof=1)) if len(sm)>1 else None} for k in keys}})
                matrices[arm]=np.array([[next(r['primary_error'] for r in ar if r['prefix_seed']==p and r['training_seed']==s) for p in prefixes] for s in seeds])
            if len(matrices)!=4:continue
            ns=len(prefixes);si=rng.integers(0,8,size=(20000,8));pi=rng.integers(0,ns,size=(20000,ns))
            for arm in ['F','W','N']:
                difference=np.broadcast_to(matrices[arm],matrices['M'].shape)-matrices['M']
                boots=difference[si[:,:,None],pi[:,None,:]].mean((1,2))
                ci=np.quantile(boots,[.05/6,1-.05/6]).tolist() if ns>=2 else None
                contrasts.append({'dataset':ds,'kind':kind,'contrast':arm+'-M','mean':float(difference.mean()),'paired_training_seed_means':difference.mean(1).tolist(),
                    'interval_98_333_percent':ci,'nominal_family':'three contrasts within this reference and estimand only',
                    'conditional_prefix_producer':'M810' if kind=='suffix' else None,'starts':prefixes,
                    'M_advantage_established_in_this_family':bool(ci and ci[0]>0),'equivalence_claim':False})
    return groups,contrasts,seedrows

def generation():
    ledger=read(RUN/'generation_ledger.json');expected={str(Path(x['output_path'])) for x in ledger['requests']}
    actual={str(p.parent) for p in (RUN/'outputs').rglob('report.json')};assert actual==expected,(len(actual),len(expected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,map(Path,sorted(actual))))
    groups,contrasts,seedrows=summarize(records)
    put(RUN/'accepted_records.json',records);put(RUN/'summaries.json',{'groups':groups,'contrasts':contrasts,'training_seed_means':seedrows})
    cols=['dataset','kind','arm','training_seed','prefix_seed','primary_error','triangle_vector_relative_l1','cycle_vector_relative_l1','triangles','typed_triangle_tv','degree_w1','target_residual_per_node','stage2_reached','connected','lcc_fraction','wall_seconds','noop_fraction','path']
    with (RUN/'per_output.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(records)
    with (RUN/'per_training_seed.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(seedrows[0]));w.writeheader();w.writerows(seedrows)
    put(RUN/'generation_acceptance.json',{'passed':True,'outputs':len(records),'full':sum(r['kind']=='full' for r in records),
        'suffix':sum(r['kind']=='suffix' for r in records),'prefix_availability':ledger['prefix_availability'],
        'verification':'independent CPU endpoints and metrics, provenance, inventories and stage2 degree invariants; not full GPU proposal replay'})

if __name__=='__main__':
    setup();p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','generation']);a=p.parse_args()
    training() if a.stage=='training' else generation()
