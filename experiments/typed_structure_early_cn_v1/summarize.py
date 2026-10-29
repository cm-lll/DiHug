"""Final EMA A/C/D/E evaluation; original and induced LCC always separate.
Frozen pure metric functions; no model import. Missing/failed cells remain.
"""
import argparse,pathlib,time,json,itertools,math,collections,csv
import numpy as np
from runtime_contract import ROOT,jobs,read,save,sha,source_binding
from lcc_eval import metrics as M
from evaluation_fields import scalar
Counter=collections.Counter

def mean(values):
    return sum(values)/len(values) if values and all(v is not None and math.isfinite(float(v)) for v in values) else None
def delta(a,b):return a-b if a is not None and b is not None else None
def write_json(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
def factory():
    v=ROOT/'lcc_eval/vendor';return M.frozen(v/'frozen_metrics.py',v/'aligned_evaluate.py',v/'hgen_utils.py')
def load_graph(path,reference=False):
    with np.load(path,allow_pickle=False) as z:
        g=M.make_graph(z['types'],z['edges'].T if reference else z['edges'],z['schema'])
        if 'families' in z:np.testing.assert_array_equal(g['families'],z['families'])
    return g
def evaluate_graph(g,ref,rs,F,names,check):
    s=M.stats(g,F,names,check);comp=M.compare(g,s,ref,rs,F,names)
    raw,dist=scalar(s,rs,comp,names);rv,_=scalar(rs,rs,M.compare(ref,rs,ref,rs,F,names),names)
    value=dict(statistics=s,comparison=comp,raw=raw,reference=rv,absolute_error={k:None if v is None or rv.get(k) is None else abs(v-rv[k]) for k,v in raw.items()},distribution_error=dist)
    value['undefined']={group:{k:'undefined_zero_denominator_missing_type_or_unimplemented_metric' for k,v in value[group].items() if v is None} for group in ('raw','reference','absolute_error','distribution_error')}
    return value
def aggregate(records,templates):
    index={(r['dataset'],r['arm'],r['sample_seed'],r['version']):r for r in records};means=[];pairs=[];pairmeans=[];clean=[]
    protocol=read(ROOT/'protocol.json');samples=protocol['sample_seeds']
    for dataset in protocol['train_seeds']:
        seed=protocol['train_seeds'][dataset]
        for version in ('original','lcc'):
            for group in ('raw','absolute_error','distribution_error'):
                keys=set(templates[dataset][group])
                for r in records:
                    if r['dataset']==dataset:keys.update(r.get(group,{}))
                for metric in sorted(keys):
                    def value(arm,sample,v=version):return index.get((dataset,arm,sample,v),{}).get(group,{}).get(metric)
                    for arm in protocol['evaluation_arms'][dataset]:
                        vals=[value(arm,s) for s in samples];m=mean(vals)
                        means.append(dict(dataset=dataset,train_seed=seed,arm=arm,version=version,group=group,metric=metric,value=m,reason=None if m is not None else 'both_samples_required_and_defined'))
                        if version=='lcc':
                            for s in samples:clean.append(dict(dataset=dataset,train_seed=seed,arm=arm,sample_seed=s,group=group,metric=metric,difference=delta(value(arm,s),value(arm,s,'original'))))
                    for left,right in protocol['contrasts'][dataset]:
                        vals=[]
                        for s in samples:
                            d=delta(value(left,s),value(right,s));vals.append(d)
                            pairs.append(dict(dataset=dataset,train_seed=seed,sample_seed=s,contrast=left+'-'+right,version=version,group=group,metric=metric,difference=d,reason=None if d is not None else 'missing_or_undefined_required_pair'))
                        pairmeans.append(dict(dataset=dataset,train_seed=seed,contrast=left+'-'+right,version=version,group=group,metric=metric,difference=mean(vals),reason=None if mean(vals) is not None else 'both_pairs_required_and_defined'))
    return dict(cell_means=means,paired=pairs,paired_means=pairmeans,lcc_minus_original=clean)
def diversity(a,b,sa=None,sb=None,F=None):
    x=set(map(tuple,a['original_ids'][a['edges']].T));y=set(map(tuple,b['original_ids'][b['edges']].T))
    na,nb=set(a['original_ids']),set(b['original_ids'])
    value=dict(node_jaccard=len(na&nb)/len(na|nb),labeled_edge_symmetric_difference=len(x^y),labeled_edge_jaccard=len(x&y)/len(x|y) if x|y else None,
        nodes=[len(a['types']),len(b['types'])],edges=[len(x),len(y)],degree_W1=M.w1(a['total'],b['total']),
        exact_labeled_duplicate=np.array_equal(a['original_ids'],b['original_ids']) and x==y,
        nonisomorphic_by_size=len(a['types'])!=len(b['types']) or len(x)!=len(y),
        limitation='labeled edge difference is not an isomorphism-invariant diversity guarantee; only two samples')
    if sa is not None and sb is not None and F is not None:
        value.update(triangle_composition_TV=F['tv'](Counter(sa['typed_triangles']),Counter(sb['typed_triangles'])),
            MP2_composition_TV=F['tv'](Counter(sa['mp'][1]['counts']),Counter(sb['mp'][1]['counts'])),
            MP3_composition_TV=F['tv'](Counter(sa['mp'][2]['counts']),Counter(sb['mp'][2]['counts'])))
    return value
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    run=pathlib.Path(a.run);out=pathlib.Path(a.output);derived=run/'derived_lcc_evaluation';derived.mkdir(exist_ok=False)
    started=time.time();records=[];inventory=[];graphs={};templates={};errors=[]
    def check():
        if time.time()>=a.deadline:raise TimeoutError('evaluation deadline')
    try:
        cfg=read(ROOT/'package/config.json');F=factory();refs={}
        for dataset in ('IMDB','PubMed'):
            check();ref=load_graph(ROOT/'package'/cfg['datasets'][dataset]['file'],True);names=cfg['datasets'][dataset]['type_names'];rs=M.stats(ref,F,names,check)
            refs[dataset]=(ref,rs,names);templates[dataset]=evaluate_graph(ref,ref,rs,F,names,check);write_json(derived/(dataset+'_reference.json'),templates[dataset])
        for job in jobs()['sampling']:
            check();folder=run/job['directory'];item=dict(job=job,status='missing',reason='sampling output absent');inventory.append(item)
            if (folder/'summary.json').exists():item['worker']=read(folder/'summary.json')
            if not (folder/'graph.npz').exists():continue
            if item.get('worker',{}).get('status')!='complete':item.update(status='incomplete',reason='worker not complete');continue
            binding=read(folder/'binding.json')
            if binding['job']!=job or sha(folder/'graph.npz')!=binding['graph_sha256']:raise ValueError('graph identity/hash mismatch')
            ref,rs,names=refs[job['dataset']];g=load_graph(folder/'graph.npz');kept,mapping=M.lcc(g)
            target=derived/job['id'];target.mkdir();mapping.update(source_sha256=sha(folder/'graph.npz'),source=str(folder/'graph.npz'))
            np.testing.assert_array_equal(kept['degree'],g['degree'][mapping['selected_local_ids']])
            np.savez_compressed(target/'lcc.npz',types=kept['types'],edges=kept['edges'],families=kept['families'],schema=kept['schema'],new_to_original=kept['original_ids'],old_to_new=np.array(mapping['old_to_new']))
            mapping['derived_sha256']=sha(target/'lcc.npz');write_json(target/'mapping.json',mapping)
            for version,graph in [('original',g),('lcc',kept)]:
                check();value=evaluate_graph(graph,ref,rs,F,names,check);value.update(dataset=job['dataset'],arm='E_earlyCN',train_seed=job['train_seed'],sample_seed=job['sample_seed'],version=version)
                records.append(value);write_json(target/(version+'.json'),value);graphs[job['dataset'],'E_earlyCN',job['sample_seed'],version]=graph
            type_counts=np.bincount(g['types'],minlength=4);kept_counts=np.bincount(kept['types'],minlength=4)
            before=np.bincount(g['families'],minlength=10);after=np.bincount(kept['families'],minlength=10)
            item.update(status='complete',reason=None,node_retention=len(kept['types'])/len(g['types']),edge_retention=kept['edges'].shape[1]/g['edges'].shape[1] if g['edges'].shape[1] else None,mapping=str(target/'mapping.json'),
                type_counts_original=type_counts.tolist(),type_counts_lcc=kept_counts.tolist(),type_nodes_lost=(type_counts-kept_counts).tolist(),
                type_retention=[int(b)/int(a) if a else None for a,b in zip(type_counts,kept_counts)],
                family_edges_original=before.tolist(),family_edges_lcc=after.tolist(),family_retention=[int(b)/int(a) if a else None for a,b in zip(before,after)])
    except BaseException as e:errors.append(repr(e))
    seen={x['job']['id'] for x in inventory}
    inventory.extend(dict(job=j,status='not_evaluated',reason='earlier_failure_or_deadline') for j in jobs()['sampling'] if j['id'] not in seen)
    new_record_count=len(records)
    try:records.extend(old_records())
    except BaseException as e:errors.append('old controls: '+repr(e))
    aggregation=aggregate(records,templates) if len(templates)==2 else dict(reason='reference evaluation incomplete')
    ds=[];record_index={(r['dataset'],r['arm'],r['sample_seed'],r['version']):r for r in records}
    for job in jobs()['training']:
        for version in ('original','lcc'):
            x=[graphs.get((job['dataset'],'E_earlyCN',s,version)) for s in (2026102821,2026102822)]
            keys=[(job['dataset'],'E_earlyCN',s,version) for s in (2026102821,2026102822)]
            valid=all(k in record_index for k in keys)
            ds.append(dict(job=job,version=version,metrics=diversity(*x,record_index[keys[0]]['statistics'],record_index[keys[1]]['statistics'],F) if valid else None,reason=None if valid else 'both_graphs_required'))
    result=dict(status='complete' if new_record_count==8 and not errors and all(x['status']=='complete' for x in inventory) else 'incomplete',records=records,inventory=inventory,diversity=ds,errors=errors,seconds=time.time()-started,source_sha256=source_binding()[1],**aggregation)
    write_json(out,result)
    for name in ('cell_means','paired','paired_means','lcc_minus_original'):
        rows=result.get(name,[])
        if rows:
            with (derived/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return 0 if result['status']=='complete' else 2
def old_records():
    b=read(ROOT/'baseline_bindings.json')['old_evaluation']
    if sha(b['path'])!=b['sha256']:raise ValueError('old result hash')
    p=read(ROOT/'protocol.json');result=[]
    for r in read(b['path'])['records']:
        d=r['dataset']
        if r['train_seed']==p['train_seeds'][d] and r['arm'] in p['evaluation_arms'][d] and r['sample_seed'] in p['sample_seeds']:
            result.append(dict(r,source='sealed old control',source_sha256=b['sha256']))
    expected={(d,a,s,v) for d in p['train_seeds'] for a in p['evaluation_arms'][d] if a!='E_earlyCN' for s in p['sample_seeds'] for v in ('original','lcc')}
    actual=[(r['dataset'],r['arm'],r['sample_seed'],r['version']) for r in result]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('missing/duplicate old controls')
    return result
if __name__=='__main__':raise SystemExit(main())
