"""D/E routing over frozen OABC receipts. No model import or graph generation."""
import argparse
import json
import pathlib
import sys
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'package'))
from runtime_plan import TRAIN_SEEDS, SAMPLE_SEEDS, task_id, sha
from summary_metrics import graph_summary, strict_mean, paired, finite

CONTRASTS = (('D','A'), ('D','B'), ('E','D'), ('E','C'), ('E','A'))


def key(dataset, train_seed, sample_seed, arm):
    if dataset not in TRAIN_SEEDS or train_seed not in TRAIN_SEEDS[dataset] or sample_seed not in SAMPLE_SEEDS or arm not in 'OABCDE':
        raise ValueError('unknown result identity')
    return dataset, train_seed, sample_seed, arm


def safe_file(root, relative):
    p = (root / relative).resolve()
    if root.resolve() not in p.parents or not p.is_file():
        raise ValueError('missing or escaping payload path: ' + str(relative))
    return p


def check_w1(metrics):
    w = metrics.get('W1_definitions', {})
    if finite(w.get('incident4')) and finite(w.get('all10_compat')):
        if not np.isclose(w['all10_compat'], .4*w['incident4'], rtol=1e-12, atol=1e-12):
            raise ValueError('incident4/all10 identity failed')


def graph_arrays(path, asset):
    with np.load(path, allow_pickle=False) as z:
        types=z['types'];edges=z['edges'];families=z['families'];schema=z['schema']
    np.testing.assert_array_equal(types, asset['types']);np.testing.assert_array_equal(schema, asset['schema'])
    if edges.ndim!=2 or edges.shape[0]!=2 or families.shape!=(edges.shape[1],):raise ValueError('graph shape')
    if edges.dtype.kind not in 'iu' or families.dtype.kind not in 'iu':raise ValueError('integer graph fields required')
    if np.any(edges<0) or np.any(edges>=len(types)) or np.any(edges[0]>=edges[1]):raise ValueError('canonical simple endpoints')
    if len(set(map(tuple,edges.T)))!=edges.shape[1]:raise ValueError('duplicate edge')
    if np.any(families<0) or np.any(families>=len(schema)):raise ValueError('family bounds')
    np.testing.assert_array_equal(np.sort(types[edges],axis=0).T,schema[families])
    return types,edges


def summarize_graph(dataset, metrics, asset, edges):
    check_w1(metrics);reference=metrics['reference_metrics'];values=graph_summary(metrics,reference)
    focus={}
    if dataset=='IMDB':
        for name,pattern in [('DAM','(0, 1, 2)'),('DAA','(0, 1, 1)'),('AAM','(1, 1, 2)')]:
            v=(metrics.get('typed_triangles') or {}).get(pattern,0) if metrics.get('typed_triangles') is not None else None
            r=(reference.get('typed_triangles') or {}).get(pattern,0) if reference.get('typed_triangles') is not None else None
            values['raw'][name]=v;values['reference'][name]=r;values['absolute_error'][name]=abs(v-r) if v is not None and r is not None else None
            focus[name]=dict(raw=v,reference=r,absolute_error=values['absolute_error'][name])
        degree=np.bincount(edges.ravel(),minlength=len(asset['types']))
        ref_degree=np.bincount(asset['edges'].ravel(),minlength=len(asset['types']))
        order=np.sort(degree[asset['types']==3])[::-1].tolist();target=np.sort(ref_degree[asset['types']==3])[::-1].tolist()
        focus['Genre_sorted_degree_spectrum']=dict(raw=order,reference=target)
        for i,(v,r) in enumerate(zip(order,target),1):
            name='Genre_sorted_degree_%02d'%i
            values['raw'][name]=v;values['reference'][name]=r;values['absolute_error'][name]=abs(v-r)
    return values,focus


def assemble(records, templates, worker_status):
    """Pure identity-based aggregation; no pairing by file ordering or graph order."""
    indexed={}
    for row in records:
        ident=key(row['dataset'],row['train_seed'],row['sample_seed'],row['arm'])
        if ident in indexed:raise ValueError('duplicate result identity: '+str(ident))
        indexed[ident]=row
    cells={};contrasts=[];mapping=[]
    for dataset,seeds in TRAIN_SEEDS.items():
        for train_seed in seeds:
            for arm in 'OABCDE':
                label='%s_s%d_%s'%(dataset,train_seed,arm)
                status=worker_status.get(label,dict(status='not_started'))
                rows=[];graphs={}
                for sample in SAMPLE_SEEDS:
                    ident=key(dataset,train_seed,sample,arm)
                    row=indexed.get(ident,dict(dataset=dataset,train_seed=train_seed,sample_seed=sample,arm=arm,status='missing_or_not_started',reason='no graph result',worker_status=status))
                    graphs[str(sample)]=row;rows.append(row)
                    mapping.append(dict(dataset=dataset,train_seed=train_seed,sample_seed=sample,arm=arm,status=row['status'],source_commit=row.get('source_commit'),graph_sha256=row.get('graph_sha256'),path=row.get('path')))
                ready=all(r['status']=='evaluated' for r in rows)
                diversity=None
                if ready and all('edge_set' in r for r in rows):
                    a,b=[set(map(tuple,r['edge_set'])) for r in rows];union=len(a|b)
                    diversity=dict(labelled_graphs_equal=a==b,edge_symmetric_difference=len(a^b),edge_jaccard=len(a&b)/union if union else 1.,scope='two labelled graphs, not population or isomorphism diversity')
                cells[label]=dict(worker_status=status,graphs=graphs,mean=strict_mean([r['values'] for r in rows]) if ready else None,
                                  mean_reason=None if ready else 'one_or_more_required_graphs_missing_or_failed',diversity=diversity)
            for left,right in CONTRASTS:
                pairs=[]
                for sample in SAMPLE_SEEDS:
                    a=indexed.get(key(dataset,train_seed,sample,left));b=indexed.get(key(dataset,train_seed,sample,right))
                    av=a.get('values') if a and a['status']=='evaluated' else None
                    bv=b.get('values') if b and b['status']=='evaluated' else None
                    pairs.append(dict(sample_seed=sample,status='complete_graph_pair' if av is not None and bv is not None else 'missing_graph',difference=paired(av,bv,templates[dataset])))
                ready=all(p['status']=='complete_graph_pair' for p in pairs)
                contrasts.append(dict(dataset=dataset,train_seed=train_seed,contrast=left+'-'+right,pairs=pairs,
                                      mean=strict_mean([p['difference'] for p in pairs]) if ready else None,
                                      mean_reason=None if ready else 'one_or_more_required_graph_pairs_missing'))
    new_ready=sum(r['status']=='evaluated' and r['arm'] in 'DE' for r in indexed.values())
    old_ready=sum(r['status']=='evaluated' and r['arm'] in 'OABC' for r in indexed.values())
    complete=new_ready==16 and old_ready==32 and all(v['worker_status'].get('status')=='complete' for v in cells.values())
    return dict(status='complete' if complete else 'incomplete',new_evaluated_graphs=new_ready,old_evaluated_graphs=old_ready,
                expected_new_graphs=16,expected_old_graphs=32,cells=cells,contrasts=contrasts,matching=mapping,
                W1_definition='incident4 primary; all10 compatibility = 0.4 * incident4',
                interpretation='left minus right; raw differences are not improvements; error differences <0 are metric-specific gains',
                scope='two existing training seeds shown separately; no new held-out or large-graph claim')


def collect(run, check=lambda:None):
    run=pathlib.Path(run).resolve();history=ROOT/'history';idx=json.loads((history/'index.json').read_text())
    # The index itself is part of approved source_binding; every consumed payload
    # is verified against the hash extracted from the fixed published ZIP.
    for relative,receipt in idx['files'].items():
        check();p=safe_file(history,relative)
        if sha(p)!=receipt['sha256'] or p.stat().st_size!=receipt['bytes']:raise ValueError('historical payload mismatch: '+relative)
    assets={};templates={};records=[];statuses={};references={}
    cfg=json.loads((ROOT/'package/config.json').read_text())
    for dataset in TRAIN_SEEDS:
        p=ROOT/'package'/cfg['datasets'][dataset]['file']
        if sha(p)!=cfg['datasets'][dataset]['sha256']:raise ValueError('reference asset hash differs')
        with np.load(p,allow_pickle=False) as z:assets[dataset]={k:z[k].copy() for k in z.files}
    for item in idx['records']:
        check();d=item['dataset'];s=item['train_seed'];arm=item['arm'];sample=item['sample_seed'];paths=item['paths'];cell=item['cell_files']
        b=json.loads((history/paths['binding.json']).read_text());m=json.loads((history/paths['metrics.json']).read_text());binding=b['binding']
        if binding['arm']!=arm or binding['config']['seed']!=s or b['seed']!=sample or binding['dataset']['sha256']!=cfg['datasets'][d]['sha256']:raise ValueError('historical identity/source differs')
        if binding['dictionary_sha256']!=sha(ROOT/'dictionaries'/('dictionary_'+d)/'final_basis.npz'):raise ValueError('historical dictionary differs')
        graph=history/paths['graph.npz']
        if sha(graph)!=b['graph_sha256']:raise ValueError('historical graph binding')
        types,edges=graph_arrays(graph,assets[d]);v,focus=summarize_graph(d,m,assets[d],edges)
        if d in references and references[d]!=m['reference_metrics']:raise ValueError('historical reference metrics differ')
        references[d]=m['reference_metrics'];templates[d]=v
        statuses['%s_s%d_%s'%(d,s,arm)]=json.loads((history/cell['summary.json']).read_text())
        records.append(dict(dataset=d,train_seed=s,sample_seed=sample,arm=arm,status='evaluated',source_commit=item['source_commit'],path='history/'+paths['graph.npz'],graph_sha256=sha(graph),values=v,focus=focus,raw_metrics=m,edge_set=edges.T.tolist(),source_limit=idx['checkpoints']))
    for d,seeds in TRAIN_SEEDS.items():
        for s in seeds:
            for arm in 'DE':
                check();label=task_id(d,s,arm);folder=run/label
                try:status=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else dict(status='not_started' if not folder.exists() else 'incomplete_no_receipt')
                except Exception as exc:status=dict(status='invalid_worker_receipt',error=repr(exc))
                statuses[label]=status
                final_hash=sha(folder/'final.pt') if (folder/'final.pt').exists() else None
                for sample in SAMPLE_SEEDS:
                    check();dest=folder/('sample_'+str(sample));row=dict(dataset=d,train_seed=s,sample_seed=sample,arm=arm,path=str(dest/'graph.npz'),worker_status=status)
                    if not all((dest/n).exists() for n in ['graph.npz','binding.json','metrics.json']):
                        row.update(status='missing_or_incomplete',reason='graph, metric or binding missing');records.append(row);continue
                    try:
                        b=json.loads((dest/'binding.json').read_text());binding=b['binding']
                        if (binding.get('dataset_name'),binding.get('train_seed'),binding.get('arm'),binding.get('task_id'),b['seed'])!=(d,s,arm,label,sample):raise ValueError('new result identity mismatch')
                        if binding!=json.loads((folder/'binding.json').read_text()):raise ValueError('sample and model binding differ')
                        if final_hash is None or b['checkpoint_sha256']!=final_hash or binding['updates']!=1000 or binding['config']['seed']!=s:raise ValueError('final checkpoint binding')
                        if sha(dest/'graph.npz')!=b['graph_sha256'] or sha(dest/'metrics.json')!=b['metrics_sha256']:raise ValueError('new graph/metric hash')
                        if binding['dataset']['sha256']!=cfg['datasets'][d]['sha256'] or binding['dictionary_sha256']!=sha(ROOT/'dictionaries'/('dictionary_'+d)/'final_basis.npz'):raise ValueError('new data/dictionary differs')
                        if binding['config']!=dict(cfg,seed=s):raise ValueError('new scientific config differs')
                        types,edges=graph_arrays(dest/'graph.npz',assets[d]);m=json.loads((dest/'metrics.json').read_text())
                        if m['reference_metrics']!=references[d]:raise ValueError('new/old fixed reference differs')
                        v,focus=summarize_graph(d,m,assets[d],edges)
                        row.update(status='evaluated',graph_sha256=b['graph_sha256'],values=v,focus=focus,raw_metrics=m,edge_set=edges.T.tolist())
                    except TimeoutError:raise
                    except Exception as exc:row.update(status='failed_evaluation',error=repr(exc))
                    records.append(row)
    result=assemble(records,templates,statuses);result['references']=references;result['historical_sources']=idx['sources']
    if (run/'execution_receipt.json').exists():result['execution_receipt']=json.loads((run/'execution_receipt.json').read_text())
    if (run/'launcher_summary.json').exists():result['launcher_receipt']=json.loads((run/'launcher_summary.json').read_text())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline-epoch',type=float);a=p.parse_args()
    output=pathlib.Path(a.output).resolve()
    if output==ROOT or ROOT in output.parents:raise ValueError('summary output must be outside immutable release')
    def check():
        if a.deadline_epoch is not None and time.time()>=a.deadline_epoch:raise TimeoutError('summary deadline reached')
    check();result=collect(a.run,check);check()
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return 0 if result['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
