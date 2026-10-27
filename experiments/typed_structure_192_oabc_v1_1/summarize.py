"""Independent raw-graph evaluation; missing/failed cells remain explicit.
Does not train, sample, repair, choose checkpoints or silently average successes.
"""
import argparse,pathlib,json,sys,hashlib
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent/'package'))
from prepare import load_case
from evaluation import evaluate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scalars(m):
    r=dict(m['primary']);r.update({k:m[k] for k in ['edges','triangles','lcc','isolates','maximum_degree','average_clustering']});r['components']=len(m['component_sizes'])
    r.update({'family_edges_'+str(i):v for i,v in enumerate(m['family_edges'])})
    if m['imdb_closure']:
        r.update({'closure_'+k:v for k,v in m['imdb_closure'].items() if v is not None})
    for i,row in enumerate(m['mp']):r['MP'+str(i+2)+'_total']=row['total']
    if 'IMDB_focus' in m:r.update(m['IMDB_focus'])
    return r

def collect(root):
    root=pathlib.Path(root);rows={};cells={};contrasts=[]
    for dataset in ['IMDB','PubMed']:
        case,cfg=load_case(dataset)
        for arm in 'OABC':
            label=dataset+'_'+arm;folder=root/label;summary=folder/'summary.json';status=json.loads(summary.read_text()) if summary.exists() else dict(status='not_started');entry=dict(worker_status=status,graphs={},mean=None,diversity=None);cells[label]=entry;sets=[]
            for seed in [2026102821,2026102822]:
                sample=folder/('sample_'+str(seed));g=sample/'graph.npz';binding=sample/'binding.json'
                if not g.exists() or not binding.exists():entry['graphs'][str(seed)]=dict(status='missing_or_incomplete');continue
                try:
                    b=json.loads(binding.read_text());assert b['seed']==seed and b['graph_sha256']==sha(g);assert b['checkpoint_sha256']==sha(folder/'final.pt');assert b['binding']['arm']==arm
                    z=np.load(g,allow_pickle=False);np.testing.assert_array_equal(z['types'],case['space'].types);np.testing.assert_array_equal(z['schema'],case['space'].pairs)
                    e=z['edges'];np.testing.assert_array_equal(case['space'].encode(e)[0],z['families']);metrics=evaluate(case,e,cfg['datasets'][dataset]['type_names']);entry['graphs'][str(seed)]=dict(status='evaluated',sha256=sha(g),metrics=metrics);rows[(dataset,arm,seed)]=scalars(metrics);sets.append(set(map(tuple,e.T)))
                except Exception as exc:entry['graphs'][str(seed)]=dict(status='failed_evaluation',error=repr(exc))
            if len(sets)==2:
                keys=set.intersection(*(set(rows[(dataset,arm,s)]) for s in [2026102821,2026102822]));entry['mean']={k:sum(rows[(dataset,arm,s)][k] for s in [2026102821,2026102822])/2 for k in keys}
                union=len(sets[0]|sets[1]);entry['diversity']=dict(labelled_graphs_equal=sets[0]==sets[1],edge_symmetric_difference=len(sets[0]^sets[1]),edge_jaccard=len(sets[0]&sets[1])/union if union else 1.,scope='two labelled samples only; no isomorphism or population-diversity certification')
        for left,right in [('A','O'),('B','O'),('B','A'),('C','B'),('C','A')]:
            pairs=[]
            for seed in [2026102821,2026102822]:
                a=rows.get((dataset,left,seed));b=rows.get((dataset,right,seed));pairs.append(dict(seed=seed,status='complete' if a is not None and b is not None else 'missing',difference={k:a[k]-b[k] for k in a.keys()&b.keys()} if a is not None and b is not None else None))
            mean=None
            if all(x['status']=='complete' for x in pairs):mean={k:sum(x['difference'][k] for x in pairs)/2 for k in pairs[0]['difference'].keys()&pairs[1]['difference'].keys()}
            contrasts.append(dict(dataset=dataset,contrast=left+'-'+right,pairs=pairs,mean=mean))
    complete=len(rows)==16 and all(c['worker_status'].get('status')=='complete' for c in cells.values())
    return dict(status='complete' if complete else 'incomplete',evaluated_graphs=len(rows),expected_graphs=16,cells=cells,contrasts=contrasts,W1_definition='per-type mean of 4 incident schema relations, node-count weighted; old E07 averaged all10. For same schema and zero nonincident dimensions, old_all10 = 0.4 * this_incident4. Main definition requires approval before formal run.',quality_claim='one training seed/dataset, two graphs/model, exploratory only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=collect(a.run)
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2)
    raise SystemExit(0 if result['status']=='complete' else 2)
