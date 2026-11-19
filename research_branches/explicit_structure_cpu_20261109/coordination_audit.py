"""Read saved deltas only: distinguish compensation, blind spots and local opportunities."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'results/explicit_structure_cpu_20261109'
OUT = ROOT/'results/structure_coordination_audit_20261109'
EPS = 1e-12
BLOCKS = ['joint', 'closure', 'low', 'hub']
EXTERNAL = ['root_path_W1', 'LCC_error', 'components_error', 'isolates_error']

def summarize(rows, weighted=False):
    weight = lambda r: r.get('stratum_weight', 1) if weighted else 1
    vals = lambda r: [r['explicit_'+b+'_change'] for b in BLOCKS]
    total = lambda pred: sum(weight(r) for r in rows if pred(r))
    rewarded = lambda r: r['explicit_change'] < -EPS
    pareto = lambda r: max(vals(r)) <= EPS and min(vals(r)) < -EPS
    out = dict(slots=total(lambda r: True), rewarded=total(rewarded),
        mixed_sign=total(lambda r: min(vals(r)) < -EPS and max(vals(r)) > EPS),
        four_block_pareto=total(pareto),
        rewarded_with_block_cost=total(lambda r: rewarded(r) and max(vals(r)) > EPS),
        pareto_with_external_cost=total(lambda r: pareto(r) and any(r[m+'_change']>EPS for m in EXTERNAL)),
        pareto_without_external_cost=total(lambda r: pareto(r) and all(r[m+'_change']<=EPS for m in EXTERNAL)),
        pareto_with_internal_component_cost=total(lambda r: pareto(r) and any(r['explicit_'+m+'_change']>EPS for m in ['joint_TV','relation_W1','node_triangle_W1','typed_triangle_L1'])),
        joint_TV_unchanged=total(lambda r: abs(r['explicit_joint_TV_change'])<=EPS),
        closure_cost_paid_by_low=total(lambda r: rewarded(r) and r['explicit_closure_change']>EPS and r['explicit_low_change'] < -EPS),
        joint_cost_paid_by_low=total(lambda r: rewarded(r) and r['explicit_joint_change']>EPS and r['explicit_low_change'] < -EPS))
    for b in BLOCKS:
        out[b+'_mean_absolute_delta']=sum(weight(r)*abs(r['explicit_'+b+'_change']) for r in rows)/out['slots']
        out[b+'_rewarded_cost']=total(lambda r: rewarded(r) and r['explicit_'+b+'_change']>EPS)
    return out

def save(name, rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    if isinstance(rows,list) and rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    files=[SRC/(n+'.json') for n in ['final_edges','conditional_edges','graph_scores']]
    binding={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    final,conditional,graphs=[json.loads(p.read_text()) for p in files]
    assert all(abs(r['explicit_change']-sum(r['explicit_'+b+'_change'] for b in BLOCKS)/4)<1e-10 for r in final+conditional)
    for name,rows,keys,weighted in [
        ('dataset',final,['dataset'],False),('seed',final,['dataset','train_seed'],False),
        ('graph',final,['dataset','train_seed','arm','sample_seed'],False),
        ('time',conditional,['dataset','t'],True)]:
        groups=sorted(set(tuple(r[k] for k in keys) for r in rows))
        save(name,[dict(zip(keys,g),**summarize([r for r in rows if tuple(r[k] for k in keys)==g],weighted)) for g in groups])
    level=[]
    for ds in ['IMDB','PubMed']:
        rr=[r for r in graphs if r['dataset']==ds]
        for b in BLOCKS:
            x=[r[b] for r in rr]
            level.append(dict(dataset=ds,block=b,graphs=len(x),mean=statistics.mean(x),min=min(x),max=max(x)))
    save('block_levels',level)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
    save('binding',binding)
    save('receipt',dict(status='complete',wall_seconds=time.monotonic()-start,models=0,GPU_hours=0,new_graphs=0,
        source_unchanged=True,limitation='Selected correlated edits, diagnostic only; no alternative objective tested.'))
    print((OUT/'dataset.json').read_text());print((OUT/'seed.json').read_text())

if __name__=='__main__':main()
