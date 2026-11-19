"""CPU-only aggregation of all fixed panels; no model imports."""
import csv,json,sys,hashlib,time
from pathlib import Path
import numpy as np

def main(path):
    out=Path(path);rows=json.loads((out/'panels.json').read_text());summary=[];rr=[]
    for ds in ['IMDB','PubMed']:
        for t in [50,500,950]:
            group=[r for r in rows if r['dataset']==ds and r['t']==t]
            base=next(r for r in group if r['rounds']==1 and r['repeat']==0)
            for rounds in (1,10,20):
                v=[r for r in group if r['rounds']==rounds]
                if len(v)!=2:continue
                s=dict(dataset=ds,t=t,rounds=rounds,n_panels=len(v),
                    p0_abs_diff_mean=np.mean([r['p0_abs_diff_mean'] for r in v]),
                    p0_abs_diff_max=max(r['p0_abs_diff_max'] for r in v),
                    q_abs_diff_mean=np.mean([r['q_abs_diff_mean'] for r in v]),
                    expected_next_edges=np.mean([r['expected_next_edges'] for r in v]),
                    delta_expected_next_edges=np.mean([r['expected_next_edges']-base['expected_next_edges'] for r in v]),
                    diagnostic_next_state_disagreements=np.mean([r['diagnostic_next_state_disagreements'] for r in v]),
                    seconds=np.mean([r['seconds'] for r in v]),
                    peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in v))
                summary.append(s)
            for r in group:
                for rel in r['relations']:
                    if rel['slots']==0:continue
                    b=next(a for a in base['relations'] if a['family']==rel['family'])
                    rr.append(dict(dataset=ds,t=t,rounds=r['rounds'],repeat=r['repeat'],**rel,
                        delta_p0_sum=rel['p0_sum']-b['p0_sum'],
                        delta_expected_next_edges=rel['expected_next_edges']-b['expected_next_edges'],
                        delta_degree_MAE=rel['expected_relation_degree_MAE']-b['expected_relation_degree_MAE']))
    for name,data in [('summary',summary),('relations',rr)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2))
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    protocol=json.loads((out/'protocol.json').read_text())
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['bindings'].items()}
    assert all(unchanged.values());(out/'post_asset_check.json').write_text(json.dumps(unchanged,indent=2))
    for r in summary:
        print(r['dataset'],r['t'],'R'+str(r['rounds']),
            'p0mean %.7f max %.5f qmean %.7f deltaEdges %+.5f different %.1f time %.3f mem %.1f'%(
                r['p0_abs_diff_mean'],r['p0_abs_diff_max'],r['q_abs_diff_mean'],r['delta_expected_next_edges'],
                r['diagnostic_next_state_disagreements'],r['seconds'],r['peak_allocated_MiB']))
    print('Completed summary and unchanged bindings',len(unchanged))

if __name__=='__main__':main(sys.argv[1])
