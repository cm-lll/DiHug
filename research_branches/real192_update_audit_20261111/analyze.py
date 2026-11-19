"""CPU-only analysis of saved vectors and conditional clean probabilities."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real192_update_audit_20261111'
NAMES=['count','joint_degree','present_triangles','absent_triangles','components']
torch.set_num_threads(1)


def dump(name,rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False))
    if isinstance(rows,list) and rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def mean_ce(rel,kind):
    count=sum(r[kind+'_n'] for r in rel)
    return sum(r[kind+'_n']*r[kind+'_CE'] for r in rel if r[kind+'_n'])/count if count else None


def main():
    started=time.time();summaries=[];evaluation=[];relrows=[];vector_checks=[];step_rows=[]
    protocol=json.loads((OUT/'protocol.json').read_text())
    for task in protocol['tasks']:
        folder=OUT/task['id'];receipt=json.loads((folder/'receipt.json').read_text()) if (folder/'receipt.json').exists() else {'status':'missing'}
        old=ROOT/'results/single_step_formal_resume_20261111'/task['id']
        steps=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()] if (folder/'updates.jsonl').exists() else []
        logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()] if (folder/'training.jsonl').exists() else []
        with np.load(task['parent']['data']) as data:types=data['types'].copy()
        for row in steps:
            step_rows.append(dict(task=task['id'],**row))
        for row in steps:
            u=row['update'];path=folder/f'vectors_u{u}.pt'
            if not path.exists():continue
            v=torch.load(path,map_location='cpu',weights_only=False)
            gradient=v['gradient'].double().numpy();delta=v['displacement'].double().numpy()
            parts=v['weighted_components'].double().numpy()
            error=np.linalg.norm(parts.sum(0)-gradient)/max(np.linalg.norm(gradient),1e-30)
            dots=parts@delta
            for i,name in enumerate(NAMES):
                recorded=row['components'][name]['dot_displacement']
                assert np.isclose(dots[i],recorded,rtol=1e-7,atol=1e-9)
            assert np.isclose(np.linalg.norm(delta),row['displacement_norm'],rtol=1e-7,atol=1e-10)
            vector_checks.append(dict(task=task['id'],update=u,component_sum_relative_error=float(error),
                                      component_dot_displacement=dots.tolist(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del v,gradient,delta,parts
        for u in (2000,2012,2020,2050):
            pjson=folder/f'validation_u{u}.json'
            if not pjson.exists():continue
            rows=json.loads(pjson.read_text())
            for row in rows:
                readout=row['readout'];t=row['t'];path=folder/f'validation_u{u}_{readout}_t{t}.npz'
                with np.load(path) as z:
                    p=z['probability'];y=z['labels'];q=z['query'];fam=z['families'];ids=z['ids']
                n=len(types);r=len(row['relations']);degree=np.zeros((n,r));refdegree=np.zeros_like(degree)
                for endpoint in (0,1):
                    np.add.at(degree,(q[endpoint],fam),p);np.add.at(refdegree,(q[endpoint],fam),y)
                # This matches expected-degree vectors, not expected hard-graph
                # matching cost; the distinction is preserved in column names.
                anonymous=[]
                for typ in np.unique(types):
                    ix=np.flatnonzero(types==typ)
                    cost=abs(degree[ix,None,:]-refdegree[ix][None,:,:]).sum(-1)
                    a,b=linear_sum_assignment(cost);anonymous.append(float(cost[a,b].mean())/(n-1))
                adjacency=np.zeros((n,n));adjacency[q[0],q[1]]=p;adjacency[q[1],q[0]]=p
                ref=np.zeros((n,n));ref[q[0],q[1]]=y;ref[q[1],q[0]]=y
                # Expected triangle count is exact for this factorized clean
                # Bernoulli output, not a statement about the reverse chain.
                tri=float(np.einsum('ij,ji->',adjacency@adjacency,adjacency)/6)
                reftri=float(np.einsum('ij,ji->',ref@ref,ref)/6)
                baseline_path=folder/f'validation_u2000_{readout}_t{t}.npz'
                with np.load(baseline_path) as z:
                    assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels'])
                initial_rows=json.loads((folder/'validation_u2000.json').read_text())
                initial=next(x for x in initial_rows if x['readout']==readout and x['t']==t)
                assert row['input_hash']==initial['input_hash']
                common=dict(task=task['id'],update=u,readout=readout,t=t,input_hash=row['input_hash'])
                evaluation.append(dict(**common,CE=row['CE'],positive_CE=mean_ce(row['relations'],'positive'),
                    negative_CE=mean_ce(row['relations'],'negative'),expected_edges=float(p.sum()),reference_edges=int(y.sum()),
                    expected_triangle_count=tri,reference_triangle_count=reftri,
                    expected_edge_abs_error=abs(float(p.sum()-y.sum())),expected_triangle_abs_error=abs(tri-reftri),
                    anonymous_expected_joint_degree_cost=float(np.mean(anonymous)),
                    expected_missed_reference_edges=float((1-p[y==1]).sum()),
                    expected_nonreference_edges=float(p[y==0].sum()),
                    below_1e8=int((p<1e-8).sum()),above_1_minus_1e8=int((p>1-1e-8).sum()),
                    mean_sigmoid_derivative=float((p*(1-p)).mean())))
                for rr in row['relations']:
                    mask=fam==rr['family']
                    relrows.append(dict(**common,**rr,expected_edges=float(p[mask].sum()),reference_edges=int(y[mask].sum())))
        if steps:
            peak=max(steps,key=lambda s:s['gradient_norm']);delta_peak=max(steps,key=lambda s:s['displacement_norm'])
            summaries.append(dict(task=task['id'],receipt=receipt,
                gradient_peak_update=peak['update'],gradient_peak=peak['gradient_norm'],
                displacement_at_gradient_peak=peak['displacement_norm'],
                displacement_peak_update=delta_peak['update'],displacement_peak=delta_peak['displacement_norm'],
                first_displacement=steps[0]['displacement_norm'],
                positive_gradient_dot_displacement_steps=sum(s['gradient_dot_displacement']>0 for s in steps),
                historical_max_risk_gap=max(s['historical_max_risk_gap'] for s in logs) if logs else None))
    dump('analysis_summary',summaries);dump('step_metrics',step_rows);dump('conditional_metrics',evaluation)
    dump('conditional_relations',relrows);dump('vector_checks',vector_checks)
    dump('analysis_receipt',dict(seconds=time.time()-started,model_calls=0,autograd_calls=0,
         new_graphs=0,GPU_hours=0,vector_checks=len(vector_checks),conditional_rows=len(evaluation),
         limitations=['Conditional clean-probability proxies, not generated graph quality.',
                      'Component gradient dot actual update is a first-order approximation.',
                      'Validation inputs are reused historical states; not independent confirmation.']))
    for s in summaries:
        print(s['task'],'grad_peak',s['gradient_peak_update'],s['gradient_peak'],
              'delta',s['displacement_at_gradient_peak'],'first_delta',s['first_displacement'])


if __name__=='__main__':main()
