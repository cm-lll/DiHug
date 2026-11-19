"""CPU aggregate of count-conditioned samples; no learned model or new draws."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/fixed_count_diagnostic_20261110'
def read(name):return list(csv.DictReader((OUT/name).open()))
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());receipt=json.loads((OUT/'receipt.json').read_text())
    batches=read('batches.csv');graphs=read('graphs.csv');sampling=read('sampling.csv')
    metrics=['edges','triangles','triangle_abs_error','components','LCC','isolates','hub','joint_degree','degree_W1','typed_triangle_present','typed_triangle_absent','component_cost','low_degree_fraction']
    aggregate=[];comparisons=[];audit=[];uniform=[]
    for task in protocol['tasks']:
        name=task['task'];rr=[x for x in batches if x['task']==name]
        for mode in ['raw','fixed_count']:
            for state in protocol['states']:
                rows=sorted([x for x in rr if x['sampling']==mode and x['state']==state],key=lambda x:int(x['t']))
                if not rows:continue
                a=dict(task=name,dataset=task['dataset'],seed=task['seed'],sampling=mode,state=state,time_count=len(rows),samples_per_time=32)
                for m in metrics:
                    a[m]=float(np.mean([float(x[m]) for x in rows]));a[m+'_mean_within_time_SD']=float(np.mean([float(x[m+'_std']) for x in rows]))
                aggregate.append(a)
        for state in ['old_update','exact_update']:
            for t in protocol['times']:
                b=[x for x in graphs if x['task']==name and x['sampling']=='fixed_count' and x['state']=='before' and int(x['t'])==t]
                a=[x for x in graphs if x['task']==name and x['sampling']=='fixed_count' and x['state']==state and int(x['t'])==t]
                if len(a)!=32 or len(b)!=32:continue
                for m in metrics:
                    delta=np.array([float(x[m])-float(y[m]) for x,y in zip(a,b)])
                    comparisons.append(dict(task=name,dataset=task['dataset'],seed=task['seed'],state=state,t=t,metric=m,
                        paired_mean_delta=float(delta.mean()),paired_sample_SD=float(delta.std(ddof=1)),Monte_Carlo_SE=float(delta.std(ddof=1)/np.sqrt(32))))
        for t in protocol['times']:
            for state in protocol['states']:
                path=(ROOT/'results/hard_clean_zero_safe_20261110' if state=='old_update' else ROOT/'results/exact_count_step_20261110_attempt2')/name/f'{"before" if state=="before" else "after"}_t{t}.npz'
                saved=np.load(path);row=next((x for x in rr if x['sampling']=='raw' and x['state']==state and int(x['t'])==t),None)
                if row:
                    for col,m in [(1,'joint_degree'),(2,'typed_triangle_present'),(3,'typed_triangle_absent'),(4,'component_cost')]:
                        np.testing.assert_allclose(float(row[m]),saved['costs'][:,col].mean(),atol=1e-12,rtol=1e-12)
        audit.append(dict(task=name,reused_raw_costs_equal=True))
    for dataset in ['IMDB','PubMed']:
        rr=[x for x in graphs if x['task']=='uniform' and x['dataset']==dataset]
        if not rr:continue
        row=dict(dataset=dataset,samples=len(rr))
        for m in metrics:
            values=np.array([float(x[m]) for x in rr]);row[m]=float(values.mean());row[m+'_std']=float(values.std(ddof=1))
        uniform.append(row)
    # Every saved conditioned graph must preserve its registered relation counts.
    verified=0
    for task in protocol['tasks']:
        targets=np.array(json.loads((OUT/f'{task["dataset"]}_reference.json').read_text())['counts'])
        import torch
        torch.set_num_threads(1)
        inp=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs'][0];order=np.argsort(inp['ids']);f=inp['f'][order]
        for path in OUT.glob(task['task']+'_*_t*.npz'):
            b=np.load(path)['bits']
            for r,m in enumerate(targets):assert np.all(b[:,f==r].sum(1)==m)
            verified+=len(b)
    unchanged=0
    for p,h in protocol['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;unchanged+=1
    table('aggregate.csv',aggregate);table('paired_changes.csv',comparisons);table('uniform_baseline.csv',uniform)
    result=dict(status=receipt['status'],aggregate=aggregate,uniform_baseline=uniform,
        checks=dict(raw=audit,conditioned_draws_count_checked=verified,unchanged_bindings=unchanged),
        conditioning_log_probability_range=[min(float(x['log_conditioning_probability']) for x in sampling),max(float(x['log_conditioning_probability']) for x in sampling)],
        uncertainty='32 paired Monte Carlo samples per time; six times not independent training repetitions; SD aggregated within times, no statistical significance claim',
        receipt=receipt)
    dump('summary.json',result)
    print(json.dumps([dict(**{k:x[k] for k in ['dataset','seed','state','sampling']},**{k:round(x[k],5) for k in ['triangles','triangle_abs_error','components','LCC','isolates','joint_degree','degree_W1','typed_triangle_present','hub']}) for x in aggregate],indent=2))
if __name__=='__main__':main()
