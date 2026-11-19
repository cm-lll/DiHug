"""CPU-only independent verification and delivery; no denoiser import."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,hashlib,datetime,time
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/pair_allocation_parameter_20261109'
OLD=ROOT/'results/joint_degree_relation_signal_20261109'
def read(n):return json.loads((OUT/(n+'.json')).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cos(a,b):
    den=float(a.norm()*b.norm());return float(a@b)/den if den else None
start=time.monotonic();receipt=read('receipt');release=read('release')
assert receipt['status']=='complete' and receipt['forwards']==2 and receipt['gradient_queries']==4
assert release['worker_exited'] and release['exit_code']==0 and release['conservative_GPU_hours']<=.15
assert all(sha(p)==h for p,h in read('binding').items())
assert all(x['p_exact'] for x in read('probability_matching'))
assert all(all(x['loaded_exact'].values()) and all(x['after_unchanged'].values()) for x in read('state_checks'))
summary=[];checks=[];groups=read('comparisons')
for parent in ['CE','PS']:
    payload=torch.load(OUT/f'{parent}_pair_gradients.pt',map_location='cpu',weights_only=False)
    old=torch.load(OLD/f'{parent}_decomposition.pt',map_location='cpu',weights_only=False)
    gg=payload['gradients'];mean=(gg['pair_draw0']+gg['pair_draw1'])/2
    assert torch.equal(mean,gg['pair_mean']) and all(torch.isfinite(x).all() for x in gg.values())
    assert payload['checkpoint_sha256']==old['checkpoint_sha256'] and payload['input_hash']==old['input_hash']
    reference=old['gradients']['common'].double();total=old['gradients']['total'].double();CE=old['gradients']['CE'].double()
    row=next(x.copy() for x in groups if x['parent']==parent and x['group']=='all')
    for k,v in [('pair_norm',float(mean.norm())),('pair_common_cosine',cos(mean,reference)),('pair_total_cosine',cos(mean,total)),('pair_CE_cosine',cos(mean,CE))]:
        assert abs(row[k]-v)<1e-12
    original_residual=old['gradients']['residual'].double()
    row['old_residual_common_cosine']=cos(original_residual,reference)
    row['gradient_none']=payload['none_gradients']
    summary.append(row)
    checks.append(dict(parent=parent,mean_exact=True,finite=True,identity=True,summary_recomputed=True))
save('summary',summary)
for name,rows in [('summary',summary),('comparisons',groups),('per_parameter',read('per_parameter'))]:
    with (OUT/(name+'.csv')).open('w',newline='') as h:
        w=csv.DictWriter(h,list(rows[0]));w.writeheader();w.writerows(rows)
save('independent_checks',dict(checks=checks,bindings=len(read('binding')),state_unchanged=True,seconds=time.monotonic()-start))
save('completion',dict(status='complete',forwards=2,VJPs=4,updates=0,new_graphs=0,
    output_hashes={str(p):sha(p) for p in OUT.glob('*_pair_gradients.pt')},receipt=receipt,release=release))
save('engineering_notes',dict(issue='Initial sandbox GPU query returned exit9 before any worker launch or model call.',
    resolution='Same guard executed with authorized direct GPU access; no protocol, source, operator or deadline change.',
    initial_model_calls=0,initial_GPU_worker_leases=0))
print(json.dumps(summary,ensure_ascii=False,indent=2))
