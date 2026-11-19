"""Independent CPU audit of saved boundaries, algebra and current CE tradeoffs."""
import os,sys,json,csv,hashlib,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=Path(sys.argv[1]);start=time.time()
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
r=read(OUT/'receipt.json');lease=read(OUT/'lease.json');assert r['status']=='complete' and lease['returncode']==0
assert (r['updates'],r['forwards'],r['gradient_probes'])==(8,56,20)
assert lease['gpu_hours']<=.25 and lease['end_epoch']<=read(OUT/'protocol.json')['science_stop_epoch']
results=read(OUT/'results.json');assert len(results)==4
old=read(ROOT/'results/reproducible_objective_20261108T135622/attempt2/probes.json')
source=read(ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json')
assert all(sha(p)==h for p,h in source.items())
inputs=read(OUT/'assets.json');assert all(sha(x['parent'])==x['sha256'] for x in inputs)
checks=[];tradeoffs=[];states=[]
for x in results:
    condition=x['condition'];dataset=x['dataset'];parent=next(q for q in inputs if q['condition']==condition)
    envelope=torch.load(parent['parent'],map_location='cpu');prior=envelope['state']
    tensors=torch.load(OUT/f'{condition}_local_tensors.pt',map_location='cpu');spans=tensors['spans']
    for t in [1,500,1000]:
        baseline=x['base'][str(t)]
        earlier=next(q for q in old if q['dataset']==dataset and q['arm']==x['parent_branch'] and q['offset']==32 and q['readout']=='raw' and q['panel']=='new_diagnostic' and q['t']==t)
        assert all(baseline[k]==earlier[k] for k in ['CE','positive_CE','negative_CE','expected_edges','input_sha256'])
    npos=sum(z['positive_count'] for z in x['train_before']['relations']);nneg=sum(z['negative_count'] for z in x['train_before']['relations'])
    for arm in ['CE','PS']:
        S=x['outcomes'][arm]['training_record']['S'];after=x['outcomes'][arm]['training_after'];before=x['train_before']
        pos=npos/S*(after['positive_CE']-before['positive_CE']);neg=nneg/S*(after['negative_CE']-before['negative_CE']);actual=after['CE']-before['CE']
        assert abs(actual-pos-neg)<2e-7
        tradeoffs.append(dict(condition=condition,arm=arm,S=S,positive_slots=npos,negative_slots=nneg,positive_to_negative_count_ratio=nneg/npos,positive_CE_contribution=pos,negative_CE_contribution=neg,actual_total_CE_change=actual,rounding_residual=actual-pos-neg))
        state=torch.load(OUT/f'{condition}_{arm}_temporary.pt',map_location='cpu')['state'];opt=state['optimizer'];b1,b2=x['betas']
        assert state['updates']==state['ema']['updates']==parent['update']
        maxerr={'exp_avg':0.,'exp_avg_sq':0.};normsq={'exp_avg':0.,'exp_avg_sq':0.};residsq={'exp_avg':0.,'exp_avg_sq':0.}
        for i,(key,span) in enumerate(zip(opt['param_groups'][0]['params'],spans)):
            if x['outcomes'][arm]['gradient_none'][i]:continue
            name,group,lo,hi=span;gradient=tensors['actual_gradients'][arm][lo:hi].double()
            for field,beta in [('exp_avg',b1),('exp_avg_sq',b2)]:
                oldmom=prior['optimizer']['state'][key][field].flatten().double()
                target=beta*oldmom+(1-beta)*(gradient if field=='exp_avg' else gradient.square())
                live=opt['state'][key][field].flatten().double();err=live-target
                assert torch.allclose(live,target,atol=2e-6,rtol=1e-5)
                maxerr[field]=max(maxerr[field],float(err.abs().max()))
                normsq[field]+=float(target.square().sum());residsq[field]+=float(err.square().sum())
        checks.append(dict(condition=condition,arm=arm,moment_formula_max_abs=maxerr,moment_formula_relative_L2={k:(residsq[k]/normsq[k])**.5 if normsq[k] else None for k in normsq}))
        states.append(dict(path=str(OUT/f'{condition}_{arm}_temporary.pt'),sha256=sha(OUT/f'{condition}_{arm}_temporary.pt')))
    del envelope,prior,tensors,state
with (OUT/'CE_tradeoff_components.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(tradeoffs[0]));writer.writeheader();writer.writerows(tradeoffs)
assert not torch.cuda.is_initialized()
summary=dict(status='passed',source_files_unchanged=len(source),parents_unchanged=4,prior_baseline_evaluations_exact=12,updates=8,forwards=56,extra_gradient_probes=20,
             model_calls_in_audit=0,gpu_hours=lease['gpu_hours'],moment_formula_checks=checks,temporary_states=states,tradeoffs=tradeoffs,seconds=time.time()-start)
(OUT/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['moment_formula_checks','temporary_states','tradeoffs']}))
