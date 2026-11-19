"""Six endpoint probability-space derivatives, CPU only; not parameter gradients."""
import os,sys,json,time,csv,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('old_worker_cpu',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
w.torch.set_num_threads(1);start=time.time();summary=[];details=[]
for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
    ds,cfg=task['dataset'],task['row']['config']
    data=ROOT/w.g.SCI.relative_to(w.g.ROOT)/'package'/cfg['datasets'][ds]['file']
    with w.np.load(data,allow_pickle=False) as z:case=w.g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
    obj=w.make_objective(case,cfg,ds,'PS')
    inputs=w.torch.load(OUT/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
    for t in [1,500,1000]:
        inp=next(inp for panel,tt,seed,inp in inputs if panel=='new_diagnostic' and tt==t)
        with w.np.load(OUT/f'{ds}_PS_32_raw_new_diagnostic_{t}.npz',allow_pickle=False) as z:p=w.torch.from_numpy(z['probability']).clone().requires_grad_()
        loss,_=obj(p,inp['query'],inp['f'],inp,logits=None)
        dp=w.torch.autograd.grad(loss,p)[0]
        aux=(dp*p*(1-p)).detach().double().numpy()
        prob=p.detach().double().numpy();ce=(prob-inp['labels'])/case['space'].total
        assert w.np.isfinite(aux).all()
        # This is the derivative of a free positive logit with negative logit fixed.
        for family in [-1]+list(range(case['n_families'])):
            for label in [0,1]:
                mask=(inp['labels']==label)&((inp['f']==family) if family>=0 else True)
                if not mask.any():continue
                cc,aa=ce[mask],aux[mask]
                opposed=cc*aa<0;total_reversed=cc*(cc+aa)<0
                details.append(dict(dataset=ds,t=t,family=family,label=label,support=int(mask.sum()),
                    CE_L1=float(abs(cc).sum()),PS_L1=float(abs(aa).sum()),PS_opposed_count=int(opposed.sum()),
                    total_sign_reversed_count=int(total_reversed.sum()),CE_L1_mass_on_opposed=float(abs(cc[opposed]).sum()),
                    mean_PS_logit_derivative=float(aa.mean()),mean_CE_logit_derivative=float(cc.mean())))
        den=float(w.np.linalg.norm(ce)*w.np.linalg.norm(aux))
        summary.append(dict(dataset=ds,t=t,PS_loss=float(loss),free_logit_gradient_cosine=float(ce@aux)/den if den else None,
                            PS_over_CE_L2=float(w.np.linalg.norm(aux)/w.np.linalg.norm(ce)),probabilities_at_zero=int((prob==0).sum()),probabilities_at_one=int((prob==1).sum())))
assert not w.torch.cuda.is_initialized()
w.fr.save(OUT/'probability_gradient_cpu.json',dict(summary=summary,groups=details,seconds=time.time()-start,network_calls=0,
    limitation='CPU derivatives wrt free positive logits at exported probabilities; not model parameter gradients or AdamW steps. t1 duplicates old validation.'))
with (OUT/'probability_gradient_groups.csv').open('w',newline='') as f:
    out=csv.DictWriter(f,fieldnames=list(details[0]));out.writeheader();out.writerows(details)
print(json.dumps(summary))
