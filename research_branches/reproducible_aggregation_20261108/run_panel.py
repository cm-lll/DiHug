"""Bounded numerical reproducibility panel, never production training."""
import os,sys,json,time,copy,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('DIAG_OUTPUT',str(ROOT/'results/reproducible_aggregation_20261108T113027')))
spec=importlib.util.spec_from_file_location('bound_diag',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
from adapter import execution_policy,VERSION
START=time.time();LIMIT=min(1794110127,START+870)
status=dict(status='loading',pid=os.getpid(),start_epoch=START,updates=0,rows=[],comparisons=[],short=[])
def save(name,value):fr.save(OUT/name,value)
def check():
    if time.time()>LIMIT:raise TimeoutError('scientific or GPU boundary')
def diff(a,b):
    result=dict(exact=True,within_original_tolerance=True,max_abs=0.,location=None,nonzero=0)
    def visit(x,y,path):
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('comparison shape/type '+path)
            same=torch.equal(x,y);result['exact'] &= same
            if x.numel():
                d=(x.double()-y.double()).abs();mx=float(d.max());result['nonzero']+=int((d!=0).sum())
                if mx>result['max_abs']:result.update(max_abs=mx,location=path+':'+str(int(d.flatten().argmax())))
                result['within_original_tolerance'] &= bool(torch.allclose(x,y,atol=2e-6,rtol=1e-5)) if x.is_floating_point() else same
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('keys '+path)
            for k in x:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):raise ValueError('length '+path)
            for i,(v,z) in enumerate(zip(x,y)):visit(v,z,path+'/'+str(i))
        elif isinstance(x,np.ndarray):
            if not isinstance(y,np.ndarray) or x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('numpy shape/type '+path)
            same=bool(np.array_equal(x,y));result['exact'] &= same
            result['within_original_tolerance'] &= bool(np.allclose(x,y,atol=2e-6,rtol=1e-5)) if np.issubdtype(x.dtype,np.floating) else same
            d=np.abs(x.astype(np.float64)-y.astype(np.float64));result['nonzero']+=int(np.count_nonzero(d))
            if d.size and float(d.max())>result['max_abs']:result.update(max_abs=float(d.max()),location=path+':'+str(int(d.argmax())))
        else:
            same=x==y;result['exact'] &= bool(same);result['within_original_tolerance'] &= bool(same)
    visit(a,b,'');return result
def main():
    fr.setup();fr.authenticate_sources()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    save('environment.json',dict(**g.common.environment('cuda'),policy=VERSION,cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    tasks=w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json')
    for task in tasks:
        check();ds=task['dataset'];cp=Path(task['checkpoint']);assert w.sha(cp)==task['checkpoint_sha256']
        parent=torch.load(cp,map_location='cpu');cfg=task['row']['config'];binding=parent['binding']
        assert parent['updates']==task['start'] and binding['config']==cfg and binding['dataset']==ds and binding['train_seed']==task['seed']
        data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
        with np.load(data,allow_pickle=False) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);r=np.random.default_rng(1)
        history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
        ps=w.make_objective(case,cfg,ds,'PS');assert ps.objective_binding()==binding['output_objective']['target']
        b=None
        def restore(payload):
            nonlocal b
            work=w.cpu_copy(payload);g.restore_payload(work,m,o,e,work['binding'],lambda x:g.rng_restore(x,r,'cuda'))
            m.train();o.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,o,e,r,'cuda',payload['updates'])
            live=b.checkpoint(binding)
            for k in ['model','optimizer','ema','rng','updates']:
                if g.digest(live[k])!=g.digest(payload[k]):raise ValueError('complete restore '+k)
        class Objective:
            def __init__(self,kind,capture):self.kind=kind;self.capture=capture;self.logits=None;self.last={}
            def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
            def __call__(self,p,q,f,inp,*,logits):
                if self.capture:self.logits=logits.detach().cpu().clone()
                return (p.sum()*0,p.new_empty(0)) if self.kind=='CE' else ps(p,q,f,inp,logits=logits)
        def step(kind,capture):
            check();u=b.completed+1;obj=Objective(kind,capture);grads={}
            def hook(*_):
                for n,p in m.named_parameters():grads[n]=None if p.grad is None else p.grad.detach().cpu().clone()
            h=o.register_step_pre_hook(hook) if capture else None
            torch.cuda.synchronize();t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            try:
                with b.step(u):
                    inp=g.one_input(case,cfg,u,r);ih=w.input_hash([inp]);rr=g.digest(g.rng_capture(r,'cuda'))
                    expected=history[u-1]
                    assert ih==expected['input_sha256'] and rr==expected['full_rng_before']
                    rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=obj)
                    assert g.digest(g.rng_capture(r,'cuda'))==expected['full_rng_after']
                torch.cuda.synchronize();seconds=time.perf_counter()-t0
            finally:
                if h:h.remove()
            status['updates']+=1
            row=dict(dataset=ds,u=u,t=inp['t'],input_hash=ih,seconds=seconds,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**{k:rec[k] for k in ['loss','unweighted_CE','auxiliary_loss','gradient_norm']})
            payload=b.checkpoint(binding)
            bundle=dict(state={k:payload[k] for k in ['model','optimizer','ema','rng','updates']},gradient=grads,logits=obj.logits,loss=torch.tensor([rec['loss'],rec['unweighted_CE'],rec['auxiliary_loss']],dtype=torch.float64))
            return row,bundle,payload
        references={}
        for kind in ['CE','PS']:
            for strict in [False,True]:
                label='strict' if strict else 'ordinary';first=None
                for rep in range(3):
                    with execution_policy(strict):
                        restore(parent);row,bundle,payload=step(kind,True)
                    row.update(kind=kind,mode=label,repeat=rep);status['rows'].append(row)
                    if rep==0:
                        first=bundle;references[kind,label]=bundle
                        torch.save(bundle,OUT/f'{ds}_{kind}_{label}_first.pt')
                    else:
                        comp={k:diff(first[k],bundle[k]) for k in ['state','gradient','logits','loss']}
                        status['comparisons'].append(dict(dataset=ds,kind=kind,mode=label,repeat=rep,comparisons=comp))
                        if strict and not all(v['exact'] for v in comp.values()):raise RuntimeError('strict repeat not bitwise exact')
                    save('progress.json',status)
                del first
            comp={k:diff(references[kind,'ordinary'][k],references[kind,'strict'][k]) for k in ['state','gradient','logits','loss']}
            status['comparisons'].append(dict(dataset=ds,kind=kind,mode='cross_execution_mode',comparisons=comp))
        del references,bundle,payload
        # Bounded 8-update tracks; duplicate track saves/reloads after update 4.
        if time.time()+150>LIMIT:raise TimeoutError('insufficient budget for complete short tracks')
        for strict in [False,True]:
            first=[]
            for rep in range(2):
                with execution_policy(strict):
                    restore(parent)
                    for i in range(8):
                        row,bundle,payload=step('PS',False)
                        hashes={k:g.digest(payload[k]) for k in ['model','optimizer','ema','rng','updates']}
                        row.update(mode='strict' if strict else 'ordinary',repeat=rep,index=i,hashes=hashes)
                        if rep==0:first.append(hashes)
                        else:
                            row['exact_vs_continuous']={k:hashes[k]==first[i][k] for k in hashes}
                            if strict and not all(row['exact_vs_continuous'].values()):raise RuntimeError('strict short continuation differs')
                        status['short'].append(row);save('progress.json',status)
                        if rep==1 and i==3:
                            path=OUT/f'{ds}_{strict}_temporary_u{b.completed}.pt';fr.atomic_checkpoint(path,payload);restore(torch.load(path,map_location='cpu'))
        assert w.sha(cp)==task['checkpoint_sha256'];del m,o,e,b,parent,ps;torch.cuda.empty_cache()
    fr.authenticate_sources();status['status']='complete'
if __name__=='__main__':
    try:main()
    except BaseException as ex:status.update(status='failed_or_stopped',error=repr(ex),traceback=traceback.format_exc())
    finally:
        status.update(end_epoch=time.time(),seconds=time.time()-START);save('receipt.json',status);print(json.dumps({k:v for k,v in status.items() if k not in ['rows','comparisons','short']}),flush=True)
    raise SystemExit(0 if status['status']=='complete' else 2)
