"""Frozen-policy terminal score diagnostic. No optimizer exists in this process."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,hashlib,traceback,signal,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research_branches/cn_multiseed_20261104'
SCI=SRC/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
sys.path[:0]=[str(SRC),str(SRC/'experiments/o_cn_matched_execute_v1'),str(SCI/'package'),str(SCI),str(ROOT/'research_branches/terminal_structure_audit_20261109')]
from hsgd.bw_preflight import common
from hsgd import runtime
from hsgd.kernel import MarginalKernel
from model_variants import StructuralDenoiser
from boundary_transactions import SamplingBoundary,rng_capture,rng_restore
from observers import RecordedGenerator,RecordedPlan,digest
from transition_objective import reverse_probability,transition_log_probability

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    out=Path(sys.argv[1]);ds=sys.argv[2];protocol=json.loads((out/'protocol.json').read_text());row=protocol['models'][ds]
    dest=out/ds;dest.mkdir(exist_ok=False);start=time.time();save(dest/'allocation.json',dict(pid=os.getpid(),start_epoch=start,visible_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    def check():
        if time.time()>=protocol['gpu_stop_epoch']:raise TimeoutError('frozen scientific/GPU stop')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('watchdog stop')))
    model=None;chain=None;identity=None;folder=None
    try:
        check();torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        for p,h in protocol['bindings'].items():
            if sha(p)!=h:raise ValueError('binding changed: '+p)
        z=torch.load(row['readout'],map_location='cpu');cfg=z['binding']['config']
        assert z['updates']==3000 and z['readout']=='ema_0.99' and z['decay']==.99 and z['train_seed']==row['train_seed']
        assert z['binding']['dataset']==ds and z['model']['joint_head.0.weight'].shape==(128,776)
        assert 'role_basis' not in z['model'] and 'cn_edge_projection.weight' not in z['model']
        assert json.loads(Path(row['readout']).with_name('final.sha256.json').read_text())['sha256']==z['checkpoint_sha256']
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        assert np.array_equal(case['space'].encode(edges)[0]+1,data['relations'])
        model=StructuralDenoiser(case,cfg,'A');model.load_state_dict(z['model'],strict=True);model=model.cuda().eval()
        initial={k:thash(v) for k,v in model.state_dict().items()};assert all(initial[k]==thash(v) for k,v in z['model'].items());del z
        save(dest/'environment.json',common.environment('cuda'))
        named=list(model.named_parameters());save(dest/'parameters.json',[dict(name=n,shape=list(p.shape),numel=p.numel(),group='CN' if 'common_neighbor' in n else 'head' if 'joint_head' in n else 'backbone_and_degree') for n,p in named])
        family=np.repeat(np.arange(case['n_families']),case['space'].counts);kernel=MarginalKernel(case['prior'],common.schedule(1000))
        original_class=runtime.StepPredictor;original_plan=common.plan_for;held=[];plans=[];selected=set()
        class GradientPredictor(original_class):
            def __call__(self,frozen,ids,fam,query):
                if self.t not in selected:return super().__call__(frozen,ids,fam,query)
                if self.closed or not np.array_equal(frozen,self.state):raise ValueError('changed frozen input')
                check()
                with torch.enable_grad():
                    p=runtime.predict(self.model,self.case['space'].types,self.edges,self.fam,query,fam,self.t/1000,self.backend,self.context,self.token)
                value=common.check_prob(p,fam,self.case);self.blocks+=1
                held.append((ids.copy(),fam.copy(),p));return value
        def plan(*a,**kw):
            p=RecordedPlan(original_plan(*a,**kw));plans.append(p);return p
        runtime.StepPredictor=GradientPredictor;common.plan_for=plan
        for ss in protocol['sample_seeds']:
            check();selected=set(protocol['times'][str(ss)]);folder=dest/str(ss);folder.mkdir();tstart=time.time()
            rng=RecordedGenerator(ss,case['space'].total);state=(rng.random(case['space'].total)<case['prior'][family]).astype(np.uint8)
            identity=dict(dataset=ds,train_seed=row['train_seed'],sample_seed=ss,readout='EMA0.99',updates=3000,readout_sha256=protocol['bindings'][row['readout']],T=1000,R=1,selected_times=sorted(selected),initial_state_sha256=digest(state))
            save(folder/'identity.json',identity);chain=SamplingBoundary(state,1000,rng,'cuda');grads=[];probe_rows=[]
            torch.save(chain.checkpoint(identity),folder/'initial.pt')
            for t in range(1000,0,-1):
                check();held.clear();plans.clear();before=chain.state.copy();calls=rng.calls
                def validate(candidate,record):
                    assert rng.calls==calls+1 and len(plans)==1 and plans[0].blocks==1
                    assert record['exact_coverage'] and record['covered_slots']==int(np.repeat(case['active'],case['space'].counts).sum())
                rec=chain.advance(lambda x,at:runtime.reverse_step(model,case,x,kernel,at,1,ss,rng,'cuda',trace=at in selected,check_time=check),validate)
                trace=rec.pop('trace',None);rec.update(query_sha=plans[0].h.hexdigest(),uniform_sha=rng.last_sha)
                if t in selected:
                    check();assert len(held)==1;ids,fam,p=held[0];saved_rng=rng_capture(rng,'cuda');tick=time.time()
                    xt=torch.as_tensor(before[ids],device='cuda');xp=torch.as_tensor(chain.state[ids],device='cuda');ft=torch.as_tensor(fam,device='cuda',dtype=torch.long)
                    q=reverse_probability(p,xt,ft,case['prior'],kernel.alpha,t-1,t)
                    qerr=float(np.max(abs(q.detach().cpu().numpy()-trace['q'][ids])))
                    assert qerr<=1e-12,'Torch/production posterior mismatch'
                    lp=transition_log_probability(p,xt,xp,ft,case['prior'],kernel.alpha,t-1,t)
                    chosen=np.where(chain.state[ids],trace['q'][ids],1-trace['q'][ids]);assert np.all(chosen>0)
                    logref=float(np.log(chosen).sum());assert abs(lp.item()-logref)<1e-8
                    gg=torch.autograd.grad(lp,[v for _,v in named],allow_unused=True)
                    flat=torch.cat([(torch.zeros_like(v) if g is None else g).detach().flatten().cpu() for (_,v),g in zip(named,gg)])
                    assert torch.isfinite(flat).all();grads.append(flat);torch.cuda.synchronize()
                    # Autograd must not consume any random stream or mutate state.
                    after_rng=rng_capture(rng,'cuda')
                    import pickle
                    # Tensor pickle byte identity is not semantic RNG identity; compare recursively.
                    def equal(a,b):
                        if isinstance(a,torch.Tensor):return torch.equal(a,b)
                        if isinstance(a,np.ndarray):return np.array_equal(a,b)
                        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                        return a==b
                    assert equal(saved_rng,after_rng),'autograd consumed RNG'
                    probe_rows.append(dict(t=t,log_probability=lp.item(),numpy_log_probability=logref,q_max_abs=qerr,gradient_norm=float(flat.norm()),none_gradients=sum(g is None for g in gg),backward_and_copy_seconds=time.time()-tick))
                    np.savez_compressed(folder/('step_%04d.npz'%t),xt=before,xprev=chain.state,p0=trace['p0'],q=trace['q'],uniform=trace['uniform'],coverage=trace['coverage'])
                    del gg,flat,q,lp,p;held.clear()
                with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if t%100==0 or t==1:torch.save(chain.checkpoint(identity),folder/'latest_complete.pt')
                if t%25==0:save(dest/'progress.json',dict(sample_seed=ss,next_t=chain.next_t,elapsed_seconds=time.time()-start,at_epoch=time.time()))
                if t==975:
                    remaining=(time.time()-tstart)/25*(975+(len(protocol['sample_seeds'])-protocol['sample_seeds'].index(ss)-1)*1000)
                    if time.time()+remaining+120>protocol['gpu_stop_epoch']:raise TimeoutError('measured throughput cannot cover remaining frozen panel')
            assert len(grads)==8
            torch.save(dict(times=[r['t'] for r in probe_rows],gradients=torch.stack(grads)),folder/'score_gradients.pt');save(folder/'probes.json',probe_rows)
            e,f=common.state_edges(case,chain.state);np.savez_compressed(folder/'graph.npz',types=case['space'].types,edges=e,families=f,schema=np.array(case['space'].pairs))
            assert all(thash(v)==initial[k] for k,v in model.state_dict().items()),'frozen model mutated'
            save(folder/'summary.json',dict(status='complete',seconds=time.time()-tstart,steps=1000,gradient_times=8,graph_sha256=sha(folder/'graph.npz')))
            del grads
        save(dest/'complete.json',dict(status='complete',chains=4,gradient_times=32,parameter_updates=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except BaseException as exc:
        save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),at_epoch=time.time()))
        if chain is not None and folder is not None:
            try:torch.save(chain.checkpoint(identity),folder/'stopped_complete_boundary.pt')
            except BaseException as se:save(dest/'save_failure.json',dict(error=repr(se)))
        raise
    finally:
        save(dest/'release.json',dict(start_epoch=start,end_epoch=time.time(),occupied_seconds=time.time()-start,visible_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
if __name__=='__main__':main()
