"""Approved reconstructed-initialization diagnostic; ZERO optimizer steps."""
import argparse,copy,hashlib,inspect,json,pathlib,random,resource,signal,sys,time,traceback,warnings
START=time.time();HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen_package'))
import numpy as np
import torch,torch_geometric,torch_scatter
from torch_geometric.nn import pool
from torch_geometric.utils import scatter
from prepare import load_case,sha
from model_variants import StructuralDenoiser,copy_common
from hsgd.bw_preflight.common import cpu_tree,check_prob_device
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from probe import traces
COMMIT='cb75b37b116a69d45aa6e78cab55787308a009db'
INPUT_SHA='9d249b717066ec78981f3e01e33405766fd0f85cbcd686c326ca3bf7b9ff4635'
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def tensor_record(t):return dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),content_sha256=tensor_hash(t))
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),numpy=np.random.get_state(),python=random.getstate())
def error(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double();diff=(a-b).abs();nz=b!=0
    return dict(exact=torch.equal(a,b),max_abs=float(diff.max()) if diff.numel() else 0.,violations=int((diff>2e-6+1e-5*b.abs()).sum()),
        at_reference_zero_max_abs=float(diff[~nz].max()) if (~nz).any() else None,
        max_relative_nonzero=float((diff[nz]/b[nz].abs()).max()) if nz.any() else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--input',required=True);ap.add_argument('--old-traces',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);ledger=json.loads((out/'activation.json').read_text());cutoff=ledger['deadline_epoch']-180
    status=dict(status='running',optimizer_steps=0,formal_training=0,new_graphs=0,model_forward_calls=0,model_backward_calls=0)
    def check():
        if time.time()>=cutoff:raise TimeoutError('compute deadline reserves180s for delivery')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stop(s,f):raise InterruptedError('signal '+str(s))
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('alarm')));signal.alarm(max(1,int(cutoff-time.time())))
    def forbidden(*args,**kwargs):status['optimizer_steps']+=1;raise RuntimeError('optimizer.step prohibited')
    for cls in [torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD]:cls.step=forbidden
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one selected GPU')
            env={m.__name__:dict(version=m.__version__,path=m.__file__,sha256=sha(m.__file__)) for m in [torch,torch_geometric,torch_scatter]}
            env.update(device=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),tf32=False)
            env['functions']={f.__module__+'.'+f.__name__:dict(path=inspect.getsourcefile(f),sha256=sha(inspect.getsourcefile(f)),source=inspect.getsource(f)) for f in [pool.global_max_pool,scatter,torch_scatter.scatter,torch_scatter.scatter_max]}
            env['extensions']={str(p):sha(p) for p in pathlib.Path(torch_scatter.__file__).parent.glob('*.so')};save(out/'environment.json',env)
            c,cfg=load_case('IMDB');assert cfg['seed']==2026102709 and c['space'].total==18336
            inp_path=pathlib.Path(a.input);assert sha(inp_path)==INPUT_SHA
            with np.load(inp_path,allow_pickle=False) as z:inp={k[2:]:z[k].copy() for k in z.files if k.startswith('0_')}
            inp['t']=500
            assert len(inp['ids'])==13120 and inp['edges'].shape==(2,313)
            expected=np.flatnonzero(np.repeat(c['active'],c['space'].counts));assert np.array_equal(np.sort(inp['ids']),expected)
            f,loc=c['space'].encode(inp['query']);assert np.array_equal(f,inp['f']) and np.array_equal(c['space'].offsets[f]+loc,inp['ids'])
            assert np.array_equal(inp['labels'],np.isin(inp['ids'],c['clean_ids']).astype(np.float32))
            basis=np.load(HERE/'frozen_package'/cfg['datasets']['IMDB']['basis_file'])['basis'] # original load; A does not use it; NO fitting
            def construct():
                torch.manual_seed(cfg['seed']);anchor=StructuralDenoiser(c,cfg,'A')
                # Original pair_inputs uses a private NumPy RNG, consumes no Torch RNG.
                # Here exact saved arrays are reused, never regenerated or substituted.
                torch.manual_seed(cfg['seed']);m=StructuralDenoiser(c,cfg,'A',basis);names=copy_common(anchor,m)
                return m,names
            m,copied=construct();s1=cpu_tree(m.state_dict());rng1=rng()
            other,copied2=construct();s2=cpu_tree(other.state_dict());assert s1.keys()==s2.keys()
            equality={k:torch.equal(s1[k],s2[k]) for k in s1};save(out/'reconstruction_equality.json',equality)
            if not all(equality.values()):raise RuntimeError('two reconstructions differ; no retry')
            torch.save(dict(model=s1,rng=rng1,config=cfg,source_commit=COMMIT,purpose='authorized RECONSTRUCTED initialization, NOT original checkpoint'),out/'reconstructed_initial.pt')
            torch.save(dict(model=s2,rng=rng(),source_commit=COMMIT,purpose='independent second reconstruction'),out/'reconstructed_initial_repeat.pt')
            save(out/'state_tensor_hashes.json',{k:tensor_record(v) for k,v in s1.items()})
            source={str(p.relative_to(HERE/'frozen_package')):sha(p) for p in (HERE/'frozen_package').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
            save(out/'binding.json',dict(source_commit=COMMIT,source=source,construction_sha256=sha(pathlib.Path(__file__)),config=cfg,
                input_sha256=INPUT_SHA,input_arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in inp.items() if isinstance(v,np.ndarray)},
                time=dict(value=500,provenance='v2 frozen probe.py force_t=500; scalar was NOT stored in NPZ'),
                reconstruction='seed/anchor/private-NumPy-input-independent/seed/A/copy_common; tensor equality not serialized byte equality',copied=copied,
                initial_file_sha256=sha(out/'reconstructed_initial.pt'),repeat_file_sha256=sha(out/'reconstructed_initial_repeat.pt')))
            del other,s2;check();m=m.to('cuda');m.train();m.zero_grad(set_to_none=True)
            captures=[];inside={'yes':False};target=m.backbone.tf_layers[0].self_attn.x_y
            target_pre=target.register_forward_pre_hook(lambda mod,args:inside.update(yes=True))
            target_post=target.register_forward_hook(lambda mod,args,result:inside.update(yes=False))
            globals_=torch_scatter.scatter.__globals__;original_max=globals_['scatter_max'];real_scatter_calls=[]
            def wrapped(*args,**kwargs):
                result=original_max(*args,**kwargs)
                if inside['yes']:
                    params=inspect.signature(original_max).bind(*args,**kwargs);params.apply_defaults();v=params.arguments
                    assert v['out'] is None
                    src=v['src'];value,index=result;src.retain_grad();value.retain_grad()
                    captures.append(dict(src=src,batch=v['index'],output=value,index=index,dim=v['dim'],dim_size=v['dim_size']))
                    real_scatter_calls.append(dict(path='PyG global_max_pool -> PyG scatter -> torch_scatter.scatter -> scatter_max -> torch.ops.torch_scatter.scatter_max',dim=v['dim'],dim_size=v['dim_size'],src=tensor_record(src),batch=tensor_record(v['index'])))
                return result
            globals_['scatter_max']=wrapped;trace=[];handles=traces(m,trace)
            original_pool=pool.global_max_pool;pool_captures=[]
            def wrapped_pool(x,batch,size=None):
                before=len(captures);value=original_pool(x,batch,size)
                if inside['yes']:
                    x.retain_grad();value.retain_grad()
                    index=captures[-1]['index'] if len(captures)>before else None
                    pool_captures.append(dict(src=x,batch=batch,output=value,index=index,dim=0,dim_size=value.shape[0],
                        dispatch=dict(with_torch_scatter=torch_geometric.typing.WITH_TORCH_SCATTER,is_cuda=x.is_cuda,requires_grad=x.requires_grad,grad_fn=type(value.grad_fn).__name__)))
                return value
            pool.global_max_pool=wrapped_pool
            ctx=FrozenTypedContext(m,c['space'].types,inp['edges'],inp['edge_families'],('rebuilt',0));begin=time.perf_counter()
            try:
                graph,logits,p0=ctx.predict(inp['query'],inp['f'],.5,('rebuilt',0),details=True);status['model_forward_calls']+=1
                check_prob_device(p0,inp['f'],c,m.prior)
                loss=supported_binary_ce(logits,torch.tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/(2*18336)
                loss.backward();status['model_backward_calls']+=1;torch.cuda.synchronize()
            finally:
                globals_['scatter_max']=original_max;pool.global_max_pool=original_pool;target_pre.remove();target_post.remove()
                for h in handles:h.remove()
                ctx.close()
            save(out/'capture_observer.json',dict(extension_calls=len(captures),pool_calls=len(pool_captures),dispatch=[x['dispatch'] for x in pool_captures]))
            # A has TWO backbone calls: full-query graph first, true-Xt CN context
            # second (CommonNeighborFeatures._context_nodes). Diagnose only the
            # first pair, retaining invocation records for all calls.
            assert len(pool_captures)==4
            captures=pool_captures[:2]
            np.savez_compressed(out/'rebuilt_forward.npz',logits=logits.detach().cpu().numpy(),p0=p0.detach().cpu().numpy(),loss=loss.detach().cpu().numpy(),ids=inp['ids'])
            old=json.loads(pathlib.Path(a.old_traces).read_text());old_first=old[0][:len(trace)]
            save(out/'old_forward_hash_comparison.json',dict(old_file_sha256=sha(pathlib.Path(a.old_traces)),first_input_only=True,
                matches=[x==y for x,y in zip(trace,old_first)],same_length=len(trace)==len(old_first),new=trace,old=old_first,
                caveat='matching selected forward hashes does NOT certify all original initial parameters; no saved per-slot old logits available'))
            frozen={}
            for name,cap in zip(['min_internal_max','max'],captures):
                frozen[name]={k:cap[k].detach().cpu().clone() for k in ['src','batch','output']}
                frozen[name].update(index=None if cap['index'] is None else cap['index'].detach().cpu().clone(),upstream=cap['output'].grad.detach().cpu().clone(),input_gradient=cap['src'].grad.detach().cpu().clone(),dim=cap['dim'],dim_size=cap['dim_size'],dispatch=cap['dispatch'])
            assert torch.equal(frozen['min_internal_max']['src'],-frozen['max']['src'])
            # For ordinary max the captured src is shared with other paths: its retained
            # gradient is TOTAL network gradient, not the isolated max contribution.
            # Only replay gradients are used for per-operator conservation/support tests.
            torch.save(frozen,out/'captured.pt');save(out/'captured_tensor_hashes.json',{name:{k:tensor_record(v) for k,v in x.items() if torch.is_tensor(v)} for name,x in frozen.items()})
            save(out/'capture_calls.json',dict(calls=real_scatter_calls,loss=float(loss.detach()),seconds=time.perf_counter()-begin,
                captured_min='internal max U=-X,V=max(U), upstream dL/dV=-dL/dmin; outer input contribution=-dL/dU',
                network_src_grad_caveat='max src participates in other network paths; captured src.grad is not isolated max gradient'))
            del captures,ctx,logits,p0,loss,graph,m,s1;torch.cuda.empty_cache();check()
            reports={}
            for name,cap in frozen.items():
                print('REPLAY',name,flush=True);begin=time.perf_counter();src=cap['src'];batch=cap['batch'];g=cap['upstream'];size=cap['dim_size'];assert cap['dim']==0
                mask=src==cap['output'][batch];counts=torch.zeros_like(cap['output'],dtype=torch.int64);counts.index_add_(0,batch,mask.long());tie=counts>1;unique=counts==1
                results=[];arrays={};base=None
                for i in range(20):
                    check();leaf=src.to('cuda').detach().clone().requires_grad_(True);bb=batch.to('cuda');gg=g.to('cuda')
                    if cap['index'] is not None:value,idx=original_max(leaf,bb,dim=0,dim_size=size)
                    else:value=original_pool(leaf,bb,size=size);idx=None
                    value.backward(gg);torch.cuda.synchronize()
                    actual=(value.detach().cpu(),None if idx is None else idx.detach().cpu(),leaf.grad.detach().cpu());arrays[str(i)+'_output']=actual[0].numpy();arrays[str(i)+'_gradient']=actual[2].numpy()
                    if actual[1] is not None:arrays[str(i)+'_index']=actual[1].numpy()
                    if base is None:base=actual
                    diff=actual[2]!=base[2];sums=torch.zeros_like(g);sums.index_add_(0,batch,actual[2]);bad_nonmax=(actual[2]!=0)&~mask
                    results.append(dict(repeat=i,forward=error(actual[0],base[0]),gradient=error(actual[2],base[2]),
                        index_changes=None if actual[1] is None else int((actual[1]!=base[1]).sum()),tie_index_changes=None if actual[1] is None else int(((actual[1]!=base[1])&tie).sum()),unique_index_changes=None if actual[1] is None else int(((actual[1]!=base[1])&unique).sum()),
                        changed_input_positions=int(diff.sum()),changed_outside_tied_extrema=int((diff&~(mask&tie[batch])).sum()),
                        unique_gradient=error(actual[2][unique[batch]],base[2][unique[batch]]),tie_gradient=error(actual[2][tie[batch]],base[2][tie[batch]]),
                        gradient_sum=error(sums,g),non_extremum_nonzero=int(bad_nonmax.sum())))
                np.savez_compressed(out/(name+'_gpu20.npz'),**arrays)
                # CPU uses the SAME installed PyG entry; its native dispatch is recorded separately.
                cpu_leaf=src.clone().requires_grad_(True);cpu_value=pool.global_max_pool(cpu_leaf,batch,size=size);cpu_value.backward(g)
                cpu_report=dict(dispatch='PyG global_max_pool -> native scatter_reduce_(amax, include_self=False)',
                    forward_vs_gpu=error(cpu_value,base[0]),gradient_vs_gpu=error(cpu_leaf.grad,base[2]),
                    caveat='CPU/GPU may select different subgradients at ties; not a same-device-repeat gate',argmax_available=False)
                np.savez_compressed(out/(name+'_cpu.npz'),output=cpu_value.detach().numpy(),gradient=cpu_leaf.grad.numpy())
                sign=None
                if name=='min_internal_max':
                    # Outer min replay via original PyG wrapper and correctly transformed upstream.
                    leaf=(-src).to('cuda').detach().requires_grad_(True);inner=-leaf;inner.retain_grad();v=pool.global_max_pool(inner,batch.to('cuda'),size=size);v.retain_grad();outer=-v;outer.retain_grad()
                    outer.backward((-g).to('cuda'));torch.cuda.synchronize()
                    sign=dict(output_vs_negative_internal=error(outer,-cap['output']),internal_upstream_vs_g=error(v.grad,g),outer_upstream_vs_negative_g=error(outer.grad,-g),
                        input_chain_rule=error(leaf.grad,-inner.grad),note='checks signs within one graph; argmax may differ from another replay')
                reports[name]=dict(unique_channels=int(unique.sum()),tied_channels=int(tie.sum()),tied_extremum_nodes=int((mask&tie[batch]).sum()),
                    extrema_multiplicity=counts.tolist(),upstream_nonzero_channels=int((g!=0).sum()),upstream_max=float(g.abs().max()),
                    gpu20=results,cpu=cpu_report,min_sign=sign,seconds=time.perf_counter()-begin)
                save(out/'operator_results.json',reports)
            status.update(status='completed_reconstructed_diagnostic',operator_results='operator_results.json',same_device_repeats=40,
                two_reconstructions_tensor_equal=True,old_forward_hashes_match=all(x==y for x,y in zip(trace,old_first)),
                peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        except BaseException as exc:
            status.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed',error=repr(exc),traceback=traceback.format_exc());raise
        finally:
            signal.alarm(0);status.update(ended_epoch=time.time(),process_seconds=time.time()-START,warnings=[str(w.message) for w in caught]);save(out/'summary.json',status)
if __name__=='__main__':main()
