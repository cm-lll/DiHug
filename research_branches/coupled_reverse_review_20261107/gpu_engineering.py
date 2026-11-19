"""Six fixed current-PB inputs: parameter gradients and cost, no updates/chains."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, time, hashlib, gc, traceback, signal, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/coupled_reverse_gpu_20261107'
PARENT = ROOT/'results/pb_confirmation_formal_20261107T005911'
sys.path.insert(0, str(ROOT/'research_branches/pb_confirmation_review_20261107'))
from runtime import science as g
from pb_degree import RelationDegreeCDF
from triangle_risk import CoupledTriangle
from global_scale import evaluate
from hsgd.losses import supported_binary_ce
from ema import cpu_copy
from integration import checkpoint_payload
torch, np = g.torch, g.np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    p=OUT/(name+'.json'); q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');q.replace(p)
def sync(): torch.cuda.synchronize()
def elapsed(start): sync(); return time.perf_counter()-start
def abort(*args): raise TimeoutError('bounded GPU engineering deadline')
def group(name):
    if name.startswith('common_neighbor'): return 'CN'
    if name.startswith('joint_head'): return 'joint_head'
    if name.startswith('backbone'): return 'backbone'
    return 'other'
def norms(names, base, aux):
    result=[]
    for label in ['all','CN','backbone','joint_head','other']:
        aa=bb=ab=0.; count=0
        for n,x,y in zip(names,base,aux):
            if label!='all' and group(n)!=label: continue
            if x is not None: aa+=float(x.double().square().sum());count+=x.numel()
            if y is not None: bb+=float(y.double().square().sum())
            if x is not None and y is not None: ab+=float((x.double()*y.double()).sum())
        result.append(dict(group=label,parameters=count,baseline_norm=aa**.5,risk_norm=bb**.5,
          ratio=(bb/aa)**.5 if aa else None,cosine=ab/(aa*bb)**.5 if aa and bb else None))
    return result

def main():
    ticket=json.loads((OUT/'registration.json').read_text())
    remain=ticket['hard_stop_epoch']-time.time()
    if remain<120: raise TimeoutError('insufficient remaining engineering window')
    signal.signal(signal.SIGALRM,abort);signal.alarm(int(remain))
    signal.signal(signal.SIGTERM,abort)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.cuda.device_count()==1
    begin=time.time();save('environment',g.common.environment('cuda'))
    hashes={}
    for p,h in json.loads((PARENT/'source_binding.json').read_text()).items():
        if sha(p)!=h:raise ValueError('parent source hash '+p)
        hashes[p]=h
    for p in Path(__file__).parent.glob('*.py'):hashes[str(p)]=sha(p)
    save('source_binding',hashes)
    rows=[];bindings=[];state_checks=[];costs=[]
    tasks=json.loads((PARENT/'tasks.json').read_text())
    for di,(ds,seed) in enumerate([('IMDB',2026110711),('PubMed',2026110721)]):
        row=next(r for r in tasks if r['dataset']==ds and r['arm']=='PB')
        cfg=row['config'];folder=PARENT/'run'/row['task_id']
        registration=json.loads((folder/'sampling_registration.json').read_text())
        readout=Path(registration['readout_file'])
        assert sha(readout)==registration['readout_sha256']
        ro=torch.load(readout,map_location='cpu')
        cp=folder/'final.pt';assert sha(cp)==ro['checkpoint_sha256']
        z=torch.load(cp,map_location='cpu');b=z['binding']
        assert z['updates']==3000 and z['ema']['updates']==3000
        assert b['dataset']==ds and b['train_seed']==seed and b['arm']=='PB' and b['config']==cfg
        data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file']
        assert sha(data)==cfg['datasets'][ds]['sha256']
        with np.load(data,allow_pickle=False) as a:
            case=g.common.make_case(a['types'],a['edges'].T,tuple(map(tuple,a['schema'])))
        assert np.array_equal(np.asarray(b['output_objective']['target']['prior']),case['prior'])
        assert b['output_objective']['weight']==1.
        model=g.StructuralDenoiser(case,cfg,'A').cuda()
        optimizer=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        ema=g.ParameterEMA(model);rng=np.random.default_rng(1)
        g.restore_payload(z,model,optimizer,ema,b,lambda r:g.rng_restore(r,rng,'cuda'))
        actual=checkpoint_payload(model,optimizer,ema,3000,g.rng_capture(rng,'cuda'),b)
        for key in ['model','optimizer','ema','rng','updates']:
            assert g.digest(actual[key])==g.digest(z[key]),key
        temporary=OUT/(ds+'_loaded_boundary.pt');torch.save(actual,temporary)
        zz=torch.load(temporary,map_location='cpu')
        g.restore_payload(dict(zz,recovery=z['recovery']),model,optimizer,ema,b,
                          lambda r:g.rng_restore(r,rng,'cuda'))
        state_checks.append(dict(dataset=ds,full_state_loaded_exact=True,file_roundtrip_exact=g.digest(actual)==g.digest(zz),updates=3000))
        assert state_checks[-1]['file_roundtrip_exact']
        before=g.digest(model.state_dict());rbefore=g.digest(g.rng_capture(rng,'cuda'))
        pb=RelationDegreeCDF(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=sha(data))
        triangle=CoupledTriangle(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=sha(data))
        kernel=g.MarginalKernel(case['prior'],g.common.schedule(1000))
        bindings.append(dict(dataset=ds,seed=seed,readout='raw',updates=3000,checkpoint=str(cp),checkpoint_sha256=sha(cp),
          data=str(data),data_sha256=sha(data),schema=[list(x) for x in case['space'].pairs],prior=case['prior'].tolist()))
        save('bindings',bindings);save('state_checks',state_checks)
        model.train();names,parameters=zip(*model.named_parameters())
        for j,t in enumerate([100,500,800]):
            if time.time()>ticket['hard_stop_epoch']-45:raise TimeoutError('reserve cleanup')
            inp=g.common.train_input(case,1000,1,2026110781+di,j,np.random.default_rng(2026110781+di*10+j),force_t=t)
            ih=g.digest({k:inp[k] for k in ['t','ids','query','edges','edge_families','labels']})
            np.savez_compressed(OUT/f'{ds}_t{t}_input.npz',**{k:inp[k] for k in ['t','ids','query','edges','edge_families','labels','f']})
            assert inp['ce_weight']==1/case['space'].total
            sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            token=(ds,t,'engineering');ctx=g.FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
            graph,logits,p=ctx.predict(inp['query'],inp['f'],t/1000,token,details=True)
            g.common.check_prob_device(p,inp['f'],case,model.prior)
            ce=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),model.prior[graph.query_families]).sum()/case['space'].total
            pb_loss,_=pb(p,inp['query'],inp['f']);base=ce+pb_loss
            baseg=torch.autograd.grad(base,parameters,retain_graph=True,allow_unused=True)
            baseline_seconds=elapsed(start)
            start=time.perf_counter();context=triangle.context(inp['edges'],t,kernel)
            risk,parts,scale=evaluate(context,p,inp['query'],inp['f'],accumulation_dtype=torch.float64)
            auxg=torch.autograd.grad(risk,parameters,allow_unused=True)
            extra_seconds=elapsed(start)
            assert torch.isfinite(base) and torch.isfinite(risk)
            assert all(torch.isfinite(v).all() for v in baseg+auxg if v is not None)
            stats=norms(names,baseg,auxg)
            np.savez_compressed(OUT/f'{ds}_t{t}_probabilities.npz',p=p.detach().cpu().numpy(),query=inp['query'],family=inp['f'])
            rec=dict(dataset=ds,seed=seed,t=t,readout='raw',input_sha256=ih,CE=float(ce),PB=float(pb_loss),risk=float(risk),
              baseline_network_FB_seconds=baseline_seconds,incremental_risk_network_B_seconds=extra_seconds,
              peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
              groups=stats,normalization=scale,source_unchanged=(before==g.digest(model.state_dict())),
              RNG_unchanged=(rbefore==g.digest(g.rng_capture(rng,'cuda'))))
            assert rec['source_unchanged'] and rec['RNG_unchanged']
            rows.append(rec);save('parameter_gradients',rows)
            print(json.dumps({k:rec[k] for k in ['dataset','t','CE','PB','risk','baseline_network_FB_seconds','incremental_risk_network_B_seconds','peak_allocated_MiB']}),flush=True)
            # Standalone operator timing isolates structure overhead from an extra network backward.
            ctx.close();del ctx,graph,logits,baseg,auxg,base,ce,pb_loss,risk,parts
            pt=p.detach().clone().requires_grad_(True);del p
            sync();start=time.perf_counter();l,_,_=evaluate(context,pt,inp['query'],inp['f'],accumulation_dtype=torch.float64)
            grad=torch.autograd.grad(l,pt)[0];sec=elapsed(start)
            costs.append(dict(dataset=ds,t=t,standalone_FP64_operator_FB_seconds=sec,gradient_norm=float(grad.norm())))
            save('operator_costs',costs)
            del pt,l,grad,context;gc.collect()
        assert before==g.digest(model.state_dict()) and rbefore==g.digest(g.rng_capture(rng,'cuda'))
        for key,value in [('optimizer',optimizer.state_dict()),('ema',ema.state_dict())]:assert g.digest(value)==g.digest(z[key])
        del model,optimizer,ema,z,actual,zz,ro,pb,triangle,parameters;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==6
    bg=np.array([r['groups'][0]['baseline_norm'] for r in rows]);ag=np.array([r['groups'][0]['risk_norm'] for r in rows])
    coefficient=.1*float(np.linalg.norm(bg)/np.linalg.norm(ag))
    assert np.isfinite(coefficient) and coefficient>0
    save('calibration',dict(rule='0.1 * RMS(norm(g_CE+PB)) / RMS(norm(g_risk)); all six states equal weight',
      coefficient=coefficient,formal_coefficient_adopted=False,
      ratios_after_calibration=(coefficient*ag/bg).tolist(),states=[dict(dataset=r['dataset'],t=r['t']) for r in rows],
      limitation='One raw trained model per dataset; not guaranteed representative of initial training or a quality improvement.'))
    assert all(sha(p)==h for p,h in hashes.items())
    save('receipt',dict(status='complete',seconds=time.time()-begin,models=2,forward_inputs=6,parameter_backward_calls=12,
       operator_backward_calls=6,optimizer_updates=0,new_graphs=0,source_unchanged=True,
       engineering_only=True,formal_started=False,ordinary_GPU_recompute_tolerance='not retested; historical failure remains'))

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
