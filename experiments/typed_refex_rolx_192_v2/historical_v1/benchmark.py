"""Engineering only. <=5 executed optimizer steps per dataset/arm. No graph sampling."""
import argparse,copy,gc,hashlib,json,os,pathlib,random,resource,signal,sys,time,traceback
START=time.time()
import numpy as np
import torch
from prepare import ROOT,load_case,sha,save
from model_variants import StructuralDenoiser,copy_common
from hsgd.bw_preflight.common import train_input,update,cpu_tree,dimensions,environment,schedule,plan_for
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel

def rng_get(rng,device):
    return dict(numpy=copy.deepcopy(rng.bit_generator.state),global_numpy=np.random.get_state(),python=random.getstate(),
                torch=torch.get_rng_state(),device=torch.cuda.get_rng_state(0) if device=='cuda' else None)
def rng_set(s,rng,device):
    rng.bit_generator.state=copy.deepcopy(s['numpy']);np.random.set_state(s['global_numpy']);random.setstate(s['python']);torch.set_rng_state(s['torch'])
    if device=='cuda':torch.cuda.set_rng_state(s['device'],0)
def sync(device):
    if device=='cuda':torch.cuda.synchronize(0)
def compare(a,b,atol=2e-6,rtol=1e-5):
    if a is None or b is None:return dict(passes=a is b,max_abs=0.,violations=int(a is not b))
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=(a-b).abs();bad=d>atol+rtol*b.abs()
    return dict(passes=not bool(bad.any()),max_abs=float(d.max()) if d.numel() else 0.,violations=int(bad.sum()))
def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--ledger',required=True);ap.add_argument('--device',choices=['cpu','cuda'],required=True)
    ap.add_argument('--cases',nargs='+',choices=[d+'_'+g for d in ['IMDB','PubMed'] for g in 'ABC'],default=[d+'_'+g for d in ['IMDB','PubMed'] for g in 'ABC']);a=ap.parse_args()
    ledger=json.loads(pathlib.Path(a.ledger).read_text());cutoff=min(START+600,ledger['activation_epoch']+ledger['budget_seconds']-180)
    out=pathlib.Path(a.output).resolve()
    if out.exists() or out.is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(out):raise ValueError('new separate output required')
    if cutoff<=time.time():raise TimeoutError('no remaining budget')
    out.mkdir(parents=True);status={'status':'running','purpose':'temporary engineering; never scientific initialization','cases':{},'started_epoch':START}
    account=pathlib.Path(a.ledger).parent/'executed_update_counts.json'
    if not account.exists():raise ValueError('explicit cumulative engineering update account required')
    def check():
        if time.time()>=cutoff:raise TimeoutError('cumulative/subtask deadline')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stop(signum,frame):raise InterruptedError('signal %d'%signum)
    for s in [signal.SIGTERM,signal.SIGHUP,signal.SIGINT]:signal.signal(s,stop)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('hard subtask alarm')));signal.alarm(max(1,int(cutoff-time.time())))
    try:
        check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        if a.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise RuntimeError('exactly one selected visible GPU required')
        env=environment(a.device);save(out/'environment.json',env)
        source={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.txt') and '__pycache__' not in str(p)}
        save(out/'freeze.json',dict(files=source,config=json.loads((ROOT/'config.json').read_text()),ledger=ledger,
            gate='FP32 atol2e-6 rtol1e-5; cross-backend gate is reported separately and never certifies formal training',
            update_plan='1,2,3 continuous; reload after1 then2,3 =5 actual calls,3 distinct disposable steps'))
        for dataset in ['IMDB','PubMed']:
            c,cfg=load_case(dataset);entry=cfg['datasets'][dataset];bp=ROOT/entry['basis_file']
            if sha(bp)!=entry['basis_sha256']:raise ValueError('basis hash mismatch')
            basis=np.load(bp,allow_pickle=False)['basis'];check()
            torch.manual_seed(cfg['seed']);anchor=StructuralDenoiser(c,cfg,'A');anchor_state=cpu_tree(anchor.state_dict())
            shared_b=None
            for arm in ['A','B','C']:
                if dataset+'_'+arm not in a.cases:continue
                check();label=dataset+'_'+arm;folder=out/label;folder.mkdir();row={'status':'running','actual_updates':0};status['cases'][label]=row;save(out/'summary.json',status)
                torch.manual_seed(cfg['seed']);m=StructuralDenoiser(c,cfg,arm,basis)
                copied=copy_common(anchor,m)
                if arm=='B':shared_b=m.refex_projection.weight.detach().clone()
                if arm=='C':
                    if shared_b is None:shared_b=StructuralDenoiser(c,cfg,'B').refex_projection.weight.detach().clone()
                    with torch.no_grad():m.refex_projection.weight.copy_(shared_b)
                    assert torch.equal(m.refex_projection.weight,shared_b)
                save(folder/'dimensions.json',dimensions(m));row['copied_common_tensor_names']=copied
                initial=cpu_tree(m.state_dict());torch.save(dict(model=initial,purpose='engineering only'),folder/'initial.pt')
                # Same real training input through original noising and query entry.
                inp=train_input(c,cfg['T'],cfg['rounds'],cfg['seed'],0,np.random.default_rng(2026102710),force_t=500)
                d_cpu={};opt_args={k:v for k,v in cfg['optimizer'].items() if k!='type'}
                opt=torch.optim.AdamW(m.parameters(),**opt_args)
                update(m,opt,c,inp,cfg['T'],'cpu',do_step=False,diagnostics=d_cpu)
                if a.device=='cuda':
                    m=m.to('cuda');opt=torch.optim.AdamW(m.parameters(),**opt_args);d_dev={}
                    update(m,opt,c,inp,cfg['T'],'cuda',do_step=False,diagnostics=d_dev)
                    checks={k:compare(d_cpu[k],d_dev[k]) for k in ['logits','p0','loss']}
                    grads={k:compare(v,d_dev['gradients'][k]) for k,v in d_cpu['gradients'].items()}
                    checks['gradients']={'passes':all(v['passes'] for v in grads.values()),'max_abs':max(v['max_abs'] for v in grads.values()),
                                         'violations':sum(v['violations'] for v in grads.values()),'per_parameter':grads}
                    save(folder/'cpu_gpu_gate.json',checks);row['cpu_gpu_gate_passed']=all(v['passes'] for v in checks.values())
                    del d_dev
                else:row['cpu_gpu_gate_passed']=None
                del d_cpu;gc.collect();m.load_state_dict(initial);opt=torch.optim.AdamW(m.parameters(),**opt_args)
                rng=np.random.default_rng(2026102711);random.seed(2026102711);np.random.seed(2026102711);torch.manual_seed(2026102711)
                if a.device=='cuda':torch.cuda.manual_seed(2026102711);torch.cuda.reset_peak_memory_stats(0)
                binding=dict(dataset=entry,arm=arm,source=source,config_sha256=sha(ROOT/'config.json'),environment=env)
                history=[];snap=None;continuous=None;continuation=[]
                for position in [1,2,3,2,3]:
                    check()
                    if len(history)==3:
                        start=time.perf_counter();ck=torch.load(folder/'after1.pt',map_location='cpu',weights_only=False)
                        if sha(folder/'after1.pt')!=row['checkpoint_sha256'] or ck['binding']!=binding:raise ValueError('checkpoint binding')
                        m.load_state_dict(ck['model']);opt.load_state_dict(copy.deepcopy(ck['optimizer']));rng_set(ck['rng'],rng,a.device)
                        assert equal_tree(cpu_tree(m.state_dict()),ck['model']) and equal_tree(cpu_tree(opt.state_dict()),ck['optimizer'])
                        assert equal_tree(rng_get(rng,a.device),ck['rng']) and ck['updates']==1
                        row['load_exact']=True;row['load_seconds']=time.perf_counter()-start
                    inp=train_input(c,cfg['T'],cfg['rounds'],cfg['seed'],position,rng,force_t={1:1,2:500,3:1000}[position])
                    inp_hash=hashlib.sha256(b''.join(np.ascontiguousarray(inp[k]).tobytes() for k in ['ids','f','query','labels','edges','edge_families'])).hexdigest()
                    if len(history)>=3:assert inp_hash==history[position-1]['input_sha256']
                    counts=json.loads(account.read_text());used=counts.get(label,0)
                    if used>=5:raise RuntimeError('cumulative five-update cap reached for '+label)
                    counts[label]=used+1;save(account,counts)
                    sync(a.device);start=time.perf_counter();record=update(m,opt,c,inp,cfg['T'],a.device);sync(a.device)
                    record.update(input_sha256=inp_hash,position=position,restored=len(history)>=3,wall_seconds=time.perf_counter()-start)
                    record['projection_gradients']={n:float(p.grad.abs().max()) for n,p in m.named_parameters() if 'projection' in n and p.grad is not None}
                    history.append(record);row['actual_updates']+=1
                    with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                    if len(history)==1:
                        start=time.perf_counter();snap=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,a.device),updates=1,binding=binding,purpose='engineering only')
                        temp=folder/'after1.tmp';torch.save(snap,temp);temp.replace(folder/'after1.pt');row['save_seconds']=time.perf_counter()-start;row['checkpoint_sha256']=sha(folder/'after1.pt')
                    if len(history)==3:continuous=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,a.device))
                recovery={k:compare(v,m.state_dict()[k]) for k,v in continuous['model'].items()}
                row['restore_continuation']=dict(passed=all(v['passes'] for v in recovery.values()),max_abs=max(v['max_abs'] for v in recovery.values()),
                    optimizer_exact=equal_tree(continuous['optimizer'],cpu_tree(opt.state_dict())),rng_exact=equal_tree(continuous['rng'],rng_get(rng,a.device)))
                save(folder/'restore_comparison.json',recovery)
                if not row['restore_continuation']['passed'] or not row['restore_continuation']['rng_exact']:raise AssertionError('continuation failed; no tolerance adjustment')
                # Read all20 blocks twice, with original posterior, NEVER sample/commit.
                sweep_inp=train_input(c,cfg['T'],20,cfg['seed'],50,np.random.default_rng(2026102712),force_t=500)
                plan=plan_for(c,[2026102712,500,61],20);kernel=MarginalKernel(c['prior'],schedule(cfg['T']))
                state=np.zeros(c['space'].total,dtype=np.uint8);f,loc=c['space'].encode(sweep_inp['edges']);state[c['space'].offsets[f]+loc]=1
                m.eval();sweeps=[];first=None
                for repeat in range(2):
                    check();sync(a.device);start=time.perf_counter();fc=m.feature_calls;feature=m.feature_seconds;encoding=m.encoding_seconds;transfer=m.transfer_seconds
                    token=('engineering_full_query',repeat);ctx=FrozenTypedContext(m,c['space'].types,sweep_inp['edges'],sweep_inp['edge_families'],token)
                    p0=np.full(len(state),np.nan);q=p0.copy();seen=np.zeros(len(state),np.uint8);network=0.;posterior=0.
                    try:
                        with torch.no_grad():
                            for r in range(20):
                                check();ids,ff,qe=plan.block(r);sync(a.device);beg=time.perf_counter()
                                pp=ctx.predict(qe,ff,.5,token).detach().cpu().numpy().astype(np.float64);sync(a.device);network+=time.perf_counter()-beg
                                beg=time.perf_counter();q[ids]=kernel.reverse(pp,state[ids],ff,499,500);posterior+=time.perf_counter()-beg;p0[ids]=pp;seen[ids]+=1
                    finally:ctx.close()
                    expected=np.repeat(c['active'].astype(np.uint8),c['space'].counts);assert np.array_equal(seen,expected)
                    if first is not None:np.testing.assert_allclose(p0,first,atol=2e-6,rtol=1e-5,equal_nan=True)
                    first=p0.copy();sync(a.device)
                    sweeps.append(dict(repeat=repeat,wall_seconds=time.perf_counter()-start,all20_blocks=True,covered=int(seen.sum()),
                         feature_calls=m.feature_calls-fc,feature_seconds=m.feature_seconds-feature,encoding_seconds=m.encoding_seconds-encoding,
                         feature_transfer_seconds=m.transfer_seconds-transfer,query_network_including_feature_seconds=network,posterior_seconds=posterior,
                         cache='within Xt only; new context, no predictions reused on repeat',generated_graph=False))
                np.savez_compressed(folder/'query_probe.npz',state=state,p0=p0,q=q,coverage=seen)
                row.update(status='completed',sweeps=sweeps,parameter_count=sum(p.numel() for p in m.parameters()),
                    stable_update_median_seconds=float(np.median([v['synchronized_update_seconds'] for v in history[1:]])),
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20 if a.device=='cuda' else None,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20 if a.device=='cuda' else None)
                save(out/'summary.json',status);print(label,json.dumps(row['restore_continuation']),row['stable_update_median_seconds'],flush=True)
                del m,opt,snap,continuous,initial;gc.collect()
                if a.device=='cuda':torch.cuda.empty_cache()
            del anchor,anchor_state;gc.collect()
        for name,h in source.items():
            check()
            if sha(ROOT/name)!=h:raise RuntimeError('source drift')
        status['status']='completed_engineering_only';status['formal_training_authorized']=False
    except BaseException as exc:
        status.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);status['ended_epoch']=time.time();status['total_seconds']=time.time()-START
        save(out/'summary.json',status)

if __name__=='__main__':main()
