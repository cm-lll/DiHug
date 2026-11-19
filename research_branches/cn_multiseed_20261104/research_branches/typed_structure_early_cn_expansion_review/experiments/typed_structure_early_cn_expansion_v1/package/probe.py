"""Bounded v2 numerical diagnosis. No scientific training or graph sampling."""
import argparse,copy,gc,hashlib,json,os,pathlib,random,resource,signal,time,traceback
START=time.time()
import numpy as np
import torch
from prepare import ROOT,load_case,save,sha
from model_variants import StructuralDenoiser,copy_common
from normal2 import normal2,pair_inputs,sync
from hsgd.bw_preflight.common import cpu_tree,environment,dimensions,schedule
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel

def rng_get(rng,device):
    return dict(numpy=copy.deepcopy(rng.bit_generator.state),global_numpy=np.random.get_state(),python=random.getstate(),torch=torch.get_rng_state(),device=torch.cuda.get_rng_state(0) if device=='cuda' else None)
def rng_set(s,rng,device):
    rng.bit_generator.state=copy.deepcopy(s['numpy']);np.random.set_state(s['global_numpy']);random.setstate(s['python']);torch.set_rng_state(s['torch'])
    if device=='cuda':torch.cuda.set_rng_state(s['device'],0)
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def metric(a,b):
    if a is None or b is None:return dict(passes=a is b,max_abs=0.,violations=int(a is not b),exact=a is b)
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=(a-b).abs();bad=d>2e-6+1e-5*b.abs();nz=b!=0
    return dict(passes=not bool(bad.any()),exact=torch.equal(a,b),max_abs=float(d.max()) if d.numel() else 0.,
        max_relative_nonzero=float((d[nz]/b[nz].abs()).max()) if nz.any() else None,
        reference_max=float(b.abs().max()) if b.numel() else 0.,reference_l2=float(torch.linalg.vector_norm(b)),
        absolute_at_reference_zero=float(d[~nz].max()) if (~nz).any() else None,violations=int(bad.sum()))
def flatten(obj,prefix=''):
    if torch.is_tensor(obj):return {prefix:obj}
    if isinstance(obj,dict):return {p:v for k,x in obj.items() for p,v in flatten(x,prefix+'/'+str(k)).items()}
    if isinstance(obj,(list,tuple)):return {p:v for k,x in enumerate(obj) for p,v in flatten(x,prefix+'/'+str(k)).items()}
    return {}
def compare_diagnostics(a,b):
    keys=['loss'];out={}
    for k in keys:out[k]=metric(a[k],b[k])
    out['inputs']=[{k:metric(x[k],y[k]) for k in ['logits','p0','loss']} for x,y in zip(a['inputs'],b['inputs'])]
    out['gradients']={k:metric(v,b['gradients'][k]) for k,v in a['gradients'].items()}
    out['passes']=out['loss']['passes'] and all(v['passes'] for x in out['inputs'] for v in x.values()) and all(v['passes'] for v in out['gradients'].values())
    return out
def input_hash(inputs):
    h=hashlib.sha256()
    for x in inputs:
        h.update(str(x['t']).encode())
        for k in ['ids','query','f','edges','edge_families','labels']:h.update(np.ascontiguousarray(x[k]).tobytes())
    return h.hexdigest()
def traces(model,into):
    handles=[]
    def hook(name):
        def record(mod,args,result):
            a=flatten(result);h=hashlib.sha256()
            for k,v in a.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            into.append(dict(module=name,sha256=h.hexdigest()))
        return record
    for name,mod in model.named_modules():
        if name in ['branch.degree_encoder','branch.degree_out','refex_projection','role_projection','backbone.lin_in_X','joint_head'] or (name.startswith('backbone.tf_layers.') and name.count('.')==2):
            handles.append(mod.register_forward_hook(hook(name)))
    return handles
def first_trace(a,b):
    return next((dict(index=i,left=x,right=y) for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--ledger',required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--cases',nargs='+',default=[d+'_'+a for d in ['IMDB','PubMed'] for a in 'ABC']);args=ap.parse_args()
    ledger=json.loads(pathlib.Path(args.ledger).read_text());deadline=ledger['activation_epoch']+ledger['budget_seconds']-300
    out=pathlib.Path(args.output).resolve()
    if out.exists() or ROOT.resolve() in out.parents or out in ROOT.resolve().parents:raise ValueError('new isolated output')
    out.mkdir(parents=True);status=dict(status='running',cases={},formal_authorized=False);account=pathlib.Path(args.ledger).parent/'executed_update_counts.json'
    def check():
        if time.time()>=deadline:raise TimeoutError('cumulative deadline minus delivery reserve')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stopped(s,f):raise InterruptedError('signal '+str(s))
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stopped)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('hard deadline')));signal.alarm(max(1,int(deadline-time.time())))
    try:
        check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise RuntimeError('one selected GPU required')
        save(out/'environment.json',environment(args.device))
        source={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.txt') and '__pycache__' not in str(p)}
        save(out/'freeze.json',dict(source=source,ledger=ledger,gate='atol2e-6 rtol1e-5 unchanged; optimizer continuation exact REQUIRED',update_cap=3,
            resource='one GPU; full query normal2 activation memory estimate <8GiB at192; no retained first graph; fail on OOM, no shrink',input_rule='same dataset pair t500 seed2026102801 for ALL arms'))
        for dataset in ['IMDB','PubMed']:
            c,cfg=load_case(dataset);entry=cfg['datasets'][dataset]
            assert entry['basis_kind']=='synthetic engineering only' and sha(ROOT/entry['basis_file'])==entry['basis_sha256']
            basis=np.load(ROOT/entry['basis_file'])['basis'];torch.manual_seed(cfg['seed']);anchor=StructuralDenoiser(c,cfg,'A')
            paired=pair_inputs(c,cfg,50,np.random.default_rng(2026102801),500);ih=input_hash(paired)
            np.savez_compressed(out/(dataset+'_unified_inputs.npz'),**{str(i)+'_'+k:v for i,x in enumerate(paired) for k,v in x.items() if isinstance(v,np.ndarray)})
            for arm in 'ABC':
                label=dataset+'_'+arm
                if label not in args.cases:continue
                check();folder=out/label;folder.mkdir();row=dict(status='running',actual_updates=0,input_sha256=ih);status['cases'][label]=row;save(out/'summary.json',status)
                print('START',label,flush=True)
                try:
                    torch.manual_seed(cfg['seed']);m=StructuralDenoiser(c,cfg,arm,basis);row['copied_common']=copy_common(anchor,m);initial=cpu_tree(m.state_dict())
                    save(folder/'dimensions.json',dimensions(m));optargs={k:v for k,v in cfg['optimizer'].items() if k!='type'}
                    opt=torch.optim.AdamW(m.parameters(),**optargs);cpu=[]
                    for repeat in range(2):
                        check();d={};normal2(m,opt,c,paired,cfg['T'],'cpu',False,d);cpu.append(d)
                    row['cpu_repeat']=compare_diagnostics(cpu[0],cpu[1]);save(folder/'cpu_repeat.json',row.pop('cpu_repeat'));row['cpu_repeat_passed']=compare_diagnostics(cpu[0],cpu[1])['passes'];del cpu[1]
                    m=m.to(args.device);opt=torch.optim.AdamW(m.parameters(),**optargs);dev=[];trace=[]
                    if args.device=='cuda':torch.cuda.reset_peak_memory_stats(0)
                    for repeat in range(2):
                        check();d={};tr=[];handles=traces(m,tr)
                        try:normal2(m,opt,c,paired,cfg['T'],args.device,False,d)
                        finally:
                            for h in handles:h.remove()
                        dev.append(d);trace.append(tr)
                    cmp=compare_diagnostics(cpu[0],dev[0]);save(folder/'cross_backend.json',cmp);row['cross_backend_passed']=cmp['passes']
                    cmp=compare_diagnostics(dev[0],dev[1]);cmp['first_forward_hash_difference']=first_trace(trace[0],trace[1]);save(folder/'same_device.json',cmp);row['same_device_passed']=cmp['passes']
                    save(folder/'repeat_forward_traces.json',trace);del cpu,dev,trace,cmp;gc.collect()
                    # Continue disposable diagnosis even if the gate failed; NEVER certify training.
                    m.load_state_dict(initial);opt=torch.optim.AdamW(m.parameters(),**optargs);rng=np.random.default_rng(2026102802);torch.manual_seed(2026102802);random.seed(2026102802);np.random.seed(2026102802)
                    if args.device=='cuda':torch.cuda.manual_seed(2026102802)
                    hist=[];binding=dict(source=source,config=cfg,dataset=entry,arm=arm,device=args.device);continuous=None;previous_diag=None;previous_trace=None
                    for seq,pos in enumerate([1,2,2]):
                        check()
                        if seq==2:
                            ck=torch.load(folder/'after1.pt',map_location='cpu',weights_only=False)
                            if sha(folder/'after1.pt')!=row['checkpoint_sha256'] or ck['binding']!=binding or ck['updates']!=1:raise ValueError('checkpoint binding')
                            # New objects; no inherited parameter/optimizer/feature cache.
                            del m,opt;gc.collect();m=StructuralDenoiser(c,cfg,arm,basis).to(args.device);m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),**optargs);opt.load_state_dict(copy.deepcopy(ck['optimizer']));rng_set(ck['rng'],rng,args.device)
                            row['load_exact']=equal(cpu_tree(m.state_dict()),ck['model']) and equal(cpu_tree(opt.state_dict()),ck['optimizer']) and equal(rng_get(rng,args.device),ck['rng'])
                            if not row['load_exact']:raise ValueError('load exact failed')
                        begin=time.perf_counter();inputs=pair_inputs(c,cfg,pos,rng,500);preparation=time.perf_counter()-begin;hh=input_hash(inputs)
                        if seq==2 and hh!=hist[1]['input_sha256']:raise ValueError('restored input differs')
                        counts=json.loads(account.read_text());used=counts.get(label,0)
                        if used>=3:raise RuntimeError('three update cap')
                        counts[label]=used+1;save(account,counts);row['actual_updates']+=1
                        d={};tr=[];handles=traces(m,tr) if seq>0 else []
                        try:rec=normal2(m,opt,c,inputs,cfg['T'],args.device,True,d)
                        finally:
                            for h in handles:h.remove()
                        rec.update(position=pos,input_sha256=hh,preparation_seconds=preparation,restored=seq==2,trace_enabled=seq>0)
                        hist.append(rec);save(folder/'updates.json',hist)
                        if seq==0:
                            begin=time.perf_counter();snap=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,args.device),updates=1,binding=binding,purpose='engineering only')
                            tmp=folder/'after1.tmp';torch.save(snap,tmp);tmp.replace(folder/'after1.pt');row['save_seconds']=time.perf_counter()-begin;row['checkpoint_sha256']=sha(folder/'after1.pt');del snap
                        if seq==1:continuous=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,args.device));previous_diag=d;previous_trace=tr
                        if seq==2:
                            comp=compare_diagnostics(previous_diag,d);comp['first_forward_hash_difference']=first_trace(previous_trace,tr)
                            comp['parameters']={k:metric(v,m.state_dict()[k]) for k,v in continuous['model'].items()}
                            optnow=cpu_tree(opt.state_dict());comp['optimizer_exact']=equal(continuous['optimizer'],optnow);comp['rng_exact']=equal(continuous['rng'],rng_get(rng,args.device))
                            comp['optimizer_tensors']={k:metric(v,flatten(optnow)[k]) for k,v in flatten(continuous['optimizer']).items()}
                            comp['passes']=comp['passes'] and all(v['passes'] for v in comp['parameters'].values()) and comp['optimizer_exact'] and comp['rng_exact']
                            save(folder/'restore.json',comp);row['restore_passed']=comp['passes'];row['optimizer_continuation_exact']=comp['optimizer_exact'];del comp
                    row['normal2_first_seconds']=hist[0]['normal2_seconds'];row['normal2_traced_second_seconds']=hist[1]['normal2_seconds']
                    row['peak_allocated_mib']=torch.cuda.max_memory_allocated(0)/2**20 if args.device=='cuda' else None
                    row['peak_reserved_mib']=torch.cuda.max_memory_reserved(0)/2**20 if args.device=='cuda' else None
                    # One actual full-query evaluation forward, no chain/sample/commit.
                    inp=paired[0];m.eval();times=[]
                    for repeat in range(2):
                        check();sync(args.device);begin=time.perf_counter();ctx=FrozenTypedContext(m,c['space'].types,inp['edges'],inp['edge_families'],repeat)
                        try:
                            with torch.no_grad():p=ctx.predict(inp['query'],inp['f'],.5,repeat).cpu().numpy().astype(np.float64)
                        finally:ctx.close()
                        sync(args.device);network=time.perf_counter()-begin;state=np.zeros(c['space'].total,np.uint8);ff,ll=c['space'].encode(inp['edges']);state[c['space'].offsets[ff]+ll]=1
                        begin=time.perf_counter();q=MarginalKernel(c['prior'],schedule(cfg['T'])).reverse(p,state[inp['ids']],inp['f'],499,500);posterior=time.perf_counter()-begin
                        times.append(dict(network_full_query_seconds=network,posterior_seconds=posterior,slots=len(p),forwards=1,generated=False))
                    np.savez_compressed(folder/'full_query.npz',ids=inp['ids'],families=inp['f'],p0=p,q=q,state=state)
                    row.update(status='completed_engineering_only',full_query=times,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,feature_seconds=m.feature_seconds,encoding_seconds=m.encoding_seconds)
                    print('END',label,'cross',row['cross_backend_passed'],'repeat',row['same_device_passed'],'restore',row['restore_passed'],flush=True)
                    del m,opt,initial,continuous,previous_diag,previous_trace,inputs,d;gc.collect()
                    if args.device=='cuda':torch.cuda.empty_cache()
                except (TimeoutError,InterruptedError):raise
                except BaseException as exc:
                    row.update(status='failed',error=repr(exc),traceback=traceback.format_exc());print('FAIL',label,repr(exc),flush=True)
                    # OOM/numerical failure does not trigger model/input changes.
                    if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                finally:save(out/'summary.json',status)
            del anchor;gc.collect()
        for name,h in source.items():
            if sha(ROOT/name)!=h:raise ValueError('source drift')
        status['status']='completed_engineering_only' if all(r['status']=='completed_engineering_only' for r in status['cases'].values()) else 'partial_failure'
    except BaseException as exc:
        status.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);status['wall_seconds']=time.time()-START;status['ended_epoch']=time.time();save(out/'summary.json',status)
if __name__=='__main__':main()
