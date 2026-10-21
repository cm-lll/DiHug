"""Small shared adapter. Original model, logits CE, kernel and sweep retained."""
import copy,hashlib,json,os,random,resource,shutil,time
from pathlib import Path
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.runtime import predict,reverse_step
from hsgd.losses import supported_binary_ce
from hsgd.kernel import MarginalKernel
from hsgd.computational import FrozenTypedContext
from role_condition.model import extract_target
from role_condition.metrics import graph,compare
from pair_model import create,paired_initial,Reader

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    p=Path(path);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');q.replace(p)
def emit(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False),flush=True)
def config():return read(ROOT/'config.json')
def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.sh') and '__pycache__' not in p.parts and p.name not in ('manifest.json',)}
def verify_package():
    m=read(ROOT/'manifest.json')
    for n,h in m['files'].items():
        p=ROOT/n
        if not p.is_file() or sha(p)!=h:raise ValueError('package hash mismatch '+n)
    return sha(ROOT/'manifest.json')
def setup(device,strict=False):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(strict,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable; no CPU fallback')
        torch.cuda.set_device(0)
    if torch.are_deterministic_algorithms_enabled()!=strict:raise RuntimeError('execution mode mismatch')
    # Runtime's rollback uses only the selected card, never every visible card.
    c.rng_save=rng_save;c.rng_restore=rng_restore
_lock_handles=[]
def acquire_guard(out,device):
    # Advisory guards only for this adapter in this container. They do not
    # inspect, kill, or claim exclusive ownership over unrelated platform jobs.
    import fcntl,tempfile
    paths=[Path(out)/'run.lock']
    if device=='cuda':paths.append(Path(tempfile.gettempdir())/'dihug_imdb_role_pair_gpu.lock')
    for path in paths:
        f=path.open('a+')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();raise RuntimeError('another adapter process owns '+str(path))
        _lock_handles.append(f)
def rng_save(rng,device):
    return {'generator':copy.deepcopy(rng.bit_generator.state),'python':random.getstate(),'numpy':np.random.get_state(),
        'torch':torch.get_rng_state().clone(),'accelerator':torch.cuda.get_rng_state(0).cpu().clone() if device=='cuda' else None}
def rng_restore(x,rng,device):
    rng.bit_generator.state=copy.deepcopy(x['generator']);random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if device=='cuda':torch.cuda.set_rng_state(x['accelerator'],0)
def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)
def digest(value):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            v=x.detach().cpu().contiguous();h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:str(k)):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()
def dataset(cfg):
    p=ROOT/cfg['data']['file']
    if sha(p)!=cfg['data']['sha256']:raise ValueError('IMDB bytes mismatch')
    with np.load(p,allow_pickle=False) as z:
        case=c.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        np.testing.assert_array_equal(case['space'].encode(case['clean'])[0],z['relations']-1)
    if len(case['space'].types)!=cfg['data']['nodes'] or case['clean'].shape[1]!=cfg['data']['edges'] or case['counts'].tolist()!=cfg['data']['family_counts']:raise ValueError('complete reference count mismatch')
    return case,extract_target(case)
def toy():
    import itertools
    return c.make_case(np.array([0,0,1,1,2,2,2,3,3]),np.array([[0,0,1,2,3,4,5],[2,4,5,4,6,7,8]]),tuple(itertools.combinations_with_replacement(range(4),2)))
def optimizer(model,cfg):
    k=dict(cfg['optimizer']);assert k.pop('type')=='AdamW';k['betas']=tuple(k['betas'])
    return torch.optim.AdamW(model.parameters(),**k)
def condition(model,target):model.set_target(target if model.arm=='role' else None)
def state_from_edges(case,edges):
    state=np.zeros(case['space'].total,np.uint8);f,local=case['space'].encode(edges);state[case['space'].offsets[f]+local]=1;return state
def input_pair(case,cfg,seed,update):
    return [c.train_input(case,cfg['T'],cfg['rounds'],seed,2*update+j,np.random.default_rng([cfg['input_seed'],seed,update,j])) for j in (0,1)]
def update(model,opt,case,inputs,cfg,profile=False,gradients=False):
    device=next(model.parameters()).device.type;c.sync(device);start=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
    parts=[];before=model.joint_head[2].weight.detach().clone()
    for inp in inputs:
        c.sync(device);tick=time.perf_counter();timings={} if profile else None
        logits,p=predict(model,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/cfg['T'],details=True,times=timings)
        c.check_prob_device(p,inp['f'],case,device_prior=model.prior)
        family=torch.as_tensor(inp['f'],device=logits.device);labels=torch.as_tensor(inp['labels'],device=logits.device)
        ce=supported_binary_ce(logits,labels,model.prior[family]);loss=ce.sum()*inp['ce_weight']
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        c.sync(device);forward=time.perf_counter()-tick;tick=time.perf_counter()
        (loss/len(inputs)).backward();c.sync(device);backward=time.perf_counter()-tick
        counts={}
        for rel in range(case['n_families']):
            counts[str(rel)]={str(y):{'count':int(((family==rel)&(labels==y)).sum()),'CE_sum':float(ce[(family==rel)&(labels==y)].detach().sum())} for y in (0,1)}
        parts.append({'t':inp['t'],'query_block':inp['round_id'],'query_count':len(inp['ids']),'loss':float(loss.detach()),'ce_weight':inp['ce_weight'],
            'query_sha256':digest([inp[k] for k in ('ids','f','query')]),'input_sha256':digest([inp[k] for k in ('edges','edge_families','query','f','labels')]),
            'noise_edges':inp['edges'].shape[1],'input_preparation_seconds':inp['preparation_seconds'],'noise_seconds':inp['noise_seconds'],
            'forward_seconds':forward,'backward_seconds':backward,'timings':timings or {},'family_label':counts})
    grads={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
    if not grads or any(not torch.isfinite(x).all() for x in grads.values()):raise FloatingPointError('nonfinite/missing participating gradients')
    norm=float(torch.sqrt(sum((g.double()**2).sum() for g in grads.values())))
    saved={k:v.detach().cpu().clone() for k,v in grads.items()} if gradients else None
    tick=time.perf_counter();opt.step();c.sync(device);optimizer_seconds=time.perf_counter()-tick
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite updated weights')
    result={'loss':sum(x['loss'] for x in parts)/len(parts),'parts':parts,'gradient_norm':norm,'parameter_changed':not torch.equal(before,model.joint_head[2].weight),
        'optimizer_seconds':optimizer_seconds,'update_seconds':time.perf_counter()-start,'lr':opt.param_groups[0]['lr']}
    return result,saved
def checkpoint(folder,payload):
    """Durable file then pointer; recover published-but-unpointed files by scan."""
    folder=Path(folder);folder.mkdir(exist_ok=True);p=folder/f"pair_u{payload['updates']:08d}.pt"
    payload=cpu(payload);payload['payload_digest']=digest(payload)
    if p.exists():
        prior=torch.load(p,map_location='cpu',weights_only=False)
        if prior is not None and prior.get('payload_digest')!=payload['payload_digest']:raise ValueError('checkpoint conflict; no overwrite')
        q=dict(prior);expected=q.pop('payload_digest')
        if digest(q)!=expected:raise ValueError('existing checkpoint content damaged')
    else:
        tmp=p.with_name(p.name+'.tmp');torch.save(payload,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        tmp.replace(p)
    save(folder/'latest.json',{'path':p.name,'sha256':sha(p),'updates':payload['updates'],'payload_digest':payload['payload_digest']})
    return p
def load_checkpoint(folder,binding):
    folder=Path(folder);valid=[]
    for parent in (folder,folder/'recovery'):
        if (parent/'latest.json').exists():
            ptr=read(parent/'latest.json');path=parent/ptr['path']
            if path.parent!=parent or not path.is_file() or sha(path)!=ptr['sha256']:raise ValueError('existing checkpoint pointer/hash conflict')
    for p in sorted(list(folder.glob('pair_u*.pt'))+list((folder/'recovery').glob('pair_u*.pt'))):
        x=torch.load(p,map_location='cpu',weights_only=False);body=dict(x);h=body.pop('payload_digest',None)
        if digest(body)!=h or x['binding']!=binding or p.name!=f"pair_u{x['updates']:08d}.pt":raise ValueError('checkpoint binding/content conflict '+str(p))
        valid.append((x['updates'],p,x))
    if not valid:raise FileNotFoundError('no valid checkpoint')
    _,p,x=max(valid,key=lambda y:y[0]);save(folder/'latest.json',{'path':p.name,'sha256':sha(p),'updates':x['updates'],'payload_digest':x['payload_digest']})
    return x
def rolling_recovery(folder,value):
    # Permanent initial/1000-update/final checkpoints stay in parent. At most
    # two extra 30-minute recovery files from THIS run are retained.
    rec=Path(folder)/'recovery';rec.mkdir(exist_ok=True);result=checkpoint(rec,value)
    old=sorted(rec.glob('pair_u*.pt'))
    for item in old[:-2]:item.unlink()
    return result
def validate(model,case,target,cfg,check):
    device=next(model.parameters()).device.type;rng=np.random.default_rng(0);saved=rng_save(rng,device);training=model.training
    start=time.perf_counter();rows=[];model.eval();condition(model,target)
    try:
        with torch.no_grad():
            for i,t in enumerate(cfg['validation_times']):
                check();inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['validation_seed'],i,np.random.default_rng([cfg['validation_seed'],t]),force_t=t)
                plan=c.plan_for(case,[cfg['validation_seed'],i,21],cfg['rounds']);counts=np.zeros((case['n_families'],2),int);loss=np.zeros_like(counts,float);expected=np.zeros(case['n_families']);cal=np.zeros((10,3));family_cal=np.zeros((case['n_families'],10,3));prior_ce=0.;coverage=np.zeros(case['space'].total,np.uint8)
                token=('validation',t);ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
                try:
                    for b in range(cfg['rounds']):
                        check();ids,f,q=plan.block(b)
                        if not len(ids):continue
                        l,p=predict(model,case['space'].types,inp['edges'],inp['edge_families'],q,f,t/cfg['T'],context=ctx,token=token,details=True)
                        y=np.isin(ids,case['clean_ids']).astype(np.float32);pt=c.check_prob(p,f,case)
                        ce=supported_binary_ce(l,torch.as_tensor(y,device=l.device),model.prior[torch.as_tensor(f,device=l.device)]).cpu().numpy()
                        np.add.at(expected,f,pt);coverage[ids]+=1
                        prior=case['prior'][f];moving=(prior>0)&(prior<1)
                        prior_ce+=float((-y[moving]*np.log(prior[moving])-(1-y[moving])*np.log1p(-prior[moving])).sum())
                        for rel in range(case['n_families']):
                            for label in (0,1):
                                ix=(f==rel)&(y==label);counts[rel,label]+=int(ix.sum());loss[rel,label]+=float(ce[ix].sum())
                        bins=np.minimum((pt*10).astype(int),9)
                        for k in range(10):
                            ix=bins==k;cal[k]+=[int(ix.sum()),float(pt[ix].sum()),float(y[ix].sum())]
                            for rel in range(case['n_families']):
                                ri=ix&(f==rel);family_cal[rel,k]+=[int(ri.sum()),float(pt[ri].sum()),float(y[ri].sum())]
                finally:ctx.close()
                if not np.array_equal(coverage,np.repeat(case['active'].astype(np.uint8),case['space'].counts)):raise ValueError('validation coverage')
                rows.append({'t':t,'CE':float(loss.sum()/case['space'].total),'constant_prior_CE':prior_ce/case['space'].total,'family_label_CE_sum':loss.tolist(),'family_label_count':counts.tolist(),
                    'expected_edges':expected.tolist(),'expected_edge_signed_error':(expected-case['counts']).tolist(),'calibration_bins_count_sumP_sumY':cal.tolist(),
                    'family_calibration_bins_count_sumP_sumY':family_cal.tolist(),'coverage_sha256':digest(coverage)})
    finally:rng_restore(saved,rng,device);model.train(training)
    return {'rows':rows,'wall_seconds':time.perf_counter()-start,'same_reference_only':True,'RNG_restored':True,'mode':'ordinary' if not torch.are_deterministic_algorithms_enabled() else 'strict'}
def environment(device):
    x=c.environment(device);x['visibility']=os.environ.get('CUDA_VISIBLE_DEVICES');x['HIP_visibility']={k:os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}
    x['free_disk_bytes']=shutil.disk_usage(ROOT).free
    if device=='cuda':
        free,total=torch.cuda.mem_get_info(0);x['device_free_bytes']=free;x['device_total_bytes']=total;x['device_total_occupied_bytes']=total-free
    return x
def bound_environment(e):
    return {k:e.get(k) for k in ('packages','python_version','torch_runtime_version','hip','cuda','device_name','visibility','HIP_visibility','matmul_allow_tf32','torch_threads','deterministic_algorithms')}
