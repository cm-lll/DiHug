import json
import os
from pathlib import Path
import signal
import time
import torch
from hsgd.bw_preflight import common as c
from .data import ROOT, digest

class Stopped(Exception): pass

def protect(output):
    p=Path(output).resolve()
    if p==ROOT or ROOT in p.parents or p in ROOT.parents:
        raise ValueError('output must be separate from package/source')
    return p

def verify_release():
    manifest=ROOT/'release_manifest.json'
    if not manifest.is_file(): raise ValueError('release manifest missing')
    data=json.loads(manifest.read_text())
    for name,sha in data.items():
        if c.sha(ROOT/name)!=sha: raise ValueError('release file changed: '+name)
    return c.sha(manifest)

def atomic_json(path,obj):
    c.json_save(path,obj)

def log(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**obj),allow_nan=False)+'\n')
    print(json.dumps(obj,allow_nan=False),flush=True)

class Control:
    def __init__(self,out,seconds=None):
        self.out=Path(out);self.signal=None;self.old={}
        self.deadline=time.monotonic()+seconds if seconds else None
    def __enter__(self):
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            self.old[sig]=signal.signal(sig,lambda s,f:setattr(self,'signal',s))
        return self
    def check(self):
        if self.signal or (self.out/'STOP').exists(): raise Stopped('signal/STOP requested')
        if self.deadline and time.monotonic()>=self.deadline: raise Stopped('bounded test deadline')
    def __exit__(self,*args):
        for s,h in self.old.items():signal.signal(s,h)

def configure(device,strict=False):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(strict,warn_only=False)
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('accelerator unavailable, no CPU fallback')
    env=c.environment(device)
    if device=='cuda':
        index=torch.cuda.current_device();p=torch.cuda.get_device_properties(index)
        env.update(device_index=index,device_name=torch.cuda.get_device_name(index),
                   device_memory_bytes=p.total_memory,device_uuid=str(getattr(p,'uuid',getattr(p,'pci_bus_id','unavailable'))),
                   visibility={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')})
    return env

def binding(case,cfg,env):
    provenance=ROOT/'source_revision.json'
    source=json.loads(provenance.read_text()) if provenance.exists() else {'source_commit':'unreleased-local-engineering'}
    return dict(version=cfg['version'],engineering_fixture=cfg['dataset']=='ENGINEERING_ONLY',release_sha256=verify_release(),config_sha256=digest(cfg),
                source_commit=source['source_commit'],
                data_sha256=cfg['data']['sha256'],runtime=c.runtime_binding(env),backend=cfg['backend'],
                device_identity={k:env.get(k) for k in ('device_index','device_uuid','visibility')})

def optimizer(model,cfg):
    x=dict(cfg['optimizer']);kind=x.pop('type')
    if kind!='AdamW':raise ValueError('optimizer mismatch')
    x['betas']=tuple(x['betas'])
    return torch.optim.AdamW(model.parameters(),**x)

def device_lock(parent):
    import fcntl
    env=configure('cuda',False)
    key=digest({k:env.get(k) for k in ('device_index','device_uuid','visibility')})
    folder=Path(parent)/'.device_leases';folder.mkdir(parents=True,exist_ok=True)
    f=(folder/key).open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:f.close();raise
    return f

def save_checkpoint(out,model,opt,rng,device,u,bind,kind='milestone',extra=None):
    out=Path(out);folder=out/'checkpoints';folder.mkdir(exist_ok=True)
    name=('u%06d.pt'%u) if kind=='milestone' else ('stop_u%06d_%d.pt'%(u,time.time_ns()))
    path=folder/name
    if path.exists(): raise ValueError('immutable checkpoint already exists')
    start=time.monotonic()
    ck=dict(purpose='engineering fixture' if bind.get('engineering_fixture') else 'fresh formal joint CN training',binding=bind,completed_updates=u,
            model=c.cpu_tree(model.state_dict()),optimizer=c.cpu_tree(opt.state_dict()),
            rng=c.cpu_tree(c.rng_save(rng,device)),extra=extra or {})
    c.torch_save(path,ck)
    receipt=dict(path=str(path.relative_to(out)),sha256=c.sha(path),completed_updates=u,
                 save_seconds=time.monotonic()-start)
    atomic_json(out/'latest.json',receipt)
    if kind=='milestone':
        ledger=json.loads((out/'milestones.json').read_text()) if (out/'milestones.json').exists() else {}
        ledger[str(u)]=receipt;atomic_json(out/'milestones.json',ledger)
    return receipt

def load_checkpoint(out,model,opt,rng,device,bind,update=None):
    out=Path(out)
    receipt=(json.loads((out/'latest.json').read_text()) if update is None else
             json.loads((out/'milestones.json').read_text())[str(update)])
    path=out/receipt['path']
    if c.sha(path)!=receipt['sha256']:raise ValueError('checkpoint hash mismatch')
    ck=c.trusted_load(path)
    purpose='engineering fixture' if bind.get('engineering_fixture') else 'fresh formal joint CN training'
    if ck['purpose']!=purpose or ck['binding']!=bind:
        raise ValueError('checkpoint source/config/runtime binding mismatch')
    if ck['completed_updates']!=receipt['completed_updates']:raise ValueError('update count mismatch')
    model.load_state_dict(ck['model'],strict=True)
    if opt is not None:opt.load_state_dict(ck['optimizer'])
    if rng is not None:c.rng_restore(ck['rng'],rng,device)
    return ck,receipt
