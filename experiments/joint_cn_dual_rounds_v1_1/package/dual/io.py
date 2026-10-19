import json
import os
from pathlib import Path
import signal
import time
import torch
from hsgd.bw_preflight import common as c
from .data import ROOT, digest
from . import transactions, migration
from .rng import rng_save,rng_restore

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
    with Path(path).open('rb') as f:os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

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
    # common.environment('cuda') inspects visible index 0; keep this entry selected-only.
    env=c.environment('cpu');env['requested_device']=device
    if device=='cuda':
        index=torch.cuda.current_device();p=torch.cuda.get_device_properties(index)
        env.update(device_index=index,device_name=torch.cuda.get_device_name(index),device_type='HIP' if torch.version.hip else 'CUDA',
                   visible_devices=torch.cuda.device_count(),
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

def save_checkpoint(out,model,opt,rng,device,u,bind,kind='milestone',extra=None,execution_binding=None):
    if kind not in ('milestone','stop'):raise ValueError('invalid checkpoint kind')
    migration.check(bind,execution_binding or bind,allow=True)
    extra=dict(extra or {},execution_binding=execution_binding or bind)
    ck=dict(purpose='engineering fixture' if bind.get('engineering_fixture') else 'fresh formal joint CN training',binding=bind,completed_updates=u,
            model=c.cpu_tree(model.state_dict()),optimizer=c.cpu_tree(opt.state_dict()),
            rng=c.cpu_tree(rng_save(rng,device)),extra=extra)
    return transactions.save(out,ck,u,bind,kind,atomic_json)

def load_checkpoint(out,model,opt,rng,device,bind,update=None,execution_binding=None,allow_device_migration=False):
    out=Path(out)
    migration.check(bind,execution_binding or bind,allow=True)
    transactions.reconcile(out,bind,atomic_json)
    receipt=(json.loads((out/'latest.json').read_text()) if update is None else
             json.loads((out/'milestones.json').read_text())[str(update)])
    path=out/receipt['path']
    if c.sha(path)!=receipt['sha256']:raise ValueError('checkpoint hash mismatch')
    ck=c.trusted_load(path)
    purpose='engineering fixture' if bind.get('engineering_fixture') else 'fresh formal joint CN training'
    if ck['purpose']!=purpose or ck['binding']!=bind:
        raise ValueError('checkpoint source/config/runtime binding mismatch')
    if ck['completed_updates']!=receipt['completed_updates']:raise ValueError('update count mismatch')
    saved_execution=ck['extra']['execution_binding'];requested=execution_binding or bind
    changed=migration.check(saved_execution,requested,allow_device_migration)
    model.load_state_dict(ck['model'],strict=True)
    if opt is not None:opt.load_state_dict(ck['optimizer'])
    if rng is not None:rng_restore(ck['rng'],rng,device)
    if changed:migration.record(out,saved_execution,requested,receipt['sha256'],
        'training resume' if rng is not None else 'checkpoint model import for sampling; training RNG not consumed',atomic_json,
        rng_restored=rng is not None)
    return ck,receipt
