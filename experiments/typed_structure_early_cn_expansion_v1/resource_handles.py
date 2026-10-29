"""Distinguish protected CUDA0/1 workers' passive handles from GPU compute use."""
import os,pathlib
from runtime_contract import ROOT,read,sha
def allowed_owner(args,visible,uid,devices,b):
    return (uid==os.getuid() and b['worker_path'] in args
        and '--ledger' in args and args[args.index('--ledger')+1]==str(pathlib.Path(b['run'])/'activation.json')
        and '--approval' in args and args[args.index('--approval')+1]==b['approval_path']
        and visible in b['protected_gpu_uuids'] and visible not in devices)
def filter_passive(handles,devices):
    if not handles.strip():return '',[]
    b=read(ROOT/'cross_batch_binding.json');ledger=pathlib.Path(b['run'])/'activation.json'
    if sha(ledger)!=b['activation_sha256'] or sha(b['approval_path'])!=b['approval_sha256']:raise ValueError('protected run binding changed')
    allowed=[];unresolved=[]
    for pid in sorted(set(handles.split())):
        if not pid.isdecimal():raise ValueError('unexpected fuser output')
        p=pathlib.Path('/proc')/pid
        try:
            args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            env=(p/'environ').read_bytes().split(b'\0')
            v=[x.decode().split('=',1)[1] for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')]
            ok=len(v)==1 and allowed_owner(args,v[0],p.stat().st_uid,devices,b)
        except FileNotFoundError:
            # Process exited between queries; no live handle remains.
            allowed.append(dict(pid=int(pid),reason='exited_before_inspection'));continue
        if ok:allowed.append(dict(pid=int(pid),visible_device=v[0],reason='bound protected worker passive handle; selected devices separately checked for compute and memory'))
        else:unresolved.append(pid)
    return ' '.join(unresolved),allowed
