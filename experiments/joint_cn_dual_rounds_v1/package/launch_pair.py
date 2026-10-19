"""Explicit TWO allocated visible devices, one sequential pipeline per device.

Does not change CUDA/HIP/ROCR visibility or reach outside the container.
Run in foreground in a persistent notebook terminal, or nohup this command.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

def validate(imdb,pubmed,root):
    if imdb==pubmed:raise ValueError('two datasets must use distinct visible device indices')
    if min(imdb,pubmed)<0:raise ValueError('negative device index')
    from dual.io import protect
    root=protect(root)
    return [(name,index,root/(name+'_seed'+str(seed))) for name,index,seed in
            [('PubMed',pubmed,2026101902),('IMDB',imdb,2026101901)]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--imdb-device',type=int,required=True)
    p.add_argument('--pubmed-device',type=int,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--accept-config-provenance-gap',action='store_true')
    a=p.parse_args()
    if not a.accept_config_provenance_gap:raise ValueError('read configuration provenance before launch; acknowledgement required')
    jobs=validate(a.imdb_device,a.pubmed_device,a.root)
    import torch
    from dual.io import verify_release,atomic_json
    verify_release()
    if not torch.cuda.is_available() or max(a.imdb_device,a.pubmed_device)>=torch.cuda.device_count():raise ValueError('requested device outside visible allocation')
    identities=[]
    for name,index,out in jobs:
        props=torch.cuda.get_device_properties(index)
        identities.append(dict(dataset=name,visible_index=index,name=props.name,
            uuid=str(getattr(props,'uuid','unavailable')),memory_bytes=props.total_memory,output=str(out)))
        if not a.resume and (out.exists() or out.with_name(out.name+'.gate').exists()):raise ValueError('fresh output/gate exists')
    uuids=[x['uuid'] for x in identities]
    if uuids[0]!='unavailable' and uuids[0]==uuids[1]:raise ValueError('both indices resolve to same device UUID')
    print(json.dumps(dict(devices=identities,visibility={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')})),flush=True)
    a.root.mkdir(parents=True,exist_ok=True)
    atomic_json(a.root/'pair_devices.json',identities)
    children=[];stopping=[];handlers={}
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):handlers[sig]=signal.signal(sig,lambda s,f:stopping.append(s))
    try:
        for name,index,out in jobs:
            cmd=[sys.executable,'-u','-m','dual.cli','pipeline','--dataset',name,'--output',str(out),'--device-index',str(index),
                 '--accept-config-provenance-gap','--resume' if a.resume else '--fresh-start']
            log=(a.root/(name+'.log')).open('a')
            proc=subprocess.Popen(cmd,cwd=Path(__file__).parent,stdout=log,stderr=subprocess.STDOUT)
            children.append((proc,out,log))
        while any(p.poll() is None for p,_,_ in children):
            if stopping:
                for proc,out,_ in children:
                    # Only these jobs. Gate gets catchable termination; formal worker gets safe STOP.
                    if proc.poll() is None:
                        if out.exists():(out/'STOP').touch()
                        else:proc.send_signal(signal.SIGTERM)
            time.sleep(1)
        return 0 if all(p.returncode==0 for p,_,_ in children) else 2
    finally:
        for proc,out,log in children:
            if proc.poll() is None:
                if out.exists():(out/'STOP').touch()
                proc.send_signal(signal.SIGTERM)
                proc.wait()  # Wait for atomic safe save; no formal deadline silently added.
            log.close()
        for s,h in handlers.items():signal.signal(s,h)

if __name__=='__main__':raise SystemExit(main())
