"""Immutable payload + durable intent, then idempotent pointer publication."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import torch
from hsgd.bw_preflight import common as c

def content_hash(obj):
    h=hashlib.sha256()
    def atom(x):
        def tag(s):
            b=s.encode();h.update(struct.pack('!Q',len(b)));h.update(b)
        if torch.is_tensor(x):
            tag('tensor:'+str(x.dtype)+':'+str(tuple(x.shape)))
            y=x.detach().cpu().contiguous()
            if y.is_floating_point() and not torch.isfinite(y).all():raise ValueError('nonfinite checkpoint tensor')
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            tag('ndarray:'+x.dtype.str+':'+str(x.shape));h.update(x.tobytes(order='C'))
        elif isinstance(x,dict):
            tag('dict:'+str(len(x)))
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):atom(k);atom(x[k])
        elif isinstance(x,(tuple,list)):
            tag(type(x).__name__+':'+str(len(x)))
            for v in x:atom(v)
        elif x is None:tag('None')
        elif isinstance(x,(bool,int,np.integer,str)):tag(type(x).__name__+':'+str(x))
        elif isinstance(x,(float,np.floating)):
            if not np.isfinite(x):raise ValueError('nonfinite checkpoint scalar')
            tag('float');h.update(struct.pack('!d',float(x)))
        else:raise TypeError('unsupported checkpoint content: '+str(type(x)))
    atom(obj);return h.hexdigest()

def payload(path,bind,u,sha,semantic):
    if c.sha(path)!=sha:raise ValueError('checkpoint byte hash mismatch')
    ck=c.trusted_load(path)
    required={'purpose','binding','completed_updates','model','optimizer','rng','extra'}
    if set(ck)!=required or ck['binding']!=bind or ck['completed_updates']!=u:
        raise ValueError('checkpoint binding/update/content mismatch')
    purpose='engineering fixture' if bind.get('engineering_fixture') else 'fresh formal joint CN training'
    if ck['purpose']!=purpose or not isinstance(ck['model'],dict) or not isinstance(ck['optimizer'],dict):
        raise ValueError('invalid checkpoint purpose/model/optimizer')
    if content_hash(ck)!=semantic:raise ValueError('complete checkpoint content hash mismatch')
    return ck

def local_path(out,relative):
    path=(out/relative).resolve()
    if path.parent!= (out/'checkpoints').resolve():raise ValueError('checkpoint path escapes its directory')
    return path

def publish(out,intent,write=c.json_save):
    receipt=intent['receipt'];u=receipt['completed_updates'];bind=intent['binding']
    path=local_path(out,receipt['path'])
    if not path.exists():
        staged=out/'transactions'/intent['staged_file']
        if staged.parent.resolve()!=(out/'transactions').resolve():raise ValueError('unsafe staged path')
        payload(staged,bind,u,receipt['sha256'],intent['content_sha256'])
        staged.replace(path)
        for parent in (path.parent,staged.parent):
            fd=os.open(str(parent),os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
    ck=payload(path,bind,u,receipt['sha256'],intent['content_sha256'])
    latest=out/'latest.json';ledger_path=out/'milestones.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if intent['kind']=='milestone' and str(u) in ledger and ledger[str(u)]!=receipt:
        raise ValueError('conflicting milestone receipt')
    old=json.loads(latest.read_text()) if latest.exists() else None
    if old:
        old_path=local_path(out,old['path'])
        if c.sha(old_path)!=old['sha256']:raise ValueError('conflicting latest checkpoint hash')
        old_ck=c.trusted_load(old_path)
        if old_ck['binding']!=bind or old_ck['completed_updates']!=old['completed_updates']:
            raise ValueError('conflicting latest binding/update')
    if old!=receipt and (old is None or old['completed_updates']<u or (old['completed_updates']==u and receipt['order_ns']>=old['order_ns'])):
        write(latest,receipt)
    if intent['kind']=='milestone' and str(u) not in ledger:
        ledger[str(u)]=receipt;write(ledger_path,ledger)
    return ck,receipt

def reconcile(out,bind,write=c.json_save):
    """Validate ALL intents and published pointers before making any repair."""
    out=Path(out);folder=out/'transactions'
    intents=[]
    for path in sorted(folder.glob('*.json')):
        item=json.loads(path.read_text());r=item['receipt']
        if item['binding']!=bind or item['kind'] not in ('milestone','stop'):raise ValueError('transaction source mismatch')
        target=local_path(out,r['path']);staged=folder/item['staged_file']
        u=r['completed_updates']
        if type(u) is not int or u<0:raise ValueError('invalid update counter')
        if ((item['kind']=='milestone' and target.name!='u%06d.pt'%u) or
            (item['kind']=='stop' and not target.name.startswith('stop_u%06d_'%u)) or path.name!=target.name+'.json'):
            raise ValueError('checkpoint filename/update/kind mismatch')
        if staged.parent.resolve()!=folder.resolve():raise ValueError('transaction staged path escape')
        payload(target if target.exists() else staged,bind,r['completed_updates'],r['sha256'],item['content_sha256'])
        intents.append(item)
    known={x['receipt']['path']:x for x in intents}
    for path in (out/'checkpoints').glob('*.pt'):
        if str(path.relative_to(out)) not in known:raise ValueError('unattested checkpoint file: '+path.name)
    for name in ('latest.json','milestones.json'):
        if not (out/name).exists():continue
        value=json.loads((out/name).read_text());rows=[value] if name=='latest.json' else list(value.values())
        for r in rows:
            item=known.get(r['path'])
            if item is None or r!=item['receipt']:raise ValueError('pointer/ledger conflicts with transaction')
        if name=='milestones.json':
            for k,r in value.items():
                if k!=str(r['completed_updates']) or known[r['path']]['kind']!='milestone':raise ValueError('milestone index mismatch')
    for item in sorted(intents,key=lambda x:(x['receipt']['completed_updates'],x['receipt']['order_ns'])):
        publish(out,item,write)
    return intents

def save(out,ck,u,bind,kind,write=c.json_save):
    out=Path(out);folder=out/'checkpoints';folder.mkdir(exist_ok=True)
    txn=out/'transactions';txn.mkdir(exist_ok=True)
    name=('u%06d.pt'%u) if kind=='milestone' else ('stop_u%06d_%d.pt'%(u,time.time_ns()))
    target=folder/name;journal=txn/(name+'.json');semantic=content_hash(ck)
    # Idempotence compares every saved value, not only model weights or update number.
    if journal.exists():
        intent=json.loads(journal.read_text())
        if intent['binding']!=bind or intent['content_sha256']!=semantic or intent['kind']!=kind:
            raise ValueError('existing immutable checkpoint content conflicts')
        reconcile(out,bind,write);return intent['receipt']
    if target.exists():
        # No intent: adoption is allowed ONLY against the caller's COMPLETE known snapshot,
        # never merely by self-hashing an unexplained file during load/recovery.
        sha=c.sha(target);payload(target,bind,u,sha,semantic)
        surviving=[]
        if (out/'latest.json').exists():surviving.append(json.loads((out/'latest.json').read_text()))
        if (out/'milestones.json').exists():surviving+=list(json.loads((out/'milestones.json').read_text()).values())
        matching=[r for r in surviving if r['path']==str(target.relative_to(out))]
        if any(r['sha256']!=sha or r['completed_updates']!=u for r in matching):raise ValueError('existing receipt conflicts')
        receipt=matching[0] if matching else dict(path=str(target.relative_to(out)),sha256=sha,
            completed_updates=u,order_ns=time.time_ns(),save_seconds=0.)
        if 'order_ns' not in receipt:raise ValueError('old receipt format is not a certified source upgrade')
        intent=dict(binding=bind,kind=kind,receipt=receipt,staged_file=name+'.adopted.prepared',
                    content_sha256=semantic,adoption='complete known in-memory snapshot matched')
        write(journal,intent);reconcile(out,bind,write);return receipt
    reconcile(out,bind,write)
    start=time.monotonic();staged=txn/(name+'.%d.prepared'%time.time_ns())
    c.torch_save(staged,ck)
    with staged.open('rb') as f:os.fsync(f.fileno())
    receipt=dict(path=str(target.relative_to(out)),sha256=c.sha(staged),completed_updates=u,
                 order_ns=time.time_ns(),save_seconds=time.monotonic()-start)
    intent=dict(binding=bind,kind=kind,receipt=receipt,staged_file=staged.name,content_sha256=semantic)
    write(journal,intent) # Durable expected byte/content hashes BEFORE checkpoint publication.
    publish(out,intent,write)
    return receipt
