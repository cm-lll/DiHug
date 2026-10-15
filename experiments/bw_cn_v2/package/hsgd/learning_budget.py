"""stdlib budget shared by gate, imports, training, validation and saving."""
import datetime
import hashlib
import json
from pathlib import Path
import time


def atomic_json(path,value):
    p=Path(path);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)


def utc(epoch):
    return datetime.datetime.fromtimestamp(epoch,datetime.timezone.utc).isoformat()


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def model_fingerprint(state):
    """Exact weights/buffers binding; called only at validation/saving, not steps."""
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(",",":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def open_budget(out,resume=False,fixture=False,budget_seconds=28800,now=None):
    now=time.time() if now is None else now;out=Path(out)
    if resume:
        b=json.loads((out/"budget.json").read_text())
        if b["engineering_fixture"]!=fixture or b["budget_seconds"]!=budget_seconds:
            raise RuntimeError("budget/mode cannot change on resume")
        if b["hard_cutoff_epoch"]-b["activation_epoch"] != budget_seconds:
            raise RuntimeError("budget cutoff inconsistent")
        if now>=b["hard_cutoff_epoch"]:raise TimeoutError("original budget already expired; no new window")
    else:
        if out.exists():raise RuntimeError("fresh independent persistent directory required")
        out.mkdir(parents=True)
        b=dict(activation_epoch=now,hard_cutoff_epoch=now+budget_seconds,budget_seconds=budget_seconds,
               activation_utc=utc(now),hard_cutoff_utc=utc(now+budget_seconds),
               engineering_fixture=fixture,scope="one wall window including preflight/imports/validation/IO/downtime",
               formal_max_updates=None,no_automatic_generation=True)
        tz=datetime.timezone(datetime.timedelta(hours=8))
        b["activation_beijing"]=datetime.datetime.fromtimestamp(now,tz).isoformat()
        b["hard_cutoff_beijing"]=datetime.datetime.fromtimestamp(now+budget_seconds,tz).isoformat()
        atomic_json(out/"budget.json",b)
    return b


def left(b):return b["hard_cutoff_epoch"]-time.time()


def used(b):return max(0.,time.time()-b["activation_epoch"])
