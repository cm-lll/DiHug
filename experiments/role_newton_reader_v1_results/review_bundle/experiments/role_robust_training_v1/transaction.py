"""STOP at complete paired-update boundaries; hard deadlines remain hard."""
from contextlib import contextmanager
from pathlib import Path
import time
from core import atomic_weights,c,fingerprint,save

class UpdateBoundary:
    def __init__(self,output,deadline,reserve=30):
        self.output=Path(output);self.deadline=deadline;self.reserve=reserve
        self.in_pair=False;self.requested=None
    def request(self,reason):self.requested=reason
    def check(self):
        if time.time()>self.deadline-self.reserve:raise TimeoutError("hard deadline exit reserve")
        if (self.output/"STOP").exists():self.requested="STOP"
        if self.requested and not self.in_pair:raise InterruptedError(self.requested)
    @contextmanager
    def pair(self):
        self.check();self.in_pair=True
        try:yield
        finally:self.in_pair=False
        self.check()

def save_pair(path,models,optimizers,rng,seed,completed,binding,budget):
    if len(set(completed.values()))!=1:raise ValueError("milestone requires complete paired update")
    for arm,m in models.items():
        parameter=m.backbone.lin_out_E.weight
        step=int(optimizers[arm].state.get(parameter,{}).get("step",0))
        if step!=completed[arm]:raise ValueError("optimizer/update counter mismatch")
    if path.exists():raise FileExistsError("checkpoint must not overwrite existing evidence")
    started=time.monotonic()
    device=next(next(iter(models.values())).parameters()).device.type
    record=dict(seed=seed,completed=dict(completed),models={a:c.cpu_tree(m.state_dict()) for a,m in models.items()},
        optimizers={a:c.cpu_tree(o.state_dict()) for a,o in optimizers.items()},rng=c.rng_save(rng,device),
        binding=binding,budget=budget,purpose="paired scientific milestone; no optimizer restart")
    h=atomic_weights(path,record)
    return dict(path=path.name,sha256=h,seed=seed,completed=dict(completed),bytes=path.stat().st_size,
                save_seconds=time.monotonic()-started)

def restore_pair(record,models,optimizers,rng,binding):
    if record["binding"]!=binding:raise ValueError("checkpoint binding mismatch")
    if len(set(record["completed"].values()))!=1:raise ValueError("not a complete paired milestone")
    for arm,m in models.items():
        m.load_state_dict(record["models"][arm],strict=True)
        optimizers[arm].load_state_dict(record["optimizers"][arm])
        assert int(optimizers[arm].state[m.backbone.lin_out_E.weight]["step"])==record["completed"][arm]
    device=next(next(iter(models.values())).parameters()).device.type
    c.rng_restore(record["rng"],rng,device)
