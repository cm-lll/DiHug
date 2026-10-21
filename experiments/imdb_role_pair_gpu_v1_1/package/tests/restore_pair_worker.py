"""Fresh-process checkpoint exercise with tiny disposable paired modules."""
import random,sys
from pathlib import Path
import numpy as np
import torch
import pair_core as p

def fresh():
    models={arm:torch.nn.Linear(3,2) for arm in ('base','role')}
    models['role'].load_state_dict(models['base'].state_dict())
    opts={arm:torch.optim.AdamW(m.parameters(),lr=2e-4) for arm,m in models.items()}
    return models,opts

def step(models,opts,rng,u):
    inp=p.input_pair(p.toy(),p.config(),7,u)
    x=torch.randn(2,3)+float(rng.random())+random.random()+float(np.random.random())
    y=torch.tensor([[1.,0.],[0.,1.]])
    for arm,m in models.items():
        opts[arm].zero_grad(set_to_none=True)
        (m(x)-y).square().sum().backward();opts[arm].step()
    return p.digest([x]+[[v[k] for k in ('edges','query','f','labels','t')] for v in inp])

def pack(models,opts,rng,u):
    return {'binding':{'fixture':'two-arm fresh-process'},'updates':u,'formal_native':True,
        'models':{a:p.cpu(m.state_dict()) for a,m in models.items()},
        'optimizers':{a:p.cpu(o.state_dict()) for a,o in opts.items()},'rng':p.rng_save(rng,'cpu')}

def main():
    p.setup('cpu',True);folder=Path(sys.argv[1]);dest=Path(sys.argv[2]);models,opts=fresh();rng=np.random.default_rng(999)
    ck=p.load_checkpoint(folder,{'fixture':'two-arm fresh-process'})
    for arm in models:models[arm].load_state_dict(ck['models'][arm]);opts[arm].load_state_dict(ck['optimizers'][arm])
    p.rng_restore(ck['rng'],rng,'cpu')
    restored=p.digest(pack(models,opts,rng,ck['updates']))
    inp=step(models,opts,rng,ck['updates']);result=pack(models,opts,rng,ck['updates']+1)
    torch.save({'restored':restored,'input':inp,'after':result},dest)

if __name__=='__main__':main()
