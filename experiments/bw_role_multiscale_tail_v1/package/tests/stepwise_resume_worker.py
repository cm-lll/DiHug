"""Fresh-process small fixture checkpoint continuation, never a BW runner."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from test_stepwise import case_guide
from stepwise_role.sampling import binding,reverse_step,load_checkpoint
torch.set_num_threads(1)
case,model,k,g,state=case_guide(False)
rng=np.random.default_rng(999999)
bound=binding(model,case,k,g,'C',20,1789)
state,s=load_checkpoint(sys.argv[1],rng,'cpu',bound,len(state),4,2)
for t in range(s,0,-1):
    state,_=reverse_step(model,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
np.savez(sys.argv[2],state=state,uniform_after=rng.random(10))
