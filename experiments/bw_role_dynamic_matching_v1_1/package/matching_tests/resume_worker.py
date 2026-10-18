import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from helpers import chain_setup
from dynamic_role.sampling import binding,reverse_step,load_checkpoint
torch.set_num_threads(1)
case,model,k,g,_=chain_setup();rng=np.random.default_rng(987)
bound=binding(model,case,k,g,'C',20,1789)
state,s=load_checkpoint(sys.argv[1],rng,'cpu',bound,case['space'].total,4,2)
for t in range(s,0,-1):state,_=reverse_step(model,case,state,k,t,20,1789,rng,'cpu',arm='C',guide=g)
np.savez(sys.argv[2],state=state,next_uniform=rng.random(10))
