"""CPU typed node-closure/path distribution scores for final HARD graphs."""
import json,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('frozen_macro_descriptors',Path(__file__).with_name('objective.py'))
obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
contexts={}
def score(graph_path,ds):
    z=np.load(graph_path);ty=z['types'];schema=z['schema'];e=z['edges'];e=e if e.shape[0]==2 else e.T
    if ds not in contexts:
        f=json.loads((ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json')).read_text())
        f={int(t):{g:dict(scale=torch.tensor(b['scale'],dtype=torch.float64),omega=torch.tensor(b['omega'],dtype=torch.float64),target=torch.tensor(np.array(b['target_real'])+1j*np.array(b['target_imag']),dtype=torch.complex128)) for g,b in groups.items()} for t,groups in f.items()}
        contexts[ds]=(obj.make_plan(schema,len(np.unique(ty))),f)
    plan,f=contexts[ds];lookup={tuple(x):r for r,x in enumerate(schema)}
    P=torch.zeros((len(schema),len(ty),len(ty)),dtype=torch.float64)
    for a,b in e.T:
        r=lookup[tuple(sorted((ty[a],ty[b])))];P[r,a,b]=P[r,b,a]=1.
    with torch.no_grad():errors=obj.distribution_errors(obj.descriptors(P,ty,plan),ty,f)
    # Do not confuse feature plug-in degree score with the old exact degree CF risk.
    rows=[dict(type=x['type'],block=x['block'],error=float(x['error'])) for x in errors if x['block']!='degree']
    return dict(closure=float(np.mean([x['error'] for x in rows if x['block']=='closure'])),
        propagation=float(np.mean([x['error'] for x in rows if x['block']=='propagation'])),groups=rows,
        scope='Hard final graph typed node-distribution scores; same frozen features, not soft conditional prediction')
