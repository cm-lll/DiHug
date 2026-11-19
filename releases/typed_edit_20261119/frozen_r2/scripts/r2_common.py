"""Original-method controls. Original dihug source is imported only from this lab."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib, time, random, copy
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dihug.core.switch_diffusion import SwitchDenoiserMLP, SwitchDiffusionDataset
from dihug.core.mixed_edit_diffusion import SCORER_ACTIONS, mixed_edit_feature_width
OLD = Path(__file__).resolve().parents[4]
HISTORY = OLD/'results/paper_revision_pr3b/pr3b_20261008_stage2_attribution_v1'
ASSETS = {
 'imdb': {'directory': 'results/imdb_exchangeable_compact_s0_3/model', 'families':5, 'types':4, 'epochs':100},
 'pubmed': {'directory': 'results/pubmed_role_corruption_gate_0716/f100_all/model', 'families':10, 'types':4, 'epochs':20},
}
ARMS = {'S': (64,1), 'D': (256,3), 'M': (256,3)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(q,p)
def save_torch(p,obj):
 p=Path(p);q=p.with_name(p.name+'.tmp');torch.save(obj,q);os.replace(q,p)
def schema(ds):
 a=ASSETS[ds];f=a['families'];t=a['types'];start=len(SCORER_ACTIONS)+f
 width=mixed_edit_feature_width('typed_closure_compact_v2',f,t)
 drop=list(range(start+16,start+20))+list(range(start+24,width))
 return dict(width=width,zero_columns=drop,keep_columns=[i for i in range(width) if i not in drop],
   action_columns=list(range(len(SCORER_ACTIONS))),family_columns=list(range(len(SCORER_ACTIONS),start)),
   endpoint_degree_columns=list(range(start,start+16)),common_neighbor_columns=list(range(start+16,start+20)),
   scalar_columns=list(range(start+20,start+24)),closure_columns=list(range(start+24,width)))
def load_data(ds,arm):
 a=ASSETS[ds];p=OLD/a['directory'];z=np.load(p/'mixed_edit_dataset.npz',allow_pickle=False)
 assert str(z['feature_profile'].item())=='typed_closure_compact_v2' and str(z['kernel'].item())=='bdrs'
 x=z['candidate_features'].copy();y=z['targets'].copy();ids=z['trajectory_ids'].copy();u=z['candidate_improvements'].copy()
 assert x.shape[1:]==(16,schema(ds)['width']) and np.isfinite(x).all()
 assert y.shape==ids.shape==(len(x),) and ((0<=y)&(y<16)).all()
 if arm in ('S','D'): x[...,schema(ds)['zero_columns']]=0
 return x,y,ids,u
def model(ds,arm,seed,device='cpu'):
 torch.manual_seed(seed); h,l=ARMS[arm];m=SwitchDenoiserMLP(schema(ds)['width'],h,l,.1).to(device)
 if arm in ('S','D'):
  with torch.no_grad():m.network[0].weight[:,schema(ds)['zero_columns']]=0
 return m
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
def update(m,o,x,y):
 m.train();o.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x),y)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);o.step()
 if not all(torch.isfinite(p).all() for p in m.parameters()):raise FloatingPointError('nonfinite parameter')
 return float(loss.detach()),float(norm)
def rng_state():
 return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy=np.random.get_state(),python=random.getstate())
def set_rng(s):
 torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python'])
 if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
def eval_model(m,x,y,device):
 m.eval();logits=[]
 with torch.no_grad():
  for k in range(0,len(x),64):logits.append(m(torch.from_numpy(x[k:k+64]).float().to(device)).cpu())
 q=torch.cat(logits);labels=torch.from_numpy(y).long();p=q.softmax(-1)
 return {'ce':float(torch.nn.functional.cross_entropy(q,labels)),'top1':float((q.argmax(1)==labels).float().mean()),'uniform_ce':float(np.log(q.shape[1]))},q.numpy(),p.numpy()
def verify_manifest(run):
 m=json.loads((Path(run)/'manifest.json').read_text())
 for p,h in m['files'].items():
  if sha(p)!=h:raise ValueError('source/asset changed '+p)
 return m
