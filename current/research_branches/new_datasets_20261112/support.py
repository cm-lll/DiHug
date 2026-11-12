"""Isolated dataset adapter; production model, CE and reverse kernel unchanged."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,importlib.util,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module('newdata_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
w,np,torch,base=s.w,s.np,s.torch,s.base
heat=module('newdata_heat',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py').heat
single=module('newdata_single',ROOT/'research_branches/cn_multiseed_20261104/experiments/late_final_single_input_v1/single_input.py')
PREP=ROOT/'data/external/new_hetero_20261112/prepared'
def read(p):return json.loads(Path(p).read_text())
def case_for(ds):
 z=np.load(PREP/ds/'small_192.npz');return w.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def config(ds,seed):
 cfg=read(ROOT/'results/cn_multiseed_20261104T042546/run/IMDB_2026110411_A_late/binding.json')['config']
 cfg.update(dataset=ds,seed=seed,steps=3000,inputs_per_update=1,rounds=1,T=1000,readout='EMA0.99')
 return cfg
class Score(base.Score):
 """Same frozen score rule, reference-derived scaling for arbitrary type count."""
 def __init__(self,ds,case,query,fam,labels):
  self.types=case['space'].types;self.n=len(self.types);self.r=case['n_families'];self.q=query;self.f=fam;self.ref=labels
  h=heat(base.heat.adj(self.n,query[:,labels.astype(bool)].T),self.types)['feat']
  d=base.joint.degrees(labels,query,fam,self.n,self.r).astype(float)
  self.s={'log_degree_scales':{},'heat_bandwidth_squared':{},'degree_bandwidth_squared':{}}
  for t in np.unique(self.types):
   mask=self.types==t;scale=np.log1p(np.maximum(1,d[mask].mean(0)));self.s['log_degree_scales'][str(t)]=scale.tolist()
   dn=np.log1p(d[mask])/scale/np.sqrt(self.r)
   self.s['heat_bandwidth_squared'][str(t)]=base.joint.median_bandwidth(h[mask]);self.s['degree_bandwidth_squared'][str(t)]=base.joint.median_bandwidth(dn)
  self.h,self.d=self.features(labels);self.target=np.bincount(fam,weights=labels,minlength=self.r)
 def features(self,b):
  h=heat(base.heat.adj(self.n,self.q[:,b.astype(bool)].T),self.types)['feat'];d=base.joint.degrees(b,self.q,self.f,self.n,self.r).astype(float)
  for t in np.unique(self.types):
   mask=self.types==t;d[mask]=np.log1p(d[mask])/np.asarray(self.s['log_degree_scales'][str(t)])/np.sqrt(self.r)
  return h,d
