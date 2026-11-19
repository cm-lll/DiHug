"""Generic inventory evaluation with historical 4-type equivalence checks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,importlib.util,itertools
from pathlib import Path
import numpy as np
import metrics as M
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);start=time.time();F=M.frozen(OLD/'vendor/frozen_metrics.py',OLD/'vendor/aligned_evaluate.py',OLD/'vendor/hgen_utils.py')
 def check():
  if time.time()-start>180:raise TimeoutError('CPU check cap')
 spec=importlib.util.spec_from_file_location('oldmetrics',OLD/'metrics.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 z=np.load(ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets/IMDB.npz');ee=z['edges'].T if z['edges'].shape[1]==2 else z['edges']
 g=M.make_graph(z['types'],ee,z['schema']);names=['Director','Actor','Movie','Genre'];a=old.stats(g,F,names,check);b=M.stats(g,F,names,check);assert a==b
 ca=old.compare(g,a,g,a,F,names);cb=M.compare(g,b,g,b,F,names)
 for x,y in [('main_incident4_W1','main_incident_W1'),('all10_compat_W1','all_schema_W1'),('incident4_actual_typeweights','incident_actual_typeweights')]:ca[y]=ca.pop(x)
 assert ca==cb
 records=[]
 for ds in ('DBLP_HGEN','ACM_HAN_full7'):
  folder=ROOT/'data/external/new_hetero_20261112/prepared'/ds;z=np.load(folder/'small_192.npz');names=read(folder/'audit.json')['type_names'];g=M.make_graph(z['types'],z['edges'].T,z['schema']);st=M.stats(g,F,names,check);selfcmp=M.compare(g,st,g,st,F,names)
  assert selfcmp['main_incident_W1']==0 and selfcmp['main_joint_degree_TV']==0
  # Arbitrary relabeling with types moved too must preserve all anonymous statistics.
  perm=np.random.default_rng(2026111203).permutation(len(z['types']));inverse=np.argsort(perm);e=np.sort(inverse[z['edges']],axis=1);e=e[np.lexsort((e[:,1],e[:,0]))]
  gp=M.make_graph(z['types'][perm],e.T,z['schema']);ps=M.stats(gp,F,names,check)
  for key in ('edges','triangles','typed_triangles','mp','typed_walks','joint_relation_degree','four_cycles','typed_four_cycles','typed_twohop_closure'):assert ps[key]==st[key],key
  # LCC missing the highest type must not index beyond bincount or misreport support.
  empty=M.make_graph(z['types'],np.empty((2,0),dtype=int),z['schema']);lc,_=M.lcc(empty);ls=M.stats(lc,F,names,check);cmp=M.compare(lc,ls,g,st,F,names);assert cmp['missing_types'] and cmp['main_incident_W1'] is None
  save(out/(ds+'_reference.json'),st);records.append(dict(dataset=ds,nodes=st['nodes'],edges=st['edges'],triangles=st['triangles'],four_cycles=st['four_cycles'],MP2=st['mp'][1]['total'],MP3=st['mp'][2]['total'],node_permutation_pass=True,missing_type_LCC_pass=True))
 save(out/'receipt.json',dict(status='passed',historical_IMDB_stats_and_comparison_exact=True,records=records,seconds=time.time()-start,model_calls=0,GPU=0))
if __name__=='__main__':main()
