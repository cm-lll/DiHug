import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/predictor_state_cross_cpu_20261110';SRC=ROOT/'results/multiobjective_equalstep_cpu_20261110'
START=time.monotonic();BIND={}
def bind(p):
 p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
def main():
 OUT.mkdir(exist_ok=False);signal.alarm(60);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 cfg=read(HERE/'protocol.json');save('protocol',cfg);bind(Path(__file__))
 for p,h in read(SRC/'bindings.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert read(SRC/'independent_audit.json')['passed']
 p=bind(ROOT/'research_branches/multiobjective_cpu_20261110/run_cpu.py')
 sp=importlib.util.spec_from_file_location('cross_finite',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 f,C,tri=m.setup();BIND.update(m.BIND)
 a=np.load(bind(SRC/'states.npz'));ix=read(SRC/'index.json');seed=cfg['seed']
 cost=np.column_stack([C,f.cost,(f.components==1).astype(float),abs(f.tri-f.tri[f.refid])])
 names=['relation_count','anonymous_joint_degree','triangle001','triangle011','components','total_cost','connected_probability','triangle_count_MAE']
 clean={};mu={};h={};errors=[];params=[]
 for step in cfg['parameter_steps']:
  n=next(i for i,x in enumerate(ix) if x==dict(seed=seed,step=step));theta=a['parameters'][n];params.append(theta)
  P,_,logclean,_=f.matrices(theta);cp=np.exp(logclean[1]);occ=f.prior.copy()
  for t in range(f.T,1,-1):occ=occ@P[t]
  errors.extend([float(np.max(abs(cp-P[1]))),float(np.max(abs(occ@cp-a['distributions'][n])))])
  pp=1/(1+np.exp(-(f.features[1]@theta)))
  independent=np.prod(np.where(f.B[None,:,:]>0,pp[:,None,:],1-pp[:,None,:]),axis=2)
  errors.append(float(np.max(abs(cp-independent))))
  clean[step]=cp;mu[step]=occ;h[step]=cp@cost
 mu['qref']=f.qref[1]
 panels=[];risk={}
 for inp in [0,20,'qref']:
  for model in [0,20]:
   dist=mu[inp]@clean[model];values=mu[inp]@h[model];orbit=np.bincount(f.orbits,weights=dist)
   risk[(inp,model)]=values
   panels.append(dict(input_source=str(inp),predictor_step=model,**dict(zip(names,values.tolist())),
      isomorph_entropy=float(-np.sum(orbit[orbit>0]*np.log(orbit[orbit>0]))),
      isomorphic_collision=float(orbit@orbit),labeled_entropy=float(-np.sum(dist[dist>0]*np.log(dist[dist>0])))))
 decomposition=[]
 for j,name in enumerate(names):
  r00=risk[(0,0)][j];r01=risk[(0,20)][j];r10=risk[(20,0)][j];r11=risk[(20,20)][j]
  pred=r01-r00;state=r10-r00;interaction=r11-r01-r10+r00
  errors.append(abs(pred+state+interaction-(r11-r00)))
  decomposition.append(dict(metric=name,total_change=float(r11-r00),predictor_at_old_states=float(pred),
     state_at_old_predictor=float(state),interaction=float(interaction),
     predictor_symmetric=float(pred+interaction/2),state_symmetric=float(state+interaction/2),
     qref_predictor_change=float(risk[('qref',20)][j]-risk[('qref',0)][j])))
 state_rows=[]
 for i in range(f.S):
  base=dict(state=i,bits=''.join(str(int(x)) for x in f.B[i]),edges=int(f.B[i].sum()),components=int(f.components[i]),
    qref=float(mu['qref'][i]),mu_old=float(mu[0][i]),mu_new=float(mu[20][i]))
  for j,name in enumerate(names):
   change=h[20][i,j]-h[0][i,j]
   state_rows.append(dict(**base,metric=name,h_old=float(h[0][i,j]),h_new=float(h[20][i,j]),change=float(change),
      weighted_qref=float(mu['qref'][i]*change),weighted_old=float(mu[0][i]*change),weighted_new=float(mu[20][i]*change)))
 groups=[]
 for field in ['edges','components']:
  for group in sorted(set(x[field] for x in state_rows)):
   for name in names:
    selected=[x for x in state_rows if x[field]==group and x['metric']==name]
    groups.append(dict(grouping=field,group=group,metric=name,states=len(selected),
      **{k:sum(x[k] for x in selected) for k in ['qref','mu_old','mu_new','weighted_qref','weighted_old','weighted_new']},
      increased_states=sum(x['change']>1e-12 for x in selected),decreased_states=sum(x['change']< -1e-12 for x in selected)))
 for j,name in enumerate(names):
  selected=[x for x in state_rows if x['metric']==name]
  errors.append(abs(sum(x['weighted_old'] for x in selected)-(risk[(0,20)][j]-risk[(0,0)][j])))
  errors.append(abs(sum(x['weighted_qref'] for x in selected)-(risk[('qref',20)][j]-risk[('qref',0)][j])))
 assert max(errors)<1e-10
 for p,hsh in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hsh,p
 save('panels',panels);save('decomposition',decomposition);save('state_details',state_rows);save('groups',groups)
 save('bindings',BIND);save('checks',dict(passed=True,count=len(errors),max_error=max(errors),original_assets_unchanged=True))
 np.savez_compressed(OUT/'distributions.npz',mu_old=mu[0],mu_new=mu[20],qref=mu['qref'],clean_old=clean[0],clean_new=clean[20],cost=cost,bits=f.B)
 save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   CPU_threads=1,GPU_hours=0,formal_model_calls=0,parameter_updates=0,new_graphs=0,panels=6))
 print(json.dumps(dict(receipt=json.loads((OUT/'receipt.json').read_text()),panels=panels,decomposition=decomposition)))
if __name__=='__main__':
 try:main()
 except BaseException:
  if OUT.exists():save('failure',dict(traceback=traceback.format_exc()))
  raise
