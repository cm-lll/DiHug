import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/noise_batch_cpu_20261110';BASE=ROOT/'results/multiobjective_equalstep_cpu_20261110';OLD=ROOT/'results/one_step_to_terminal_cpu_20261110'
START=time.monotonic();BIND={};COUNT=0
def bind(p):
 p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
def module():
 p=bind(ROOT/'research_branches/multiobjective_cpu_20261110/run_cpu.py');s=importlib.util.spec_from_file_location('noise_finite',p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atoms(f,C,theta):
 _,_,clean,pred=f.matrices(theta);loss=[];grad=[]
 for t in range(1,f.T+1):
  cp=np.exp(clean[t]);loss.append(cp@C)
  dz=np.einsum('ij,jk,ije->kie',cp,C,f.B[None,:,:]-pred[t][:,None,:])
  grad.append(np.einsum('kie,ied->ikd',dz,f.features[t]))
 return np.concatenate(loss),np.concatenate(grad)
def main():
 global COUNT
 OUT.mkdir(exist_ok=False);signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 cfg=read(HERE/'protocol.json');save('protocol',cfg);bind(Path(__file__))
 for p,h in read(BASE/'bindings.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert read(BASE/'independent_audit.json')['passed']
 m=module();f,C,tri=m.setup();BIND.update(m.BIND)
 a=np.load(bind(OLD/'states_and_distributions.npz'));index=read(OLD/'distribution_index.json');bu=read(OLD/'updates.json')
 read(BASE/'curves.json');read(BASE/'updates.json')
 prob=np.concatenate([f.qref[t]/f.T for t in range(1,f.T+1)]);assert abs(prob.sum()-1)<1e-12
 banks={};initials={};lengths={};gates=[]
 for seed in cfg['initializations']:
  n=next(i for i,x in enumerate(index) if x['seed']==seed and x['step']==0 and x['arm']=='exact_forward_clean_structure');initials[seed]=a['parameters'][n].copy()
  lengths[seed]={x['step']:x['displacement_norm'] for x in bu if x['seed']==seed and x['arm']=='exact_forward_clean_structure'}
  al,ag=atoms(f,C,initials[seed]);loss,G=m.local(f,C,initials[seed])
  err=max(float(np.max(abs(prob@al-loss))),float(np.max(abs(np.einsum('i,ikd->kd',prob,ag)-G))))
  assert err<1e-10;gates.append(dict(seed=seed,atom_exact_error=err))
  for stream in cfg['noise_streams']:
   rng=np.random.default_rng(np.random.SeedSequence([seed,stream]));banks[f'{seed}_{stream}']=rng.choice(len(prob),size=(20,32),p=prob)
 np.savez_compressed(OUT/'noise_banks.npz',**banks,atom_probability=prob)
 save('gate',dict(passed=True,checks=gates));save('noise_identity',dict(atom_encoding='atom=(t-1)*64+state',bank_sha256=hashlib.sha256((OUT/'noise_banks.npz').read_bytes()).hexdigest()))
 names=['relation_count','anonymous_joint_degree','triangle001','triangle011','components']
 updates=[];curves=[];tasks=[];saved=[];params=[];distributions=[];inputs=[]
 for seed in cfg['initializations']:
  for stream in cfg['noise_streams']:
   bank=banks[f'{seed}_{stream}']
   for K in cfg['batch_sizes']:
    theta=initials[seed].copy();meta=dict(seed=seed,noise_stream=stream,K=K);status='complete';reason='';completed=0;exposure=0;tic=time.monotonic()
    for step in range(21):
     al,ag=atoms(f,C,theta);true_loss=prob@al;trueG=np.einsum('i,ikd->kd',prob,ag)
     if step in [0,5,10,15,20]:
      ev=f.evaluate(theta);dist=ev['distribution'];orbit=np.bincount(f.orbits,weights=dist);metric=dict(ev['metrics']);metric['generated_state_clean_risk']=metric.pop('single_step_clean_risk')
      metric.update(isomorph_entropy=float(-np.sum(orbit[orbit>0]*np.log(orbit[orbit>0]))),forward_clean_risk=float((true_loss[0]+true_loss[1]+(true_loss[2]+true_loss[3])/2+true_loss[4])/4))
      for j,name in enumerate(names):metric['forward_'+name]=float(true_loss[j]);metric['terminal_'+name]=float(dist@C[:,j])
      curves.append(dict(**meta,step=step,**metric));saved.append(dict(**meta,step=step));params.append(theta.copy());distributions.append(dist)
     if step==20:break
     ids=bank[step,:K];G=ag[ids].mean(0);emp_loss=al[ids].mean(0);exposure+=K
     inputs.extend(dict(**meta,step=step+1,draw=j,t=int(atom//f.S+1),state=int(atom%f.S)) for j,atom in enumerate(ids))
     try:v,lam,res=m.minnorm(G)
     except Exception as ex:status='stopped_solver';reason=repr(ex);break
     norm=float(np.linalg.norm(v))
     if not np.isfinite(norm):status='stopped_nonfinite';reason='nonfinite direction';break
     if norm<=1e-12:status='stopped_zero_direction';reason='cannot match prescribed nonzero baseline length';break
     delta=-lengths[seed][step+1]*v/norm;nxt=theta+delta;actual=nxt-theta
     assert abs(np.linalg.norm(actual)-lengths[seed][step+1])<1e-12 and np.isfinite(nxt).all()
     nextal,_=atoms(f,C,nxt);after=prob@nextal
     updates.append(dict(**meta,step=step+1,coefficients=lam.tolist(),gradient_error=float(np.linalg.norm(G-trueG)),
       exact_gradient_norm=float(np.linalg.norm(trueG)),empirical_gradient_norm=float(np.linalg.norm(G)),
       KKT=res,displacement=float(np.linalg.norm(actual)),true_changes=(after-true_loss).tolist(),
       empirical_changes=(nextal[ids].mean(0)-emp_loss).tolist(),true_all_decreased=bool(np.all(after<true_loss)),
       **{'change_'+name:float(after[j]-true_loss[j]) for j,name in enumerate(names)}))
     theta=nxt;completed+=1;COUNT+=1
    if status!='complete':
     # Keep exact stop-boundary parameter even if it is not an evaluation grid point.
     np.save(OUT/f'stopped_{seed}_{stream}_{K}.npy',theta)
    tasks.append(dict(**meta,status=status,reason=reason,completed=completed,noise_inputs=exposure,seconds=time.monotonic()-tic))
    save('tasks',tasks);save('updates',updates);save('curves',curves)
 save('inputs',inputs);save('state_index',saved);np.savez_compressed(OUT/'states.npz',parameters=np.array(params),distributions=np.array(distributions))
 for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 save('bindings',BIND)
 save('receipt',dict(status='complete_stopped',tasks=len(tasks),complete_trajectories=sum(x['status']=='complete' for x in tasks),
    seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,CPU_threads=1,GPU_hours=0,
    formal_model_calls=0,toy_updates=COUNT,forward_noise_inputs_used=len(inputs),new_model_generated_graphs=0))
 print(json.dumps(dict(receipt=read(OUT/'receipt.json'),tasks=tasks)))
if __name__=='__main__':
 try:main()
 except BaseException:
  if OUT.exists():save('failure',dict(traceback=traceback.format_exc(),updates=COUNT))
  raise
