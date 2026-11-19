import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, importlib.util, itertools, json, resource, signal, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/multiobjective_cpu_20261110'; OLD=ROOT/'results/one_step_to_terminal_cpu_20261110'
START=time.monotonic(); BIND={}; COUNT=0
def bind(p):
 p=Path(p); BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def module(name,p):
 s=importlib.util.spec_from_file_location(name,bind(p)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r))); w.writeheader(); w.writerows(x)

def minnorm(G):
 H=G@G.T; scale=max(float(np.max(abs(H))),1e-30); Hs=H/scale; n=len(G); candidates=[]
 for size in range(1,n+1):
  for ix0 in itertools.combinations(range(n),size):
   ix=list(ix0); h=Hs[np.ix_(ix,ix)]
   A=np.block([[h,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
   rhs=np.r_[np.zeros(size),1.]; sol=np.linalg.lstsq(A,rhs,rcond=1e-12)[0]
   if np.max(abs(A@sol-rhs))>1e-8 or np.min(sol[:-1]) < -1e-10: continue
   lam=np.zeros(n); lam[ix]=np.maximum(sol[:-1],0); lam/=lam.sum()
   candidates.append((float(lam@Hs@lam),lam))
 assert candidates,'No feasible simplex face'
 _,lam=min(candidates,key=lambda x:x[0]); v=G.T@lam; hv=Hs@lam; val=float(lam@hv)
 residual=max(float(max(0,-np.min(hv-val))),float(np.max(abs((hv-val)[lam>1e-9]))),abs(lam.sum()-1),float(max(0,-lam.min())))
 assert residual<=1e-8,('KKT failure',residual)
 return v,lam,residual

def setup():
 m=module('multiobjective_finite',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py')
 m.kernel_module=module('multiobjective_kernel',m.KERNEL)
 f=m.FiniteDiffusion(json.loads(bind(m.PROTOCOL).read_text()))
 tri=np.zeros((f.S,2))
 for s,A in enumerate(f.adj):
  for nodes in itertools.combinations(range(4),3):
   if all(A[i,j] for i,j in itertools.combinations(nodes,2)): tri[s,int(f.types[list(nodes)].sum())-1]+=1
 tc=(tri-tri[f.refid])**2
 assert np.max(abs(tc.mean(1)-f.blocks[:,2]))<1e-14
 C=np.column_stack([f.blocks[:,:2],tc,f.blocks[:,3]])
 return f,C,tri

def local(f,C,theta):
 _,_,clean,pred=f.matrices(theta); loss=np.zeros(5); grad=np.zeros((5,f.D))
 for t in range(1,f.T+1):
  joint=f.qref[t][:,None]*np.exp(clean[t])/f.T
  loss+=np.einsum('ij,jk->k',joint,C)
  dz=np.einsum('ij,jk,ije->kie',joint,C,f.B[None,:,:]-pred[t][:,None,:])
  grad+=np.einsum('kie,ied->kd',dz,f.features[t])
 return loss,grad

def main():
 global COUNT
 OUT.mkdir(exist_ok=False); signal.alarm(180); resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 cfg=json.loads(bind(HERE/'protocol.json').read_text()); save('protocol',cfg); bind(Path(__file__))
 f,C,tri=setup(); a=np.load(bind(OLD/'states_and_distributions.npz'))
 ix=json.loads(bind(OLD/'distribution_index.json').read_text()); bind(OLD/'curves.json')
 names=cfg['objectives']; checks=[]
 for G,expected in [(np.eye(3),np.ones(3)/3),(np.array([[1.,0],[1,0]]),np.array([1.,0])),(np.array([[1.,0],[-1,0]]),np.zeros(2)),(np.zeros((3,2)),np.zeros(2))]:
  v,lam,res=minnorm(G); err=float(np.max(abs(v-expected))); assert err<1e-8
  checks.append(dict(test='solver_synthetic',error=err,KKT=res))
 initials={}
 for seed in cfg['seeds']:
  n=next(i for i,r in enumerate(ix) if r['seed']==seed and r['step']==0 and r['arm']=='exact_forward_clean_structure')
  theta=a['parameters'][n].copy(); initials[seed]=theta
  loss,G=local(f,C,theta)
  for k in range(f.D):
   d=np.eye(f.D)[k]*1e-5; fd=(local(f,C,theta+d)[0]-local(f,C,theta-d)[0])/2e-5
   err=float(np.max(abs(fd-G[:,k]))); assert err<1e-7
   checks.append(dict(test='finite_difference',seed=seed,parameter=k,error=err))
 save('checks',checks); save('gate',dict(passed=True,checks=len(checks),quality_gate=False))
 curves=[]; updates=[]; states=[]; distributions=[]; index=[]
 for seed in cfg['seeds']:
  theta=initials[seed].copy()
  for step in range(cfg['updates_each']+1):
   loss,G=local(f,C,theta); ev=f.evaluate(theta); dist=ev['distribution']; orbit=np.bincount(f.orbits,weights=dist)
   metrics=dict(ev['metrics']); metrics['generated_state_clean_risk']=metrics.pop('single_step_clean_risk')
   metrics.update(forward_clean_risk=float((loss[0]+loss[1]+(loss[2]+loss[3])/2+loss[4])/4),
      isomorph_entropy=float(-np.sum(orbit[orbit>0]*np.log(orbit[orbit>0]))),
      parameter_displacement=float(np.linalg.norm(theta-initials[seed])),
      reference_graph_probability=float(dist[f.refid]),max_graph_probability=float(dist.max()))
   for j,name in enumerate(names): metrics['forward_'+name]=float(loss[j]); metrics['terminal_'+name]=float(dist@C[:,j])
   for j in range(2): metrics['triangle'+str(j)+'_mean']=float(dist@tri[:,j])
   curves.append(dict(seed=seed,arm='MGDA_five_raw_gradients',step=step,**metrics))
   states.append(theta.copy()); distributions.append(dist); index.append(dict(seed=seed,step=step))
   if step==cfg['updates_each']: break
   t0=time.monotonic(); v,lam,res=minnorm(G); solver_seconds=time.monotonic()-t0
   delta=-cfg['learning_rate']*v; nxt=theta+delta
   assert np.isfinite(nxt).all()
   after,_=local(f,C,nxt)
   updates.append(dict(seed=seed,step=step+1,coefficients=lam.tolist(),gradient_norms=np.linalg.norm(G,axis=1).tolist(),
      combined_norm=float(np.linalg.norm(v)),displacement=float(np.linalg.norm(delta)),
      zero_direction=bool(np.linalg.norm(v)<=1e-12),KKT=res,solver_seconds=solver_seconds,
      predicted_changes=(G@delta).tolist(),actual_changes=(after-loss).tolist(),
      all_actual_decreased=bool(np.all(after-loss<0)),
      **{'change_'+name:float(after[j]-loss[j]) for j,name in enumerate(names)}))
   theta=nxt; COUNT+=1
  save('curves',curves); save('updates',updates)
 save('index',index); np.savez_compressed(OUT/'states.npz',parameters=np.array(states),distributions=np.array(distributions))
 for p,h in BIND.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 save('bindings',BIND)
 save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    CPU_threads=1,GPU_hours=0,formal_model_calls=0,toy_updates=COUNT,new_sampled_graphs=0,
    all_local_objectives_decreased_updates=sum(r['all_actual_decreased'] for r in updates),
    max_KKT_residual=max(r['KKT'] for r in updates),max_derivative_error=max(r['error'] for r in checks)))
 print(json.dumps(dict(receipt=json.loads((OUT/'receipt.json').read_text()),finals=[r for r in curves if r['step']==20])))

if __name__=='__main__':
 try: main()
 except BaseException:
  if OUT.exists(): save('failure',dict(traceback=traceback.format_exc(),updates=COUNT))
  raise
