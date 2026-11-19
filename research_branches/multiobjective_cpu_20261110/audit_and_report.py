"""Independent product-probability/state-update audit and CPU tables; no optimization."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib, json, resource, signal, time
from pathlib import Path
import numpy as np
import run_cpu as r

START=time.monotonic(); signal.alarm(180); resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
OUT=r.OUT; OLD=r.OLD
f,C,tri=r.setup(); a=np.load(OUT/'states.npz'); index=json.loads((OUT/'index.json').read_text())
rows=json.loads((OUT/'curves.json').read_text()); updates=json.loads((OUT/'updates.json').read_text())
old=np.load(OLD/'states_and_distributions.npz'); oldix=json.loads((OLD/'distribution_index.json').read_text())
oldrows=json.loads((OLD/'curves.json').read_text()); oldu=json.loads((OLD/'updates.json').read_text())
errs=[]; mass=[]; kkt=[]; rotation=[]
def independent(theta):
 loss=np.zeros(5); G=np.zeros((5,f.D)); P={}
 for t in range(1,f.T+1):
  p=1/(1+np.exp(-(f.features[t]@theta)))
  cp=np.prod(np.where(f.B[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2)
  conditional=cp@C; loss+=f.qref[t]@conditional/f.T
  for e in range(f.M):
   cov=(cp*f.B[:,e][None,:])@C-p[:,e,None]*conditional
   G+=np.einsum('i,ik,id->kd',f.qref[t]/f.T,cov,f.features[t][:,e,:])
  q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0])
  P[t]=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
  mass.append(float(np.max(abs(P[t].sum(1)-1))))
 dist=f.prior.copy()
 for t in range(f.T,0,-1): dist=dist@P[t]
 return loss,G,dist

for n,meta in enumerate(index):
 theta=a['parameters'][n]; loss,G,dist=independent(theta)
 row=rows[n]
 errs.extend([float(np.max(abs(dist-a['distributions'][n]))),abs(float(dist@f.cost)-row['terminal_cost'])])
 errs.extend(abs(float(loss[j])-row['forward_'+name]) for j,name in enumerate(r.json.loads((OUT/'protocol.json').read_text())['objectives']))
 if meta['step']==0:
  oi=next(i for i,x in enumerate(oldix) if x['seed']==meta['seed'] and x['step']==0 and x['arm']=='exact_forward_clean_structure')
  assert np.array_equal(theta,old['parameters'][oi])
 if meta['step']==20: continue
 u=next(x for x in updates if x['seed']==meta['seed'] and x['step']==meta['step']+1)
 lam=np.array(u['coefficients']);v=G.T@lam; expected=theta-.2*v
 errs.append(float(np.max(abs(expected-a['parameters'][n+1]))))
 hg=G@v; vv=float(v@v); scale=max(float(np.max(abs(G@G.T))),1e-30)
 kkt.append(float(max(0,np.max(vv-hg))/scale))
 assert abs(lam.sum()-1)<1e-10 and np.min(lam)>=0
 ag=(G[0]+G[1]+(G[2]+G[3])/2+G[4])/4
 rotation.append(dict(**meta,cosine_to_same_state_old_average=float(v@ag/(np.linalg.norm(v)*np.linalg.norm(ag))),
                      norm_ratio_to_same_state_old_average=float(np.linalg.norm(v)/np.linalg.norm(ag))))
assert max(errs)<1e-10 and max(mass)<1e-10 and max(kkt)<1e-8
for path,sha in json.loads((OUT/'bindings.json').read_text()).items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
r.save('independent_audit',dict(passed=True,states=len(index),update_reconstructions=len(updates),max_error=max(errs),
   max_row_mass_error=max(mass),max_KKT_violation=max(kkt),sources_unchanged=True,
   seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
r.save('direction_comparison',rotation)

paired=[]; paths=[]
for seed in sorted(set(x['seed'] for x in rows)):
 for arm,step in [('initial',0),('reference_edge_CE',20),('exact_forward_clean_structure',20),('MGDA_five_raw_gradients',20)]:
  if arm=='MGDA_five_raw_gradients':
   row=next(x for x in rows if x['seed']==seed and x['step']==step).copy(); dist=a['distributions'][index.index(dict(seed=seed,step=step))]
  else:
   name='exact_forward_clean_structure' if arm=='initial' else arm
   row=next(x for x in oldrows if x['seed']==seed and x['step']==step and x['arm']==name).copy()
   oi=next(i for i,x in enumerate(oldix) if x['seed']==seed and x['step']==step and x['arm']==name); dist=old['distributions'][oi]
  row['arm']=arm
  for j,name in enumerate(['triangle001','triangle011']): row['terminal_'+name]=float(dist@C[:,j+2]); row[name+'_mean']=float(dist@tri[:,j])
  paired.append(row)
 us=[x for x in updates if x['seed']==seed]; bs=[x for x in oldu if x['seed']==seed and x['arm']=='exact_forward_clean_structure']
 lengths=[sum(x['displacement'] for x in us),sum(x['displacement_norm'] for x in bs)]
 paths.append(dict(seed=seed,candidate_path_length=lengths[0],baseline_path_length=lengths[1],ratio=lengths[0]/lengths[1],
   mean_coefficients=np.mean([x['coefficients'] for x in us],axis=0).tolist(),
   min_cosine_to_old_average=min(x['cosine_to_same_state_old_average'] for x in rotation if x['seed']==seed),
   max_cosine_to_old_average=max(x['cosine_to_same_state_old_average'] for x in rotation if x['seed']==seed)))
r.save('paired_final',paired);r.save('path_and_coefficients',paths)
metrics=['terminal_cost','relation_count','anonymous_joint_degree','typed_triangles','components','triangle_MAE','connected_probability','isomorph_entropy','isomorphic_collision','reference_graph_probability','terminal_triangle001','terminal_triangle011']
summary=[]
for arm in dict.fromkeys(x['arm'] for x in paired):
 for key in metrics:
  vals=[x[key] for x in paired if x['arm']==arm]
  summary.append(dict(arm=arm,metric=key,mean=float(np.mean(vals)),sample_std=float(np.std(vals,ddof=1)),n=3,unit='toy_initialization_not_real_training_seed'))
r.save('summary_mean_std',summary)
print(json.dumps(dict(audit=json.loads((OUT/'independent_audit.json').read_text()),paths=paths)))
