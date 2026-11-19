"""Independent CPU arithmetic audit of saved trajectory, no optimization."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource,signal,time
import numpy as np
import run_cpu as r
START=time.monotonic();signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
m=r.load_module();f,C,tri=m.setup();out=r.OUT
def read(p):return json.loads(p.read_text())
a=np.load(out/'states.npz');ix=read(out/'index.json');rows=read(out/'curves.json');updates=read(out/'updates.json')
old=np.load(r.OLD/'states_and_distributions.npz');oldix=read(r.OLD/'distribution_index.json');oldu=read(r.OLD/'updates.json')
errors=[];kkts=[];masses=[];lengths=[];stage_rows=[]
names=['relation_count','anonymous_joint_degree','triangle001','triangle011','components']
for n,meta in enumerate(ix):
 theta=a['parameters'][n];loss=np.zeros(5);G=np.zeros((5,f.D));P={};clean={}
 for t in range(1,f.T+1):
  p=1/(1+np.exp(-(f.features[t]@theta)))
  cp=np.prod(np.where(f.B[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2);clean[t]=cp
  conditional=cp@C;loss+=f.qref[t]@conditional/f.T
  for e in range(f.M):
   cov=(cp*f.B[:,e][None,:])@C-p[:,e,None]*conditional
   G+=np.einsum('i,ik,id->kd',f.qref[t]/f.T,cov,f.features[t][:,e,:])
  q=f.k[t][0]+p*(f.k[t][1]-f.k[t][0]);P[t]=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
  masses.append(float(np.max(abs(P[t].sum(1)-1))))
 occ={f.T:f.prior.copy()}
 for t in range(f.T,0,-1):occ[t-1]=occ[t]@P[t]
 dist=occ[0];errors.append(float(np.max(abs(dist-a['distributions'][n]))))
 errors.extend(abs(float(loss[j])-rows[n]['forward_'+name]) for j,name in enumerate(names))
 errors.append(abs(float(dist@f.cost)-rows[n]['terminal_cost']))
 if meta['step'] in [0,20]:
  for t in range(1,f.T+1):
   fq=f.qref[t]@clean[t]@C;gq=occ[t]@clean[t]@C
   stage_rows.append(dict(**meta,t=t,TV_generated_vs_forward=float(abs(occ[t]-f.qref[t]).sum()/2),
      **{'forward_'+name:float(fq[j]) for j,name in enumerate(names)},
      **{'generated_state_'+name:float(gq[j]) for j,name in enumerate(names)}))
 if meta['step']==0:
  oi=next(i for i,x in enumerate(oldix) if x['seed']==meta['seed'] and x['step']==0 and x['arm']=='exact_forward_clean_structure')
  assert np.array_equal(theta,old['parameters'][oi])
 if meta['step']==20:continue
 u=next(x for x in updates if x['seed']==meta['seed'] and x['step']==meta['step']+1)
 b=next(x for x in oldu if x['seed']==meta['seed'] and x['step']==meta['step']+1 and x['arm']=='exact_forward_clean_structure')
 lam=np.array(u['coefficients']);v=G.T@lam;delta=-b['displacement_norm']*v/np.linalg.norm(v)
 errors.append(float(np.max(abs(theta+delta-a['parameters'][n+1]))))
 lengths.append(abs(float(np.linalg.norm(a['parameters'][n+1]-theta))-b['displacement_norm']))
 assert abs(lam.sum()-1)<1e-10 and lam.min()>=0
 scale=max(float(np.max(abs(G@G.T))),1e-30);kkts.append(float(max(0,np.max(v@v-G@v))/scale))
assert max(errors)<1e-10 and max(lengths)<1e-12 and max(kkts)<1e-8 and max(masses)<1e-10
for p,h in read(out/'bindings.json').items():assert hashlib.sha256(r.Path(p).read_bytes()).hexdigest()==h,p
r.save('independent_audit',dict(passed=True,states=len(ix),updates=len(updates),max_arithmetic_error=max(errors),
   max_step_length_error=max(lengths),max_KKT_violation=max(kkts),max_row_mass_error=max(masses),
   original_hashes_unchanged=True,seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
r.save('stage_distribution_analysis',stage_rows)
paired=read(r.PREV/'paired_final.json')+[x for x in rows if x['step']==20]
r.save('paired_final',paired)
keys=['terminal_cost','relation_count','anonymous_joint_degree','typed_triangles','components','triangle_MAE','triangle_mean','edges',
      'connected_probability','isomorph_entropy','isomorphic_collision','reference_graph_probability','reference_isomorph_mass',
      'terminal_triangle001','terminal_triangle011']
summary=[]
for arm in dict.fromkeys(x['arm'] for x in paired):
 for key in keys:
  values=[x[key] for x in paired if x['arm']==arm]
  summary.append(dict(arm=arm,metric=key,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),n=3,unit='toy_initialization'))
r.save('summary_mean_std',summary)
perseed=[]
for seed in sorted(set(x['seed'] for x in rows)):
 start=next(x for x in rows if x['seed']==seed and x['step']==0);end=next(x for x in rows if x['seed']==seed and x['step']==20)
 us=[x for x in updates if x['seed']==seed]
 perseed.append(dict(seed=seed,path_length=sum(x['actual_length'] for x in us),
   scale_min=min(x['scale_over_raw_mgda'] for x in us),scale_max=max(x['scale_over_raw_mgda'] for x in us),
   all_costs_decreased=sum(x['all_actual_decreased'] for x in us),
   **{'change_'+key:end[key]-start[key] for key in keys},
   **{'forward_change_'+key:end['forward_'+key]-start['forward_'+key] for key in names}))
r.save('per_seed_change',perseed)
print(json.dumps(dict(audit=read(out/'independent_audit.json'),changes=perseed)))
