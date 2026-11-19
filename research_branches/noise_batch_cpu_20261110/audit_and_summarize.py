"""Read-only independent probability and paired-input audit; no repeated optimization."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource,signal,time
import numpy as np
import run_cpu as r
START=time.monotonic();signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
p=r.OUT
def read(name):return json.loads((p/(name+'.json')).read_text())
cfg=read('protocol');tasks=read('tasks');curves=read('curves');updates=read('updates');inputs=read('inputs');index=read('state_index')
bank=np.load(p/'noise_banks.npz');a=np.load(p/'states.npz');m=r.module();f,C,tri=m.setup();errs=[]
prob=np.concatenate([f.qref[t]/f.T for t in range(1,f.T+1)])
assert np.array_equal(prob,bank['atom_probability'])
for seed in cfg['initializations']:
 for stream in cfg['noise_streams']:
  draws=np.random.default_rng(np.random.SeedSequence([seed,stream])).choice(len(prob),size=(20,32),p=prob)
  assert np.array_equal(draws,bank[f'{seed}_{stream}'])
for row in inputs:
 expected=int(bank[f"{row['seed']}_{row['noise_stream']}"][row['step']-1,row['draw']])
 assert expected==(row['t']-1)*64+row['state']
old=np.load(r.OLD/'states_and_distributions.npz');oldix=json.loads((r.OLD/'distribution_index.json').read_text())
for n,meta in enumerate(index):
 theta=a['parameters'][n];P={};loss=np.zeros(5)
 for t in range(1,f.T+1):
  pp=1/(1+np.exp(-(f.features[t]@theta)))
  cp=np.prod(np.where(f.B[None,:,:]>0,pp[:,None,:],1-pp[:,None,:]),axis=2)
  loss+=f.qref[t]@cp@C/f.T
  q=f.k[t][0]+pp*(f.k[t][1]-f.k[t][0])
  P[t]=np.prod(np.where(f.B[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
 dist=f.prior.copy()
 for t in range(f.T,0,-1):dist=dist@P[t]
 errs.append(float(np.max(abs(dist-a['distributions'][n]))))
 errs.append(abs(float(dist@f.cost)-curves[n]['terminal_cost']))
 for j,name in enumerate(['relation_count','anonymous_joint_degree','triangle001','triangle011','components']):errs.append(abs(float(loss[j])-curves[n]['forward_'+name]))
 if meta['step']==0:
  oi=next(i for i,x in enumerate(oldix) if x['seed']==meta['seed'] and x['step']==0 and x['arm']=='exact_forward_clean_structure')
  assert np.array_equal(theta,old['parameters'][oi])
assert max(errs)<1e-10
for path,sha in read('bindings').items():assert hashlib.sha256(r.Path(path).read_bytes()).hexdigest()==sha,path
r.save('independent_audit',dict(passed=True,saved_states=len(index),noise_records=len(inputs),reproduced_banks=9,
 max_error=max(errs),original_hashes_unchanged=True,seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
 scope='All saved parameter states/probabilities and all paired sampled input identities; not a second optimization replay'))
metrics=['terminal_cost','relation_count','anonymous_joint_degree','typed_triangles','components','triangle_MAE','triangle_mean','edges',
 'connected_probability','isomorph_entropy','isomorphic_collision','reference_isomorph_mass','terminal_triangle001','terminal_triangle011','forward_clean_risk']
finals=[x for x in curves if x['step']==20];r.save('finals',finals)
perinit=[];aggregate=[];progress=[]
for K in cfg['batch_sizes']:
 for seed in cfg['initializations']:
  ts=[x for x in tasks if x['K']==K and x['seed']==seed]
  xs=[x for x in finals if x['K']==K and x['seed']==seed]
  for metric in metrics:
   vals=[x[metric] for x in xs]
   perinit.append(dict(K=K,seed=seed,metric=metric,completed=len(xs),requested=len(ts),
      mean=float(np.mean(vals)) if len(vals)==len(ts) else None,
      noise_stream_std=float(np.std(vals,ddof=1)) if len(vals)==len(ts) else None))
 for metric in metrics:
  rs=[x for x in perinit if x['K']==K and x['metric']==metric]
  ok=all(x['mean'] is not None for x in rs)
  aggregate.append(dict(K=K,metric=metric,initializations=len(rs),noise_streams_per_initialization=3,
    mean=float(np.mean([x['mean'] for x in rs])) if ok else None,
    between_initialization_std=float(np.std([x['mean'] for x in rs],ddof=1)) if ok else None,
    mean_within_initialization_std=float(np.mean([x['noise_stream_std'] for x in rs])) if ok else None))
 us=[x for x in updates if x['K']==K];draws=[x for x in inputs if x['K']==K]
 progress.append(dict(K=K,updates=len(us),all_five_true_costs_decreased=sum(x['true_all_decreased'] for x in us),
    mean_gradient_error_on_actual_paths=float(np.mean([x['gradient_error'] for x in us])),
    mean_true_gradient_norm_on_actual_paths=float(np.mean([x['exact_gradient_norm'] for x in us])),
    noise_inputs=len(draws),counts_by_t={str(t):sum(x['t']==t for x in draws) for t in [1,2,3]}))
r.save('per_initialization',perinit);r.save('hierarchical_summary',aggregate);r.save('progress_summary',progress)
# Report actual t1 sparse-state input proportion; not an acceptance gate.
coverage=[]
for K in cfg['batch_sizes']:
 for seed in cfg['initializations']:
  for stream in cfg['noise_streams']:
   xs=[x for x in inputs if x['K']==K and x['seed']==seed and x['noise_stream']==stream and x['t']==1]
   sparse=sum(f.B[x['state']].sum()<=2 for x in xs)
   coverage.append(dict(K=K,seed=seed,noise_stream=stream,t1_inputs=len(xs),sparse_inputs=int(sparse),
     sparse_fraction=float(sparse/len(xs)) if xs else None))
r.save('noise_coverage',coverage)
print(json.dumps(dict(audit=read('independent_audit'),progress=progress,
 summary=[x for x in aggregate if x['metric'] in ['terminal_cost','typed_triangles','connected_probability']])))
