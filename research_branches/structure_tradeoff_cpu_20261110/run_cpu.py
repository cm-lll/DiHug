import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,importlib.util,json,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/structure_tradeoff_cpu_20261110'
OLD=ROOT/'results/one_step_to_terminal_cpu_20261110'
START=time.monotonic();BIND={}
def bind(p):
 p=Path(p);BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,bind(p));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
 if isinstance(x,list) and x:
  with (OUT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in x for k in r)));w.writeheader();w.writerows(x)
def main():
 OUT.mkdir(exist_ok=False);signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 save('protocol',json.loads(bind(HERE/'protocol.json').read_text()));bind(Path(__file__))
 m=mod('tradeoff_finite',ROOT/'research_branches/terminal_structure_audit_20261109/audit_cpu.py')
 m.kernel_module=mod('tradeoff_kernel',m.KERNEL)
 cfg=json.loads(bind(m.PROTOCOL).read_text());f=m.FiniteDiffusion(cfg)
 a=np.load(bind(OLD/'states_and_distributions.npz'));index=json.loads(bind(OLD/'distribution_index.json').read_text())
 # Independent typed triangle enumeration, not inferred from total counts.
 tri=np.zeros((f.S,2))
 import itertools
 for s,A in enumerate(f.adj):
  for nodes in itertools.combinations(range(4),3):
   if all(A[i,j] for i,j in itertools.combinations(nodes,2)):tri[s,int(f.types[list(nodes)].sum())-1]+=1
 tc=(tri-tri[f.refid])**2
 C=np.column_stack([f.blocks,tc]);names=f.block_names+['triangle001','triangle011']
 assert np.max(abs(tc.mean(1)-f.blocks[:,2]))<1e-14
 def evaluate(theta):
  _,_,clean,pred=f.matrices(theta);loss=np.zeros(6);z=[];g=np.zeros((6,f.D))
  for t in range(1,f.T+1):
   cp=np.exp(clean[t]);joint=f.qref[t][:,None]*cp/f.T
   loss+=np.einsum('ij,jk->k',joint,C)
   dz=np.einsum('ij,jk,ije->kie',joint,C,f.B[None,:,:]-pred[t][:,None,:])
   z.append(dz.reshape(6,-1));g+=np.einsum('kie,ied->kd',dz,f.features[t])
  return loss,np.concatenate(z,axis=1),g
 rows=[];cross=[];lp_rows=[];witness=[];checks=[]
 for n,item in enumerate(index):
  if item['arm']!='exact_forward_clean_structure':continue
  theta=a['parameters'][n];loss,z,g=evaluate(theta)
  meta=dict(seed=item['seed'],step=item['step'])
  for space,G in [('independent_logits',z),('shared_parameters',g)]:
   avg=G[:4].mean(0);norms=np.linalg.norm(G,axis=1)
   for k,name in enumerate(names):
    rows.append(dict(**meta,space=space,component=name,loss=float(loss[k]),gradient_norm=float(norms[k]),
                     derivative_under_average_descent=float(-G[k]@avg),
                     normalized_average_alignment=float(G[k]@avg/max(norms[k]*np.linalg.norm(avg),1e-30))))
    for j,other in enumerate(names[:4]):
     cross.append(dict(**meta,space=space,target=name,source=other,
                       cosine=float(G[k]@G[j]/max(norms[k]*norms[j],1e-30)),
                       contribution_to_average_direction=float(-G[k]@G[j]/4)))
   for objective,ix in [('four_blocks',[0,1,2,3]),('split_five',[0,1,4,5,3])]:
    gradients=G[ix];nn=np.linalg.norm(gradients,axis=1)
    assert np.all(nn>1e-14),'Zero gradients require explicit separate handling'
    U=gradients/nn[:,None];D=G.shape[1]
    opt=linprog(np.r_[np.zeros(D),-1.],A_ub=np.column_stack([U,np.ones(len(ix))]),
                b_ub=np.zeros(len(ix)),bounds=[(-1,1)]*D+[(0,None)],method='highs',
                options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    assert opt.success,opt.message
    d=opt.x[:-1];margin=opt.x[-1];res=float(max(0,np.max(U@d+margin)))
    assert res<1e-8
    unit=d/max(np.linalg.norm(d),1e-30)
    r=dict(**meta,space=space,objectives=objective,margin=float(margin),residual=res,
           min_unit_direction_improvement=float(np.min(-U@unit)),strict_common_direction=bool(margin>1e-8))
    if space=='shared_parameters' and margin>1e-8:
     lp,_,_=evaluate(theta+1e-4*unit);lm,_,_=evaluate(theta-1e-4*unit)
     fd=(lp-lm)/2e-4;error=float(np.max(abs(fd-g@unit)))
     assert error<1e-7
     checks.append(dict(**meta,objectives=objective,finite_difference_error=error))
     r['all_selected_finite_changes_negative']=bool(np.all(lp[ix]<loss[ix]))
     r['terminal_total_change']=float(f.evaluate(theta+1e-4*unit)['metrics']['terminal_cost']-
                                      f.evaluate(theta)['metrics']['terminal_cost'])
     for k,name in enumerate(names):witness.append(dict(**meta,objectives=objective,component=name,
         derivative=float(g[k]@unit),finite_change=float(lp[k]-loss[k])))
    lp_rows.append(r)
  # Exact finite derivative of all component objectives in each parameter coordinate.
  if item['step'] in [0,20]:
   for k in range(f.D):
    v=np.eye(f.D)[k]*1e-5
    fd=(evaluate(theta+v)[0]-evaluate(theta-v)[0])/2e-5
    error=float(np.max(abs(fd-g[:,k])));assert error<1e-7
    checks.append(dict(**meta,parameter=k,finite_difference_error=error))
 save('gradients',rows);save('cross_component_effects',cross);save('common_descent',lp_rows)
 save('finite_witness',witness);save('checks',checks)
 for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 save('bindings',BIND)
 summary={}
 for space in ['independent_logits','shared_parameters']:
  summary[space]={}
  for objective in ['four_blocks','split_five']:
   r=[x for x in lp_rows if x['space']==space and x['objectives']==objective]
   summary[space][objective]=dict(conditions=len(r),common_descent=sum(x['strict_common_direction'] for x in r),
                                   min_margin=min(x['margin'] for x in r),max_margin=max(x['margin'] for x in r))
  for name in names:
   r=[x for x in rows if x['space']==space and x['component']==name]
   summary[space][name+'_harmed_by_average']=sum(x['derivative_under_average_descent']>1e-12 for x in r)
 save('summary',summary)
 save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,
      RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,states=15,LPs=len(lp_rows),
      formal_model_calls=0,GPU_hours=0,optimization_updates=0,new_graphs=0,CPU_threads=1))
 print(json.dumps(summary))
if __name__=='__main__':
 try:main()
 except BaseException:
  if OUT.exists():save('failure',dict(traceback=traceback.format_exc()))
  raise
