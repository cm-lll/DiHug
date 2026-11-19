"""CPU exact endpoint attribution and independent anonymous cloud-distance audit.

Reuses frozen 222 probability directions. No model calls, autograd or sampling.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,csv,traceback,resource,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import closure_semantics_audit as old
ROOT=old.ROOT;SRC=old.SRC;PANEL=old.PAIRED;PREV=old.OUT
OUT=ROOT/'results/closure_score_attribution_20261116'
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,rows):
 dump(n+'.json',rows)
 with (OUT/(n+'.csv')).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def cloud_cost(x,y):
 # Exact uniform empirical W2^2 using a type-preserving bijection, fixed scale.
 cost=((x[:,None,:]-y[None,:,:])**2).mean(2)
 a,b=linear_sum_assignment(cost)
 return float(cost[a,b].mean())
def main():
 start=time.time();OUT.mkdir(exist_ok=False);bindings={}
 def read(path):bindings[str(path)]=old.sha(path);return json.loads(Path(path).read_text())
 def load(path):bindings[str(path)]=old.sha(path);return np.load(path)
 proto=read(SRC/'protocol.json');manifest=read(PANEL/'input_manifest.json');directions=read(PREV/'direction_results.json');sigma=read(ROOT/'results/hetero_micro_macro_calibration_20261116/frozen_scales.json')['sigma_squared']
 dump('protocol.json',dict(panel='all frozen222 complete directions, 9 previous no-contrast conditions retained separately',new_directions=0,model_calls=0,GPU_hours=0,autograd=0,threads=1,CPU_seconds_max=180,
  attribution='Exact symmetric endpoint CF difference; per-node/per-feature decomposition along linear descriptor phase interpolation, not causal attribution',
  independent_check='Within each node type exact equal-weight minimum squared-distance assignment of 50-dimensional joint degree+closure descriptors; original reference RMS scales retained; no identity alignment; diagnostic only',
  groups='current Xt per-type top ceil(10%) hub; remaining degree<=1 low_nonhub; rest other. Disjoint for exact accounting.',
  numerical_sign_tolerance=dict(loss=1e-12,cloud=1e-12,triangle=1e-10),selection='No changes to frequencies, weights, scales or epsilon'))
 ctx={};contexts=[]
 for ds in ['IMDB','PubMed']:
  row=next(r for r in proto['parents'] if r['dataset']==ds);z=load(row['data']);ty=z['types'];schema=z['schema'];e=z['edges'];e=e if e.shape[0]==2 else e.T
  ref=np.zeros((len(schema),len(ty),len(ty)));lookup={tuple(r):i for i,r in enumerate(schema)}
  for i,j in e.T:ref[lookup[tuple(sorted((ty[i],ty[j])))],i,j]=ref[lookup[tuple(sorted((ty[i],ty[j])))],j,i]=1
  plan=old.obj.make_plan(schema,len(np.unique(ty)));rd,rc,_=old.counts(ref,ty,plan);refx=np.concatenate([rd,rc],1)
  frozen=read(ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json'))
  ctx[ds]=(ty,schema,plan,refx,frozen,row['config']['datasets'][ds]['type_names'])
  for typ in np.unique(ty):
   b=frozen[str(typ)]['closure'];y=refx[ty==typ]/np.array(b['scale']);assert cloud_cost(y,y)==0 and cloud_cost(y[::-1],y)==0
 rows=[];types=[];features=[];nodes=[];maxres=0.;check_cloud=0
 for en in manifest:
  if en['kind']!='free':continue
  z=load(en['input']);pr=load(PANEL/(en['key']+'_calibrated_micro_macro_prediction.npz'))['p'];ds=en['dataset'];ty,schema,plan,refx,frozen,names=ctx[ds]
  q=z['query'];fam=z['f'];fq=z['fullquery'];X=np.zeros((len(ty),len(ty)));X[fq[0],fq[1]]=z['xt'];X+=X.T;xd=X.sum(1);CN=X@X;cn=CN[q[0],q[1]];absent=z['xt'][z['ids']]==0
  P=np.zeros((len(schema),len(ty),len(ty)));P[fam,q[0],q[1]]=pr;P[fam,q[1],q[0]]=pr
  bd,bc,_=old.counts(P,ty,plan);base=np.concatenate([bd,bc],1);R=len(schema);cache={};groups=np.full(len(ty),'other',dtype='<U12')
  for typ in np.unique(ty):
   ix=np.flatnonzero(ty==typ);hub=ix[np.argsort(xd[ix],kind='stable')[-max(1,int(np.ceil(len(ix)*.1))):]];groups[ix[xd[ix]<=1]]='low_nonhub';groups[hub]='hub'
   b=frozen[str(typ)]['closure'];scale=np.array(b['scale']);omega=np.array(b['omega']);target=np.array(b['target_real'])+1j*np.array(b['target_imag']);x=base[ix];phase=(x/scale)@omega;cf=np.exp(1j*phase).mean(0);ref=refx[ix]/scale
   cache[int(typ)]=(ix,scale,omega,target,phase,cf,float(np.mean(abs(cf-target)**2)),cloud_cost(x/scale,ref),ref)
  selected=[r for r in directions if r['status']=='complete' and r['train_seed']==en['train_seed'] and r['sample_seed']==en['sample_seed']]
  for prev in selected:
   if time.time()-start>180:raise TimeoutError('CPU budget')
   r=prev['family'];eps=prev['epsilon'];m=(fam==r)&absent;hi=m&(cn==cn[m].max());lo=m&(cn==cn[m].min());src,dst=(hi,lo) if prev['direction']=='high_to_low' else (lo,hi)
   pp=pr.copy();pp[src]-=eps*pr[src]/pr[src].sum();pp[dst]+=eps*(1-pr[dst])/(1-pr[dst]).sum();assert abs(pp[fam==r].sum()-pr[fam==r].sum())<1e-10
   Q=np.zeros_like(P);Q[fam,q[0],q[1]]=pp;Q[fam,q[1],q[0]]=pp;nd,nc,_=old.counts(Q,ty,plan);new=np.concatenate([nd,nc],1)
   ident={k:prev[k] for k in ['dataset','train_seed','sample_seed','t','family','direction']};total=0.;cloud=0.;ceq=0.;deg=0.;clo=0.;max_marginal=0.;badfeatures=0;allfeatures=0
   for typ in np.unique(ty):
    ix,scale,omega,target,phase,cf,e0,w0,ref=cache[int(typ)];nphase=(new[ix]/scale)@omega;ncf=np.exp(1j*nphase).mean(0);e1=float(np.mean(abs(ncf-target)**2));delta=e1-e0
    factor=1/(4*sigma[ds][str(typ)]['closure'])/len(np.unique(ty))
    # exp(i b)-exp(i a) = i exp(i(a+b)/2) sinc((b-a)/2) (b-a).
    coefficient=np.real(np.conj(ncf+cf-2*target)[None,:]*1j*np.exp(1j*(phase+nphase)/2)*np.sinc((nphase-phase)/(2*np.pi)))
    attribution=(coefficient@omega.T)*(new[ix]-base[ix])/(len(ix)*omega.shape[1]*scale)
    res=abs(attribution.sum()-delta);maxres=max(maxres,res);assert res<1e-12
    scaled=attribution*factor;total+=delta*factor;degr=float(scaled[:,:R].sum());clos=float(scaled[:,R:].sum());deg+=degr;clo+=clos
    w1=cloud_cost(new[ix]/scale,ref);cloud+=(w1-w0)*factor;ceq+=(w1-w0)/len(np.unique(ty))
    # Marginal sorted-feature distances do not detect cross-feature association.
    oldm=np.mean(abs(np.sort(base[ix]/scale,axis=0)-np.sort(ref,axis=0)),axis=0);newm=np.mean(abs(np.sort(new[ix]/scale,axis=0)-np.sort(ref,axis=0)),axis=0);md=newm-oldm
    max_marginal=max(max_marginal,float(md.max()));badfeatures+=int((md>1e-12).sum());allfeatures+=len(md)
    types.append(dict(**ident,type=int(typ),type_name=names[int(typ)],n=len(ix),base_error=e0,change_error=delta,scaled_contribution=delta*factor,degree_contribution=degr,closure_contribution=clos,base_joint_W2_squared=w0,change_joint_W2_squared=w1-w0,scaled_cloud_contribution=(w1-w0)*factor))
    for j in range(new.shape[1]):
     if np.max(abs(new[ix,j]-base[ix,j]))<1e-16:continue
     ff=dict(**ident,type=int(typ),type_name=names[int(typ)],feature=j,block='degree' if j<R else 'closure',relations=str([j] if j<R else plan['closures'][j-R]),scaled_contribution=float(scaled[:,j].sum()),change_marginal_W1=float(md[j]),change_mean=float((new[ix,j]-base[ix,j]).mean()))
     for group in ['hub','low_nonhub','other']:ff[group+'_contribution']=float(scaled[groups[ix]==group,j].sum())
     features.append(ff)
    for k,node in enumerate(ix):nodes.append(dict(**ident,node=int(node),type=int(typ),group=str(groups[node]),current_degree=float(xd[node]),degree_contribution=float(scaled[k,:R].sum()),closure_contribution=float(scaled[k,R:].sum()),total_contribution=float(scaled[k].sum())))
   assert abs(total-prev['change_scaled_closure'])<1e-12
   assert abs(nc.sum()/3-prev['base_triangles']-prev['change_triangles'])<1e-8
   rows.append(dict(**ident,epsilon=eps,triangle_error_change=prev['change_triangle_absolute_error'],scaled_score_change=total,degree_contribution=deg,closure_contribution=clo,scaled_joint_W2_change=cloud,equal_type_joint_W2_change=ceq,marginal_features_worse=badfeatures,marginal_features_total=allfeatures,max_scaled_marginal_W1_increase=max_marginal))
 table('directions',rows);table('type_contributions',types);table('feature_contributions',features);table('node_contributions',nodes)
 dump('previous_skips.json',[r for r in directions if r['status']!='complete'])
 summaries=[]
 for seed in sorted(set(r['train_seed'] for r in rows)):
  rr=[r for r in rows if r['train_seed']==seed];dis=[r for r in rr if r['triangle_error_change']<-1e-10 and r['scaled_score_change']>1e-12]
  summaries.append(dict(train_seed=seed,directions=len(rr),triangle_better_score_worse=len(dis),of_these_joint_W2_better=sum(r['scaled_joint_W2_change']<-1e-12 for r in dis),joint_W2_worse=sum(r['scaled_joint_W2_change']>1e-12 for r in dis),any_marginal_feature_worse=sum(r['marginal_features_worse']>0 for r in dis),degree_term_positive=sum(r['degree_contribution']>1e-12 for r in dis),closure_term_positive=sum(r['closure_contribution']>1e-12 for r in dis)))
 table('summary',summaries)
 for f in [__file__,old.__file__,Path(old.__file__).with_name('objective.py')]:bindings[str(f)]=old.sha(f)
 dump('sources.json',bindings);dump('receipt.json',dict(status='complete',seconds=time.time()-start,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,directions=len(rows),max_exact_attribution_residual=maxres,network_calls=0,autograd=0,GPU_hours=0,new_graphs=0,source_assets_verified=all(old.sha(k)==v for k,v in bindings.items())))
def reporting():
 """Summarize old hard graphs and saved attribution without model calls."""
 cc=json.loads((PREV/'closure_channels_with_support.json').read_text());res=[]
 for seed in [2026110421,2026110422]:
  for arm in ['parent','CE','calibrated_micro_macro']:
   for typ,rels,name in [(3,(3,0,3),'Species-Gene-Gene'),(3,(3,1,6),'Species-Gene-Disease'),(3,(3,2,8),'Species-Gene-Chemical'),(2,(5,4,5),'Chemical-Disease-Disease')]:
    r=[r for r in cc if r['train_seed']==seed and r['kind']=='terminal' and r['arm']==arm and r['type']==typ and (r['r'],r['s'],r['u'])==rels];assert len(r)==4
    values=[r['closures'] for r in r];res.append(dict(train_seed=seed,arm=arm,pattern=name,n=4,reference=r[0]['ref_closures'],mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),values=values))
 table('terminal_pattern_check',res)
 rows=json.loads((OUT/'directions.json').read_text());ts=json.loads((OUT/'type_contributions.json').read_text());fs=json.loads((OUT/'feature_contributions.json').read_text());cases=[]
 for family in [3,5]:
  def select(r):return r['train_seed']==2026110421 and r['sample_seed']==1666973446 and r['family']==family and r['direction']=='high_to_low'
  cases.append(dict(direction=next(r for r in rows if select(r)),types=[r for r in ts if select(r)],features=sorted([r for r in fs if select(r)],key=lambda r:abs(r['scaled_contribution']),reverse=True)))
 dump('illustrative_cases.json',cases)
 dump('report_source.json',dict(script_sha256=old.sha(__file__),main_receipt_unchanged=True,reporting_only=True))
if __name__=='__main__':
 try:
  if '--report' in sys.argv:reporting()
  else:main();reporting()
 except BaseException:
  if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()))
  raise
