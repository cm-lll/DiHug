"""CPU analysis of frozen probabilities; no model imports or graph sampling."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/paired_response_check_20261116'
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,rows):
 dump(n+'.json',rows)
 if rows:
  with (OUT/(n+'.csv')).open('w') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def mat(v,e,n):
 a=np.zeros((n,n));a[e[0],e[1]]=v;a[e[1],e[0]]=v;return a
def tri(a):return float(np.sum((a@a)*a)/6)
def main():
 start=time.time();manifest=json.loads((OUT/'input_manifest.json').read_text());rows=[];relations=[];nodes=[];composition=[]
 dump('analysis_protocol.json',dict(current_degree_groups='per node type: low degree<=1; hub top ceil(10% n) current Xt total degree (ties broken by ID only for bookkeeping); overlap retained and disclosed',CN_bins=['0','1-2','3-7','8+'],identity_errors='only meaningful restoration errors for normal forward states; free states anonymous distribution comparison primary',expectation='Independent edges conditional on Xt and fixed R1 query, not expected nonlinear structural loss',selection='all frozen inputs, no exclusions'))
 for entry in manifest:
  z=np.load(entry['input']);x=z['xt'];ids=z['ids'];fam=z['f'];query=z['query'];fullq=z['fullquery'];ty=z['types'];n=len(ty);e=z['reference_edges'];X=mat(x,fullq,n);R=mat(np.ones(e.shape[1]),e,n);CN=X@X;cn=CN[query[0],query[1]];labels=R[query[0],query[1]].astype(int);xx=x[ids];rd=R.sum(1);xd=X.sum(1)
  ident={k:entry[k] for k in ['model','dataset','train_seed','sample_seed','kind','t']}
  for arm in ['parent','CE','calibrated_micro_macro']:
   path=OUT/(entry['key']+'_'+arm+'_prediction.npz')
   if not path.exists():continue
   zz=np.load(path);p=zz['p'];q=zz['q'];logits=zz['logits'];P=mat(p,query,n);Q=mat(q,query,n)
   # Fixed zero families are absent in active queries; support verified by worker.
   assert np.all(z['prior'][z['fullfamily'][np.setdiff1d(np.arange(len(x)),ids)]]==0)
   ce=np.logaddexp(logits[:,0],logits[:,1])-logits[np.arange(len(labels)),labels]
   birth=q*(1-xx);death=(1-q)*xx;bt=tri(Q)-tri(Q*X);dt=tri(X)-tri(Q*X)
   norm=[np.mean(abs(np.sort(P.sum(1)[ty==a])-np.sort(rd[ty==a]))) for a in np.unique(ty)]
   r=dict(**ident,arm=arm,current_edges=float(x.sum()),current_triangles=tri(X),reference_edges=float(R.sum()/2),reference_triangles=tri(R),clean_edges=float(p.sum()),clean_triangles=tri(P),clean_triangle_abs_error=abs(tri(P)-tri(R)),clean_type_degree_W1=float(np.mean(norm)),next_edges=float(q.sum()),next_triangles=tri(Q),triangle_change=tri(Q)-tri(X),triangle_births=bt,triangle_deaths=dt,edge_births=float(birth.sum()),edge_deaths=float(death.sum()),birth_CN=float(birth@cn/birth.sum()) if birth.sum() else None,death_CN=float(death@cn/death.sum()) if death.sum() else None,CE=float(ce.mean()),positive_CE=float(ce[labels==1].mean()),negative_CE=float(ce[labels==0].mean()),high_CN_births=float(birth[cn>=3].sum()),low_CN_deaths=float(death[cn<=2].sum()))
   a=lambda t:float(np.cos((t/1000+.008)/1.008*np.pi/2)**2/np.cos(.008/1.008*np.pi/2)**2)
   Pi=mat(z['prior'][z['fullfamily']],fullq,n);F=a(entry['t'])*R+(1-a(entry['t']))*Pi;Fprev=a(entry['t']-1)*R+(1-a(entry['t']-1))*Pi
   r.update(forward_expected_triangles=tri(F),forward_previous_expected_triangles=tri(Fprev),current_forward_excess=tri(X)-tri(F),next_forward_excess=tri(Q)-tri(Fprev))
   rows.append(r)
   for f in np.unique(fam):
    mask=fam==f;fe=query[:,mask];pr=P.copy()*0;pr[fe[0],fe[1]]=p[mask];pr[fe[1],fe[0]]=p[mask]
    for label in [0,1]:
     m=mask&(labels==label);count=int(m.sum())
     relations.append(dict(**ident,arm=arm,family=int(f),reference_label=label,support=count,CE=float(ce[m].mean()) if count else None,probability_mass=float(p[m].sum()),miss_or_false_mass=float((1-p[m]).sum()) if label else float(p[m].sum()),birth_mass=float(birth[m].sum()),death_mass=float(death[m].sum()),linear_triangle_birth=float(birth[m]@cn[m]),linear_triangle_death=float(death[m]@cn[m])))
   for a in np.unique(ty):
    ix=np.flatnonzero(ty==a);hub=ix[np.argsort(xd[ix],kind='stable')[-max(1,int(np.ceil(len(ix)*.1))):]];low=ix[xd[ix]<=1]
    for group,idx in [('all',ix),('current_low',low),('current_hub',hub)]:
     nodes.append(dict(**ident,arm=arm,type=int(a),group=group,support=len(idx),current_degree_mean=float(xd[idx].mean()) if len(idx) else None,expected_clean_degree_mean=float(P.sum(1)[idx].mean()) if len(idx) else None,expected_next_degree_mean=float(Q.sum(1)[idx].mean()) if len(idx) else None,restoration_degree_MAE=float(abs(P.sum(1)[idx]-rd[idx]).mean()) if len(idx) and entry['kind']=='normal' else None))
   # Type composition via ordered type masks, each unordered triangle counted six times.
   before_composition=len(composition)
   for types in itertools.combinations_with_replacement(np.unique(ty),3):
    vals=[]
    for M in [X,R,P,Q]:
     total=0.
     for a,b,c in set(itertools.permutations(types)):
      ia=np.flatnonzero(ty==a);ib=np.flatnonzero(ty==b);ic=np.flatnonzero(ty==c)
      total+=float(np.sum((M[np.ix_(ia,ib)]@M[np.ix_(ib,ic)])*M[np.ix_(ia,ic)]))
     vals.append(total/6)
    composition.append(dict(**ident,arm=arm,types='-'.join(map(str,types)),current=vals[0],reference=vals[1],clean=vals[2],next=vals[3]))
   for key,M in [('current',X),('reference',R),('clean',P),('next',Q)]:assert abs(sum(c[key] for c in composition[before_composition:])-tri(M))<1e-8
 table('responses',rows);table('relation_label',relations);table('node_groups',nodes);table('triangle_composition',composition)
 summary=[]
 for seed in sorted(set(r['train_seed'] for r in rows)):
  for kind in ['free','normal']:
   for arm in ['parent','CE','calibrated_micro_macro']:
    rr=[r for r in rows if r['train_seed']==seed and r['kind']==kind and r['arm']==arm]
    if not rr:continue
    d=dict(dataset=rr[0]['dataset'],train_seed=seed,kind=kind,arm=arm,n=len(rr),times=[r['t'] for r in rr])
    for k in ['current_edges','current_triangles','clean_edges','clean_triangles','clean_triangle_abs_error','clean_type_degree_W1','triangle_change','triangle_births','triangle_deaths','edge_births','edge_deaths','birth_CN','death_CN','high_CN_births','low_CN_deaths','CE','positive_CE','negative_CE']:
     v=[r[k] for r in rr if r[k] is not None];d[k]=float(np.mean(v)) if v else None
    summary.append(d)
 table('summary',summary);dump('analysis_receipt.json',dict(seconds=time.time()-start,responses=len(rows),CPU_only=True))
if __name__=='__main__':main()
