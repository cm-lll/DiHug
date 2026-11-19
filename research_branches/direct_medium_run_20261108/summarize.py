"""Descriptive matched summaries; samples and local histories are not training repeats."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,sys,math,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(sys.argv[1])
def read(name):
 with (O/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
 if not rows:return
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
samples=read('conditional_samples.csv');pred=read('predictions.csv');relations=read('relations.csv');nodes=read('nodegroups.csv')
metrics=['DIRECT','total_degree_W1','relation_degree_W1','joint_TV','triangles','triangle_abs_error','degree_component','joint_component','triangle_component','sampled_edges','sampled_edge_abs_error']
pm=['CE','positive_CE','negative_CE','expected_edges','edge_abs_error','expected_degree_MAE','expected_triangles','expected_triangle_abs_error']
groups={};pg={}
for x in samples:groups.setdefault((x['condition'],x['arm'],int(x['offset'])),[]).append(x)
for x in pred:
 if x['panel']=='new':pg.setdefault((x['condition'],x['arm'],int(x['offset'])),[]).append(x)
def matrix(rows,k):
 a=np.full((20,32),np.nan)
 for x in rows:a[int(x['index']),int(x['sample'])]=float(x[k])
 assert np.isfinite(a).all();return a
def pvec(rows,k):
 a=np.full(20,np.nan)
 for x in rows:a[int(x['index'])]=float(x[k])
 assert np.isfinite(a).all();return a
def mcse(a):return float(np.sqrt(np.sum(np.var(a,axis=1,ddof=1)/32))/20)
def strata_se(v):return float(np.sqrt(np.sum(np.var(v.reshape(10,2),axis=1,ddof=1)/2)/100))
curves=[];comparisons=[];input_pairs=[];triplets=[]
for key,rows in sorted(groups.items()):
 c,arm,offset=key;s=dict(condition=c,arm=arm,offset=offset,n_inputs=20,draws_per_input=32)
 for k in metrics:
  a=matrix(rows,k);s[k]=float(a.mean());s[k+'_MC_SE']=mcse(a);s[k+'_input_SE']=strata_se(a.mean(1))
 for k in pm:s[k]=float(pvec(pg[key],k).mean())
 s['CE_plus_lambda_DIRECT']=s['CE']+.001*s['DIRECT'];curves.append(s)
conditions=sorted({k[0] for k in groups})
for c in conditions:
 start=groups[c,'start',0]
 for offset in [8,32,64]:
  ka=(c,'CE',offset);kb=(c,'CE_STRUCT',offset)
  if ka not in groups or kb not in groups:continue
  rowsa,rowsb=groups[ka],groups[kb];s=dict(condition=c,offset=offset)
  for k in metrics:
   a,b,z=matrix(rowsa,k),matrix(rowsb,k),matrix(start,k);d=b-a
   s[k+'_CE']=float(a.mean());s[k+'_candidate']=float(b.mean());s[k+'_delta']=float(d.mean());s[k+'_paired_MC_SE']=mcse(d)
   s[k+'_paired_input_SE']=strata_se(d.mean(1));s[k+'_improved_inputs']=int((d.mean(1)<0).sum())
   s[k+'_CE_minus_origin']=float((a-z).mean());s[k+'_candidate_minus_origin']=float((b-z).mean())
  for k in pm:s[k+'_delta']=float((pvec(pg[kb],k)-pvec(pg[ka],k)).mean())
  comparisons.append(s)
  for idx in range(20):
   source=next(x for x in rowsa if int(x['index'])==idx);row=dict(condition=c,offset=offset,index=idx,layer=idx//2+1,t=int(source['t']))
   for k in metrics:
    a,b=matrix(rowsa,k)[idx],matrix(rowsb,k)[idx];row[k+'_CE']=float(a.mean());row[k+'_candidate']=float(b.mean());row[k+'_delta']=float((b-a).mean());row[k+'_paired_MC_SE']=float((b-a).std(ddof=1)/np.sqrt(32))
   for k in pm:row[k+'_delta']=float(pvec(pg[kb],k)[idx]-pvec(pg[ka],k)[idx])
   input_pairs.append(row)
  for arm,rr in [('CE',rowsa),('CE_STRUCT',rowsb)]:
   ref=45 if c.startswith('IMDB') else 1937
   for idx in range(20):
    a=matrix(rr,'triangles')[idx];mae=abs(a-ref).mean();bias=abs(a.mean()-ref)
    triplets.append(dict(condition=c,arm=arm,offset=offset,index=idx,mean=float(a.mean()),sd=float(a.std(ddof=1)),reference=ref,mean_absolute_error=float(mae),absolute_mean_error=float(bias),cross_reference_gap=float(mae-bias),below=int((a<ref).sum()),above=int((a>ref).sum())))
def paired_long(rows,ids,fields,name):
 lookup={tuple(x[k] for k in ['condition','arm','offset','panel','index']+ids):x for x in rows};out=[]
 for key,a in lookup.items():
  if a['arm']!='CE':continue
  other=list(key);other[1]='CE_STRUCT';b=lookup.get(tuple(other))
  if b is None:continue
  z={k:a[k] for k in ['condition','offset','panel','index','t']+ids}
  for k in fields:
   av=float(a[k]) if a[k]!='' else None;bv=float(b[k]) if b[k]!='' else None
   z[k+'_CE']=av;z[k+'_candidate']=bv;z[k+'_delta']=bv-av if av is not None and bv is not None else None
  out.append(z)
 write(name,out)
paired_long(relations,['family'],['positive_count','negative_count','expected_edges','missed_positive_mass','false_positive_mass','positive_CE','negative_CE','expected_relation_degree_MAE'],'paired_relations.csv')
paired_long(nodes,['type','group'],['support','degree_MAE','triangle_participation_MAE'],'paired_nodegroups.csv')
write('curves.csv',curves);write('comparisons.csv',comparisons);write('paired_inputs.csv',input_pairs);write('triangle_distribution.csv',triplets)
write('legacy_predictions.csv',[x for x in pred if x['panel']=='legacy'])
summary=dict(scope='raw conditional independent-edge probabilities, not full generation; two training seeds, four local histories',
 curves=curves,comparisons=comparisons,MC_SE_note='conditional auxiliary draws only',input_SE_note='ten strata with two inputs each; not training-seed uncertainty or significance test',
 GPU=json.loads((O/'receipt.json').read_text()),lease=json.loads((O/'lease.json').read_text()),CPU=json.loads((O/'cpu_receipt.json').read_text()),created_epoch=time.time())
(O/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,4,figsize=(16,12))
for j,c in enumerate(conditions):
 for i,k in enumerate(['DIRECT','total_degree_W1','joint_TV','triangle_abs_error']):
  for arm,color in [('CE','tab:blue'),('CE_STRUCT','tab:orange')]:
   cr=[next(x for x in curves if x['condition']==c and x['arm']=='start')]+sorted([x for x in curves if x['condition']==c and x['arm']==arm],key=lambda x:x['offset'])
   axes[j,i].plot([x['offset'] for x in cr],[x[k] for x in cr],marker='o',color=color,label=arm)
  axes[j,i].set_title(c+' / '+k);axes[j,i].set_xlabel('local update offset');axes[j,i].grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Raw conditional graph scores: 20 stratified inputs, 32 paired draws each (not T1000)')
fig.tight_layout();fig.savefig(O/'curves.png',dpi=150);plt.close(fig)
print(json.dumps([dict(condition=x['condition'],offset=x['offset'],DIRECT_delta=x['DIRECT_delta'],MC_SE=x['DIRECT_paired_MC_SE'],input_SE=x['DIRECT_paired_input_SE'],degree=x['total_degree_W1_delta'],joint=x['joint_TV_delta'],triangle=x['triangle_abs_error_delta']) for x in comparisons],indent=2))
