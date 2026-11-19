"""CPU summary and checks for the fixed eight-step window."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sys,hashlib,math
from pathlib import Path
OUT=Path(sys.argv[1]);ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
r=read(OUT/'receipt.json');lease=read(OUT/'lease.json');p=read(OUT/'protocol.json')
assert r['status']=='complete' and (r['updates'],r['forwards'])==(16,17)
assert lease['returncode']==0 and lease['gpu_hours']<=p['gpu_hours_limit'] and lease['end_epoch']<=p['science_stop_epoch']
source=read(ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json')
assert all(sha(f)==h for f,h in source.items())
binding=read(OUT/'asset_binding.json');assert all(sha(x['parent'])==x['parent_sha256'] for x in binding)
rows=read(OUT/'probes.json');updates=read(OUT/'updates.json');base=rows[0]
idx={(x['arm'],x['offset']):x for x in rows};paired=[];relations=[]
for step in range(1,9):
 a,b=idx['CE',step],idx['PS',step]
 u=[x for x in updates if x['offset']==step];assert len(u)==2 and u[0]['input_sha256']==u[1]['input_sha256']
 rec={'offset':step,'update':2000+step,'training_t':u[0]['t']}
 for m in ['CE','positive_CE','negative_CE','expected_edges']:
  rec[m+'_CE']=a[m];rec[m+'_PS']=b[m];rec[m+'_gap']=b[m]-a[m]
  prior=base if step==1 else idx['CE',step-1];priorb=base if step==1 else idx['PS',step-1]
  rec[m+'_change_CE']=a[m]-prior[m];rec[m+'_change_PS']=b[m]-priorb[m]
  rec[m+'_gap_change']=rec[m+'_change_PS']-rec[m+'_change_CE']
 paired.append(rec)
 for x,y in zip(a['relations'],b['relations']):
  assert x['family']==y['family']
  z={k:x[k] for k in ['family','types','slots','positive_count','negative_count']};z.update(offset=step,update=2000+step,training_t=u[0]['t'])
  for m in ['positive_CE','negative_CE','expected_edges','missed_positive_mass','false_positive_mass','expected_relation_degree_MAE']:
   z[m+'_CE']=x[m];z[m+'_PS']=y[m];z[m+'_gap']=None if x[m] is None else y[m]-x[m]
  relations.append(z)
dump('paired_steps.csv',paired);dump('paired_relations.csv',relations);dump('training_updates.csv',updates)
dump('validation_total.csv',[{k:v for k,v in x.items() if k!='relations'} for x in rows])
dump('validation_relations.csv',[dict(arm=x['arm'],offset=x['offset'],**z) for x in rows for z in x['relations']])
summary={'status':'passed','source_files_unchanged':len(source),'parent_hash_unchanged':True,'gpu_hours':lease['gpu_hours'],'peak_reserved_MiB':r['peak_reserved_MiB'],'steps':paired,'endpoint_relations':[x for x in relations if x['offset']==8],'gates':r['gates'],'limits':['one development seed','PS-trained parent for both local arms','fixed t500 raw only','no causal separation of parameters and moments','not generation']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(10,7))
for axis,metric in zip(ax.flat,['CE','positive_CE','negative_CE','expected_edges']):
 for arm in ['CE','PS']:
  q=[base]+[idx[arm,i] for i in range(1,9)]
  axis.plot([2000+i for i in range(9)],[x[metric] for x in q],marker='o',label=arm)
 axis.set_title(metric);axis.set_xlabel('temporary update');axis.grid(alpha=.2);axis.legend()
fig.suptitle('PubMed 0422: fixed t500 raw conditional prediction, not generated graphs')
fig.tight_layout();fig.savefig(OUT/'paired_steps.png',dpi=150);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False))
