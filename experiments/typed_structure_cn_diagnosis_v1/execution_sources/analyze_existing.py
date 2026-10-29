"""Read existing audited metrics and validation; no model or graph evaluation."""
import pathlib,json,csv,collections,time
B=pathlib.Path(__file__).resolve().parent
ROOT=B.parents[1];RUN=ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/run'
SRC=ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_final_screen_v1'
def read(p):return json.loads(p.read_text())
def write(name,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (B/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mean(xs):return sum(xs)/len(xs) if xs and all(x is not None for x in xs) else None
def diff(x,y):return None if x is None or y is None else x-y
e=read(RUN/'unified_evaluation.json');cfg=read(SRC/'package/config.json');families=list(__import__('itertools').combinations_with_replacement(range(4),2))
rr=[];tr=[];cl=[];vr=[];tot=[];diffs=[];tail=[]
for row in e['records']:
 d=row['dataset'];names=cfg['datasets'][d]['type_names'];s=row['statistics'];identity={k:row[k] for k in ['dataset','arm','train_seed','sample_seed','version']};ref=read(RUN/'derived_lcc_evaluation'/f'{d}_reference.json')['statistics']
 tot.append(dict(**identity,edges=s['edges'],reference_edges=ref['edges'],W1=row['comparison']['main_incident4_W1'],joint_TV=row['comparison']['main_joint_degree_TV'],triangles=s['triangles'],reference_triangles=ref['triangles'],lcc=s['lcc']))
 for f,(a,b) in enumerate(families):
  rr.append(dict(**identity,family=f,relation=names[a]+'--'+names[b],edges=s['family_edges'][f],reference_edges=ref['family_edges'][f],signed_error=s['family_edges'][f]-ref['family_edges'][f],absolute_error=abs(s['family_edges'][f]-ref['family_edges'][f])))
 for t,name in enumerate(names):
  sp=s['typed_degree'][name];rp=ref['typed_degree'][name]
  tail.append(dict(**identity,type=name,sorted_degrees=json.dumps(sp['sorted_degrees']),reference_sorted_degrees=json.dumps(rp['sorted_degrees'])))
  for k in ['maximum','second','zero','low_le1','mean','p90','p99','gini_standard','top5_degree_share']:
   tr.append(dict(**identity,type=name,metric=k,value=sp[k],reference=rp[k],error=None if sp[k] is None or rp[k] is None else abs(sp[k]-rp[k])))
  for k,v in [('total_degree_W1',row['comparison']['type_total_degree_W1'][name]),('joint_degree_TV',row['comparison']['joint_TV_by_type'][name])]:tr.append(dict(**identity,type=name,metric=k,value=v,reference=0,error=v))
  for f,pair in enumerate(families):
   if t not in pair:continue
   q=sp['by_relation'][f];qr=rp['by_relation'][f]
   for k in ['mean','max','zero','low_le1','p90','p99']:
    tr.append(dict(**identity,type=name,metric=f'relation{f}_{k}',value=q[k],reference=qr[k],error=None if q[k] is None or qr[k] is None else abs(q[k]-qr[k])))
 for key,value in s['typed_twohop_closure'].items():
  a,b,c=__import__('ast').literal_eval(key);r=ref['typed_twohop_closure'][key]
  cl.append(dict(**identity,pattern=key,label=names[a]+'--'+names[b]+'--'+names[c],**value,**{'reference_'+k:v for k,v in r.items()}))
for d in ['IMDB','PubMed']:
 for arm in ['A','C','D','E']:
  seed=2026102911 if d=='IMDB' else 2026102912;folder=RUN/f'{d}_s{seed}_{arm}'
  for node in read(folder/'validation.json'):
   for kind in ['raw','ema_0.99']:
    for x in node[kind]['rows']:
     for y in x['family']:
      f=y['family'];a,b=families[f];names=cfg['datasets'][d]['type_names']
      vr.append(dict(dataset=d,arm=arm,train_seed=seed,update=node['updates'],readout=kind,t=x['t'],input_sha256=x['input_sha256'],total_CE=x['ce_sum_over_S'],relation=names[a]+'--'+names[b],p0_minus_reference=y['expected_edges']-y['true_edges'],q_available=False,**y))
for d,left,right in [('PubMed','E','C'),('IMDB','D','A'),('IMDB','E','D')]:
 for f in range(10):
  rows=lambda a:[x for x in rr if x['dataset']==d and x['arm']==a and x['family']==f and x['version']=='original']
  a,b=rows(left),rows(right);diffs.append(dict(dataset=d,contrast=left+'-'+right,family=f,relation=a[0]['relation'],reference=a[0]['reference_edges'],left=mean([x['edges'] for x in a]),right=mean([x['edges'] for x in b]),difference=mean([x['edges'] for x in a])-mean([x['edges'] for x in b]),absolute_error_difference=mean([x['absolute_error'] for x in a])-mean([x['absolute_error'] for x in b])))
for name,rows in [('graph_totals.csv',tot),('graph_relations.csv',rr),('typed_degrees_and_joint.csv',tr),('degree_spectra.csv',tail),('typed_closure.csv',cl),('fixed_validation.csv',vr),('relation_contrasts.csv',diffs)]:write(name,rows)
summary=dict(status='complete',read_only=True,graph_records=len(e['records']),validation_relation_rows=len(vr),source_run=str(RUN),missing=['fixed validation saved aggregates, not original logits/p0/q arrays','validation ordinary state arrays not separately persisted; frozen seed recipe and input hashes available','no neuron/CN intermediate attribution collected'],relation_contrasts=diffs)
(B/'existing_summary.json').write_text(json.dumps(summary,indent=2))
print('PubMed E-C',json.dumps(diffs[:10],indent=2))
for t in [1,500,1000]:
 print('PubMed final EMA t',t)
 for arm in ['A','C','D','E']:
  r=[x for x in vr if x['dataset']=='PubMed' and x['arm']==arm and x['update']==3000 and x['readout']=='ema_0.99' and x['t']==t];print(arm,'p0',sum(x['expected_edges'] for x in r),'CE',r[0]['total_CE'],'by_relation',[(x['relation'],round(x['p0_minus_reference'],3)) for x in r])
