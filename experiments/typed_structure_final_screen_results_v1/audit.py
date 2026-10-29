"""Read-only CPU audit of sealed runs; never constructs or invokes a model."""
import argparse,collections,hashlib,json,math,pathlib,time,itertools
import numpy as np
import networkx as nx
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def eq(a,b):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def tv(a,b):
 na,nb=sum(a.values()),sum(b.values())
 return .5*sum(abs(a[k]/na-b[k]/nb) for k in a.keys()|b.keys()) if na and nb else float(bool(na or nb))
def w1(a,b):
 if not len(a) or not len(b):return None
 n=int(max(max(a),max(b)))+1
 return abs(np.cumsum(np.bincount(a,minlength=n)/len(a)-np.bincount(b,minlength=n)/len(b))).sum()
def stats(types,edges,schema):
 assert edges.shape[0]==2 and np.all(edges[0]<edges[1]) and np.all(edges>=0) and np.all(edges<len(types))
 assert len(set(map(tuple,edges.T)))==edges.shape[1]
 lookup={tuple(x):i for i,x in enumerate(schema)}
 fam=np.array([lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T])
 deg=np.zeros((len(types),len(schema)),dtype=np.int64)
 np.add.at(deg,(edges[0],fam),1);np.add.at(deg,(edges[1],fam),1)
 g=nx.Graph();g.add_nodes_from(range(len(types)));g.add_edges_from(edges.T)
 comps=sorted((sorted(c) for c in nx.connected_components(g)),key=lambda x:(-len(x),x));tr=collections.Counter()
 for u,v in g.edges():
  for w in set(g[u])&set(g[v]):
   if u<v<w:tr[str(tuple(sorted(map(int,types[[u,v,w]]))))]+=1
 total=deg.sum(1);mp2=sum(int(d)*(int(d)-1)//2 for d in total)
 mp3=sum((int(total[u])-1)*(int(total[v])-1)-len(set(g[u])&set(g[v])) for u,v in edges.T)
 a=nx.to_numpy_array(g,dtype=np.int64);common=a@a
 four=sum(int(common[u,v])*(int(common[u,v])-1)//2 for u in range(len(types)) for v in range(u+1,len(types)))//2
 scalars=dict(nodes=len(types),edges=edges.shape[1],components=len(comps),lcc=len(comps[0]),isolates=int((total==0).sum()),maximum_degree=int(max(total)),triangles=sum(tr.values()),four_cycles=four,average_clustering=nx.average_clustering(g))
 return dict(types=types,edges=edges,schema=schema,families=fam,degree=deg,comps=comps,triangles=tr,scalars=scalars,mp=[edges.shape[1],mp2,mp3])
def load(p,reference=False):
 with np.load(p,allow_pickle=False) as z:
  s=stats(z['types'],z['edges'].T if reference else z['edges'],z['schema'])
  if 'families' in z:np.testing.assert_array_equal(z['families'],s['families'])
 return s
def metrics(s,r,row):
 for k,v in s['scalars'].items():eq(v,row['statistics'][k])
 assert [len(c) for c in s['comps']]==row['statistics']['component_sizes']
 assert dict(s['triangles'])==row['statistics']['typed_triangles']
 assert s['mp']==[x['total'] for x in row['statistics']['mp']]
 np.testing.assert_array_equal(np.bincount(s['families'],minlength=10),row['statistics']['family_edges'])
 W=J=0.;missing=False
 for t in range(4):
  a=s['degree'][s['types']==t];b=r['degree'][r['types']==t];weight=float((r['types']==t).mean())
  if not len(a):missing=True;continue
  rel=[i for i,p in enumerate(s['schema']) if t in p]
  W+=weight*np.mean([w1(a[:,i],b[:,i]) for i in rel])
  J+=weight*tv(collections.Counter(map(tuple,a)),collections.Counter(map(tuple,b)))
 d=row['distribution_error'];eq(None if missing else W,d['main_incident4_W1']);eq(None if missing else .4*W,d['all10_compat_W1']);eq(None if missing else J,d['main_joint_degree_TV']);eq(tv(s['triangles'],r['triangles']),d['triangle_composition_TV'])
 for k,v in row['raw'].items():
  rv=row['reference'].get(k);eq(row['absolute_error'][k],None if v is None or rv is None else abs(v-rv))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--public-only',action='store_true');a=p.parse_args()
 run=pathlib.Path(a.run);src=pathlib.Path(a.source);out=pathlib.Path(a.output);assert not out.exists();started=time.time()
 activation=read(run/'activation.json');summary=read(run/'launcher_summary.json');ev=read(run/'unified_evaluation.json');protocol=read(src/'protocol.json')
 source_files={str(f.relative_to(src)):sha(f) for f in sorted(src.rglob('*')) if f.is_file() and not any(x in f.parts for x in ('__pycache__','.pytest_cache','historical_multi_ema','evidence','tests')) and f.suffix in ('.py','.json','.npz') and f.name not in ('cost.json','release.json','source_manifest.json')}
 assert hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()==activation['source_sha256']
 assert summary['status']==ev['status']=='complete' and not summary['primary_failures'] and not summary['unstarted'] and not ev['errors']
 assert len(summary['jobs'])==24 and all(j['exit_code']==0 for j in summary['jobs'])
 assert sha(run/'unified_evaluation.json')==summary['evaluation']['sha256']
 records={(r['dataset'],r['arm'],r['sample_seed'],r['version']):r for r in ev['records']};assert len(records)==32
 refs={d:load(src/'package/assets'/f'{d}.npz',True) for d in protocol['train_seeds']}
 weights=[];training=[];init_states={};inputs={};validation_inputs={};sampling_inputs={};graph_receipts=[]
 for item in summary['jobs']:
  job=item['job'];folder=run/job['directory'];receipt=item['receipt'];bind=read(folder/'binding.json');assert bind['source_sha256']==activation['source_sha256']
  if job['mode']=='train':
   assert receipt['completed_updates']==receipt['ema_updates']==3000
   rows=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()];assert [r['update'] for r in rows]==list(range(1,3001))
   assert all(r['optimizer_steps']==1 and r['ema_updates']==r['update'] and len(r['inputs'])==2 and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and all(x['slots']==(13120 if job['dataset']=='IMDB' else 18336) for x in r['inputs']) for r in rows)
   key=job['dataset'];stream=[(r['input_sha256'],r['t']) for r in rows]
   if key in inputs:assert inputs[key]==stream
   else:inputs[key]=stream
   val=read(folder/'validation.json');assert [x['updates'] for x in val]==[0,1000,2000,3000]
   vi=[(r['t'],r['input_sha256']) for x in val for kind in ('raw','ema_0.99') for r in x[kind]['rows']]
   assert len(set(vi))==3
   if key in validation_inputs:assert validation_inputs[key]==vi
   else:validation_inputs[key]=vi
   training.append(dict(**job,seconds=receipt['finished_epoch']-receipt['started_epoch'],normal2_seconds=sum(r['normal2_seconds'] for r in rows),total_update_seconds=sum(r['total_update_seconds'] for r in rows),peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib']))
   if not a.public_only:
    import torch
    for name,u in [('initial',0),('checkpoint_u1000',1000),('checkpoint_u2000',2000),('final',3000),('readout_raw',3000),('readout_ema_0.99',3000)]:
     rr=read(folder/(name+'.sha256.json'));f=folder/(name+'.pt');assert rr['updates']==u and sha(f)==rr['sha256'];weights.append(dict(path=str(f),bytes=f.stat().st_size,sha256=rr['sha256'],updates=u))
    initial=torch.load(folder/'initial.pt',map_location='cpu');assert initial['updates']==initial['ema']['updates']==0
    assert all(torch.equal(initial['model'][k],v) for k,v in initial['ema']['shadow'].items())
    init_states[(key,job['arm'])]={k:(tuple(v.shape),str(v.dtype),hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in initial['model'].items()}
    final=torch.load(folder/'final.pt',map_location='cpu');assert final['updates']==final['ema']['updates']==3000 and final['ema']['decay']==.99
    for kind in ('raw','ema_0.99'):
     payload=torch.load(folder/('readout_'+kind+'.pt'),map_location='cpu');expected=dict(final['model'])
     if kind!='raw':expected.update(final['ema']['shadow'])
     assert payload['readout']==kind and payload['updates']==3000 and payload['checkpoint_sha256']==sha(folder/'final.pt')
     assert payload['model'].keys()==expected.keys() and all(torch.equal(v,expected[k]) for k,v in payload['model'].items())
    del initial,final,payload,expected
  else:
   assert bind['job']==job and bind['readout']=='ema_0.99'
   assert sha(folder/'graph.npz')==bind['graph_sha256'] and sha(folder/'metrics.json')==bind['metrics_sha256']
   parent=run/job['depends_on'];assert bind['model_sha256']==read(parent/'readout_ema_0.99.sha256.json')['sha256'] and bind['checkpoint_sha256']==read(parent/'final.sha256.json')['sha256']
   steps=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()];assert [x['time'] for x in steps]==list(range(1000,0,-1))
   assert all(x['remaining_time']==x['time']-1 and x['exact_coverage'] and x['blocks']==1 and x['covered_slots']==(13120 if job['dataset']=='IMDB' else 18336) for x in steps)
   key=(job['dataset'],job['sample_seed']);si=(bind['initial_state_sha256'],bind['initial_rng'])
   if key in sampling_inputs:assert sampling_inputs[key]==si
   else:sampling_inputs[key]=si
   s=load(folder/'graph.npz');r=refs[job['dataset']];np.testing.assert_array_equal(s['types'],r['types']);np.testing.assert_array_equal(s['schema'],r['schema'])
   metrics(s,r,records[job['dataset'],job['arm'],job['sample_seed'],'original'])
   derived=run/'derived_lcc_evaluation'/job['id'];mapping=read(derived/'mapping.json');keep=s['comps'][0]
   assert mapping['selected_local_ids']==keep and mapping['source_sha256']==bind['graph_sha256'] and mapping['derived_sha256']==sha(derived/'lcc.npz')
   l=load(derived/'lcc.npz');old=np.full(len(s['types']),-1);old[keep]=range(len(keep));mask=np.isin(s['edges'][0],keep)&np.isin(s['edges'][1],keep)
   np.testing.assert_array_equal(l['types'],s['types'][keep]);np.testing.assert_array_equal(l['edges'],old[s['edges'][:,mask]])
   metrics(l,r,records[job['dataset'],job['arm'],job['sample_seed'],'lcc'])
   graph_receipts.append(dict(**job,graph_sha256=bind['graph_sha256'],seconds=bind['seconds'],lcc_nodes=len(keep),lcc_edges=l['scalars']['edges']))
 shared=0
 for dataset in protocol['train_seeds']:
  for left,right in itertools.combinations(protocol['arms'],2):
   if a.public_only:continue
   x,y=init_states[dataset,left],init_states[dataset,right]
   for k in x.keys()&y.keys():
    if x[k][:2]==y[k][:2]:assert x[k]==y[k],(dataset,left,right,k);shared+=1
 checked=0
 def val(d,arm,s,v,g,m):return records[d,arm,s,v][g].get(m)
 for row in ev['cell_means']:
  vs=[val(row['dataset'],row['arm'],s,row['version'],row['group'],row['metric']) for s in protocol['sample_seeds']]
  eq(row['value'],None if None in vs else sum(vs)/2);checked+=1
 for row in ev['paired']+ev['paired_means']:
  left,right=row['contrast'].split('-');samples=[row['sample_seed']] if 'sample_seed' in row else protocol['sample_seeds'];diffs=[]
  for s in samples:
   x=val(row['dataset'],left,s,row['version'],row['group'],row['metric']);y=val(row['dataset'],right,s,row['version'],row['group'],row['metric']);diffs.append(None if x is None or y is None else x-y)
  eq(row['difference'],None if None in diffs else sum(diffs)/len(diffs));checked+=1
 for row in ev['lcc_minus_original']:
  x=val(row['dataset'],row['arm'],row['sample_seed'],'lcc',row['group'],row['metric']);y=val(row['dataset'],row['arm'],row['sample_seed'],'original',row['group'],row['metric']);eq(row['difference'],None if x is None or y is None else x-y);checked+=1
 result=dict(status='passed',started_epoch=started,finished_epoch=time.time(),seconds=time.time()-started,scientific_finished_epoch=summary['finished_epoch'],post_budget_audit=True,public_only=a.public_only,models=8,updates=24000,chains=16,graph_versions=32,weights=weights,training=training,graphs=graph_receipts,common_initial_tensor_comparisons=shared,numeric_aggregate_checks=checked,source_sha256=activation['source_sha256'],scope='independent discrete counts, four-cycles, MP1/2/3 totals, clustering, fixed-reference W1/joint-TV, triangle-TV, LCC mapping; all aggregate arithmetic; sealed training/input/chain contracts',limits=['MP/closure composition retained from frozen evaluator, not independently enumerated here','CPU/GPU historical failure not changed','one training seed per dataset, two generated samples','late audit/publication authorized after disk recovery; original science clock unchanged'])
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','seconds','models','chains','graph_versions','common_initial_tensor_comparisons','numeric_aggregate_checks']}))
if __name__=='__main__':main()
