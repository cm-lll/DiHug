"""Independent CPU audit of the six completed tasks and read-only cross-batch union."""
import argparse,hashlib,importlib.util,json,math,pathlib,sys,time
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True);p.add_argument('--public-only',action='store_true');a=p.parse_args()
 start=time.time();run=pathlib.Path(a.run);src=pathlib.Path(a.source)
 spec=importlib.util.spec_from_file_location('independent_counts',a.library);L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
 read,sha,eq=L.read,L.sha,L.eq
 sys.path.insert(0,str(src));import runtime_contract as contract
 protocol=read(src/'protocol.json');act=read(run/'activation.json');summary=read(run/'launcher_summary.json');ev=read(run/'unified_evaluation.json')
 assert contract.source_binding()[1]==act['source_sha256']
 assert summary['status']==ev['status']=='complete' and not summary['primary_failures'] and not summary['unstarted'] and not ev['errors'] and not ev['cross_batch_errors'] and not ev['missing_identities'] and ev['comparison_status']=='complete'
 assert len(summary['jobs'])==18 and all(x['exit_code']==0 for x in summary['jobs'])
 assert sha(run/'unified_evaluation.json')==summary['evaluation']['sha256']
 records={(r['dataset'],r['train_seed'],r['arm'],r['sample_seed'],r['version']):r for r in ev['records']};assert len(records)==len(ev['records'])==52
 refs={d:L.load(src/'package/assets'/f'{d}.npz',True) for d in protocol['train_seeds']}
 weights=[];training=[];graphs=[];streams={};validation={};sample_streams={};initials={};pair_checks=[]
 for item in summary['jobs']:
  job=item['job'];folder=run/job['directory'];receipt=item['receipt'];bind=read(folder/'binding.json');assert bind['source_sha256']==act['source_sha256']
  if job['mode']=='train':
   rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
   assert [r['update'] for r in rows]==list(range(1,3001)) and receipt['completed_updates']==receipt['ema_updates']==3000
   assert all(r['optimizer_steps']==1 and r['ema_updates']==r['update'] and len(r['inputs'])==2 and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and all(i['slots']==(13120 if job['dataset']=='IMDB' else 18336) for i in r['inputs']) for r in rows)
   key=(job['dataset'],job['train_seed']);st=[(r['input_sha256'],r['t']) for r in rows]
   if key in streams:assert st==streams[key];pair_checks.append(dict(kind='actual_training_inputs',dataset=key[0],train_seed=key[1],updates=3000))
   else:streams[key]=st
   val=read(folder/'validation.json');assert [x['updates'] for x in val]==[0,1000,2000,3000]
   vi=[(r['t'],r['input_sha256']) for x in val for k in ('raw','ema_0.99') for r in x[k]['rows']];assert len(set(vi))==3
   if key in validation:assert vi==validation[key]
   else:validation[key]=vi
   training.append(dict(job=job,seconds=receipt['finished_epoch']-receipt['started_epoch'],normal2_seconds=sum(x['normal2_seconds'] for x in rows),total_update_seconds=sum(x['total_update_seconds'] for x in rows),peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib']))
   if not a.public_only:
    import torch
    for name,u in [('initial',0),('checkpoint_u1000',1000),('checkpoint_u2000',2000),('final',3000),('readout_raw',3000),('readout_ema_0.99',3000)]:
     f=folder/(name+'.pt');b=read(folder/(name+'.sha256.json'));assert b['updates']==u and sha(f)==b['sha256'];weights.append(dict(path=str(f),bytes=f.stat().st_size,sha256=b['sha256'],updates=u))
    ini=torch.load(folder/'initial.pt',map_location='cpu');assert ini['updates']==ini['ema']['updates']==0 and not ini['optimizer']['state']
    assert all(torch.equal(ini['model'][k],v) for k,v in ini['ema']['shadow'].items())
    initials[(job['dataset'],job['train_seed'],job['arm'])]=ini['model']
    final=torch.load(folder/'final.pt',map_location='cpu');assert final['updates']==final['ema']['updates']==3000 and final['ema']['decay']==.99
    for kind in ('raw','ema_0.99'):
     export=torch.load(folder/('readout_'+kind+'.pt'),map_location='cpu');expected=dict(final['model'])
     if kind!='raw':expected.update(final['ema']['shadow'])
     assert export['readout']==kind and export['updates']==3000 and export['checkpoint_sha256']==sha(folder/'final.pt')
     assert export['model'].keys()==expected.keys() and all(torch.equal(v,expected[k]) for k,v in export['model'].items())
    del ini,final,export,expected
  else:
   assert bind['job']==job and bind['readout']=='ema_0.99'
   assert sha(folder/'graph.npz')==bind['graph_sha256'] and sha(folder/'metrics.json')==bind['metrics_sha256']
   par=run/job['depends_on'];assert bind['model_sha256']==read(par/'readout_ema_0.99.sha256.json')['sha256'] and bind['checkpoint_sha256']==read(par/'final.sha256.json')['sha256']
   rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
   assert [r['time'] for r in rows]==list(range(1000,0,-1)) and all(r['remaining_time']==r['time']-1 and r['exact_coverage'] and r['blocks']==1 and r['covered_slots']==(13120 if job['dataset']=='IMDB' else 18336) for r in rows)
   key=(job['dataset'],job['sample_seed']);ss=(bind['initial_state_sha256'],bind['initial_rng'])
   if key in sample_streams:assert ss==sample_streams[key]
   else:sample_streams[key]=ss
   s=L.load(folder/'graph.npz');ref=refs[job['dataset']]
   L.np.testing.assert_array_equal(s['types'],ref['types']);L.np.testing.assert_array_equal(s['schema'],ref['schema'])
   rk=(job['dataset'],job['train_seed'],job['arm'],job['sample_seed']);L.metrics(s,ref,records[(*rk,'original')])
   derived=run/'derived_lcc_evaluation'/job['id'];mp=read(derived/'mapping.json');keep=s['comps'][0]
   assert mp['selected_local_ids']==keep and mp['source_sha256']==bind['graph_sha256'] and mp['derived_sha256']==sha(derived/'lcc.npz')
   l=L.load(derived/'lcc.npz');old=L.np.full(len(s['types']),-1);old[keep]=range(len(keep));mask=L.np.isin(s['edges'][0],keep)&L.np.isin(s['edges'][1],keep)
   L.np.testing.assert_array_equal(l['types'],s['types'][keep]);L.np.testing.assert_array_equal(l['edges'],old[s['edges'][:,mask]])
   L.metrics(l,ref,records[(*rk,'lcc')]);graphs.append(dict(job=job,sha256=bind['graph_sha256'],seconds=bind['seconds'],lcc_nodes=len(keep),lcc_edges=l['scalars']['edges']))
 # Compare actual scientific theta0 files; do not rebuild any model.
 if not a.public_only:
  import torch
  for d,new in protocol['new_train_seeds'].items():
   early=initials[d,new,'E_earlyCN'];late=initials[d,new,'E_lateCN'];count=0
   for k,v in early.items():
    if k=='cn_edge_projection.weight':assert torch.count_nonzero(v)==0
    else:
     expected=late[k][:,:739] if k=='joint_head.0.weight' else late[k]
     assert torch.equal(v,expected),(d,k);count+=1
   pair_checks.append(dict(kind='early_late_actual_theta0',dataset=d,train_seed=new,exact_tensors=count,projection_zero=True))
  baseline=read(src/'baseline_bindings.json')
  for d,old in protocol['train_seeds'].items():
   b=baseline['initial'][d];assert sha(b['path'])==b['sha256'];late=torch.load(b['path'],map_location='cpu')['model'];early=initials[d,old,'D_earlyCN'];count=0
   assert not any('role_' in k for k in early)
   for k,v in early.items():
    if k=='cn_edge_projection.weight':assert torch.count_nonzero(v)==0
    else:assert torch.equal(v,late[k][:,:739] if k=='joint_head.0.weight' else late[k]),(d,k);count+=1
   pair_checks.append(dict(kind='D_sealed_theta0_migration',dataset=d,train_seed=old,exact_tensors=count))
 # Compare original-seed input streams with the protected completed E-early batch.
 cross=read(src/'cross_batch_binding.json');protected=pathlib.Path(cross['run'])
 if not a.public_only and protected.exists():
  assert sha(protected/'activation.json')==cross['activation_sha256']
  for d,seed in protocol['train_seeds'].items():
   oldrows=[json.loads(x) for x in (protected/f'{d}_s{seed}_E'/'training.jsonl').read_text().splitlines()]
   assert [(x['input_sha256'],x['t']) for x in oldrows]==streams[d,seed]
   for sample in protocol['sample_seeds']:
    b=read(protected/f'{d}_s{seed}_E/ema_0.99/sample_{sample}/binding.json')
    assert (b['initial_state_sha256'],b['initial_rng'])==sample_streams[d,sample]
   pair_checks.append(dict(kind='protected_E_actual_input_pairing',dataset=d,train_seed=seed,updates=3000,sampling_seeds=protocol['sample_seeds']))
 # Existing external records must be byte-for-value identical after declared naming normalization.
 import cross_batch
 if not a.public_only:
  external,inventory=cross_batch.external_records()
  for row in external:assert records[cross_batch.key(row)]==row
 else:
  # Public audit recomputes the new graphs and all stored arithmetic without
  # requiring private historical absolute paths. Provenance was checked by
  # the private server audit and its manifest-bound receipt is retained.
  external=[r for r in ev['records'] if r.get('batch')!='new_six']
 assert len(external)==28
 checked=0
 def value(r,arm,sample,version=None):return records[r['dataset'],r['train_seed'],arm,sample,version or r['version']][r['group']].get(r['metric'])
 for row in ev['cell_means']:
  vals=[value(row,row['arm'],s) for s in protocol['sample_seeds']];eq(row['value'],None if None in vals else sum(vals)/2);checked+=1
 for row in ev['paired']+ev['paired_means']:
  left,right=row['contrast'].split('-');ss=[row['sample_seed']] if 'sample_seed' in row else protocol['sample_seeds'];ds=[]
  for sample in ss:
   x=value(row,left,sample);y=value(row,right,sample);ds.append(None if x is None or y is None else x-y)
  eq(row['difference'],None if None in ds else sum(ds)/len(ds));checked+=1
 for row in ev['lcc_minus_original']:
  x=value(row,row['arm'],row['sample_seed'],'lcc');y=value(row,row['arm'],row['sample_seed'],'original');eq(row['difference'],None if x is None or y is None else x-y);checked+=1
 result=dict(status='passed',started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,public_only=a.public_only,models=6,updates=18000,chains=12,new_graph_versions=24,combined_models=8,combined_chains=16,external_records=28,numeric_aggregate_checks=checked,weights=weights,training=training,graphs=graphs,pair_checks=pair_checks,source_sha256=act['source_sha256'],scientific_complete_epoch=summary['finished_epoch'],scope='CPU file/state audit without model construction or forward; independent discrete graph statistics and aggregation arithmetic; actual theta0 and external provenance checked in private mode only',limits=['MP/closure composition inherited from frozen evaluator, not independently enumerated','Two development training seeds for E early/late, one seed for D early/RolX early comparison','CPU/GPU equivalence not asserted','Public mode does not reopen private weights or historical absolute paths; it recomputes new graph statistics and all archived aggregation arithmetic'])
 with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:result[k] for k in ['status','seconds','models','chains','numeric_aggregate_checks']}))
if __name__=='__main__':main()
