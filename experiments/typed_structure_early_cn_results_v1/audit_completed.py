"""CPU-only independent audit of the completed two-task early-CN batch."""
import argparse, hashlib, importlib.util, json, math, pathlib, sys, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--source',required=True)
    p.add_argument('--audit-library',required=True);p.add_argument('--output',required=True)
    p.add_argument('--public-only',action='store_true');a=p.parse_args()
    started=time.time();run=pathlib.Path(a.run);src=pathlib.Path(a.source)
    spec=importlib.util.spec_from_file_location('independent_audit_library',a.audit_library)
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    read,sha,eq=lib.read,lib.sha,lib.eq
    sys.path.insert(0,str(src));import runtime_contract as contract
    act=read(run/'activation.json');summ=read(run/'launcher_summary.json')
    ev=read(run/'unified_evaluation.json');protocol=read(src/'protocol.json')
    assert contract.source_binding()[1]==act['source_sha256']
    assert summ['status']==ev['status']=='complete'
    assert not summ['primary_failures'] and not summ['unstarted'] and not ev['errors']
    assert len(summ['jobs'])==6 and all(x['exit_code']==0 for x in summ['jobs'])
    assert sha(run/'unified_evaluation.json')==summ['evaluation']['sha256']
    records={(r['dataset'],r['arm'],r['sample_seed'],r['version']):r for r in ev['records']}
    assert len(records)==len(ev['records'])==24
    refs={d:lib.load(src/'package/assets'/f'{d}.npz',True) for d in protocol['train_seeds']}
    weights=[];graphs=[];training=[];input_contracts={}
    for item in summ['jobs']:
        job=item['job'];folder=run/job['directory'];receipt=item['receipt'];binding=read(folder/'binding.json')
        assert binding['source_sha256']==act['source_sha256']
        if job['mode']=='train':
            rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            assert [x['update'] for x in rows]==list(range(1,3001))
            assert receipt['completed_updates']==receipt['ema_updates']==3000
            assert all(x['optimizer_steps']==1 and x['ema_updates']==x['update'] and len(x['inputs'])==2 and math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']) and all(y['slots']==(13120 if job['dataset']=='IMDB' else 18336) for y in x['inputs']) for x in rows)
            val=read(folder/'validation.json');assert [x['updates'] for x in val]==[0,1000,2000,3000]
            vi=[(r['t'],r['input_sha256']) for x in val for k in ('raw','ema_0.99') for r in x[k]['rows']]
            assert len(set(vi))==3
            input_contracts[job['dataset']]=dict(training_inputs=len(rows),validation_inputs=vi)
            training.append(dict(job=job,seconds=receipt['finished_epoch']-receipt['started_epoch'],peak_rss_mib=receipt['peak_rss_mib'],peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib']))
            if not a.public_only:
                import torch
                for name,u in [('initial',0),('checkpoint_u1000',1000),('checkpoint_u2000',2000),('final',3000),('readout_raw',3000),('readout_ema_0.99',3000)]:
                    f=folder/(name+'.pt');rr=read(folder/(name+'.sha256.json'))
                    assert rr['updates']==u and sha(f)==rr['sha256']
                    weights.append(dict(path=str(f),bytes=f.stat().st_size,sha256=rr['sha256'],updates=u))
                initial=torch.load(folder/'initial.pt',map_location='cpu')
                assert initial['updates']==initial['ema']['updates']==0
                assert all(torch.equal(initial['model'][k],v) for k,v in initial['ema']['shadow'].items())
                final=torch.load(folder/'final.pt',map_location='cpu')
                assert final['updates']==final['ema']['updates']==3000 and final['ema']['decay']==.99
                for kind in ('raw','ema_0.99'):
                    payload=torch.load(folder/('readout_'+kind+'.pt'),map_location='cpu')
                    expected=dict(final['model'])
                    if kind!='raw':expected.update(final['ema']['shadow'])
                    assert payload['readout']==kind and payload['updates']==3000 and payload['checkpoint_sha256']==sha(folder/'final.pt')
                    assert payload['model'].keys()==expected.keys() and all(torch.equal(v,expected[k]) for k,v in payload['model'].items())
                del initial,final,payload,expected
        else:
            assert binding['job']==job and binding['readout']=='ema_0.99'
            assert sha(folder/'graph.npz')==binding['graph_sha256'] and sha(folder/'metrics.json')==binding['metrics_sha256']
            parent=run/job['depends_on']
            assert binding['model_sha256']==read(parent/'readout_ema_0.99.sha256.json')['sha256']
            assert binding['checkpoint_sha256']==read(parent/'final.sha256.json')['sha256']
            steps=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
            assert [x['time'] for x in steps]==list(range(1000,0,-1))
            assert all(x['remaining_time']==x['time']-1 and x['exact_coverage'] and x['blocks']==1 and x['covered_slots']==(13120 if job['dataset']=='IMDB' else 18336) for x in steps)
            s=lib.load(folder/'graph.npz');ref=refs[job['dataset']]
            lib.np.testing.assert_array_equal(s['types'],ref['types']);lib.np.testing.assert_array_equal(s['schema'],ref['schema'])
            lib.metrics(s,ref,records[job['dataset'],'E_earlyCN',job['sample_seed'],'original'])
            derived=run/'derived_lcc_evaluation'/job['id'];mapping=read(derived/'mapping.json');keep=s['comps'][0]
            assert mapping['selected_local_ids']==keep and mapping['source_sha256']==binding['graph_sha256'] and mapping['derived_sha256']==sha(derived/'lcc.npz')
            l=lib.load(derived/'lcc.npz');old=lib.np.full(len(s['types']),-1);old[keep]=range(len(keep))
            mask=lib.np.isin(s['edges'][0],keep)&lib.np.isin(s['edges'][1],keep)
            lib.np.testing.assert_array_equal(l['types'],s['types'][keep]);lib.np.testing.assert_array_equal(l['edges'],old[s['edges'][:,mask]])
            lib.metrics(l,ref,records[job['dataset'],'E_earlyCN',job['sample_seed'],'lcc'])
            graphs.append(dict(job=job,sha256=binding['graph_sha256'],seconds=binding['seconds'],lcc_nodes=len(keep),lcc_edges=l['scalars']['edges']))
    checked=0
    def value(d,arm,s,v,g,m):return records[d,arm,s,v][g].get(m)
    for row in ev['cell_means']:
        vs=[value(row['dataset'],row['arm'],s,row['version'],row['group'],row['metric']) for s in protocol['sample_seeds']]
        eq(row['value'],None if None in vs else sum(vs)/2);checked+=1
    for row in ev['paired']+ev['paired_means']:
        left,right=row['contrast'].split('-');samples=[row['sample_seed']] if 'sample_seed' in row else protocol['sample_seeds'];ds=[]
        for s in samples:
            x=value(row['dataset'],left,s,row['version'],row['group'],row['metric']);y=value(row['dataset'],right,s,row['version'],row['group'],row['metric']);ds.append(None if x is None or y is None else x-y)
        eq(row['difference'],None if None in ds else sum(ds)/len(ds));checked+=1
    for row in ev['lcc_minus_original']:
        x=value(row['dataset'],row['arm'],row['sample_seed'],'lcc',row['group'],row['metric']);y=value(row['dataset'],row['arm'],row['sample_seed'],'original',row['group'],row['metric'])
        eq(row['difference'],None if x is None or y is None else x-y);checked+=1
    result=dict(status='passed',started_epoch=started,finished_epoch=time.time(),seconds=time.time()-started,public_only=a.public_only,models=2,updates=6000,chains=4,new_graph_versions=8,all_records_with_controls=24,weights=weights,training=training,graphs=graphs,input_contracts=input_contracts,numeric_aggregate_checks=checked,source_sha256=act['source_sha256'],scientific_complete_epoch=summ['finished_epoch'],limits=['One training seed per dataset and two samples; new-seed replication still running in separate batch','MP/closure composition inherited from frozen evaluator, not independently enumerated in this audit','CPU/GPU equivalence not asserted; no model constructed, forward or sampling invoked'])
    out=pathlib.Path(a.output)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['status','seconds','models','chains','new_graph_versions','numeric_aggregate_checks']}))
if __name__=='__main__':main()
