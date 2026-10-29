"""New runtime-only CPU checks; no optimizer, CUDA, forward or chain."""
import pathlib,json,time,os,copy
import pytest
import runtime_contract as rc
import cross_batch as cb
import run

def test_queue_exact_scope():
    q=rc.jobs();t=q['training'];s=q['sampling']
    assert len(t)==6 and len(s)==12
    assert len({j['id'] for j in t+s})==18
    assert len({j['directory'] for j in t+s})==18
    assert {j['slot'] for j in t}==set(range(6))
    assert all(j['updates']==3000 for j in t)
    assert all(j['T']==1000 and j['readout']=='ema_0.99' for j in s)
    for j in t:
        children=[x for x in s if x['depends_on']==j['id']]
        assert len(children)==2 and {x['sample_seed'] for x in children}==set(rc.SAMPLES)
        assert all(x['slot']==j['slot'] and x['train_seed']==j['train_seed'] and x['arm']==j['arm'] for x in children)
        assert not (j['arm']=='E_earlyCN' and j['train_seed']==rc.TRAIN_SEEDS[j['dataset']])

def simulate(tmp_path,monkeypatch,fail=False,stop=False):
    out=tmp_path/'run';out.mkdir();approval=tmp_path/'fake.json';rc.save(approval,{'not_authorized_test_only':True})
    ledger=out/'activation.json';rc.save(ledger,dict(start_epoch=time.time(),deadline_epoch=time.time()+2000,source_sha256='mock'))
    if stop:(out/'STOP').touch()
    starts=[];finished=[]
    class Process:
        pid=999999999
        def __init__(self,job):self.job=job;self.returncode=None;self.count=0
        def poll(self):
            self.count+=1
            if self.job['slot']==1 and self.job['mode']=='train' and self.count<4:return None
            if self.returncode is None:
                self.returncode=3 if fail and self.job['slot']==0 else 0
                folder=out/self.job['directory'];folder.mkdir(parents=True,exist_ok=True)
                rc.save(folder/'summary.json',dict(status='failed' if self.returncode else 'complete',completed_updates=3000,completed_chains=1,error='sentinel_original_failure' if self.returncode else None))
                finished.append(self.job['id'])
            return self.returncode
        def wait(self,timeout=None):self.returncode=-15;return self.returncode
    def spawn(args,**kw):
        assert 'run_worker.py' in args[1]
        ticket=json.loads(os.read(kw['pass_fds'][0],100000));job=ticket['job']
        starts.append((job,finished.copy()));return Process(job)
    monkeypatch.setattr(run.os,'killpg',lambda *a:None)
    code=run.supervise(out,approval,ledger,[str(x) for x in range(6)],spawn=spawn,summarizer=lambda *a:dict(status='complete'),poll=0)
    return code,starts,rc.read(out/'launcher_summary.json')

def test_per_slot_no_global_barrier(tmp_path,monkeypatch):
    code,starts,r=simulate(tmp_path,monkeypatch)
    assert code==0 and len(starts)==18 and r['status']=='complete'
    slow=next(j['id'] for j in rc.jobs()['training'] if j['slot']==1)
    first=next((j,done) for j,done in starts if j['mode']=='sample' and j['slot']==0)
    assert slow not in first[1]
    for j,done in starts:
        if j['mode']=='sample':assert j['depends_on'] in done

def test_failure_no_restart_and_original_reason(tmp_path,monkeypatch):
    code,starts,r=simulate(tmp_path,monkeypatch,fail=True)
    assert code==2 and r['status']=='incomplete' and r['primary_failures']
    assert len({j['id'] for j,_ in starts})==len(starts)==6
    assert 'sentinel_original_failure' in json.dumps(r)
    assert len(r['unstarted'])==12

def test_stop_no_spawn(tmp_path,monkeypatch):
    code,starts,r=simulate(tmp_path,monkeypatch,stop=True)
    assert code==2 and not starts and len(r['unstarted'])==18

def records():
    rows=[]
    for d,seed,arm in cb.cells():
        # Seed-dependent offsets make accidental cross-seed pairing visible.
        offset=100 if seed==rc.PROTOCOL['new_train_seeds'][d] else 0
        val={'D_earlyCN':8,'D_lateCN':10,'E_earlyCN':6,'E_lateCN':11,'C':7}[arm]+offset
        for sample in rc.SAMPLES:
            for version in ('original','lcc'):
                raw=val+(1 if sample==rc.SAMPLES[1] else 0)
                rows.append(dict(dataset=d,train_seed=seed,arm=arm,sample_seed=sample,version=version,raw={'edge':raw},absolute_error={'edge':abs(raw-9)},distribution_error={'W1':val/100,'null':None}))
    return rows

def test_cross_batch_seed_matching_and_direction():
    r=cb.aggregate(records(),{})
    assert r['comparison_status']=='complete' and len(r['missing_identities'])==0
    for x in r['paired']:
        if x['group']=='raw' and x['contrast']=='E_earlyCN-E_lateCN':assert x['difference']==-5
    means=[x for x in r['cell_means'] if x['dataset']=='IMDB' and x['arm']=='E_earlyCN' and x['version']=='original' and x['group']=='absolute_error']
    assert {x['train_seed']:x['value'] for x in means}=={2026102911:2.5,2026102931:97.5}

def test_missing_no_partial_mean_and_null():
    rows=records();gone=rows.pop(0);r=cb.aggregate(rows,{})
    assert r['comparison_status']=='incomplete' and len(r['missing_identities'])==1
    target=[x for x in r['cell_means'] if (x['dataset'],x['train_seed'],x['arm'],x['version'])==(gone['dataset'],gone['train_seed'],gone['arm'],gone['version'])]
    assert all(x['value'] is None for x in target)
    assert all(x['difference'] is None for x in r['paired'] if x['metric']=='null')

def test_duplicate_and_wrong_seed_rejected():
    rows=records()
    with pytest.raises(ValueError):cb.aggregate(rows+[rows[0]],{})
    rows[0]['train_seed']=123
    with pytest.raises(ValueError):cb.aggregate(rows,{})

def test_readout_identity_hash_and_completion(tmp_path):
    j=rc.jobs()['sampling'][0];p=rc.readout_path(tmp_path,j);p.parent.mkdir(parents=True)
    p.write_bytes(b'CPU identity fixture, not scientific weights');(p.parent/'final.pt').write_bytes(b'checkpoint')
    r=dict(dataset=j['dataset'],arm=j['arm'],train_seed=j['train_seed'],updates=3000,files={'ema_0.99':{'sha256':rc.sha(p)}},checkpoint_sha256=rc.sha(p.parent/'final.pt'))
    rc.save(p.parent/'readouts.json',r);rc.save(p.parent/'summary.json',{'status':'complete'})
    assert rc.verify_readout(tmp_path,j)[0]==p
    r['arm']='E_lateCN';rc.save(p.parent/'readouts.json',r)
    with pytest.raises(ValueError):rc.verify_readout(tmp_path,j)
    r['arm']=j['arm'];rc.save(p.parent/'readouts.json',r);p.write_bytes(b'tampered')
    with pytest.raises(ValueError):rc.verify_readout(tmp_path,j)

def test_resource_handles_only_bound_protected_workers():
    from resource_handles import allowed_owner
    b=rc.read(rc.ROOT/'cross_batch_binding.json');args=['python',b['worker_path'],'--approval',b['approval_path'],'--ledger',str(pathlib.Path(b['run'])/'activation.json')]
    assert allowed_owner(args,b['protected_gpu_uuids'][0],os.getuid(),['newGPU'],b)
    assert not allowed_owner(args,'newGPU',os.getuid(),['newGPU'],b)
    args[1]='unknown_worker.py'
    assert not allowed_owner(args,b['protected_gpu_uuids'][0],os.getuid(),['newGPU'],b)

def test_gpu_resource_permission_and_low_utilization():
    devices=['GPU-'+str(i) for i in range(6)];rows={d:[str(i),d,'RTX 2080 Ti','11000','0','1'] for i,d in enumerate(devices)}
    a=dict(allocation_check_complete=True,reservation_conflicts=[],allowed_uuids=devices,minimum_free_mib=6000)
    assert len(run.resources(devices,a,rows,'',''))==6
    with pytest.raises(RuntimeError):run.resources(devices,a,rows,devices[0]+',1,100','')
    with pytest.raises(PermissionError):run.resources(devices,dict(a,allocation_check_complete=False),rows,'','')
    with pytest.raises(PermissionError):run.resources([rc.PROTOCOL['protected_gpu_uuids'][0]]+devices[1:],a,rows,'','')

def test_no_activation_in_dry_run(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['run.py','dry-run','--output',str(tmp_path/'does_not_exist')])
    monkeypatch.setattr(run,'supervise',lambda *a:pytest.fail('dry run launched supervisor'))
    assert run.main()==0 and not (tmp_path/'does_not_exist').exists()
    assert json.loads(capsys.readouterr().out)['formal_activated'] is False

def test_bound_summary_and_protocol():
    f,h=rc.source_binding()
    assert all(x in f for x in ['summarize.py','cross_batch.py','run_worker.py','run.py','model_io.py','protocol.json','cross_batch_binding.json'])
    assert rc.PROTOCOL['formal_authorized'] is False and rc.PROTOCOL['budget_seconds'] is None

def test_external_sealed_controls_only():
    r,inv=cb.external_records()
    assert len(r) in (20,28) and len({cb.key(x) for x in r})==len(r)
    assert {x['train_seed'] for x in r}<={2026102911,2026102912}

def test_summary_missing_outputs_preserves_all_identities(tmp_path,monkeypatch):
    import summarize as s
    out=tmp_path/'empty_run';out.mkdir();target=out/'unified_evaluation.json'
    monkeypatch.setattr(s,'factory',lambda:{})
    monkeypatch.setattr(s,'load_graph',lambda *a,**k:{})
    monkeypatch.setattr(s.M,'stats',lambda *a,**k:{})
    monkeypatch.setattr(s,'evaluate_graph',lambda *a,**k:dict(raw={'edges':926},reference={'edges':926},absolute_error={'edges':0},distribution_error={'W1':0}))
    monkeypatch.setattr('sys.argv',['summarize.py','--run',str(out),'--output',str(target),'--deadline',str(time.time()+20)])
    assert s.main()==2
    r=rc.read(target)
    assert r['status']=='incomplete' and len(r['inventory'])==12
    assert all(x['status']=='missing' for x in r['inventory'])
    assert len([x for x in r['missing_identities'] if x['arm']=='D_earlyCN'])==8
    assert all(x['difference'] is None for x in r['paired_means'] if 'early' in x['contrast'])

def test_cross_batch_rejects_unbound_new_summary(tmp_path,monkeypatch):
    p=tmp_path/'wrong.json';rc.save(p,dict(source_sha256='wrong',records=[]))
    monkeypatch.setattr('sys.argv',['cross_batch.py','--new-summary',str(p),'--output',str(tmp_path/'new.json')])
    with pytest.raises(ValueError,match='source binding'):cb.main()

def test_formal_requires_real_authorization(tmp_path):
    a=tmp_path/'not_approval.json';c=tmp_path/'not_cost.json';rc.save(a,{'user_formal_authorized':False});rc.save(c,{})
    with pytest.raises(PermissionError):rc.approval_gate(a,c,tmp_path/'never_activated')
