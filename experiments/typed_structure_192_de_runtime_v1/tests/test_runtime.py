import ast
import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'package'))
import launch
import runtime_plan as rp
import admission
import summarize as sm


def protocol():return json.loads((ROOT/'proposal.json').read_text())
def cost():return json.loads((ROOT/'evidence/execution/cost_proposal.json').read_text())


def test_exact_tasks_and_dependency_paths(tmp_path):
    p=rp.plan(protocol(),tmp_path/'unstarted')
    assert (p['task_count'],p['sample_count'])==(8,16)
    assert len({r['task_id'] for r in p['tasks']})==8
    assert len({s['output'] for r in p['tasks'] for s in r['samples']})==16
    for r in p['tasks']:
        assert str(r['train_seed']) in r['output'] and r['dataset'] in r['output'] and r['output'].endswith('_'+r['arm'])
        assert r['updates']==1000 and r['checkpoint_updates']==[0,500,1000]
        assert r['role_basis']==(r['arm']=='E')
        assert all(s['dependency']==r['task_id']+':final1000' and s['T']==1000 for s in r['samples'])
        assert [s['sample_seed'] for s in r['samples']]==list(rp.SAMPLE_SEEDS)
    assert not (tmp_path/'unstarted').exists()


@pytest.mark.parametrize('field,value',[('updates_per_model',500),('rounds',20),('sample_seeds',[1,2])])
def test_scientific_task_config_rejected(field,value):
    p=protocol();p[field]=value
    with pytest.raises(ValueError):rp.plan(p,'/tmp/unused')


def test_budget_queue_is_execution_queue():
    c=cost();queues,loads=admission.assignment(c,1000,list(range(8)))
    assert [x[0] for x in queues]==rp.QUEUE
    assert loads==[r['estimated_worker_seconds'] for r in c['queue']]
    estimate=max(loads)*c['concurrency_planning_factor']+c['evaluation_delivery_reserve_seconds']+c['failure_reserve_seconds']+c['unmeasured_state_reserve_seconds']
    assert estimate==c['planned_seconds_before_rounding'] and estimate<8700
    c['queue'][0],c['queue'][1]=c['queue'][1],c['queue'][0]
    with pytest.raises(ValueError):admission.assignment(c,1000,list(range(8)))
    with pytest.raises(ValueError):admission.assignment(cost(),1000,list(range(7)))


def test_public_dry_cli_no_workers_gpu_or_activation(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('subprocess/GPU must not be touched')
    for name in ['Popen','run','check_output']:monkeypatch.setattr(launch.subprocess,name,forbidden)
    output=tmp_path/'no_run';manifest=tmp_path/'dry.json'
    monkeypatch.setattr(sys,'argv',['launch.py','dry-run','--protocol',str(ROOT/'proposal.json'),'--cost',str(ROOT/'evidence/execution/cost_proposal.json'),'--output',str(output),'--manifest-output',str(manifest)])
    assert launch.main()==0
    d=json.loads(manifest.read_text());assert not output.exists() and d['worker_launches']==d['GPU_calls']==d['optimizer_steps']==0
    assert d['formal_start'] is None and d['formal_deadline'] is None


def test_dry_cli_real_subprocess(tmp_path):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r=subprocess.run([sys.executable,str(ROOT/'package/launch.py'),'dry-run','--protocol',str(ROOT/'proposal.json'),'--output',str(tmp_path/'never_created')],env=env,capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['sample_count']==16 and not (tmp_path/'never_created').exists()


@pytest.mark.parametrize('arm',['D','E'])
def test_initialization_routes_and_no_scientific_constructor(arm):
    basis=object();calls=[]
    def initializer(*args):calls.append(args);return 'mock'
    assert rp.initialization_for_task(initializer,'IMDB',arm,2026102811,basis)=='mock'
    assert calls==[('IMDB',arm,2026102811,basis if arm=='E' else None)]
    if arm=='E':
        with pytest.raises(ValueError):rp.initialization_for_task(initializer,'IMDB',arm,2026102811,None)


def test_worker_parser_and_real_routing_static():
    s=(ROOT/'package/worker.py').read_text();tree=ast.parse(s)
    assert "choices=list('DE')" in s and "--train-seed" in s
    assert 'initialization_for_task(initialization,dataset,arm,seed,basis)' in s
    assert "arm=='C'" not in s and "seed=protocol['train_seeds'][dataset]" not in s
    # Existing sequence still constructs/copies, then resets training streams,
    # then persists initial before validation/training.
    assert s.index('model,case,cfg,proof=')<s.index('rng,rng_proof=reset_training_streams')<s.index("atomic_checkpoint(out/'initial.pt'")<s.index('validation_rows=')
    assert any(isinstance(x,ast.For) and ast.unparse(x.target)=='sample_seed' for x in ast.walk(tree))


def test_dictionary_approval_actual_files_and_dimensions():
    approved={'dictionaries':{}}
    for dataset in rp.TRAIN_SEEDS:
        root=ROOT/'dictionaries'/('dictionary_'+dataset);r=json.loads((root/'selection.json').read_text())
        approved['dictionaries'][dataset]=dict(files={n:rp.sha(root/n) for n in ['final_basis.npz','selection.json','panel.json','frozen_splits.json']},K=r['K'],offsets=r['offsets'],dimension=r['offsets'][-1])
    admission.dictionaries(ROOT/'dictionaries',approved)
    approved['dictionaries']['IMDB']['files']['final_basis.npz']='0'*64
    with pytest.raises(ValueError):admission.dictionaries(ROOT/'dictionaries',approved)


def test_runtime_binding_covers_summary_worker_protocol_history(monkeypatch):
    files,base=rp.source_binding(ROOT/'package')
    for relative in ['summarize.py','summary_metrics.py','proposal.json','package/launch.py','package/worker.py','package/runtime_plan.py','history/index.json']:
        assert relative in files
    original=rp.sha
    for relative in ['summarize.py','summary_metrics.py','proposal.json','package/worker.py']:
        target=ROOT/relative
        monkeypatch.setattr(rp,'sha',lambda p: 'f'*64 if pathlib.Path(p)==target else original(p))
        assert rp.source_binding(ROOT/'package')[1]!=base
    monkeypatch.setattr(rp,'sha',original)
    with pytest.raises(ValueError):admission.verify_summary(ROOT/'package',{'summarize_sha256':'0'*64})


def test_ticket_training_seed_and_output_binding(tmp_path):
    f=tmp_path/'fixture_not_approval';f.write_text('CPU fixture only, NOT authorized')
    a=SimpleNamespace(dataset='IMDB',train_seed=2026102811,arm='E',output=str(tmp_path/'IMDB_s2026102811_E'),approval=str(f),cost=str(f))
    t=dict(parent_pid=123,dataset=a.dataset,train_seed=a.train_seed,arm=a.arm,task_id=rp.task_id(a.dataset,a.train_seed,a.arm),device='mock',output=a.output,approval_sha256=rp.sha(f),cost_sha256=rp.sha(f),admitted_epoch=10.)
    rp.verify_ticket(t,a,123,'mock',11.)
    t['train_seed']=2026102834
    with pytest.raises(PermissionError):rp.verify_ticket(t,a,123,'mock',11.)


def template():
    return dict(raw={'edges':10.,'closure_path_ratio':None},reference={'edges':10.,'closure_path_ratio':None},absolute_error={'edges':0.,'closure_path_ratio':None},distribution_error={'degree_W1_incident4':0.,'MP2_composition_TV':0.,'MP3_composition_TV':0.},undefined={})


def synthetic_records():
    rows=[];statuses={}
    for d,seeds in rp.TRAIN_SEEDS.items():
        for si,seed in enumerate(seeds):
            for arm in 'OABCDE':
                statuses['%s_s%d_%s'%(d,seed,arm)]={'status':'complete'}
                for j,sample in enumerate(rp.SAMPLE_SEEDS):
                    value=100*si+10*'OABCDE'.index(arm)+j
                    v=template();v['raw']['edges']=value;v['absolute_error']['edges']=abs(value-10);v['distribution_error']['degree_W1_incident4']=value/10
                    rows.append(dict(dataset=d,train_seed=seed,sample_seed=sample,arm=arm,status='evaluated',values=v))
    return rows,statuses


def test_five_contrasts_exact_seed_identity_and_direction():
    rows,status=synthetic_records();r=sm.assemble(list(reversed(rows)),{d:template() for d in rp.TRAIN_SEEDS},status)
    assert r['status']=='complete' and len(r['contrasts'])==20
    for c in r['contrasts']:
        left,right=c['contrast'].split('-');si=rp.TRAIN_SEEDS[c['dataset']].index(c['train_seed'])
        expected=[]
        for j,p in enumerate(c['pairs']):
            a=100*si+10*'OABCDE'.index(left)+j;b=100*si+10*'OABCDE'.index(right)+j
            assert p['difference']['raw_difference']['edges']==a-b
            assert p['difference']['absolute_error_difference']['edges']==abs(a-10)-abs(b-10)
            expected.append(abs(a-10)-abs(b-10))
        assert c['mean']['absolute_error_difference']['edges']==sum(expected)/2
    assert r['cells']['IMDB_s2026102834_D']['mean']['raw']['edges']!=r['cells']['IMDB_s2026102811_D']['mean']['raw']['edges']


def test_graph_error_before_mean_null_and_missing_propagate():
    rows,status=synthetic_records()
    sel=[x for x in rows if (x['dataset'],x['train_seed'],x['arm'])==('IMDB',2026102811,'E')]
    for row,val in zip(sel,[8,12]):row['values']['raw']['edges']=val;row['values']['absolute_error']['edges']=2
    r=sm.assemble(rows,{d:template() for d in rp.TRAIN_SEEDS},status)
    mean=r['cells']['IMDB_s2026102811_E']['mean'];assert mean['raw']['edges']==10 and mean['absolute_error']['edges']==2
    assert mean['raw']['closure_path_ratio'] is None
    rows.remove(sel[1]);r=sm.assemble(rows,{d:template() for d in rp.TRAIN_SEEDS},status)
    assert r['status']=='incomplete' and r['cells']['IMDB_s2026102811_E']['mean'] is None
    for c in r['contrasts']:
        if c['dataset']=='IMDB' and c['train_seed']==2026102811 and c['contrast'].startswith('E-'):
            assert c['mean'] is None and c['pairs'][1]['difference']['raw_difference']['edges'] is None


def test_duplicate_and_unknown_seed_rejected():
    rows,status=synthetic_records()
    with pytest.raises(ValueError):sm.assemble(rows+[rows[0]],{d:template() for d in rp.TRAIN_SEEDS},status)
    rows[0]['train_seed']=123
    with pytest.raises(ValueError):sm.assemble(rows,{d:template() for d in rp.TRAIN_SEEDS},status)


@pytest.mark.parametrize('mode',['success','failure','STOP','deadline','summary_failure'])
def test_real_cpu_mock_processes_same_supervisor(tmp_path,mode):
    out=tmp_path/'mock_run';out.mkdir();f=tmp_path/'not_authorized.txt';f.write_text('CPU mock only')
    args=SimpleNamespace(output=str(out),approval=str(f),cost=str(f),dictionary_dir=str(ROOT/'dictionaries'),ledger=str(f),protocol=str(ROOT/'proposal.json'))
    devices=['MOCK-'+str(i) for i in range(8)];queues,loads=rp.assignment(cost(),1000,devices);tasks=dict(zip(devices,queues))
    now=time.time();ledger=dict(start_epoch=now,deadline_epoch=now+1500);approval=dict(updates=1000,resource_authorization={'fixture_only':True})
    children=[];commands=[]
    def spawn(cmd,**kwargs):
        assert pathlib.Path(cmd[1]).name=='worker.py'
        assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert all(kwargs['env'][k]=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
        commands.append(cmd)
        behavior='success' if mode in ['success','summary_failure'] else 'fail' if mode=='failure' and len(commands)==1 else 'sleep'
        process=subprocess.Popen([cmd[0],str(ROOT/'tests/mock_worker.py')]+cmd[2:]+['--behavior',behavior],**kwargs);children.append(process)
        if mode=='STOP':(out/'STOP').write_text('CPU fixture stop')
        if mode=='deadline':ledger['deadline_epoch']=time.time()+1199
        return process
    def summary(*unused):
        if mode=='summary_failure':raise RuntimeError('synthetic summary failure')
        return dict(status='complete' if mode=='success' else 'incomplete_summary_saved')
    rc=launch.supervise(ROOT/'package',args,ledger,approval,cost(),devices,tasks,loads,1.,[],spawn=spawn,summarizer=summary,poll_seconds=.01)
    r=json.loads((out/'launcher_summary.json').read_text())
    assert all(p.poll() is not None for p in children) and len(commands)<=8
    labels=[c[c.index('--output')+1] for c in commands];assert len(set(labels))==len(labels)
    assert rc==(0 if mode=='success' else 2)
    if mode=='success':
        assert len(commands)==8 and r['status']=='complete'
        for cmd,expected in zip(commands,rp.QUEUE):
            ticket=json.loads((pathlib.Path(cmd[cmd.index('--output')+1])/'fixture_ticket.json').read_text())
            assert (ticket['dataset'],ticket['train_seed'],ticket['arm'])==expected
            assert ticket['task_id']==rp.task_id(*expected)
    if mode=='failure':assert any(x.get('exit_code')==9 for x in r['primary_failures'])
    if mode in ['STOP','deadline']:
        assert len(commands)==1 and sum(map(len,r['unstarted'].values()))==7
        assert r['primary_failures'][0]['reason']==('STOP' if mode=='STOP' else 'delivery_reserve_deadline')
    if mode=='summary_failure':assert 'synthetic summary failure' in r['evaluation']['error'] and r['execution_complete'] is True
    assert json.loads((out/'execution_receipt.json').read_text())['primary_failures']==r['primary_failures']


def test_history_readonly_real_records_and_no_new_run(tmp_path):
    r=sm.collect(tmp_path/'not_started')
    assert r['old_evaluated_graphs']==32 and r['new_evaluated_graphs']==0 and r['status']=='incomplete'
    assert not (tmp_path/'not_started').exists() and len(r['matching'])==48
    assert len(r['contrasts'])==20 and all(c['mean'] is None for c in r['contrasts'])
    cell=r['cells']['IMDB_s2026102834_C'];f=cell['graphs']['2026102821']['focus']
    assert {'DAM','DAA','AAM','Genre_sorted_degree_spectrum'}<=set(f)
    assert len(f['Genre_sorted_degree_spectrum']['raw'])==10
    assert cell['graphs']['2026102821']['source_commit']=='143e6bf80a9858ec21e512da2f1a03e02e3a5a38'


def test_resource_permission_not_utilization_mock_only():
    ids=['GPU-'+str(i) for i in range(8)];rows={d:[str(i),d,'NVIDIA GeForce RTX 2080 Ti','11000','0','1'] for i,d in enumerate(ids)}
    auth=dict(basis='explicit_user_eight_card_authorization',allowed_uuids=ids,reservation_conflicts=[],allocation_check_complete=True,minimum_free_mib=4000)
    assert len(launch.resource_check(ids,auth,rows,'',''))==8
    auth['allocation_check_complete']=False
    with pytest.raises(PermissionError):launch.resource_check(ids,auth,rows,'','')


@pytest.mark.parametrize('fault',[None,'wrong_seed','metric_hash','missing_graph','worker_failed'])
def test_new_result_loader_with_explicit_engineering_replay(tmp_path,fault):
    # Copy an OLD graph as a clearly marked CPU I/O fixture, never a D/E result.
    # No model weights loaded and no new graph generated.
    import shutil
    out=tmp_path/'synthetic_loader_fixture';out.mkdir();cfg=json.loads((ROOT/'package/config.json').read_text())
    folder=out/'IMDB_s2026102811_E';folder.mkdir();source=ROOT/'history/round1/IMDB_C'
    b=json.loads((source/'binding.json').read_text());b.update(dataset_name='IMDB',train_seed=2026102811,arm='E',task_id=folder.name,config=dict(cfg,seed=2026102811),fixture_only=True)
    (folder/'binding.json').write_text(json.dumps(b));(folder/'final.pt').write_bytes(b'NOT A CHECKPOINT: CPU I/O fixture only')
    (folder/'summary.json').write_text(json.dumps(dict(status='failed_or_incomplete' if fault=='worker_failed' else 'complete',error='synthetic_primary_failure' if fault=='worker_failed' else None,fixture_only=True)))
    for seed in rp.SAMPLE_SEEDS:
        dest=folder/('sample_'+str(seed));dest.mkdir();old=source/dest.name
        for name in ['graph.npz','metrics.json']:shutil.copyfile(old/name,dest/name)
        sample=dict(seed=seed,checkpoint_sha256=rp.sha(folder/'final.pt'),graph_sha256=rp.sha(dest/'graph.npz'),metrics_sha256=rp.sha(dest/'metrics.json'),binding=b,fixture_only=True)
        if fault=='wrong_seed':sample['seed']=2026102822 if seed==2026102821 else 2026102821
        if fault=='metric_hash':sample['metrics_sha256']='0'*64
        (dest/'binding.json').write_text(json.dumps(sample))
    if fault=='missing_graph':(folder/'sample_2026102822/graph.npz').unlink()
    r=sm.collect(out);cell=r['cells'][folder.name]
    assert r['status']=='incomplete' # other seven models are absent in this fixture
    if fault in ['wrong_seed','metric_hash']:assert r['new_evaluated_graphs']==0 and cell['mean'] is None
    elif fault=='missing_graph':assert r['new_evaluated_graphs']==1 and cell['mean'] is None
    else:assert r['new_evaluated_graphs']==2
    if fault=='worker_failed':assert cell['worker_status']['error']=='synthetic_primary_failure'


def test_run_summary_tampering_does_not_launch_process(tmp_path,monkeypatch):
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not execute')))
    receipt=launch.run_summary(ROOT/'package',tmp_path,{'summarize_sha256':'0'*64},time.time()+60)
    assert receipt['status']=='failed' and 'hash mismatch' in receipt['error']


def test_unapproved_formal_entry_rejects_before_device_access(tmp_path,monkeypatch):
    # Deliberately invalid fixture, never an activation-capable approval file.
    fixture=tmp_path/'REJECTED_CPU_FIXTURE.json';fixture.write_text(json.dumps({'independent_review':'not_authorized','user_continue_confirmed':False}))
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no device calls')))
    monkeypatch.setattr(sys,'argv',['launch.py','formal','--approval',str(fixture),'--protocol',str(ROOT/'proposal.json'),'--dictionary-dir',str(ROOT/'dictionaries'),'--cost',str(ROOT/'evidence/execution/cost_proposal.json'),'--output',str(tmp_path/'not_activated'),'--gpu-uuids','not-a-device'])
    with pytest.raises(PermissionError):launch.main()
    assert not (tmp_path/'not_activated').exists()
