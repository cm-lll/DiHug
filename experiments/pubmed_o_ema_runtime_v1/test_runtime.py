import json,pathlib,sys,time,copy,os,subprocess
import pytest
from runtime_contract import jobs,cell,save,sha,read,verify_readout,verify_ticket,output_check,BUDGET,ROOT
from run import supervise,resources
from summarize import assemble
from summary_metrics import graph_summary
from assessment import SEEDS,SAMPLES

def test_two_trajectories_eight_correct_chains_and_cli():
    p=jobs();assert len(p['training'])==2 and len(p['sampling'])==8
    assert len({j['directory'] for j in p['training']+p['sampling']})==10
    assert all(j['updates']==1000 for j in p['training']) and all(j['T']==1000 for j in p['sampling'])
    assert all(j['depends_on']==cell(j['train_seed']) and j['readout'] in ('raw','ema') for j in p['sampling'])
    r=subprocess.run([sys.executable,str(ROOT/'run.py'),'dry-run','--output','/tmp/not_created_ema_dryrun'],capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)['formal_activated'] is False
    r=subprocess.run([sys.executable,str(ROOT/'run.py'),'formal','--output','/tmp/not_created_ema_dryrun'],capture_output=True,text=True)
    assert r.returncode!=0 and not pathlib.Path('/tmp/not_created_ema_dryrun').exists()

def test_real_export_files_and_readout_tampering(tmp_path):
    import torch
    from ema import ParameterEMA
    from run_worker import export_readouts
    from worker import atomic_checkpoint
    m=torch.nn.Linear(2,2);e=ParameterEMA(m)
    with torch.no_grad():m.weight.add_(2)
    e.updates=1000;folder=tmp_path/cell(SEEDS[0]);folder.mkdir()
    atomic_checkpoint(folder/'final.pt',dict(updates=1000));export_readouts(folder,m,e,1000,dict(train_seed=SEEDS[0]))
    save(folder/'summary.json',dict(status='complete'))
    actual={}
    for r in ('raw','ema'):
        j=next(j for j in jobs()['sampling'] if j['train_seed']==SEEDS[0] and j['readout']==r)
        p,_=verify_readout(tmp_path,j);actual[r]=torch.load(p,map_location='cpu')
        assert actual[r]['readout']==r
    assert not torch.equal(actual['raw']['model']['weight'],actual['ema']['model']['weight'])
    (folder/'readout_ema.pt').write_bytes((folder/'readout_raw.pt').read_bytes())
    with pytest.raises(ValueError):verify_readout(tmp_path,j)

def setup_launcher(tmp_path):
    out=tmp_path/'run';out.mkdir();approval=tmp_path/'approval.json';save(approval,{'fixture':True})
    ledger=out/'activation.json';save(ledger,dict(start_epoch=time.time(),deadline_epoch=time.time()+7800,source_sha256='fixture'))
    return out,approval,ledger
def fake_spawner(out,seen,fail=None):
    class Process:
        def __init__(self,cmd,**kwargs):
            fd=kwargs['pass_fds'][0];t=json.loads(os.read(fd,32768));j=t['job'];seen.append(j)
            self.returncode=2 if j['id']==fail else 0
            folder=out/j['directory'];folder.mkdir(parents=True)
            save(folder/'summary.json',dict(status='complete' if self.returncode==0 else 'incomplete',completed_updates=1000 if j['mode']=='train' else 0,completed_chains=1 if j['mode']=='sample' else 0,error='fixture failure' if self.returncode else None))
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    return Process
def test_schedule_barrier_success(tmp_path):
    out,a,l=setup_launcher(tmp_path);seen=[]
    rc=supervise(out,a,l,['gpu0','gpu1'],spawn=fake_spawner(out,seen),summarizer=lambda *a:dict(status='complete'),poll=0)
    assert rc==0 and len(seen)==10 and [j['mode'] for j in seen[:2]]==['train','train']
    for slot in (0,1):assert [j['readout'] for j in seen if j['mode']=='sample' and j['slot']==slot]==['raw','raw','ema','ema']
def test_failure_preserves_cause_and_does_not_sample(tmp_path):
    out,a,l=setup_launcher(tmp_path);seen=[]
    def bad_summary(*a):raise RuntimeError('summary failure')
    rc=supervise(out,a,l,['gpu0','gpu1'],spawn=fake_spawner(out,seen,cell(SEEDS[0])),summarizer=bad_summary,poll=0)
    r=read(out/'launcher_summary.json');assert rc==2 and r['primary_failures'] and r['evaluation']['status']=='failed'
    assert all(j['mode']=='train' for j in seen) and len(r['unstarted'])==8
@pytest.mark.parametrize('why',['STOP','deadline'])
def test_stop_never_restarts_or_launches(tmp_path,why):
    out,a,l=setup_launcher(tmp_path);seen=[]
    if why=='STOP':(out/'STOP').touch()
    else:
        x=read(l);x['deadline_epoch']=time.time()+1100;save(l,x)
    assert supervise(out,a,l,['gpu0','gpu1'],spawn=fake_spawner(out,seen),summarizer=lambda *a:dict(status='incomplete'),poll=0)==2
    assert seen==[] and len(read(out/'launcher_summary.json')['unstarted'])==10

def fixture_records():
    ref=read(ROOT/'reference.json');rs=[]
    for seed in SEEDS:
        for r in ('raw','ema'):
            for s in SAMPLES:
                m=copy.deepcopy(ref);m.update(W1_definitions={'incident4':1.,'all10_compat':.4},primary={'type_weighted_joint_degree_TV':.4,'typed_triangle_composition_TV':.1},comparison={'mp_frequency_TV':[.1,.2,.3]})
                change=10 if r=='raw' else 5
                if seed==SEEDS[1]:change*=2
                m['edges']+=change;m['family_edges'][0]+=change
                rs.append(dict(train_seed=seed,readout=r,sample_seed=s,status='complete',metrics=m,values=graph_summary(m,ref),edge_set=[[0,1]]))
    return rs,ref
def test_pairing_mean_and_missing_propagation():
    rs,ref=fixture_records();r=assemble(list(reversed(rs)),ref)
    assert r['status']=='complete'
    assert [c['mean']['absolute_error_difference']['edges'] for c in r['contrasts']]==[-5.,-10.]
    assert r['assessment']['accuracy_conclusion']=='consistent_accuracy_gain_in_this_panel'
    rs[0]=dict(train_seed=SEEDS[0],readout='raw',sample_seed=SAMPLES[0],status='incomplete',reason='fixture missing')
    r=assemble(rs,ref);assert r['status']=='incomplete' and r['contrasts'][0]['mean'] is None and r['cells'][cell(SEEDS[0])+'_raw']['mean'] is None
    assert r['assessment']['pooled_error_difference'] is None
def test_resource_utilization_is_not_authority_or_failure():
    ds=['GPU-a','GPU-b'];rows={d:[str(i),d,'RTX 2080 Ti','11000','0','1'] for i,d in enumerate(ds)}
    a=dict(allocation_check_complete=True,allowed_uuids=ds,minimum_free_mib=9000)
    assert len(resources(ds,a,rows,'',''))==2
    with pytest.raises(PermissionError):resources(ds,dict(a,allocation_check_complete=False),rows,'','')
    with pytest.raises(RuntimeError):resources(ds,a,rows,'GPU-a, 123, 100','')
def test_output_overlap_rejected_before_mkdir(tmp_path):
    src=tmp_path/'source';src.mkdir();before=list(src.iterdir())
    with pytest.raises(ValueError):output_check(src/'bad',[src])
    assert list(src.iterdir())==before
