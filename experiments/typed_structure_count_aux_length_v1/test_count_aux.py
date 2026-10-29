import sys,pathlib,copy,json,importlib.util
import pytest
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'package'))
import torch
from count_aux import count_loss
from runtime_contract import jobs,PROTOCOL,approval_gate
from cross_batch import aggregate

def test_hand_count_and_gradient():
    p=torch.tensor([.2,.8,.4],dtype=torch.float64,requires_grad=True)
    loss,r=count_loss(p,[0,0,1],[1,0],[2,1])
    assert float(loss)==pytest.approx(.08)
    loss.backward();torch.testing.assert_close(p.grad,torch.tensor([0,0,.4],dtype=torch.float64))
    assert r['relation_denominator']==2
def test_empty_family_excluded_and_fixed_one():
    p=torch.tensor([.5],dtype=torch.float64,requires_grad=True)
    loss,r=count_loss(p,[0],[1,3,0],[1,0,0],[0,3,0])
    assert float(loss)==.25 and r['relation_denominator']==1
def test_capacity_normalization():
    loss,_=count_loss(torch.tensor([.25,.25],dtype=torch.float64),[0,0],[1],[2])
    assert float(loss)==.125
def test_no_active():
    p=torch.empty(0,dtype=torch.float64,requires_grad=True)
    loss,r=count_loss(p,[],[0,2],[0,0],[0,2]);loss.backward()
    assert loss==0 and r['relation_denominator']==0
@pytest.mark.parametrize('p,f,t,n', [([.5,.5],[0,0],[1],[1]),([float('nan')],[0],[1],[1]),([1.2],[0],[1],[1]),([.5],[1],[1],[1]),([.5],[0],[2],[1])])
def test_invalid_count(p,f,t,n):
    with pytest.raises((ValueError,FloatingPointError)):count_loss(torch.tensor(p),f,t,n)
def test_zero_lambda_formula():
    x=torch.tensor([[.2,.9],[.5,-.3]],requires_grad=True);labels=torch.tensor([1,0]);ce=torch.nn.functional.cross_entropy(x,labels,reduction='sum')/18336
    aux,_=count_loss(x.softmax(-1)[:,1],[0,0],[1],[2])
    g=torch.autograd.grad(ce/2,x,retain_graph=True)[0]
    gg=torch.autograd.grad(ce/2+0*aux/2,x)[0]
    assert torch.equal(g,gg)
def test_queue():
    j=jobs();assert len(j['training'])==8 and len(j['sampling'])==48
    alljobs=j['training']+j['sampling']
    assert len({x['id'] for x in alljobs})==56 and len({x['directory'] for x in alljobs})==56
    for t in j['training']:
        s=[x for x in j['sampling'] if x['depends_on']==t['id']]
        assert t['updates']==9000 and len(s)==6 and {x['checkpoint'] for x in s}=={3000,6000,9000}
        assert all(x['slot']==t['slot'] and x['train_seed']==t['train_seed'] for x in s)
def test_aggregation_identity_null_direction():
    templates={d:{'absolute_error':{'edges':0},'distribution_error':{'main_incident4_W1':0},'raw':{'edges':0}} for d in PROTOCOL['train_seeds']}
    rows=[]
    for j in jobs()['sampling']:
        for v in ('original','lcc'):
            value=j['checkpoint']/1000+(10 if j['arm']=='CE' else 2)+(j['train_seed']%10)*100+(j['sample_seed']%10)
            rows.append(dict(**j,version=v,absolute_error={'edges':value},raw={'edges':value},distribution_error={'main_incident4_W1':None}))
    a=aggregate(rows,templates)
    assert all(x['difference']==-8 for x in a['paired_means'] if x['metric']=='edges')
    assert all(x['difference']==3 for x in a['length_means'] if x['metric']=='edges')
    assert all(x['difference'] is None for x in a['paired_means'] if x['metric']=='main_incident4_W1')
    first=rows.pop(0);a=aggregate(rows,templates)
    assert len(a['missing_identities'])==1
    assert any(x['value'] is None for x in a['cell_means'] if x['dataset']==first['dataset'] and x['train_seed']==first['train_seed'] and x['arm']==first['arm'] and x['checkpoint']==first['checkpoint'] and x['version']==first['version'])
def test_scientific_source_frozen():
    import hashlib
    for rel,expected in json.loads((ROOT/'frozen_science_bindings.json').read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
def test_inference_isolation_and_exception():
    from ema import ParameterEMA,cpu_copy
    from probe import rng_get,equal
    from stage import isolated_readout
    import numpy as np
    m=torch.nn.Linear(2,2);opt=torch.optim.AdamW(m.parameters());ema=ParameterEMA(m);rng=np.random.default_rng(7)
    before=cpu_copy(m.state_dict());r=rng_get(rng,'cpu')
    with pytest.raises(RuntimeError):
        with isolated_readout(m,opt,ema,rng,'cpu'):
            with torch.no_grad():m.weight.add_(1)
            rng.random();torch.rand(3);raise RuntimeError('simulated sampling failure')
    assert equal(before,m.state_dict()) and equal(r,rng_get(rng,'cpu'))

def test_mock_continuous_three_checkpoint_worker(tmp_path,monkeypatch):
    # Actual run_worker.train orchestration; scientific update/sampler replaced.
    # No optimizer.step and no generated graph, even in this simulation.
    import run_worker as w
    import numpy as np
    from probe import rng_get,equal
    from runtime_contract import jobs,verify_readout
    from ema import cpu_copy
    job=jobs()['training'][0];out=tmp_path/job['directory'];out.mkdir()
    model=torch.nn.Linear(2,2);model.count_aux_weight=0.;opt=torch.optim.AdamW(model.parameters())
    rng=np.random.default_rng(123);expected=np.random.default_rng(123).random(3).tolist();seen=[];sampled=[];weights=[]
    monkeypatch.setattr(w,'UPDATES',3);monkeypatch.setattr(w,'SAVE_EVERY',1)
    monkeypatch.setitem(w.PROTOCOL,'generation_checkpoints',[1,2,3])
    monkeypatch.setattr(w,'fresh',lambda *a:(model,opt,{}, {'T':1000},rng,{}))
    monkeypatch.setattr(w,'rng_get',lambda r,d:rng_get(r,'cpu'))
    monkeypatch.setattr(w,'paired_validation',lambda *a:{'raw':{'test_only':True},'ema_0.99':{'test_only':True}})
    monkeypatch.setattr(w,'input_hash',lambda i:str(i))
    def inputs(*a):
        seen.append(float(rng.random()));return [{'t':500},{'t':500}]
    monkeypatch.setattr(w,'pair_inputs',inputs)
    def update(fn,m,opt,case,inputs,T,device,ema,completed):
        if weights:assert equal(m.state_dict(),weights[-1])
        with torch.no_grad():m.weight.add_(.01)
        ema.update(m,completed+1);weights.append(cpu_copy(m.state_dict()))
        return {'optimizer_steps':1,'simulated':True}
    monkeypatch.setattr(w,'complete_normal2',update)
    from stage import isolated_readout
    monkeypatch.setattr(w,'isolated_readout',lambda m,o,e,r,d:isolated_readout(m,o,e,r,'cpu'))
    def sample(j,folder,run,approval,check,progress,loaded):
        path,receipt=verify_readout(run,j);p=torch.load(path,map_location='cpu')
        assert p['updates']==j['checkpoint'] and p['readout']=='ema_0.99'
        loaded[0].load_state_dict(p['model']);rng.random();torch.rand(2)
        sampled.append((j['checkpoint'],j['sample_seed']));progress['completed_chains']=1
    monkeypatch.setattr(w,'sample',sample)
    progress={'completed_chains':0};ledger={'output':str(tmp_path),'approval_sha256':'simulation'}
    w.train(job,out,{},ledger,lambda:None,progress)
    assert seen==expected and len(sampled)==6 and progress['completed_updates']==3 and progress['completed_chains']==6
    assert len(list(out.glob('checkpoint_u*.pt')))==3 and (out/'initial.pt').exists()
    assert opt.state_dict()['state']=={}

def test_dry_run_never_spawns(monkeypatch,tmp_path,capsys):
    import run,sys
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('spawned')))
    monkeypatch.setattr(sys,'argv',['run.py','dry-run','--output',str(tmp_path/'not_created')])
    assert run.main()==0 and not (tmp_path/'not_created').exists()
    value=json.loads(capsys.readouterr().out);assert not value['formal_activated'] and len(value['sampling'])==48

def test_allocation_not_utilization_gate():
    from run import resources
    devices=['GPU-'+str(i) for i in range(8)]
    rows={d:[str(i),d,'NVIDIA GeForce RTX 2080 Ti','11011','0','1'] for i,d in enumerate(devices)}
    authority=dict(allocation_check_complete=True,reservation_conflicts=[],allowed_uuids=devices,minimum_free_mib=8000)
    assert len(resources(devices,authority,rows,'',''))==8
    with pytest.raises(PermissionError):resources(devices,dict(authority,allocation_check_complete=False),rows,'','')
    with pytest.raises(RuntimeError):resources(devices,authority,rows,devices[0]+',123,100','')

@pytest.mark.parametrize('failure',[False,True])
def test_supervisor_mock_no_restart(tmp_path,failure):
    import run,os,time
    from runtime_contract import save
    output=tmp_path/'run';output.mkdir();a=tmp_path/'NOT_AN_AUTHORIZATION.json';save(a,{'simulation_only':True})
    ledger=output/'activation.json';save(ledger,{'start_epoch':time.time(),'deadline_epoch':time.time()+3600,'source_sha256':'simulation'})
    calls=[]
    class Fake:
        def __init__(self,command,**kw):
            ticket=json.loads(os.read(kw['pass_fds'][0],32768));job=ticket['job'];calls.append(job)
            folder=output/job['directory'];folder.mkdir()
            self.returncode=2 if failure and len(calls)==1 else 0;self.pid=-1
            save(folder/'summary.json',dict(status='incomplete' if self.returncode else 'complete',completed_updates=9000,completed_chains=6,error='simulated_worker_failure' if self.returncode else None))
        def poll(self):return self.returncode
        def wait(self,**kw):return self.returncode
    def summary(*args):
        if failure:raise RuntimeError('simulated_summary_failure')
        return {'status':'complete','simulation_only':True}
    result=run.supervise(output,a,ledger,['test'+str(i) for i in range(8)],spawn=Fake,summarizer=summary,poll=0)
    receipt=json.loads((output/'launcher_summary.json').read_text())
    assert len(calls)==8 and len({x['id'] for x in calls})==8
    assert result==(2 if failure else 0)
    if failure:
        assert receipt['primary_failures'] and receipt['evaluation']['status']=='failed'
        assert 'simulated_worker_failure' in json.dumps(receipt['primary_failures'])

def test_no_approval_refused_before_launch(tmp_path,monkeypatch):
    import runtime_contract as c
    monkeypatch.setattr(c,'verify_assets',lambda:None)
    a=tmp_path/'not_approval.json';b=tmp_path/'cost.json';a.write_text('{"user_formal_authorized":false}');b.write_text('{}')
    with pytest.raises(PermissionError):c.approval_gate(a,b,tmp_path/'output')
