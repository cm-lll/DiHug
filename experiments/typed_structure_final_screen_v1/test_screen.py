import json,sys,pathlib,copy,os,subprocess,time
import numpy as np
import pytest
import torch
import runtime_contract as R
import run
from ema import ParameterEMA
from integration import final_readouts,restore_payload,checkpoint_payload,ema_validation
from summarize import aggregate,mean,factory,load_graph,evaluate_graph,diversity
from lcc_eval import metrics as M

def test_exact_queue_and_dependencies():
    j=R.jobs();assert len(j['training'])==8 and len(j['sampling'])==16
    assert len({x['id'] for x in j['training']+j['sampling']})==24
    assert len({x['directory'] for x in j['training']+j['sampling']})==24
    for t in j['training']:
        c=[s for s in j['sampling'] if s['depends_on']==t['id']]
        assert t['updates']==3000 and len(c)==2 and all(s['slot']==t['slot'] and s['readout']=='ema_0.99' and s['T']==1000 for s in c)
    assert {x['train_seed'] for x in j['training']}=={2026102911,2026102912}

def test_asset_dictionary_bindings():
    R.verify_assets()
    assert R.dictionary_expected('IMDB')['K']==[5,4,3,2]
    assert R.dictionary_expected('PubMed')['K']==[8,8,8,6]

def test_no_formal_budget_or_approval():
    assert R.read(R.ROOT/'protocol.json')['budget_seconds'] is None
    assert R.read(R.ROOT/'protocol.json')['formal_authorized'] is False

def test_ema_formula_theta0_and_rng():
    m=torch.nn.Linear(2,1);e=ParameterEMA(m);old={k:v.clone() for k,v in e.shadow.items()};rng=torch.get_rng_state()
    with torch.no_grad():
        for p in m.parameters():p.add_(1)
    e.update(m,1)
    for n,p in m.named_parameters():torch.testing.assert_close(e.shadow[n],old[n]*.99+p*.01)
    assert torch.equal(rng,torch.get_rng_state())

def test_final3000_only_and_independent_exports():
    m=torch.nn.Linear(1,1);e=ParameterEMA(m)
    with pytest.raises(ValueError):final_readouts(m,e,1000)
    e.updates=3000;x=final_readouts(m,e,3000);x['ema']['weight'].zero_()
    assert not torch.equal(x['ema']['weight'],e.shadow['weight'])

def test_ema_validation_exception_restores():
    m=torch.nn.Linear(1,1);e=ParameterEMA(m);old=copy.deepcopy(m.state_dict());mode=m.training
    def broken(*args):raise RuntimeError('mock')
    with pytest.raises(RuntimeError):ema_validation(broken,m,e)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.items()) and mode==m.training

def test_mock_real_train_loop_milestones(monkeypatch,tmp_path):
    import run_worker as w
    m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters());j=R.jobs()['training'][0];snaps=[];valid=[]
    monkeypatch.setattr(w,'fresh',lambda *a:(m,o,{}, {},None,{}))
    monkeypatch.setattr(w,'rng_get',lambda *a:{})
    monkeypatch.setattr(w,'atomic_checkpoint',lambda p,x:snaps.append((p.name,x['updates'],x['ema']['updates'])))
    monkeypatch.setattr(w,'paired_validation',lambda m,e,*a:valid.append(e.updates) or {})
    monkeypatch.setattr(w,'pair_inputs',lambda *a:[{'t':1},{'t':500}])
    monkeypatch.setattr(w,'input_hash',lambda *a:'mock')
    def step(fn,m,o,c,inp,T,device,e,u):e.updates=u+1;return {'optimizer_steps':1}
    monkeypatch.setattr(w,'complete_normal2',step)
    monkeypatch.setattr(w,'export_readouts',lambda out,m,e,u,b:valid.append(('export',u,e.updates)))
    w.train(j,tmp_path,{},dict(approval_sha256='mock'),lambda:None,{})
    assert snaps==[('initial.pt',0,0),('checkpoint_u1000.pt',1000,1000),('checkpoint_u2000.pt',2000,2000),('final.pt',3000,3000)]
    assert valid==[0,1000,2000,3000,('export',3000,3000)]

def test_readout_wrong_identity_or_bytes_rejected(tmp_path):
    j=R.jobs()['sampling'][0];folder=tmp_path/j['depends_on'];folder.mkdir()
    p=folder/'readout_ema_0.99.pt';p.write_bytes(b'EMA');(folder/'final.pt').write_bytes(b'checkpoint')
    receipt=dict(dataset=j['dataset'],arm=j['arm'],train_seed=j['train_seed'],updates=3000,checkpoint_sha256=R.sha(folder/'final.pt'),files={'ema_0.99':{'sha256':R.sha(p)}})
    R.save(folder/'readouts.json',receipt);R.save(folder/'summary.json',{'status':'complete'})
    assert R.verify_readout(tmp_path,j)[0]==p
    p.write_bytes(b'raw')
    with pytest.raises(ValueError):R.verify_readout(tmp_path,j)
    p.write_bytes(b'EMA');receipt['arm']='D';R.save(folder/'readouts.json',receipt)
    with pytest.raises(ValueError):R.verify_readout(tmp_path,j)

def test_summary_is_in_source_binding():
    f,_=R.source_binding();assert 'summarize.py' in f and 'evaluation_fields.py' in f and 'lcc_eval/metrics.py' in f

def test_output_overlap_rejected(tmp_path):
    protected=tmp_path/'input';protected.mkdir()
    with pytest.raises(ValueError):R.output_check(protected/'new',[protected])
    assert not (protected/'new').exists()

def test_original_lcc_induced_and_actual_capacities():
    types=np.array([0,1,2,3,0]);edges=np.array([[0,1,3],[1,2,4]])
    schema=np.array([(a,b) for a in range(4) for b in range(a,4)])
    g=M.make_graph(types,edges,schema);k,m=M.lcc(g)
    assert m['selected_local_ids']==[0,1,2] and k['edges'].shape[1]==2
    assert sum(M.capacity(k))==3 and sum(M.capacity(g))==10
    np.testing.assert_array_equal(k['degree'],g['degree'][m['selected_local_ids']])

def test_scalar_error_first_and_null_propagation():
    templates={d:{'raw':{'edges':10},'absolute_error':{'edges':0},'distribution_error':{'W1':0}} for d in R.TRAIN_SEEDS}
    rows=[]
    for s,n in zip(R.SAMPLES,[8,12]):
        rows.append(dict(dataset='IMDB',arm='A',sample_seed=s,version='original',raw={'edges':n},absolute_error={'edges':abs(n-10)},distribution_error={'W1':1}))
    out=aggregate(rows,templates)
    row=next(x for x in out['cell_means'] if x['dataset']=='IMDB' and x['arm']=='A' and x['version']=='original' and x['group']=='absolute_error' and x['metric']=='edges');assert row['value']==2
    assert all(x['difference'] is None for x in out['paired'])
    assert mean([1,None]) is None

def test_mp_labels_and_zero_denominator():
    names=['a','b','c','d'];types=np.arange(4);schema=np.array([(a,b) for a in range(4) for b in range(a,4)])
    g=M.make_graph(types,np.array([[0,1,2],[1,2,3]]),schema);F=factory();s=M.stats(g,F,names,lambda:None);v=evaluate_graph(g,g,s,F,names,lambda:None)
    assert [v['raw'][f'MP{i}_total'] for i in (1,2,3)]==[3,2,1]
    assert v['distribution_error']['MP2_composition_TV']==0
    assert any(r['path_ratio'] is None for r in s['typed_twohop_closure'].values())

def test_missing_type_distribution_is_null():
    schema=np.array([(a,b) for a in range(4) for b in range(a,4)]);F=factory();names=['a','b','c','d']
    ref=M.make_graph(np.arange(4),np.array([[0,1,2],[1,2,3]]),schema);g=M.make_graph(np.array([0,1]),np.array([[0],[1]]),schema)
    rs=M.stats(ref,F,names,lambda:None);v=evaluate_graph(g,ref,rs,F,names,lambda:None)
    assert v['distribution_error']['main_incident4_W1'] is None and len(v['comparison']['missing_types'])==2

def test_w1_compatibility_and_exact_self_reference():
    g=load_graph(R.ROOT/'package/assets/IMDB.npz',True);F=factory();names=['Director','Actor','Movie','Genre'];s=M.stats(g,F,names,lambda:None);v=evaluate_graph(g,g,s,F,names,lambda:None)
    assert v['distribution_error']['main_incident4_W1']==0 and v['distribution_error']['all10_compat_W1']==0
    assert s['component_sizes']==[188,2,2]

@pytest.mark.parametrize('fail',[False,True])
def test_supervisor_mock_success_failure(monkeypatch,tmp_path,fail):
    ap=tmp_path/'approval.json';lp=tmp_path/'ledger.json';R.save(ap,{});R.save(lp,dict(start_epoch=time.time(),deadline_epoch=time.time()+2000,source_sha256='mock'))
    launched=[]
    class P:
        def __init__(self,cmd,**kw):
            ticket=json.loads(os.read(kw['pass_fds'][0],32768));j=ticket['job'];launched.append(j);self.returncode=2 if fail else 0
            folder=tmp_path/j['directory'];folder.mkdir(parents=True)
            R.save(folder/'summary.json',dict(status='failed' if fail else 'complete',completed_updates=3000 if j['mode']=='train' else 0,completed_chains=1 if j['mode']=='sample' else 0))
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    code=run.supervise(tmp_path,ap,lp,list(map(str,range(8))),spawn=P,summarizer=lambda *a:dict(status='complete'),poll=0)
    assert code==(2 if fail else 0)
    assert len(launched)==(8 if fail else 24)
    if not fail:assert all(j['mode']=='train' for j in launched[:8])

def test_stop_does_not_spawn(tmp_path):
    ap=tmp_path/'a';lp=tmp_path/'l';R.save(ap,{});R.save(lp,dict(start_epoch=time.time(),deadline_epoch=time.time()+2000,source_sha256='mock'));(tmp_path/'STOP').touch()
    def forbidden(*a,**kw):raise AssertionError('spawned after STOP')
    assert run.supervise(tmp_path,ap,lp,list(map(str,range(8))),spawn=forbidden,summarizer=lambda *a:dict(status='incomplete'),poll=0)==2

def test_public_dry_run_never_creates_output(tmp_path):
    out=tmp_path/'not_created';p=subprocess.run([sys.executable,str(R.ROOT/'run.py'),'dry-run','--output',str(out)],capture_output=True,text=True,check=True)
    v=json.loads(p.stdout);assert not v['formal_activated'] and len(v['sampling'])==16 and not out.exists()

def test_public_missing_run_summary_keeps_all16(tmp_path):
    target=tmp_path/'summary.json'
    p=subprocess.run([sys.executable,str(R.ROOT/'summarize.py'),'--run',str(tmp_path),'--output',str(target),'--deadline',str(time.time()+60)],capture_output=True,text=True)
    assert p.returncode==2,p.stderr
    v=json.loads(target.read_text());assert v['status']=='incomplete' and len(v['inventory'])==16 and len(v['records'])==0
    assert all(x['value'] is None for x in v['cell_means'])
    assert all(x['difference'] is None for x in v['paired_means'])

def test_loaded_ema_then_next_formula_matches():
    m=torch.nn.Linear(1,1);e=ParameterEMA(m);opt=torch.optim.AdamW(m.parameters())
    with torch.no_grad():m.weight.add_(2)
    e.update(m,1);p=checkpoint_payload(m,opt,e,1,{},dict(test=True))
    other=torch.nn.Linear(1,1);oe=ParameterEMA(other);oo=torch.optim.AdamW(other.parameters())
    restore_payload(p,other,oo,oe,dict(test=True),lambda s:None)
    with torch.no_grad():m.weight.add_(1);other.weight.add_(1)
    e.update(m,2);oe.update(other,2)
    assert all(torch.equal(e.shadow[k],oe.shadow[k]) for k in e.shadow)
