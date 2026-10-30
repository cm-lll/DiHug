import pathlib,copy,contextlib,types,json
import pytest
import torch,numpy as np
from runtime_contract import ROOT,jobs,PROTOCOL,generation_checkpoints
from cross_batch import aggregate
from resume_io import validate_parent

def test_queue():
    q=jobs();assert len(q['training'])==8 and len(q['sampling'])==24
    assert sum(j['updates']-500 for j in q['training'])==32000
    assert len({j['directory'] for j in q['sampling']})==24
    assert [j['slot'] for j in q['training']]==list(range(8))
    assert all(j['resume_updates']==500 for j in q['training'])
    assert generation_checkpoints('IMDB')==[3000] and generation_checkpoints('PubMed')==[3000,6000]
def fixture_records():
    rows=[]
    for j in jobs()['sampling']:
        for version in ['original','lcc']:
            x=j['checkpoint']/1000+(2 if j['arm']=='CE_COUNT' else 0)
            rows.append(dict(**j,version=version,raw={'edges':x},absolute_error={'edges':abs(x-10)},distribution_error={'main_incident4_W1':x}))
    templates={d:dict(raw={'edges':0},absolute_error={'edges':0},distribution_error={'main_incident4_W1':0}) for d in PROTOCOL['train_seeds']}
    return rows,templates
def test_comparison_direction_and_no_phantom_imdb6000():
    rows,t=fixture_records();r=aggregate(rows,t)
    assert r['comparison_status']=='complete'
    assert not any(x['dataset']=='IMDB' and x['checkpoint']==6000 for x in r['cell_means'])
    assert all(x['difference']==-2 for x in r['paired_means'] if x['group']=='absolute_error')
    assert all(x['dataset']=='PubMed' and x['difference']==-3 for x in r['length_means'] if x['group']=='absolute_error')
def test_missing_sample_is_not_averaged():
    rows,t=fixture_records();bad=next(x for x in rows if x['dataset']=='PubMed' and x['arm']=='CE_COUNT' and x['checkpoint']==3000 and x['version']=='original');rows.remove(bad)
    r=aggregate(rows,t);assert r['comparison_status']=='incomplete'
    a=[x for x in r['paired_means'] if x['dataset']==bad['dataset'] and x['train_seed']==bad['train_seed'] and x['checkpoint']==3000 and x['version']=='original']
    assert a and all(x['difference'] is None for x in a)
def test_parent_source_and_counter_cannot_be_bypassed():
    j=jobs()['training'][0];entry=json.loads((ROOT/'parent_checkpoints.json').read_text())[j['id']];b=json.loads(pathlib.Path(entry['binding_path']).read_text())
    p=dict(binding=b,updates=500,ema=dict(updates=500,decay=.99),rng=dict.fromkeys(['numpy','global_numpy','python','torch','device']))
    validate_parent(p,b,j,b['config'])
    a=copy.deepcopy(p);a['binding']['source_sha256']='bad'
    with pytest.raises(ValueError):validate_parent(a,a['binding'],j,b['config'])
    a=copy.deepcopy(p);a['updates']=499
    with pytest.raises(ValueError):validate_parent(a,b,j,b['config'])

@pytest.mark.parametrize('dataset',['IMDB','PubMed'])
def test_actual_loop_with_mock_updates_and_mock_sampling(monkeypatch,tmp_path,dataset):
    import run_worker as w
    job=next(j for j in jobs()['training'] if j['dataset']==dataset and j['arm']=='CE_COUNT')
    out=tmp_path/job['directory'];out.mkdir()
    m=torch.nn.Linear(1,1);m.count_aux_weight=.001
    opt=types.SimpleNamespace(state_dict=lambda:{})
    ema=types.SimpleNamespace(updates=500,at_update_boundary=True,state_dict=lambda:{'updates':ema.updates})
    rng=np.random.default_rng(5);events=[];counters=[]
    monkeypatch.setattr(w,'resume_parent',lambda job:(m,opt,ema,{}, {},rng,{'parent':{'sha256':'parent'}}))
    monkeypatch.setattr(w,'rng_get',lambda r,d:copy.deepcopy(r.bit_generator.state))
    monkeypatch.setattr(w,'atomic_checkpoint',lambda path,payload:events.append(('save',payload['updates'])))
    monkeypatch.setattr(w,'pair_inputs',lambda c,f,p,r:[{'t':1,'position':p},{'t':2,'position':p}])
    monkeypatch.setattr(w,'input_hash',lambda x:str(x[0]['position']))
    def step(fn,model,opt,case,inputs,T,device,e,completed):
        assert completed==e.updates and inputs[0]['position']==completed+1
        e.updates+=1;counters.append(e.updates)
        return {'optimizer_steps':1,'gradient_observation':None}
    monkeypatch.setattr(w,'complete_normal2',step)
    monkeypatch.setattr(w,'paired_validation',lambda *a:dict(raw={},ema_099={}))
    monkeypatch.setattr(w,'export_readouts',lambda out,m,e,u,b:events.append(('readout',u)))
    @contextlib.contextmanager
    def isolated(*args):
        before=copy.deepcopy(rng.bit_generator.state);events.append(('isolate',ema.updates))
        try:yield
        finally:rng.bit_generator.state=before
    monkeypatch.setattr(w,'isolated_readout',isolated)
    def sample(j,*args,**kwargs):events.append(('sample',j['checkpoint'],j['sample_seed']));rng.random()
    monkeypatch.setattr(w,'sample',sample)
    progress={'completed_chains':0}
    w.train(job,out,{'next_input501_hashes':{job['id']:'501'}},{'approval_sha256':'approval','output':str(tmp_path)},lambda:None,progress)
    assert counters==list(range(501,job['updates']+1))
    assert progress['completed_updates']==job['updates']
    assert [x[1] for x in events if x[0]=='readout']==generation_checkpoints(dataset)
    assert progress['completed_chains']==(2 if dataset=='IMDB' else 4)
    assert [x[1] for x in events if x[0]=='save']==[500]+[u for u in PROTOCOL['checkpoints'] if u<=job['updates']]
