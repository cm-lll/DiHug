import sys,pathlib,json,time,tempfile
import torch
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from gradient_observation import GradientObservation
from count_aux import count_loss,LAMBDA
from runtime_contract import jobs,PROTOCOL,RESERVE,sha
from summarize import paired_rows,build

def test_frozen_queue_and_no_generation():
    q=jobs()
    assert len(q['training'])==8 and not q['sampling']
    assert len({r['directory'] for r in q['training']})==8
    assert [r['slot'] for r in q['training']]==list(range(8))
    assert all(r['updates']==500 for r in q['training'])
    assert PROTOCOL['checkpoints']==[0,100,250,500]
    assert PROTOCOL['gradient_observation_updates']==[1,250,500]
    assert LAMBDA==PROTOCOL['loss']['lambda_count']==.001 and RESERVE==900
    assert 'def sample(' not in (ROOT/'run_worker.py').read_text()

def test_formula_is_reviewed():
    assert sha(ROOT/'package/count_aux.py')==sha(ROOT.parent/'typed_structure_count_aux_bounded_v2/package/count_aux.py')
    assert sha(ROOT/'formula.json')==sha(ROOT.parent/'typed_structure_count_aux_bounded_v2/formula.json')

def test_observer_noninvasive_half_accumulation_and_alias():
    torch.manual_seed(2026103001)
    m=torch.nn.Linear(2,1);m.shared=m.weight
    before=torch.get_rng_state().clone();obs=GradientObservation(m)
    for x in [torch.tensor([[1.,2.]]),torch.tensor([[3.,4.]])]:
        y=m(x).squeeze();ce=(y-1).square()/2;aux=.001*y.square()/2
        expect=torch.autograd.grad(ce+aux,tuple(m.parameters()),retain_graph=True)
        obs.add(ce,aux)
        assert all(p.grad is None for p in m.parameters())
        got=torch.autograd.grad(ce+aux,tuple(m.parameters()))
        assert all(torch.equal(a,b) for a,b in zip(got,expect))
    assert torch.equal(before,torch.get_rng_state())
    stats=obs.result()
    assert stats['named_parameter_count']==stats['unique_parameter_ids']==2
    assert stats['all_parameters']['weighted_aux_l2']>0

def test_pairing_direction_missing_and_seed():
    def r(seed,arm,x):return dict(dataset='PubMed',train_seed=seed,arm=arm,update=500,readout='raw',t=500,input_sha256='same',absolute_error=x)
    rows=[r(2026102912,'CE',5.),r(2026102912,'CE_COUNT',2.),r(2026102932,'CE',7.)]
    out=paired_rows(rows,['absolute_error'],False)
    a=next(x for x in out if (x['dataset'],x['train_seed'],x['update'],x['readout'],x['t'])==('PubMed',2026102912,500,'raw',500))
    b=next(x for x in out if (x['dataset'],x['train_seed'],x['update'],x['readout'],x['t'])==('PubMed',2026102932,500,'raw',500))
    assert a['absolute_error_difference']==-3 and b['absolute_error_difference'] is None

def test_empty_run_is_incomplete(tmp_path):
    r=build(tmp_path,time.time()+30)
    assert r['status']=='incomplete' and len(r['cells'])==8
    assert all(x['ce_difference'] is None for x in r['paired_validation'])

def test_constant_support_and_half_loss():
    z=torch.tensor([-.3,.4,1.],dtype=torch.float64,requires_grad=True);p=z.sigmoid()
    aux,c=count_loss(p,[0,0,1],[1.,1.,2.],[2,1,0],[0,0,2.])
    manual=(torch.hypot(torch.ones(2,dtype=torch.float64),c['signed_error'][:2])-1).mean()
    assert torch.allclose(aux,manual)
    a=torch.autograd.grad(.001*aux/2,z,retain_graph=True)[0]
    b=torch.autograd.grad(.001*aux,z)[0]
    assert torch.equal(a*2,b)
