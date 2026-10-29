import sys,pathlib,copy,types,importlib.util
import numpy as np
import pytest,torch
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'package'))
torch.set_num_threads(1)
from early_cn import canonical_pairs,edge_cn_residual,install
from runtime_contract import jobs,source_binding,read,ROOT
from summarize import aggregate,old_records

def test_queue():
    q=jobs();assert len(q['training'])==2 and len(q['sampling'])==4
    alljobs=q['training']+q['sampling'];assert len({x['directory'] for x in alljobs})==6
    assert all(j['updates']==3000 for j in q['training'])
    assert all(j['readout']=='ema_0.99' and j['T']==1000 for j in q['sampling'])
    assert {j['depends_on'] for j in q['sampling']}=={j['id'] for j in q['training']}

def test_pair_mapping():
    e=torch.tensor([[2,0,1,0,2,1],[0,2,2,2,1,2]]);f=torch.tensor([1,1,3,1,3,3])
    q,g,i=canonical_pairs(e,f,4)
    assert q.tolist()==[[0,1],[2,2]] and g.tolist()==[1,3] and i.tolist()==[0,0,1,0,1,1]
    values=torch.arange(2*37,dtype=torch.float32).reshape(2,37).requires_grad_()
    projection=torch.nn.Linear(37,64,bias=False)
    fake=types.SimpleNamespace(cn_edge_projection=projection,common_neighbor=types.SimpleNamespace(aggregate=lambda *args:values))
    ctx=types.SimpleNamespace(types=np.zeros(4),x=torch.zeros(4,4))
    graph=types.SimpleNamespace(edges=e,families=f)
    out=edge_cn_residual(fake,ctx,graph,.5)
    assert torch.equal(out,projection(values)[i])
    assert torch.equal(out[0],out[1]) and torch.equal(out[0],out[3])
    out.sum().backward();assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.equal(edge_cn_residual(fake,ctx,graph,.5,True),torch.zeros(6,64))

def test_empty_and_conflicting():
    q,f,i=canonical_pairs(torch.empty(2,0,dtype=torch.long),torch.empty(0,dtype=torch.long),4)
    assert q.shape==(2,0) and i.numel()==0
    with pytest.raises(ValueError):canonical_pairs(torch.tensor([[0,1],[1,0]]),torch.tensor([0,1]),3)
    with pytest.raises(ValueError):canonical_pairs(torch.tensor([[0],[0]]),torch.tensor([0]),3)

def small():
    from hsgd.model import TypedSparseDenoiser
    m=TypedSparseDenoiser(4,10,layers=1,width=256,edge_width=64,heads=8,global_width=128,cn_hidden=32,prior=np.full(10,.2))
    m.arm='E';return m

def test_real_xt_support_and_nonzero_early():
    from hsgd.computational import FrozenTypedContext
    m=install(small());m.eval()
    with torch.no_grad():m.cn_edge_projection.weight.fill_(.01)
    typ=np.zeros(4,dtype=np.int64);e=np.array([[0,1],[2,2]]);f=np.zeros(2,dtype=np.int64)
    query=np.array([[0,0,1],[1,3,3]]);qf=np.zeros(3,dtype=np.int64)
    c=FrozenTypedContext(m,typ,e,f,'x')
    try:
        g=c.assemble(c.transfer_query(c.prepare_query(query,qf,'x'),'x'),'x')
        cn=m.common_neighbor.aggregate(c,torch.tensor(query),torch.tensor(qf),.5)
        assert cn[0,-1]==1 and cn[0,-5].item()==pytest.approx(np.log(2),abs=1e-7)
        assert torch.equal(cn[1:],torch.zeros_like(cn[1:])) # query edges cannot create paths
        residual=edge_cn_residual(m,c,g,.5);assert residual.shape==(g.edges.shape[1],64)
        assert residual.abs().sum()>0
        # Exactly two shared first-layer calls: real Xt context, then main.
        c.close()
    finally:c.close()
    seen=[];h=m.backbone.tf_layers[0].register_forward_pre_hook(lambda mod,args:seen.append(args[1].shape[1]))
    c=FrozenTypedContext(m,typ,e,f,'y')
    try:
        _,l,p=c.predict(query,qf,.5,'y',details=True)
        assert len(seen)==2 and seen[0]==4 and seen[1]==10
        assert l.shape==(3,2);assert m.joint_head[0].in_features==739
        l.sum().backward()
        assert m.cn_edge_projection.weight.grad is not None
    finally:h.remove();c.close()

def test_backbone_absent_residual_unchanged():
    from hsgd.holders import PlaceHolder
    src=ROOT/'tests/fixtures/original_conv_transformer_model.py'
    spec=importlib.util.spec_from_file_location('sealed_backbone',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m=small();new=m.backbone
    old=mod.GraphTransformerConv(1,PlaceHolder(X=4,E=21,y=1,charge=0),dict(dx=256,de=64,dy=128,n_head=8,dim_ffX=512,dim_ffE=128),PlaceHolder(X=4,E=11,y=0,charge=0),dropout=0.,sn_hidden_dim=0)
    old.load_state_dict(new.state_dict());old.eval();new.eval()
    x=torch.eye(4);edges=torch.tensor([[0,1],[1,0]]);attr=torch.zeros(2,21);attr[:,0]=1;attr[:,11]=1
    args=(x,attr,edges,torch.tensor([[.5]]),torch.zeros(4,dtype=torch.long))
    with torch.no_grad():
        a,ah=old(*args,return_hidden=True);b,bh=new(*args,return_hidden=True)
        for k in ah:assert torch.equal(ah[k],bh[k])
        assert torch.equal(a.edge_attr,b.edge_attr)
        _,ch=new(*args,return_hidden=True,edge_residual=torch.zeros(2,64))
        for k in bh:assert torch.equal(bh[k],ch[k])

def test_old_controls_and_comparisons():
    old=old_records();assert len(old)==16
    rec=[];templates={d:{g:{'test':0} for g in ('raw','absolute_error','distribution_error')} for d in ('IMDB','PubMed')}
    for d,seed in read(ROOT/'protocol.json')['train_seeds'].items():
        for s in (2026102821,2026102822):
            for v in ('original','lcc'):
                for arm,val in [('E_earlyCN',3),('E',7),('C' if d=='PubMed' else 'D',2)]:
                    rec.append(dict(dataset=d,train_seed=seed,arm=arm,sample_seed=s,version=v,**{g:{'test':val} for g in templates[d]}))
    a=aggregate(rec,templates)
    assert all(r['difference']==(-4 if r['contrast']=='E_earlyCN-E' else 1) for r in a['paired'])
    rec=[r for r in rec if not(r['dataset']=='IMDB' and r['arm']=='E_earlyCN' and r['sample_seed']==2026102822)]
    a=aggregate(rec,templates)
    assert all(r['difference'] is None for r in a['paired_means'] if r['dataset']=='IMDB')

def test_bindings_and_no_dropout():
    files,_=source_binding()
    for p in ('run.py','run_worker.py','summarize.py','model_io.py','protocol.json','baseline_bindings.json','package/early_cn.py'):
        assert p in files
    assert not (ROOT/'cn_dropout.py').exists()
    assert 'cn_dropout' not in (ROOT/'package/hsgd/model.py').read_text()

@pytest.mark.parametrize('fail',[False,True])
def test_mock_supervisor_no_real_worker(tmp_path,fail):
    import os,time,json
    from run import supervise
    out=tmp_path/'run';out.mkdir()
    approval=tmp_path/'not_authorized.json';approval.write_text(json.dumps({'test_only':True,'user_formal_authorized':False}))
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(start_epoch=time.time(),deadline_epoch=time.time()+3600,source_sha256='mock')))
    launched=[];summaries=[]
    class Dummy:
        def __init__(self,job):self.job=job;self.returncode=2 if fail else 0
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    def spawn(cmd,**kwargs):
        # Consume the actual ticket, but never execute cmd or import worker.
        fd=kwargs['pass_fds'][0];ticket=json.loads(os.read(fd,32768));j=ticket['job'];launched.append(j)
        p=out/j['directory'];p.mkdir(parents=True,exist_ok=True)
        (p/'summary.json').write_text(json.dumps(dict(status='failed' if fail else 'complete',completed_updates=3000,completed_chains=1)))
        return Dummy(j)
    def summary(*args):summaries.append(args);return {'status':'incomplete' if fail else 'complete'}
    code=supervise(out,approval,ledger,['mock0','mock1'],spawn=spawn,summarizer=summary,poll=0)
    r=read(out/'launcher_summary.json')
    if not fail:
        assert code==0 and len(launched)==6
        assert [j['mode'] for j in launched[:2]]==['train','train']
        assert all(j['mode']=='sample' for j in launched[2:])
    else:
        assert code==2 and len(launched)==2 and r['primary_failures'] and r['unstarted']
    assert len(summaries)==1

def test_formal_not_authorized(tmp_path):
    from runtime_contract import approval_gate
    p=tmp_path/'unapproved.json';p.write_text('{"user_formal_authorized":false}')
    c=tmp_path/'cost.json';c.write_text('{}')
    with pytest.raises(PermissionError):approval_gate(p,c,tmp_path/'unused')
