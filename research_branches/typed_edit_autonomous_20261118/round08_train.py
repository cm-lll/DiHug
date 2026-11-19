"""Current-state plus candidate consequences; unchanged inverse CE and C replay."""
import round01 as r
import round03 as r3
import round06_cache_fast as cache
import round08_cache as delta_cache
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r08_train'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


class GlobalModel(r.ContextModel):
    def __init__(self,seed):
        super().__init__(seed);old=self.base.network[0];self.global_width=b.read(cache.OUT/'schema.json')['width']
        layer=torch.nn.Linear(135+2*self.global_width,256)
        with torch.no_grad():
            layer.weight.zero_();layer.weight[:,:135].copy_(old.weight);layer.bias.copy_(old.bias)
        self.base.network[0]=layer

    def forward(self,x):
        local=x[:,:,:45];glob=x[:,:,45:];assert glob.shape[-1]==2*self.global_width
        mask=(local[:,:,0]<.5).to(x.dtype).unsqueeze(-1);den=mask.sum(1,keepdim=True).clamp_min(1)
        mean=(local*mask).sum(1,keepdim=True)/den
        var=((local-mean).square()*mask).sum(1,keepdim=True)/den
        ctx=torch.cat([mean,torch.sqrt(var+1e-8)],-1).expand(-1,x.shape[1],-1)
        return self.base(torch.cat([local,ctx,glob],-1))


def augment(z,name):
    q=np.load(cache.OUT/f'{name}.npz');assert np.array_equal(q['ids'],z['ids'])
    ctx=np.broadcast_to(q['context'][:,None,:],(len(z['y']),16,q['context'].shape[1]))
    d=np.load(delta_cache.OUT/f'{name}.npz');assert np.array_equal(d['ids'],z['ids'])
    return {**z,'x':np.concatenate([z['x'],ctx,d['delta']],-1)}


def data():
    old,news,splits=t.data()
    return augment(old,'old_train'),augment(news['B'],'new_B_train'),{name:augment(z,name) for name,z in splits.items()}


def verify():
    delta_cache.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();delta_cache.verify();assert b.read(delta_cache.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND08.md',delta_cache.OUT/'acceptance.json',SRC/'candidate_consequences.py',cache.OUT/'cache_acceptance.json',cache.OUT/'schema.json',
        *[cache.OUT/f'{name}.npz' for name in ['old_train','old_assessment','new_B_train','new_assessment']],
        r3.OUT/'summary.json',r3.OUT/'acceptance.json',
        *[delta_cache.OUT/f'{name}.npz' for name in ['old_train','old_assessment','new_B_train','new_assessment']]]
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},steps_per_seed=800,seed_list=t.SEEDS,supervision='original inverse CE'))
    old,new,splits=data();orders={seed:r3.order(seed,old,new) for seed in t.SEEDS};maxerr=0.;tic=time.monotonic()
    for seed in t.SEEDS:
        m=GlobalModel(seed).eval()
        for name,z in splits.items():
            _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu')
            saved=np.load(t.OUT/f'fits/{seed}/base/{name}.npz')['logits'];maxerr=max(maxerr,float(abs(q-saved).max()))
    assert maxerr<=2e-5
    m=GlobalModel(810).eval();x,y=t.batch(old,new,orders[810][0])
    with torch.no_grad():
        m.base.network[0].weight[:,45:].normal_(0,.01);perm=torch.randperm(16)
        pe=float((m(x)[:,perm]-m(x[:,perm])).abs().max())
    assert pe<=2e-5
    states=[]
    for _ in range(2):
        m=GlobalModel(810);o=t.optimizer(m);t.seed_rng(810)
        for ix in orders[810][:8]:t.update(m,o,*t.batch(old,new,ix))
        states.append(t.snapshot(m,o))
    assert t.equal(*states);torch.save(states[1],OUT/'engineering_resume.pt')
    t.update(m,o,*t.batch(old,new,orders[810][8]));expected=t.snapshot(m,o)
    cp=torch.load(OUT/'engineering_resume.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    t.update(m,o,*t.batch(old,new,orders[810][8]));assert t.equal(expected,t.snapshot(m,o))
    start=time.monotonic()
    for ix in orders[810][9:17]:t.update(m,o,*t.batch(old,new,ix))
    cost=2*1600*(time.monotonic()-start)/8+35
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,engineering_updates=26,
        parent_logit_error=maxerr,permutation_logit_error=pe,repeat_and_serialized_next_step_exact=True,seconds=time.monotonic()-tic))
    assert cost<=600;verify();start=time.monotonic()
    for seed in t.SEEDS:
        m=GlobalModel(seed);o=t.optimizer(m);t.seed_rng(seed);history=[]
        for step,ix in enumerate(orders[seed],1):
            assert time.monotonic()-start<600
            loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
        dest=OUT/f'fits/{seed}';dest.mkdir(parents=True,exist_ok=False)
        torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
        put(f'fits/{seed}/report.json',dict(seed=seed,updates=800,checkpoint_sha256=b.sha(dest/'checkpoint.pt'),history=history))
        print('trained consequence',seed,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=1600,seconds=time.monotonic()-start,new_from_scratch_fits=0))
    start=time.monotonic();rows=b.read(r3.OUT/'summary.json')['per_fit'];error=0.
    for seed in t.SEEDS:
        m=GlobalModel(seed);o=t.optimizer(m);t.seed_rng(seed)
        for ix in orders[seed]:t.update(m,o,*t.batch(old,new,ix))
        dest=OUT/f'fits/{seed}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert b.sha(dest/'checkpoint.pt')==b.read(dest/'report.json')['checkpoint_sha256']
        assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
        for name,z in splits.items():
            saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
            _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');error=max(error,float(abs(q-fresh).max()));assert error<=2e-5
            rows.append(dict(seed=seed,kind='consequenceC800',split=name,groups=r.ev.score(q,z)))
        print('accepted consequence',seed,round(time.monotonic()-start,2),flush=True)
    decisions=r.screen(rows,'consequenceC800')
    put('summary.json',dict(per_fit=rows,decisions=decisions,classification_screen_passed=all(x['passed'] for x in decisions),development_only=True))
    previous=r.OUT;r.OUT=OUT;r.tables(rows);r.OUT=previous
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=1600,logit_error=error,seconds=time.monotonic()-start))
    print('classification screen',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
