"""CE-trained state-potential residual versus matched full-model continuation."""
import round10_train_fix as prev
import round10_deploy as dep
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
r=prev.r;OUT=ROOT/'results/typed_edit_autonomous_20261118/r11_train'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def raw_bank(name):
    cache=prev.cache.OUT
    if name in ['old_train','old_assessment']:
        kind='train' if name=='old_train' else 'assessment';ids=range(15) if kind=='train' else range(8,16)
        return np.array([row['raw'] for j in ids for row in b.read(cache/f'mixed/{kind}_{j}/counts.json')],dtype=np.int64)
    keys=[('U' if i%2 else 'B',i) for i in range(8)] if name=='new_B_train' else [('U',i) for i in range(8,12)]+[('B',8),('B',10)]
    return np.array([row['raw'] for kind,tid in keys for row in b.read(cache/f'corruption/{kind}_{tid}/counts.json')],dtype=np.int64)


def data():
    old,new,splits=prev.data();raw=[raw_bank('new_B_train')]
    for root,keys in [(prev.deep.previous.OUT,[(k,i) for i in [100,102] for k in ['U','B']]),(prev.deep.OUT,[('H',i) for i in range(4)])]:
        raw.append(np.array([row['current_raw'] for k,i in keys for row in b.read(root/f'cache/{k}_{i}/record.json')['records']],dtype=np.int64))
    nr=np.concatenate(raw);assert nr.shape==(1280,26)
    def augment(z,raw):
        expected=(np.log1p(raw)/np.log1p(1565)).astype(np.float32)
        assert np.array_equal(expected,z['x'][:,0,45:71])
        return {**z,'x':np.concatenate([z['x'],np.broadcast_to(raw[:,None,:],(len(raw),16,26)).astype(np.float32)],-1)}
    return augment(old,raw_bank('old_train')),augment(new,nr),{k:augment(z,raw_bank(k)) for k,z in splits.items()},nr


class Model(torch.nn.Module):
    def __init__(self,seed,arm):
        super().__init__();self.arm=arm;self.base=prev.GlobalModel(seed)
        self.base.load_state_dict(torch.load(prev.OUT/f'fits/{seed}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
        norm=b.read(OUT/'normalization.json');self.register_buffer('mu',torch.tensor(norm['mean'],dtype=torch.float32))
        self.register_buffer('sd',torch.tensor(norm['sd'],dtype=torch.float32));self.register_buffer('delta_scale',torch.tensor([np.sqrt(1565)]*20+[1565]*6,dtype=torch.float32))
        if arm=='P':
            for p in self.base.parameters():p.requires_grad_(False)
            self.weights=torch.nn.Parameter(torch.zeros(52))

    def phi(self,x):
        z=(x[:,:,97:]-self.mu)/self.sd;d=x[:,:,71:97]*self.delta_scale/self.sd
        return torch.cat([-d,-(2*z*d+d*d)],-1)

    def forward(self,x):
        q=self.base(x[:,:,:97])
        return q+(self.phi(x)*self.weights).sum(-1) if self.arm=='P' else q


def optimizer(m):return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01 if m.arm=='P' else 1e-4,weight_decay=1e-4)


def verify():
    prev.verify();dep.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();prev.verify();assert b.read(dep.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    old,new,splits,raw=data();mu=raw.mean(0);sd=np.maximum(raw.std(0),1)
    put('normalization.json',dict(mean=mu.tolist(),sd=sd.tolist(),rows=1280,training_only=True))
    files=[Path(__file__),SRC/'ROUND11.md',prev.OUT/'acceptance.json',prev.OUT/'summary.json',dep.OUT/'acceptance.json',dep.OUT/'summary.json',OUT/'normalization.json']
    for seed in t.SEEDS:files.append(prev.OUT/f'fits/{seed}/checkpoint.pt')
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},arms=['P','C'],seeds=t.SEEDS,steps_per_arm_seed=800))
    orders={seed:prev.order(seed,old,new) for seed in t.SEEDS};error=0.;start=time.monotonic();times=[]
    for seed in t.SEEDS:
        for arm in ['P','C']:
            m=Model(seed,arm).eval()
            for name,z in splits.items():
                _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');saved=np.load(prev.OUT/f'fits/{seed}/{name}.npz')['logits']
                error=max(error,float(abs(q-saved).max()));assert error<=2e-5
    x,y=t.batch(old,new,orders[810][0]);m=Model(810,'P').eval()
    xx=x.numpy();zz=(xx[:,:,97:]-mu)/sd;dd=xx[:,:,71:97]*np.array([np.sqrt(1565)]*20+[1565]*6)/sd
    independent=np.concatenate([-dd,-(2*zz*dd+dd*dd)],-1);pe=float(abs(independent-m.phi(x).numpy()).max());assert pe<=2e-5
    assert (m.phi(x)[x[:,:,0]>.5]==0).all()
    with torch.no_grad():
        m.weights.normal_();permutation=torch.randperm(16);eq=float((m(x)[:,permutation]-m(x[:,permutation])).abs().max())
    assert eq<=2e-5
    for arm in ['P','C']:
        states=[]
        for _ in range(2):
            m=Model(810,arm);o=optimizer(m);t.seed_rng(810);frozen={k:v.clone() for k,v in m.base.state_dict().items()}
            for ix in orders[810][:8]:t.update(m,o,*t.batch(old,new,ix))
            if arm=='P':assert t.equal(frozen,m.base.state_dict())
            states.append(t.snapshot(m,o))
        assert t.equal(*states);torch.save(states[1],OUT/f'engineering_{arm}.pt')
        t.update(m,o,*t.batch(old,new,orders[810][8]));wanted=t.snapshot(m,o)
        cp=torch.load(OUT/f'engineering_{arm}.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
        t.update(m,o,*t.batch(old,new,orders[810][8]));assert t.equal(wanted,t.snapshot(m,o))
        tic=time.monotonic()
        for ix in orders[810][9:17]:t.update(m,o,*t.batch(old,new,ix))
        times.append((time.monotonic()-tic)/8)
    cost=2*1600*sum(times)+35;put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,engineering_updates=52,
        initial_logit_error=error,independent_phi_error=pe,permutation_error=eq,frozen_weights_exact=True,seconds=time.monotonic()-start))
    assert cost<=600;verify();start=time.monotonic()
    for seed in t.SEEDS:
        for arm in ['P','C']:
            m=Model(seed,arm);o=optimizer(m);t.seed_rng(seed);frozen={k:v.clone() for k,v in m.base.state_dict().items()};history=[]
            for step,ix in enumerate(orders[seed],1):
                assert time.monotonic()-start<600
                loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            if arm=='P':assert t.equal(frozen,m.base.state_dict())
            dest=OUT/f'fits/{seed}/{arm}';dest.mkdir(parents=True,exist_ok=False);torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
            put(f'fits/{seed}/{arm}/report.json',dict(seed=seed,arm=arm,updates=800,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                checkpoint_sha256=b.sha(dest/'checkpoint.pt'),history=history))
            print('trained potential comparison',seed,arm,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=3200,seconds=time.monotonic()-start,new_from_scratch_fits=0))
    start=time.monotonic();rows=b.read(prev.OUT/'summary.json')['per_fit'];maxerr=0.
    for seed in t.SEEDS:
        for arm in ['P','C']:
            m=Model(seed,arm);o=optimizer(m);t.seed_rng(seed)
            for ix in orders[seed]:t.update(m,o,*t.batch(old,new,ix))
            dest=OUT/f'fits/{seed}/{arm}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
            for name,z in splits.items():
                saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
                _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');maxerr=max(maxerr,float(abs(q-fresh).max()));assert maxerr<=2e-5
                rows.append(dict(seed=seed,kind='potential800' if arm=='P' else 'continued800',split=name,groups=r.ev.score(q,z)))
            print('accepted potential comparison',seed,arm,round(time.monotonic()-start,2),flush=True)
    decisions={arm:r.screen(rows,kind) for arm,kind in [('P','potential800'),('C','continued800')]}
    put('summary.json',dict(per_fit=rows,decisions=decisions,development_only=True));saved_output=r.OUT;r.OUT=OUT;r.tables(rows);r.OUT=saved_output
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=3200,logit_error=maxerr,seconds=time.monotonic()-start))
    print('classification decisions',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
