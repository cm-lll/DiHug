"""Independently fitted IMDB potential scorer with a fixed inverse-CE recipe."""
import round17_cache as cache
from potential import Potential
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r18_train';SEEDS=[812,813,814]


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def data():
    return [{k:z[k] for k in z.files} for z in [np.load(cache.OUT/'old_train.npz'),np.load(cache.OUT/'new_train.npz')]]


def model():
    schema=b.read(cache.OUT/'schema.json');norm=b.read(cache.OUT/'normalization.json')
    return Potential(schema['n'],schema['columns'],norm['mean'],norm['sd'])


def orders(seed,old,new):
    ro=np.random.default_rng(920000+seed);rn=np.random.default_rng(930000+seed);rr=np.random.default_rng(940000+seed)
    noop=np.flatnonzero(old['target_action']==0);neg=np.flatnonzero(new['target_triangle_delta']<0);pos=np.flatnonzero(new['target_triangle_delta']>0);out=[]
    for _ in range(800):
        keep=ro.choice(len(old['y']),16,replace=False);extra=rn.choice(np.setdiff1d(noop,keep),16,replace=False)
        out.append((np.concatenate([keep,extra]),np.concatenate([rr.choice(neg,16,replace=False),rr.choice(pos,16,replace=False)])))
    return out


def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=1e-4)


def verify():
    cache.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();cache.verify();assert b.read(cache.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND18.md',SRC/'potential.py',cache.OUT/'acceptance.json',cache.OUT/'schema.json',cache.OUT/'normalization.json',cache.OUT/'old_train.npz',cache.OUT/'new_train.npz']
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},seeds=SEEDS,updates_per_fit=800,coefficient_count=16,new_coefficient_fits=3,new_original_MLP_fits=0))
    old,new=data();seq={seed:orders(seed,old,new) for seed in SEEDS};start=time.monotonic();torch.manual_seed(2026111818)
    x,y=t.batch(old,new,seq[812][0]);m=model();assert (m(x)==0).all()
    with torch.no_grad():m.weights.normal_();perm=torch.randperm(16);eq=float((m(x)[:,perm]-m(x[:,perm])).abs().max())
    xx=x.numpy();mu=m.mu.numpy().astype(float);sd=m.sd.numpy().astype(float);z=(xx[:,:,:m.k]-mu)/sd;d=xx[:,:,m.k:]*m.delta_scale.numpy().astype(float)/sd
    expected=(np.concatenate([-d,-(2*z*d+d*d)],-1)*m.weights.detach().numpy()).sum(-1)
    with torch.no_grad():err=float(abs(expected-m(x).numpy()).max())
    assert eq<=2e-5 and err<=2e-5
    states=[]
    for _ in range(2):
        m=model();o=optimizer(m);t.seed_rng(812)
        for ix in seq[812][:8]:t.update(m,o,*t.batch(old,new,ix))
        states.append(t.snapshot(m,o))
    assert t.equal(*states);torch.save(states[-1],OUT/'engineering.pt');t.update(m,o,*t.batch(old,new,seq[812][8]));wanted=t.snapshot(m,o)
    cp=torch.load(OUT/'engineering.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    t.update(m,o,*t.batch(old,new,seq[812][8]));assert t.equal(wanted,t.snapshot(m,o))
    tic=time.monotonic()
    for ix in seq[812][9:17]:t.update(m,o,*t.batch(old,new,ix))
    cost=2*2400*(time.monotonic()-tic)/8+20;put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_updates=26,independent_score_error=err,permutation_error=eq,repeat_and_resume_exact=True,seconds=time.monotonic()-start))
    assert cost<=300;verify();start=time.monotonic()
    for seed in SEEDS:
        m=model();o=optimizer(m);t.seed_rng(seed);history=[]
        for step,ix in enumerate(seq[seed],1):
            assert time.monotonic()-start<300
            loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
        dest=OUT/f'fits/{seed}';dest.mkdir(parents=True,exist_ok=False);torch.save(t.snapshot(m,o),dest/'checkpoint.pt')
        diagnostics={}
        for name,z in [('old_train',old),('new_train',new)]:
            metric,q,p=b.c.eval_model(m,z['x'],z['y'],'cpu');np.savez_compressed(dest/f'{name}.npz',logits=q,targets=z['y']);diagnostics[name]=metric
        put(f'fits/{seed}/report.json',dict(seed=seed,updates=800,trainable_parameters=16,history=history,training_only_diagnostics=diagnostics,checkpoint_sha256=b.sha(dest/'checkpoint.pt')))
        print('IMDB scorer trained',seed,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=2400,new_from_scratch_fits=3,new_original_MLP_fits=0,seconds=time.monotonic()-start))
    start=time.monotonic();maxerr=0.;reports=[]
    for seed in SEEDS:
        m=model();o=optimizer(m);t.seed_rng(seed)
        for ix in seq[seed]:t.update(m,o,*t.batch(old,new,ix))
        dest=OUT/f'fits/{seed}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
        for name,z in [('old_train',old),('new_train',new)]:
            saved=np.load(dest/f'{name}.npz');assert np.array_equal(saved['targets'],z['y']);metric,q,p=b.c.eval_model(m,z['x'],z['y'],'cpu')
            maxerr=max(maxerr,float(abs(q-saved['logits']).max()));assert maxerr<=2e-5;reports.append(dict(seed=seed,bank=name,training_only=True,metrics=metric))
        print('IMDB scorer accepted',seed,round(time.monotonic()-start,2),flush=True)
    put('summary.json',dict(training_diagnostics=reports,new_coefficient_fits=3,decisions=None,graph_quality_not_yet_evaluated=True))
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=2400,logit_error=maxerr,seconds=time.monotonic()-start))


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
