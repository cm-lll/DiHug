"""Projected nonnegative quadratic coefficients, matched to uniform replay."""
import round19_train as prior
import round19_deploy as dep
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r20_train';SEEDS=prior.SEEDS;DATASETS=prior.DATASETS
data=prior.data;model=prior.model;orders=prior.orders


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def update(m,o,x,y):
    loss,norm=t.update(m,o,x,y)
    with torch.no_grad():m.weights[m.k:].clamp_(min=0.)
    assert torch.isfinite(m.weights).all() and (m.weights[m.k:]>=0).all()
    return loss,norm


def verify():
    dep.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def audit():
    rows=[]
    for ds in DATASETS:
        _,_,columns,_=prior.prior.spec(ds)
        for arm in ['S','U']:
            for seed in SEEDS:
                m=dep.model(ds,f'{arm}_{seed}');k=m.k;w=m.weights.detach().numpy();mu=m.mu.numpy();sd=m.sd.numpy()
                for j,col in enumerate(columns):
                    stationary=None if w[k+j]==0 else float(mu[j]-sd[j]*w[j]/(2*w[k+j]))
                    rows.append(dict(dataset=ds,arm=arm,seed=seed,coordinate=col,linear=float(w[j]),quadratic=float(w[k+j]),mean=float(mu[j]),sd=float(sd[j]),stationary=stationary,stationary_kind='minimum' if w[k+j]>0 else 'maximum' if w[k+j]<0 else 'linear'))
    put('curvature_audit.json',dict(rows=rows,interpretation='Continuous-coordinate curvature only; no discrete-graph convergence or target claim.'))


def main():
    b.setup();dep.verify();assert b.read(dep.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND20.md',SRC/'potential.py',dep.OUT/'summary.json',dep.OUT/'acceptance.json',prior.OUT/'acceptance.json']
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},datasets=DATASETS,seeds=SEEDS,updates_per_fit=800,new_coefficient_fits=6,parameter_constraint='quadratic >= 0 after each optimizer step'))
    audit();banks={ds:data(ds) for ds in DATASETS};seq={ds:{s:orders(ds,s,*banks[ds]) for s in SEEDS} for ds in DATASETS};cost=20.
    for ds in DATASETS:
        old,new=banks[ds];states=[]
        for _ in range(2):
            m=model(ds);o=prior.imdb.optimizer(m);t.seed_rng(812)
            for ix in seq[ds][812][:8]:update(m,o,*t.batch(old,new,ix))
            states.append(t.snapshot(m,o))
        assert t.equal(*states);torch.save(states[-1],OUT/f'engineering_{ds}.pt');update(m,o,*t.batch(old,new,seq[ds][812][8]));wanted=t.snapshot(m,o)
        cp=torch.load(OUT/f'engineering_{ds}.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
        update(m,o,*t.batch(old,new,seq[ds][812][8]));assert t.equal(wanted,t.snapshot(m,o));tic=time.monotonic()
        for ix in seq[ds][812][9:17]:update(m,o,*t.batch(old,new,ix))
        cost+=2*2400*(time.monotonic()-tic)/8
    put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_updates=52,repeat_resume_exact=True,nonnegative_checked_after_every_update=True));assert cost<=300
    verify();tic=time.monotonic()
    for ds in DATASETS:
        old,new=banks[ds]
        for seed in SEEDS:
            m=model(ds);o=prior.imdb.optimizer(m);t.seed_rng(seed);history=[]
            for step,ix in enumerate(seq[ds][seed],1):
                assert time.monotonic()-tic<300
                loss,norm=update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            folder=OUT/f'fits/{ds}/{seed}';folder.mkdir(parents=True,exist_ok=False);torch.save(t.snapshot(m,o),folder/'checkpoint.pt');diagnostics={}
            for name,z in [('old_train',old),('new_train',new)]:
                met,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');np.savez_compressed(folder/f'{name}.npz',logits=q,targets=z['y']);diagnostics[name]=met
            put(f'fits/{ds}/{seed}/report.json',dict(dataset=ds,seed=seed,updates=800,history=history,training_only_diagnostics=diagnostics,minimum_quadratic=float(m.weights[m.k:].min()),checkpoint_sha256=b.sha(folder/'checkpoint.pt')))
            print('projected trained',ds,seed,round(time.monotonic()-tic,2),flush=True)
    put('completion.json',dict(completed=True,scientific_updates=4800,new_from_scratch_fits=6,new_original_MLP_fits=0,seconds=time.monotonic()-tic));tic=time.monotonic();error=0.
    for ds in DATASETS:
        old,new=banks[ds]
        for seed in SEEDS:
            m=model(ds);o=prior.imdb.optimizer(m);t.seed_rng(seed)
            for ix in seq[ds][seed]:update(m,o,*t.batch(old,new,ix))
            folder=OUT/f'fits/{ds}/{seed}';cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
            for name,z in [('old_train',old),('new_train',new)]:
                _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');saved=np.load(folder/f'{name}.npz');assert np.array_equal(z['y'],saved['targets']);error=max(error,float(abs(q-saved['logits']).max()));assert error<=2e-5
            print('projected accepted',ds,seed,round(time.monotonic()-tic,2),flush=True)
    verify();put('summary.json',dict(decisions=None,new_coefficient_fits=6,training_only=True));put('acceptance.json',dict(passed=True,exact_replay_updates=4800,logit_error=error,seconds=time.monotonic()-tic))


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
