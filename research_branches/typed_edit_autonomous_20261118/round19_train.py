"""Matched uniform replay of the complete new corruption bank, inverse CE only."""
import round18_train as imdb
import round18_deploy as prior
from potential import Potential
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r19_train'
SEEDS=[812,813,814];DATASETS=['pubmed','imdb']


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def data(ds):
    if ds=='imdb':return imdb.data()
    old,new,_,_=prior.old.train.r11.data()
    def convert(z):return {**z,'x':np.concatenate([z['x'][:,:,97:],z['x'][:,:,71:97]],-1)}
    return convert(old),convert(new)


def model(ds):
    ref,_,columns,norm=prior.spec(ds);return Potential(ref.num_nodes,columns,norm['mean'],norm['sd'])


def orders(ds,seed,old,new):
    if ds=='imdb':previous=imdb.orders(seed,old,new)
    else:
        oo,nn,_,_=prior.old.train.r11.data();previous=prior.old.train.r11.prev.order(seed,oo,nn)
    rng=np.random.default_rng(950000+seed)
    return [(oi,rng.choice(len(new['y']),32,replace=False)) for oi,_ in previous]


def verify():
    prior.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();prior.verify();assert b.read(prior.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND19.md',SRC/'potential.py',prior.OUT/'summary.json',prior.OUT/'acceptance.json']
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},datasets=DATASETS,seeds=SEEDS,updates_per_fit=800,new_coefficient_fits=6,new_original_MLP_fits=0))
    banks={ds:data(ds) for ds in DATASETS};seq={ds:{s:orders(ds,s,*banks[ds]) for s in SEEDS} for ds in DATASETS};audits={};cost=20.;parity=0.
    for ds in DATASETS:
        old,new=banks[ds];ref,_,columns,_=prior.spec(ds);m=model(ds);k=m.k;nt=sum(c[0]=='typed_triangles' for c in columns)
        dd=np.rint(new['x'][np.arange(len(new['y'])),new['y'],k:]*m.delta_scale.numpy());tri=dd[:,:nt].sum(1);cyc=np.abs(dd[:,nt:]).sum(1)
        audit=dict(records=len(tri),negative=int((tri<0).sum()),positive=int((tri>0).sum()),zero=int((tri==0).sum()),zero_triangle_nonzero_cycle=int(((tri==0)&(cyc>0)).sum()),sampling={})
        for seed in SEEDS:
            ix=np.concatenate([v[1] for v in seq[ds][seed]]);hist=np.bincount(ix,minlength=len(tri));np.save(OUT/f'{ds}_{seed}_new_sample_counts.npy',hist)
            audit['sampling'][str(seed)]=dict(draws=len(ix),distinct=int((hist>0).sum()),negative=int((tri[ix]<0).sum()),positive=int((tri[ix]>0).sum()),zero=int((tri[ix]==0).sum()),zero_triangle_nonzero_cycle=int(((tri[ix]==0)&(cyc[ix]>0)).sum()))
        if ds=='pubmed':
            legacy=prior.old.train.Model();cp=torch.load(prior.old.train.OUT/'fits/812/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'];legacy.load_state_dict(cp);m.load_state_dict(cp)
            oo,nn,_,_=prior.old.train.r11.data()
            with torch.no_grad():parity=float((legacy(torch.from_numpy(nn['x'][:64]))-m(torch.from_numpy(new['x'][:64]))).abs().max())
            assert parity<=2e-5
        states=[]
        for _ in range(2):
            m=model(ds);o=imdb.optimizer(m);t.seed_rng(812)
            for ix in seq[ds][812][:8]:t.update(m,o,*t.batch(old,new,ix))
            states.append(t.snapshot(m,o))
        assert t.equal(*states);torch.save(states[-1],OUT/f'engineering_{ds}.pt');t.update(m,o,*t.batch(old,new,seq[ds][812][8]));wanted=t.snapshot(m,o)
        cp=torch.load(OUT/f'engineering_{ds}.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
        t.update(m,o,*t.batch(old,new,seq[ds][812][8]));assert t.equal(wanted,t.snapshot(m,o))
        tic=time.monotonic()
        for ix in seq[ds][812][9:17]:t.update(m,o,*t.batch(old,new,ix))
        cost+=2*2400*(time.monotonic()-tic)/8;audits[ds]=audit
    put('replay_coverage.json',audits);put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_updates=52,legacy_input_logit_error=parity,repeat_resume_exact=True));assert cost<=300
    verify();tic=time.monotonic()
    for ds in DATASETS:
        old,new=banks[ds]
        for seed in SEEDS:
            m=model(ds);o=imdb.optimizer(m);t.seed_rng(seed);history=[]
            for step,ix in enumerate(seq[ds][seed],1):
                assert time.monotonic()-tic<300
                loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            folder=OUT/f'fits/{ds}/{seed}';folder.mkdir(parents=True,exist_ok=False);torch.save(t.snapshot(m,o),folder/'checkpoint.pt');diagnostics={}
            for name,z in [('old_train',old),('new_train',new)]:
                met,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');np.savez_compressed(folder/f'{name}.npz',logits=q,targets=z['y']);diagnostics[name]=met
            put(f'fits/{ds}/{seed}/report.json',dict(dataset=ds,seed=seed,updates=800,history=history,training_only_diagnostics=diagnostics,checkpoint_sha256=b.sha(folder/'checkpoint.pt')))
            print('uniform trained',ds,seed,round(time.monotonic()-tic,2),flush=True)
    put('completion.json',dict(completed=True,scientific_updates=4800,new_from_scratch_fits=6,new_original_MLP_fits=0,seconds=time.monotonic()-tic));tic=time.monotonic();error=0.
    for ds in DATASETS:
        old,new=banks[ds]
        for seed in SEEDS:
            m=model(ds);o=imdb.optimizer(m);t.seed_rng(seed)
            for ix in seq[ds][seed]:t.update(m,o,*t.batch(old,new,ix))
            folder=OUT/f'fits/{ds}/{seed}';cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
            for name,z in [('old_train',old),('new_train',new)]:
                _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');saved=np.load(folder/f'{name}.npz');assert np.array_equal(z['y'],saved['targets']);error=max(error,float(abs(q-saved['logits']).max()));assert error<=2e-5
            print('uniform accepted',ds,seed,round(time.monotonic()-tic,2),flush=True)
    verify();put('summary.json',dict(decisions=None,new_coefficient_fits=6,training_only=True));put('acceptance.json',dict(passed=True,exact_replay_updates=4800,logit_error=error,seconds=time.monotonic()-tic))


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
