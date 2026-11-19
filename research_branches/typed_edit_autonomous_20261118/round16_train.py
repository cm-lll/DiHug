"""Frozen standalone recipe: three new optimization-order fits for other starts."""
import round11 as r11
import round15 as prior
import csv
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
r=r11.r;OUT=ROOT/'results/typed_edit_autonomous_20261118/r16_train'
SEEDS=[812,813,814]


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


class Model(torch.nn.Module):
    def __init__(self,seed=None):
        super().__init__();norm=b.read(r11.OUT/'normalization.json')
        self.register_buffer('mu',torch.tensor(norm['mean'],dtype=torch.float32));self.register_buffer('sd',torch.tensor(norm['sd'],dtype=torch.float32))
        self.register_buffer('delta_scale',torch.tensor([np.sqrt(1565)]*20+[1565]*6,dtype=torch.float32));self.weights=torch.nn.Parameter(torch.zeros(52))
    def phi(self,x):return r11.Model.phi(self,x)
    def forward(self,x):return (self.phi(x)*self.weights).sum(-1)


def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=1e-4)


def verify():
    r11.verify();prior.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();r11.verify();assert b.read(prior.OUT/'acceptance.json')['passed'];assert b.read(prior.OUT/'summary.json')['new_stream_development_passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND16.md',r11.OUT/'normalization.json',r11.OUT/'raw_precision.json',prior.OUT/'summary.json',prior.OUT/'acceptance.json']
    for sid in [106,107]:files.extend([b.RUN/f'prefixes/pubmed/seed{sid}/{name}' for name in ['initial.json','target.json','binding.json']])
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},steps_per_seed=800,seeds=SEEDS,new_coefficient_fits=3,random_initializations=0,supervision='true inverse cross entropy'))
    old,new,splits,_=r11.data();orders={seed:r11.prev.order(seed,old,new) for seed in SEEDS};tic=time.monotonic()
    x,y=t.batch(old,new,orders[812][0]);m=Model().eval()
    with torch.no_grad():m.weights.normal_();perm=torch.randperm(16);eq=float((m(x)[:,perm]-m(x[:,perm])).abs().max())
    assert eq<=2e-5
    norm=b.read(r11.OUT/'normalization.json');xx=x.numpy();z=(xx[:,:,97:]-np.array(norm['mean']))/np.array(norm['sd'])
    d=xx[:,:,71:97]*np.array([np.sqrt(1565)]*20+[1565]*6)/np.array(norm['sd'])
    expected=(np.concatenate([-d,-(2*z*d+d*d)],-1)*m.weights.detach().numpy()).sum(-1)
    with torch.no_grad():err=float(abs(expected-m(x).numpy()).max())
    assert err<=2e-5
    states=[]
    for _ in range(2):
        m=Model();o=optimizer(m);t.seed_rng(812)
        for ix in orders[812][:8]:t.update(m,o,*t.batch(old,new,ix))
        states.append(t.snapshot(m,o))
    assert t.equal(*states);torch.save(states[-1],OUT/'engineering.pt');t.update(m,o,*t.batch(old,new,orders[812][8]));wanted=t.snapshot(m,o)
    cp=torch.load(OUT/'engineering.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    t.update(m,o,*t.batch(old,new,orders[812][8]));assert t.equal(wanted,t.snapshot(m,o))
    start=time.monotonic()
    for ix in orders[812][9:17]:t.update(m,o,*t.batch(old,new,ix))
    cost=2*2400*(time.monotonic()-start)/8+20;put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_updates=26,
        independent_score_error=err,permutation_error=eq,repeat_and_resume_exact=True,seconds=time.monotonic()-tic))
    assert cost<=300;verify();start=time.monotonic()
    for seed in SEEDS:
        m=Model();o=optimizer(m);t.seed_rng(seed);history=[]
        for step,ix in enumerate(orders[seed],1):
            assert time.monotonic()-start<300
            loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
        dest=OUT/f'fits/{seed}';dest.mkdir(parents=True,exist_ok=False);torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
        put(f'fits/{seed}/report.json',dict(seed=seed,updates=800,trainable_parameters=52,history=history,checkpoint_sha256=b.sha(dest/'checkpoint.pt')))
        print('trained standalone potential',seed,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=2400,seconds=time.monotonic()-start,new_from_scratch_fits=3,new_original_MLP_fits=0))
    rows=[];start=time.monotonic();maxerr=0.
    for seed in SEEDS:
        m=Model();o=optimizer(m);t.seed_rng(seed)
        for ix in orders[seed]:t.update(m,o,*t.batch(old,new,ix))
        dest=OUT/f'fits/{seed}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
        for name,z in splits.items():
            saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
            _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');maxerr=max(maxerr,float(abs(q-fresh).max()));assert maxerr<=2e-5
            rows.append(dict(seed=seed,kind='standalone800_new_order',split=name,groups=r.ev.score(q,z)))
        print('accepted standalone potential',seed,round(time.monotonic()-start,2),flush=True)
    put('summary.json',dict(per_fit=rows,decisions=None,old_parent_screen_applicable=False,new_coefficient_fits=3))
    flat=[dict(seed=v['seed'],split=v['split'],group=k,**q) for v in rows for k,q in v['groups'].items()]
    with (OUT/'per_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    agg=[]
    for split,group in sorted({(v['split'],v['group']) for v in flat}):
        rr=[v for v in flat if v['split']==split and v['group']==group];row=dict(split=split,group=group,optimization_repeats=3,n_per_repeat=rr[0]['n'])
        for metric in ['accuracy','restricted_accuracy','ce','conditional_ce']:
            values=[v[metric] for v in rr if v[metric] is not None];row[metric+'_mean']=float(np.mean(values)) if values else None;row[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        agg.append(row)
    with (OUT/'mean_sd.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=2400,logit_error=maxerr,seconds=time.monotonic()-start))
    print('new optimization fits accepted; no corresponding original MLP parent screen',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
