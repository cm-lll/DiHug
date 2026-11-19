"""Matched 800-step optimization check; retain all earlier checkpoints and outcomes."""
import round01 as r
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r02'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(seed,arm):return r.ContextModel(seed) if arm=='context800' else t.model(seed)
def source(seed,arm):return (r.OUT/f'fits/{seed}/checkpoint.pt') if arm=='context800' else (t.OUT/f'fits/{seed}/B/checkpoint.pt')


def order(seed):
    rng=np.random.default_rng(920000+seed)
    result=[(rng.choice(3000,32,replace=False),rng.choice(256,32,replace=False)) for _ in range(800)]
    for (i,j),(a,c) in zip(result,t.batches(seed)):assert np.array_equal(i,a) and np.array_equal(j,c)
    return result


def resume(seed,arm):
    m=model(seed,arm);o=t.optimizer(m);cp=torch.load(source(seed,arm),map_location='cpu',weights_only=False)
    m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    return m,o


def verify():
    r.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();r.verify();assert b.read(r.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND02.md',r.OUT/'summary.json',r.OUT/'acceptance.json']
    for seed in t.SEEDS:
        for arm in ['B800','context800']:files.append(source(seed,arm))
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},existing_updates_per_fit=200,extra_updates_per_fit=600))
    old,news,splits=t.data();new=news['B'];times=[]
    for arm in ['B800','context800']:
        m,o=resume(810,arm);start=time.monotonic()
        for ix in order(810)[200:208]:t.update(m,o,*t.batch(old,new,ix))
        times.append((time.monotonic()-start)/8)
    cost=2*2400*max(times)+35
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,engineering_updates=16,update_seconds=times))
    assert cost<=600;verify();start=time.monotonic()
    for seed in t.SEEDS:
        for arm in ['B800','context800']:
            m,o=resume(seed,arm);history=[]
            for step,ix in enumerate(order(seed)[200:],201):
                assert time.monotonic()-start<600
                loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            dest=OUT/f'fits/{seed}/{arm}';dest.mkdir(parents=True,exist_ok=False)
            torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
            put(f'fits/{seed}/{arm}/report.json',dict(seed=seed,arm=arm,total_updates=800,new_updates=600,
                checkpoint_sha256=b.sha(dest/'checkpoint.pt'),history=history))
            print('trained',seed,arm,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,new_scientific_updates=2400,seconds=time.monotonic()-start))
    start=time.monotonic();rows=b.read(t.OUT/'summary.json')['per_fit'];error=0.
    for seed in t.SEEDS:
        for arm in ['B800','context800']:
            m=model(seed,arm);o=t.optimizer(m);t.seed_rng(seed)
            for ix in order(seed):t.update(m,o,*t.batch(old,new,ix))
            dest=OUT/f'fits/{seed}/{arm}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert b.sha(dest/'checkpoint.pt')==b.read(dest/'report.json')['checkpoint_sha256']
            assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
            for name,z in splits.items():
                saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
                _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');error=max(error,float(abs(q-fresh).max()));assert error<=2e-5
                rows.append(dict(seed=seed,kind=arm,split=name,groups=r.ev.score(q,z)))
            print('accepted',seed,arm,round(time.monotonic()-start,2),flush=True)
    decisions=r.screen(rows,'context800')
    for d in decisions:
        seed=d['seed'];newr={v['kind']:v['groups'] for v in rows if v['seed']==seed and v['split']=='new_assessment'}
        oldr={v['kind']:v['groups'] for v in rows if v['seed']==seed and v['split']=='old_assessment'}
        c,n=newr['context800'],newr['B800']
        d['checks'].update(matched_negative_plus_two=c['negative']['restricted_correct']>=n['negative']['restricted_correct']+2,
            matched_positive_preserved=c['positive']['restricted_accuracy']>=n['positive']['restricted_accuracy']-.05,
            matched_old_ce_preserved=oldr['context800']['all']['ce']<=oldr['B800']['all']['ce']*1.05)
        d['passed']=all(d['checks'].values())
    put('summary.json',dict(per_fit=rows,decisions=decisions,passed=all(d['passed'] for d in decisions)))
    saved_out=r.OUT;r.OUT=OUT;r.tables(rows);r.OUT=saved_out
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=3200,logit_error=error,seconds=time.monotonic()-start))
    print('quality',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
