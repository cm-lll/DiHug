"""Complete the fixed-budget context by replay-composition development comparison."""
import round01 as r
import round02 as r2
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
import importlib.util
rp=ROOT/'research_branches/typed_edit_replay_composition_20261118/run.py'
sp=importlib.util.spec_from_file_location('frozen_replay',rp);replay=importlib.util.module_from_spec(sp);sp.loader.exec_module(replay)
OUT=ROOT/'results/typed_edit_autonomous_20261118/r03'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(seed,arm):return r.ContextModel(seed) if arm=='contextC800' else t.model(seed)


def order(seed,old,new):
    target=old['x'][:,:,:5].argmax(-1)[np.arange(len(old['y'])),old['y']];pool=np.flatnonzero(target==0)
    delta=np.rint(new['x'][np.arange(len(new['y'])),new['y'],43].astype(float)*np.sqrt(1565));neg=np.flatnonzero(delta<0);pos=np.flatnonzero(delta>0)
    rng=np.random.default_rng(930000+seed);rr=[]
    for i,_ in r2.order(seed):
        keep=i[:16];j=rng.choice(np.setdiff1d(pool,keep),16,replace=False)
        k=np.concatenate([rng.choice(neg,16,replace=False),rng.choice(pos,16,replace=False)])
        rr.append((np.concatenate([keep,j]),k))
    for (i,j),(a,c) in zip(rr,replay.orders(seed,old,new)):assert np.array_equal(i,a) and np.array_equal(j,c)
    return rr


def resume(seed,arm):
    m=model(seed,arm);o=t.optimizer(m)
    if arm=='C800':
        cp=torch.load(replay.OUT/f'fits/{seed}/C/checkpoint.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);offset=200
    else:t.seed_rng(seed);offset=0
    return m,o,offset


def verify():
    r2.verify();replay.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def main():
    b.setup();r2.verify();replay.verify();OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND03.md',rp,replay.OUT/'acceptance.json',r2.OUT/'summary.json',r2.OUT/'acceptance.json']
    for seed in t.SEEDS:files.append(replay.OUT/f'fits/{seed}/C/checkpoint.pt')
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},new_updates=2800))
    old,news,splits=t.data();new=news['B'];times=[]
    for arm in ['C800','contextC800']:
        m,o,offset=resume(810,arm);seq=order(810,old,new);start=time.monotonic()
        for ix in seq[offset:offset+8]:t.update(m,o,*t.batch(old,new,ix))
        times.append((time.monotonic()-start)/8)
    cost=2*2800*max(times)+35
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,engineering_updates=16,update_seconds=times))
    assert cost<=600;verify();start=time.monotonic()
    for seed in t.SEEDS:
        for arm in ['C800','contextC800']:
            m,o,offset=resume(seed,arm);history=[]
            for step,ix in enumerate(order(seed,old,new)[offset:],offset+1):
                assert time.monotonic()-start<600
                loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
            dest=OUT/f'fits/{seed}/{arm}';dest.mkdir(parents=True,exist_ok=False)
            torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
            put(f'fits/{seed}/{arm}/report.json',dict(seed=seed,arm=arm,total_updates=800,new_updates=800-offset,
                checkpoint_sha256=b.sha(dest/'checkpoint.pt'),history=history))
            print('trained',seed,arm,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,new_scientific_updates=2800,seconds=time.monotonic()-start))
    start=time.monotonic();rows=b.read(r2.OUT/'summary.json')['per_fit'];error=0.
    for seed in t.SEEDS:
        for arm in ['C800','contextC800']:
            m=model(seed,arm);o=t.optimizer(m);t.seed_rng(seed)
            for ix in order(seed,old,new):t.update(m,o,*t.batch(old,new,ix))
            dest=OUT/f'fits/{seed}/{arm}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert b.sha(dest/'checkpoint.pt')==b.read(dest/'report.json')['checkpoint_sha256']
            assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
            for name,z in splits.items():
                saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
                _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');error=max(error,float(abs(q-fresh).max()));assert error<=2e-5
                rows.append(dict(seed=seed,kind=arm,split=name,groups=r.ev.score(q,z)))
            print('accepted',seed,arm,round(time.monotonic()-start,2),flush=True)
    decisions=r.screen(rows,'contextC800')
    for d in decisions:
        seed=d['seed'];newr={v['kind']:v['groups'] for v in rows if v['seed']==seed and v['split']=='new_assessment'}
        oldr={v['kind']:v['groups'] for v in rows if v['seed']==seed and v['split']=='old_assessment'}
        c,n=newr['contextC800'],newr['C800']
        d['checks'].update(matched_negative_plus_two=c['negative']['restricted_correct']>=n['negative']['restricted_correct']+2,
            matched_positive_preserved=c['positive']['restricted_accuracy']>=n['positive']['restricted_accuracy']-.05,
            matched_old_ce_preserved=oldr['contextC800']['all']['ce']<=oldr['C800']['all']['ce']*1.05)
        d['passed']=all(d['checks'].values())
    put('summary.json',dict(per_fit=rows,decisions=decisions,passed=all(d['passed'] for d in decisions)))
    previous=r.OUT;r.OUT=OUT;r.tables(rows);r.OUT=previous
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=3200,logit_error=error,seconds=time.monotonic()-start))
    print('quality',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
