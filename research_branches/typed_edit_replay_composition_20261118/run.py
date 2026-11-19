"""Frozen replay-composition experiment; reuse and exactly replay the old B control."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREV=ROOT/'research_branches/typed_edit_corruption_coverage_20261118'
sys.path.insert(0,str(PREV))
import train as t
import accept_train as ev
import numpy as np
import torch
import time
import json
import csv
import hashlib
b=t.b
OUT=ROOT/'results/typed_edit_replay_composition_20261118'
SRC=Path(__file__).parent


def put(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def verify():
    t.verify_train()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def orders(seed,old,new):
    target=old['x'][:,:,:5].argmax(-1)[np.arange(len(old['y'])),old['y']]
    pool=np.flatnonzero(target==0)
    delta=np.rint(new['x'][np.arange(len(new['y'])),new['y'],43].astype(float)*np.sqrt(1565))
    neg=np.flatnonzero(delta<0);pos=np.flatnonzero(delta>0)
    assert len(neg)==153 and len(pos)==97 and len(pool)>=32
    rng=np.random.default_rng(930000+seed);result=[]
    for i,_ in t.batches(seed):
        keep=i[:16]
        j=rng.choice(np.setdiff1d(pool,keep),16,replace=False)
        k=np.concatenate([rng.choice(neg,16,replace=False),rng.choice(pos,16,replace=False)])
        result.append((np.concatenate([keep,j]),k))
    return result


def ohash(order):return hashlib.sha256(b''.join(i.tobytes()+j.tobytes() for i,j in order)).hexdigest()


def gate(old,new,splits):
    runs=[];tic=time.monotonic();error=0.;order=orders(810,old,new)
    for seed in t.SEEDS:
        m=t.model(seed);_,q,_=b.c.eval_model(m,splits['old_assessment']['x'],splits['old_assessment']['y'],'cpu')
        oldq=np.load(t.OUT/f'fits/{seed}/base/old_assessment.npz')['logits']
        error=max(error,float(abs(q-oldq).max()))
    assert error<=2e-5
    for _ in range(2):
        m=t.model(810);o=t.optimizer(m);t.seed_rng(810)
        for ix in order[:8]:t.update(m,o,*t.batch(old,new,ix))
        runs.append(t.snapshot(m,o))
    assert t.equal(*runs)
    torch.save(runs[1],OUT/'engineering_resume.pt')
    t.update(m,o,*t.batch(old,new,order[8]));expected=t.snapshot(m,o)
    cp=torch.load(OUT/'engineering_resume.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    t.update(m,o,*t.batch(old,new,order[8]));assert t.equal(expected,t.snapshot(m,o))
    start=time.monotonic()
    for ix in order[9:17]:t.update(m,o,*t.batch(old,new,ix))
    per=(time.monotonic()-start)/8;start=time.monotonic()
    for z in splits.values():b.c.eval_model(m,z['x'],z['y'],'cpu')
    cost=2*(400*per+2*(time.monotonic()-start))+30
    put('engineering.json',dict(passed=cost<=240,cost_bound_seconds=cost,engineering_updates=26,
        duplicate_eight_exact=True,serialized_next_step_exact=True,parent_logit_error=error,seconds=time.monotonic()-tic))
    assert cost<=240


def train():
    b.setup();t.verify_train();assert not (OUT/'manifest.json').exists()
    assert b.read(OUT/'connectivity.json')['completed']
    files=[*SRC.glob('*.py'),SRC/'PROTOCOL.md',OUT/'connectivity.json',t.OUT/'summary.json',t.OUT/'training_acceptance.json']
    for seed in t.SEEDS:
        for kind in ['base','U','B']:
            files+=list((t.OUT/f'fits/{seed}/{kind}').glob('*'))
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},previous_manifest_sha256=b.sha(t.OUT/'training_manifest.json'),
        previous_input_files=b.read(t.OUT/'training_manifest.json')['files'],seed_list=t.SEEDS,
        scientific_updates=400,updates_per_continuation=200,old_control_reused=True))
    old,news,splits=t.data();new=news['B'];gate(old,new,splits);verify();start=time.monotonic()
    for seed in t.SEEDS:
        order=orders(seed,old,new);dest=OUT/f'fits/{seed}/C';dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'batch_indices.npz',old=np.array([i for i,j in order]),new=np.array([j for i,j in order]))
        m=t.model(seed);o=t.optimizer(m);t.seed_rng(seed);before=t.rng_hash();history=[]
        for step,ix in enumerate(order,1):
            assert time.monotonic()-start<240
            loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
        cp=t.snapshot(m,o);torch.save(cp,dest/'checkpoint.pt');after=t.rng_hash()
        assert before==b.read(t.OUT/f'fits/{seed}/B/report.json')['dropout_rng_start']
        assert after==b.read(t.OUT/f'fits/{seed}/B/report.json')['dropout_rng_end']
        t.predict(m,splits,dest)
        put(Path(f'fits/{seed}/C/report.json'),dict(seed=seed,updates=200,initial_sha256=b.sha(t.checkpoint(seed)),
            checkpoint_sha256=b.sha(dest/'checkpoint.pt'),order_sha256=ohash(order),rng_start=before,rng_end=after,history=history))
        print('trained C',seed,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=400,engineering_updates=26,
        new_independent_fits=0,surrogate_chains=0,gpu_calls=0,seconds=time.monotonic()-start))


def accept():
    b.setup();verify();assert b.read(OUT/'completion.json')['completed'];start=time.monotonic()
    old,news,splits=t.data();new=news['B'];rows=[];error=0.;updates=0
    for seed in t.SEEDS:
        for kind in ['base','U','B','C']:
            dest=(OUT if kind=='C' else t.OUT)/f'fits/{seed}/{kind}'
            m=t.model(seed)
            if kind in ['B','C']:
                order=orders(seed,old,new) if kind=='C' else t.batches(seed)
                if kind=='C':
                    saved=np.load(dest/'batch_indices.npz')
                    assert np.array_equal(saved['old'],np.array([i for i,j in order]))
                    assert np.array_equal(saved['new'],np.array([j for i,j in order]))
                cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert b.sha(dest/'checkpoint.pt')==b.read(dest/'report.json')['checkpoint_sha256']
                o=t.optimizer(m);t.seed_rng(seed)
                for ix in order:t.update(m,o,*t.batch(old,new,ix));updates+=1
                assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer'])
                assert torch.equal(torch.get_rng_state(),cp['rng'])
            elif kind=='U':
                m.load_state_dict(torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
            for name,z in splits.items():
                saved=np.load(dest/f'{name}.npz');q=saved['logits']
                assert np.array_equal(saved['targets'],z['y']) and np.array_equal(saved['ids'],z['ids'])
                _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');error=max(error,float(abs(fresh-q).max()));assert error<=2e-5
                rows.append(dict(seed=seed,kind=kind,split=name,groups=ev.score(q,z)))
            print('accepted',seed,kind,round(time.monotonic()-start,2),flush=True)
    decisions=[]
    for seed in t.SEEDS:
        n={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='new_assessment'}
        oldr={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='old_assessment'}
        c=n['C'];oc=oldr['C'];p=oldr['base'];checks={}
        for k in ['base','U']:
            checks['new_negative_plus_two_vs_'+k]=c['negative']['restricted_correct']>=n[k]['negative']['restricted_correct']+2
            checks['new_positive_retained_vs_'+k]=c['positive']['restricted_accuracy']>=n[k]['positive']['restricted_accuracy']-.05
        checks.update(new_negative_retained_vs_B=c['negative']['restricted_accuracy']>=n['B']['negative']['restricted_accuracy']-.05,
            new_positive_plus_two_vs_B=c['positive']['restricted_correct']>=n['B']['positive']['restricted_correct']+2,
            new_switch_ce_vs_U=c['switch']['conditional_ce']<=n['U']['switch']['conditional_ce'],
            old_negative_retained=oc['negative']['restricted_correct']>=p['negative']['restricted_correct'],
            old_positive_retained=oc['positive']['restricted_accuracy']>=p['positive']['restricted_accuracy']-.05,
            old_noop_retained=oc['noop']['accuracy']>=p['noop']['accuracy']-.05,
            old_ce_retained=oc['all']['ce']<=p['all']['ce']*1.05,
            old_bridge_ce_retained=oc['bridge']['ce']<=p['bridge']['ce']*1.05)
        decisions.append(dict(seed=seed,checks=checks,passed=all(checks.values())))
    put('summary.json',dict(per_fit=rows,decisions=decisions,short_generation_gate_passed=all(x['passed'] for x in decisions),development_only=True))
    flat=[dict(seed=r['seed'],kind=r['kind'],split=r['split'],group=g,**v) for r in rows for g,v in r['groups'].items()]
    with (OUT/'per_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    agg=[]
    for split in splits:
        for kind in ['base','U','B','C']:
            for group in rows[0]['groups']:
                rr=[r for r in flat if (r['split'],r['kind'],r['group'])==(split,kind,group)]
                z=dict(split=split,kind=kind,group=group,n_per_seed=rr[0]['n'],existing_parent_seeds=2,new_from_scratch_fits=0)
                for k in ['accuracy','restricted_accuracy','ce','conditional_ce']:
                    vals=[r[k] for r in rr if r[k] is not None]
                    z[k+'_mean']=float(np.mean(vals)) if vals else None
                    z[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                agg.append(z)
    with (OUT/'mean_sd.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    verify();put('acceptance.json',dict(passed=True,exact_replay_updates=updates,B_replay_updates=400,C_replay_updates=400,
        logit_max_error=error,seconds=time.monotonic()-start,short_generation_gate_passed=all(x['passed'] for x in decisions)))


if __name__=='__main__':
    try: {'train':train,'accept':accept}[sys.argv[1]]()
    except BaseException as exc:
        put('failure_'+sys.argv[1]+'.json',dict(type=type(exc).__name__,message=str(exc)));raise
