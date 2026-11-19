"""Candidate-context augmentation of the original typed-edit MLP; bounded paired trial."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/typed_edit_corruption_coverage_20261118'))
import train as t
import accept_train as ev
import torch
import numpy as np
import time,json,csv
b=t.b
SRC=Path(__file__).parent
OUT=ROOT/'results/typed_edit_autonomous_20261118/r01'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


class ContextModel(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();self.base=t.model(seed);old=self.base.network[0]
        layer=torch.nn.Linear(135,256)
        with torch.no_grad():
            layer.weight.zero_();layer.weight[:,:45].copy_(old.weight);layer.bias.copy_(old.bias)
        self.base.network[0]=layer

    def forward(self,x):
        mask=(x[:,:,0]<.5).to(x.dtype).unsqueeze(-1);den=mask.sum(1,keepdim=True).clamp_min(1)
        mean=(x*mask).sum(1,keepdim=True)/den
        var=((x-mean).square()*mask).sum(1,keepdim=True)/den
        ctx=torch.cat([mean,torch.sqrt(var+1e-8)],-1).expand(-1,x.shape[1],-1)
        return self.base(torch.cat([x,ctx],-1))


def verify():
    t.verify_train()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def screen(rows,kind):
    decisions=[]
    for seed in t.SEEDS:
        n={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='new_assessment'}
        old={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='old_assessment'}
        c=n[kind];oc=old[kind];p=old['base'];checks={}
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
    return decisions


def tables(rows):
    flat=[dict(seed=r['seed'],kind=r['kind'],split=r['split'],group=g,**v) for r in rows for g,v in r['groups'].items()]
    with (OUT/'per_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    agg=[]
    for key in sorted({(r['kind'],r['split'],r['group']) for r in flat}):
        rr=[r for r in flat if (r['kind'],r['split'],r['group'])==key]
        row=dict(kind=key[0],split=key[1],group=key[2],n_per_seed=rr[0]['n'],parent_seeds=2,new_from_scratch_fits=0)
        for metric in ['accuracy','restricted_accuracy','ce','conditional_ce']:
            vals=[r[metric] for r in rr if r[metric] is not None]
            row[metric+'_mean']=float(np.mean(vals)) if vals else None
            row[metric+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        agg.append(row)
    with (OUT/'mean_sd.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)


def main():
    b.setup();t.verify_train();OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND01.md',SRC/'OBJECTIVE.md',t.OUT/'summary.json',t.OUT/'training_acceptance.json']
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},previous_files=b.read(t.OUT/'training_manifest.json')['files']))
    old,news,splits=t.data();new=news['B'];order=t.batches(810);tic=time.monotonic();maxerr=0.;permerr=0.
    for seed in t.SEEDS:
        m=ContextModel(seed).eval();parent=t.model(seed).eval()
        for z in splits.values():
            _,q,_=b.c.eval_model(m,z['x'],z['y'],'cpu');_,p,_=b.c.eval_model(parent,z['x'],z['y'],'cpu')
            maxerr=max(maxerr,float(abs(q-p).max()))
    assert maxerr<=2e-5
    x,y=t.batch(old,new,order[0]);m=ContextModel(810).eval()
    with torch.no_grad():
        m.base.network[0].weight[:,45:].normal_(0,.01)
        perm=torch.randperm(16);permerr=float((m(x)[:,perm]-m(x[:,perm])).abs().max())
    assert permerr<=2e-5
    rr=[]
    for _ in range(2):
        m=ContextModel(810);o=t.optimizer(m);t.seed_rng(810)
        for ix in order[:8]:t.update(m,o,*t.batch(old,new,ix))
        rr.append(t.snapshot(m,o))
    assert t.equal(*rr);torch.save(rr[1],OUT/'engineering_resume.pt')
    t.update(m,o,*t.batch(old,new,order[8]));expected=t.snapshot(m,o)
    cp=torch.load(OUT/'engineering_resume.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(cp['state_dict']);o.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng'])
    t.update(m,o,*t.batch(old,new,order[8]));assert t.equal(expected,t.snapshot(m,o))
    start=time.monotonic()
    for ix in order[9:17]:t.update(m,o,*t.batch(old,new,ix))
    cost=2*400*(time.monotonic()-start)/8+35
    put('engineering.json',dict(passed=cost<=240,parent_logit_error=maxerr,permutation_logit_error=permerr,
        duplicate_and_serialized_next_step_exact=True,engineering_updates=26,cost_bound_seconds=cost,seconds=time.monotonic()-tic))
    assert cost<=240;verify();start=time.monotonic()
    for seed in t.SEEDS:
        m=ContextModel(seed);o=t.optimizer(m);t.seed_rng(seed);history=[]
        for step,ix in enumerate(t.batches(seed),1):
            assert time.monotonic()-start<240
            loss,norm=t.update(m,o,*t.batch(old,new,ix));history.append(dict(step=step,loss=loss,grad_norm=norm))
        dest=OUT/f'fits/{seed}';dest.mkdir(parents=True,exist_ok=False)
        torch.save(t.snapshot(m,o),dest/'checkpoint.pt');t.predict(m,splits,dest)
        put(f'fits/{seed}/report.json',dict(seed=seed,updates=200,checkpoint_sha256=b.sha(dest/'checkpoint.pt'),history=history))
        print('trained',seed,round(time.monotonic()-start,2),flush=True)
    verify();put('completion.json',dict(completed=True,scientific_updates=400,seconds=time.monotonic()-start,new_from_scratch_fits=0,gpu_calls=0))
    start=time.monotonic();rows=b.read(t.OUT/'summary.json')['per_fit'];error=0.
    for seed in t.SEEDS:
        m=ContextModel(seed);o=t.optimizer(m);t.seed_rng(seed)
        for ix in t.batches(seed):t.update(m,o,*t.batch(old,new,ix))
        dest=OUT/f'fits/{seed}';cp=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert b.sha(dest/'checkpoint.pt')==b.read(dest/'report.json')['checkpoint_sha256']
        assert t.equal(m.state_dict(),cp['state_dict']) and t.equal(o.state_dict(),cp['optimizer']) and torch.equal(torch.get_rng_state(),cp['rng'])
        for name,z in splits.items():
            saved=np.load(dest/f'{name}.npz');q=saved['logits'];assert np.array_equal(saved['targets'],z['y'])
            _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');error=max(error,float(abs(q-fresh).max()));assert error<=2e-5
            rows.append(dict(seed=seed,kind='R01',split=name,groups=ev.score(q,z)))
    decisions=screen(rows,'R01');put('summary.json',dict(per_fit=rows,decisions=decisions,passed=all(d['passed'] for d in decisions)))
    tables(rows);verify();put('acceptance.json',dict(passed=True,exact_replay_updates=400,logit_error=error,seconds=time.monotonic()-start))
    print('quality',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
