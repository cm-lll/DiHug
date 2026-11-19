"""Saved-score evaluation and exact CPU replay of the matched coverage continuations."""
from train import *
import csv


def score(q,z):
    x,y=z['x'],z['y'];ac=x[:,:,:5].argmax(-1);ix=np.arange(len(y));target=ac[ix,y]
    delta=np.rint(x[ix,y,43].astype(float)*np.sqrt(1565))
    pq=q.astype(float)-q.max(1,keepdims=True);ce=np.log(np.exp(pq).sum(1))-pq[ix,y]
    masked=np.where((ac==0)|(ac==4),q,-np.inf);rq=masked.astype(float)-masked.max(1,keepdims=True)
    rc=np.log(np.exp(rq).sum(1))-rq[ix,y]
    groups=dict(all=np.ones(len(y),bool),bridge=(z['ids']%2==0 if 'origin' not in z else np.zeros(len(y),bool)),switch=target==4,
        negative=(target==4)&(delta<0),zero=(target==4)&(delta==0),positive=(target==4)&(delta>0),noop=target==0)
    result={}
    for name,mask in groups.items():
        eligible=mask&((target==4)|(target==0))
        result[name]=dict(n=int(mask.sum()),correct=int((q.argmax(1)==y)[mask].sum()),
            restricted_correct=int((masked.argmax(1)==y)[mask].sum()),
            accuracy=float((q.argmax(1)==y)[mask].mean()) if mask.any() else None,
            restricted_accuracy=float((masked.argmax(1)==y)[mask].mean()) if mask.any() else None,
            ce=float(ce[mask].mean()) if mask.any() else None,
            conditional_ce=float(rc[eligible].mean()) if eligible.any() else None)
    return result


def main():
    b.setup();verify_train();old,new,splits=data();rows=[];updates=0;error=0.;tic=time.monotonic()
    assert b.read(OUT/'training_completion.json')['completed']
    for seed in SEEDS:
        for kind in ['base','U','B']:
            m=model(seed);folder=OUT/f'fits/{seed}/{kind}'
            if kind!='base':
                cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);report=b.read(folder/'report.json')
                assert b.sha(folder/'checkpoint.pt')==report['checkpoint_sha256']
                o=optimizer(m);seed_rng(seed);assert rng_hash()==report['dropout_rng_start']
                for ix in batches(seed):update(m,o,*batch(old,new[kind],ix));updates+=1
                assert equal(m.state_dict(),cp['state_dict']) and equal(o.state_dict(),cp['optimizer'])
                assert torch.equal(torch.get_rng_state(),cp['rng']) and rng_hash()==report['dropout_rng_end']
            for name,z in splits.items():
                saved=np.load(folder/f'{name}.npz');q=saved['logits']
                assert np.array_equal(saved['targets'],z['y']) and np.array_equal(saved['ids'],z['ids'])
                _,fresh,_=b.c.eval_model(m,z['x'],z['y'],'cpu');err=float(np.max(abs(q-fresh)))
                error=max(error,err);assert err<=2e-5
                rows.append(dict(seed=seed,kind=kind,split=name,groups=score(q,z)))
            print('accepted',seed,kind,round(time.monotonic()-tic,2),flush=True)
    decisions=[]
    for seed in SEEDS:
        newr={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='new_assessment'}
        oldr={r['kind']:r['groups'] for r in rows if r['seed']==seed and r['split']=='old_assessment'}
        c=newr['B'];oc=oldr['B'];base=oldr['base'];checks={}
        for name in ['base','U']:
            r=newr[name]
            checks['new_negative_plus_two_vs_'+name]=c['negative']['restricted_correct']>=r['negative']['restricted_correct']+2
            checks['new_positive_preserved_vs_'+name]=c['positive']['restricted_accuracy']>=r['positive']['restricted_accuracy']-.05
        checks.update(new_conditional_ce_vs_U=c['switch']['conditional_ce']<=newr['U']['switch']['conditional_ce'],
            old_negative_preserved=oc['negative']['restricted_correct']>=base['negative']['restricted_correct'],
            old_positive_preserved=oc['positive']['restricted_accuracy']>=base['positive']['restricted_accuracy']-.05,
            old_all_ce_preserved=oc['all']['ce']<=base['all']['ce']*1.05,
            old_bridge_ce_preserved=oc['bridge']['ce']<=base['bridge']['ce']*1.05)
        decisions.append(dict(seed=seed,checks=checks,passed=all(checks.values())))
    put('summary.json',dict(per_fit=rows,decisions=decisions,short_generation_gate_passed=all(d['passed'] for d in decisions),
        development_only=True,new_independent_fits=0))
    flat=[]
    for r in rows:
        for group,v in r['groups'].items():flat.append(dict(seed=r['seed'],kind=r['kind'],split=r['split'],group=group,**v))
    with (OUT/'per_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    agg=[]
    for split in splits:
        for kind in ['base','U','B']:
            for group in rows[0]['groups']:
                rr=[r for r in flat if (r['split'],r['kind'],r['group'])==(split,kind,group)]
                row=dict(split=split,kind=kind,group=group,existing_fitting_seeds=2,new_independent_fits=0)
                for metric in ['accuracy','restricted_accuracy','ce','conditional_ce']:
                    values=[r[metric] for r in rr if r[metric] is not None]
                    row[metric+'_mean']=float(np.mean(values)) if values else None
                    row[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
                agg.append(row)
    with (OUT/'mean_sd.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    verify_train()
    put('training_acceptance.json',dict(passed=True,verification_updates=updates,full_replay_exact=True,
        saved_logit_max_error=error,inputs_unchanged=True,seconds=time.monotonic()-tic,
        short_generation_gate_passed=all(d['passed'] for d in decisions),source_sha256=b.sha(__file__)))


if __name__=='__main__':main()
