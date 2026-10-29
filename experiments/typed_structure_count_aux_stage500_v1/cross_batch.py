"""Identity-safe loss and length comparisons. No historical controls mixed in."""
import math
from runtime_contract import PROTOCOL,jobs
def mean(x):return sum(x)/len(x) if x and all(v is not None and math.isfinite(float(v)) for v in x) else None
def difference(a,b):return a-b if a is not None and b is not None else None
def key(r):return (r['dataset'],r['train_seed'],r['arm'],r['checkpoint'],r['sample_seed'],r['version'])
def aggregate(records,templates):
    expected={key(dict(**j,version=v)) for j in jobs()['sampling'] for v in ('original','lcc')}
    index={}
    for r in records:
        k=key(r)
        if k not in expected or k in index:raise ValueError('unexpected/duplicate identity')
        index[k]=r
    missing=[dict(dataset=d,train_seed=s,arm=a,checkpoint=u,sample_seed=z,version=v,reason='not_present_or_failed') for d,s,a,u,z,v in sorted(expected-set(index))]
    means=[];paired=[];paired_means=[];length=[];length_means=[];cleanup=[]
    for d,seeds in PROTOCOL['train_seeds'].items():
        for v in ('original','lcc'):
            for g in ('raw','absolute_error','distribution_error'):
                metrics=set(templates.get(d,{}).get(g,{}))
                for r in records:
                    if r['dataset']==d:metrics.update(r.get(g,{}))
                for metric in sorted(metrics):
                    def get(seed,arm,u,z,version=v):return index.get((d,seed,arm,u,z,version),{}).get(g,{}).get(metric)
                    def base(seed):return dict(dataset=d,train_seed=seed,version=v,group=g,metric=metric)
                    for seed in seeds:
                        for u in PROTOCOL['generation_checkpoints']:
                            for arm in PROTOCOL['arms']:
                                vals=[get(seed,arm,u,z) for z in PROTOCOL['sample_seeds']];m=mean(vals)
                                means.append(dict(**base(seed),arm=arm,checkpoint=u,value=m,reason=None if m is not None else 'both_samples_required_and_defined'))
                                if v=='lcc':
                                    for z in PROTOCOL['sample_seeds']:cleanup.append(dict(**base(seed),arm=arm,checkpoint=u,sample_seed=z,difference=difference(get(seed,arm,u,z),get(seed,arm,u,z,'original'))))
                            vals=[]
                            for z in PROTOCOL['sample_seeds']:
                                diff=difference(get(seed,'CE_COUNT',u,z),get(seed,'CE',u,z));vals.append(diff)
                                paired.append(dict(**base(seed),checkpoint=u,sample_seed=z,contrast='CE_COUNT-CE',difference=diff,reason=None if diff is not None else 'missing_or_undefined_pair'))
                            m=mean(vals);paired_means.append(dict(**base(seed),checkpoint=u,contrast='CE_COUNT-CE',difference=m,reason=None if m is not None else 'both_pairs_required_and_defined'))
                        for arm in PROTOCOL['arms']:
                            cps=PROTOCOL['generation_checkpoints']
                            for later,earlier in [(cps[i],cps[i-1]) for i in range(1,len(cps))]:
                                vals=[]
                                for z in PROTOCOL['sample_seeds']:
                                    diff=difference(get(seed,arm,later,z),get(seed,arm,earlier,z));vals.append(diff)
                                    length.append(dict(**base(seed),arm=arm,later=later,earlier=earlier,sample_seed=z,difference=diff,reason=None if diff is not None else 'missing_or_undefined_pair'))
                                m=mean(vals);length_means.append(dict(**base(seed),arm=arm,later=later,earlier=earlier,difference=m,reason=None if m is not None else 'both_pairs_required_and_defined'))
    return dict(cell_means=means,paired=paired,paired_means=paired_means,length_paired=length,length_means=length_means,lcc_minus_original=cleanup,missing_identities=missing,comparison_status='complete' if not missing else 'incomplete')
