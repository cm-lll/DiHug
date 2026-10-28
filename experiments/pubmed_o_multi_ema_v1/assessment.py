"""Predeclared decay-specific, equally weighted TRAINING-seed summaries."""
import math,statistics
from runtime_contract import SEEDS,OLD_SEEDS,NEW_SEEDS,SAMPLES,READOUTS,DECAYS
REFERENCE_FAMILY_EDGES=(39,137,159,17,83,267,27,161,33,3)
def mean(values):return sum(values)/len(values)
def assess(records):
    index={}
    for r in records:
        key=(r['train_seed'],r['readout'],r['sample_seed'])
        if key in index or key[0] not in SEEDS or key[1] not in READOUTS or key[2] not in SAMPLES:raise ValueError('duplicate/unknown identity')
        index[key]=r
    cells={}
    for s in SEEDS:
        for kind in READOUTS:
            rows=[index.get((s,kind,k)) for k in SAMPLES];value=None
            if all(r is not None and r.get('status')=='complete' for r in rows):
                if all(all(r.get(k) is not None and math.isfinite(float(r[k])) for k in ['edges','W1_incident4','LCC']) and r.get('family_edges') is not None for r in rows):
                    for r in rows:
                        if len(r['family_edges'])!=10 or sum(r['family_edges'])!=r['edges']:raise ValueError('family counts')
                    value=dict(edge_absolute_error=mean([abs(r['edges']-926) for r in rows]),signed_bias=mean([r['edges']-926 for r in rows]),
                        W1=mean([r['W1_incident4'] for r in rows]),LCC=mean([r['LCC'] for r in rows]),
                        family_errors=[mean([abs(r['family_edges'][j]-target) for r in rows]) for j,target in enumerate(REFERENCE_FAMILY_EDGES)])
            cells[s,kind]=value
    result=dict(status='complete' if all(x is not None for x in cells.values()) else 'incomplete',primary_decay='ema_0.99',decays={},original_records=records,recommend_adoption=False,
                interpretation='accuracy first; smoothing can lag or preserve bias; warnings are not safety guarantees; no pooled decay selection')
    for kind,decay in DECAYS.items():
        pairs=[]
        for s in SEEDS:
            raw,ema=cells[s,'raw'],cells[s,kind];pair=dict(train_seed=s,group='old_development' if s in OLD_SEEDS else 'new',raw=raw,ema=ema,status='complete' if raw is not None and ema is not None else 'incomplete',difference=None,warnings=None)
            if pair['status']=='complete':
                pair.update(difference=ema['edge_absolute_error']-raw['edge_absolute_error'],family_error_difference=[e-r for e,r in zip(ema['family_errors'],raw['family_errors'])],
                            warnings={'W1_gt_10_percent':ema['W1']>1.1*raw['W1'],'LCC_drop_gt_19_2':raw['LCC']-ema['LCC']>19.2})
            pairs.append(pair)
        groups={}
        for label,seeds in [('all_eight',SEEDS),('old_two',OLD_SEEDS),('new_six',NEW_SEEDS)]:
            rows=[x for x in pairs if x['train_seed'] in seeds];complete=all(x['status']=='complete' for x in rows)
            entry=dict(status='complete' if complete else 'incomplete',expected_seeds=len(seeds),mean_difference=None,median_difference=None,improved=None,tied=None,worsened=None)
            if complete:
                d=[x['difference'] for x in rows];bias={k:[x[k]['signed_bias'] for x in rows] for k in ('raw','ema')}
                entry.update(mean_difference=mean(d),median_difference=statistics.median(d),improved=sum(v<0 for v in d),tied=sum(v==0 for v in d),worsened=sum(v>0 for v in d),
                             signed_bias={k:dict(values=v,mean=mean(v),range=max(v)-min(v),population_sd=statistics.pstdev(v)) for k,v in bias.items()})
            groups[label]=entry
        result['decays'][kind]=dict(decay=decay,primary=kind=='ema_0.99',per_seed=pairs,groups=groups,
                                    warning_interpretation='report each seed; a warning does not discard all panel information')
    return result
