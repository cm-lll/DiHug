"""Frozen per-seed interpretation rules, not a new graph evaluator.

Caller adapts the existing evaluator to edges, W1_incident4, LCC, family_edges.
All original metrics remain in the returned records; no success-only average.
"""
import math
SEEDS=(2026102812,2026102833)
SAMPLES=(2026102821,2026102822)
REFERENCE_FAMILY_EDGES=(39,137,159,17,83,267,27,161,33,3)
def assess(records):
    lookup={}
    for r in records:
        key=(r['train_seed'],r['readout'],r['sample_seed'])
        if key in lookup:raise ValueError('duplicate identity')
        if key[0] not in SEEDS or key[1] not in ('raw','ema') or key[2] not in SAMPLES:raise ValueError('unexpected identity')
        lookup[key]=r
    result={'per_seed':[],'original_records':records,'scope':'exploratory two training seeds; no statistical guarantee'}
    for seed in SEEDS:
        cell={'train_seed':seed,'status':'complete','raw':None,'ema':None,'redlines':None}
        for readout in ('raw','ema'):
            rs=[lookup.get((seed,readout,s)) for s in SAMPLES]
            if any(r is None or r.get('status')!='complete' for r in rs):
                cell['status']='incomplete';continue
            keys=['edges','W1_incident4','LCC']
            if any(any(r.get(k) is None or not math.isfinite(float(r[k])) for k in keys) or r.get('family_edges') is None for r in rs):
                cell['status']='incomplete';continue
            for r in rs:
                if len(r['family_edges'])!=10 or sum(r['family_edges'])!=r['edges']:raise ValueError('relation counts do not sum to total')
            cell[readout]=dict(edge_absolute_error=sum(abs(r['edges']-926) for r in rs)/2,
                              signed_edge_bias=sum(r['edges']-926 for r in rs)/2,
                              W1_incident4=sum(r['W1_incident4'] for r in rs)/2,
                              LCC=sum(r['LCC'] for r in rs)/2,
                              family_absolute_errors=[sum(abs(r['family_edges'][i]-REFERENCE_FAMILY_EDGES[i]) for r in rs)/2 for i in range(10)])
        if cell['status']=='complete':
            raw,ema=cell['raw'],cell['ema'];w=raw['W1_incident4']
            cell['edge_error_difference']=ema['edge_absolute_error']-raw['edge_absolute_error']
            cell['family_error_differences']=[e-r for e,r in zip(ema['family_absolute_errors'],raw['family_absolute_errors'])]
            cell['redlines']=dict(W1_exceeds_10_percent=ema['W1_incident4']>1.1*w,
                                  LCC_drop_exceeds_19_2=raw['LCC']-ema['LCC']>19.2)
            cell['redline_note']='human-chosen non-adoption warnings; not proven safety or sufficient structural acceptance'
        result['per_seed'].append(cell)
    if any(c['status']!='complete' for c in result['per_seed']):
        result.update(status='incomplete',accuracy_conclusion=None,pooled_error_difference=None,recommend_adoption=False)
        return result
    diffs=[c['edge_error_difference'] for c in result['per_seed']]
    if all(d<0 for d in diffs):conclusion='consistent_accuracy_gain_in_this_panel'
    elif any(d<0 for d in diffs):conclusion='mixed_or_partial_not_consistent'
    else:conclusion='accuracy_hypothesis_not_supported_in_this_panel'
    result.update(status='complete',accuracy_conclusion=conclusion,pooled_error_difference=sum(diffs)/2,
                  non_adoption_redline=any(any(c['redlines'].values()) for c in result['per_seed']),
                  recommend_adoption=False,
                  interpretation='no automatic adoption: inspect tails/triangles/other structure even if accuracy improves and redlines do not fire')
    result['between_seed_signed_bias_gap']={r:abs(result['per_seed'][0][r]['signed_edge_bias']-result['per_seed'][1][r]['signed_edge_bias']) for r in ('raw','ema')}
    return result
