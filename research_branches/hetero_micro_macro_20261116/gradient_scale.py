"""Fixed development-panel gradient calibration, with no model dependency.

This is a proposed engineering calibration, not an optimizer stability theorem.
Each macro block already averages the frozen per-type residual-scaled errors.
The old global loss-value lambda is REPLACED, never applied a second time.
"""
import math

BLOCKS=('degree','closure','propagation')

def freeze_coefficients(rows):
    """Accept one row per parent/time with squared *unclipped* parameter norms.

    Required balanced panel: two datasets, two seeds per dataset, six fixed times.
    All trainable parameters use the same norm convention for CE and each block.
    Norms must come from the same forward/state, before clipping or AdamW.
    """
    expected={'IMDB':{2026110411,2026110412},'PubMed':{2026110421,2026110422}}
    times={1,75,250,550,800,1000}
    keys={(r['dataset'],int(r['seed']),int(r['t'])) for r in rows}
    assert len(rows)==len(keys)==24
    assert keys=={(d,s,t) for d,ss in expected.items() for s in ss for t in times}
    means={}
    for key in ('CE',)+BLOCKS:
        values=[float(r['norm_squared'][key]) for r in rows]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('nonfinite/negative gradient norm')
        means[key]=math.fsum(values)/len(values)
        if means[key]<=0:raise ValueError('zero calibration gradient: do not amplify or drop this objective silently')
    rms={k:math.sqrt(v) for k,v in means.items()}
    # RMS of each coefficient-scaled block on this panel equals CE_RMS/3.
    # Triangle inequality gives RMS of the SUM of scaled macro gradients <= CE_RMS.
    coefficients={b:rms['CE']/rms[b]/len(BLOCKS) for b in BLOCKS}
    if any(not math.isfinite(v) or v<=0 for v in coefficients.values()):raise ValueError('invalid coefficient')
    return dict(coefficients=coefficients,CE_coefficient=1.0,panel_RMS=rms,
        common_across_datasets_and_seeds=True,old_lambda_applied=False,
        scope='RMS norm budget on calibration states only; no pointwise, AdamW, heldout or generation guarantee')

def compose(ce,block_losses,frozen):
    assert set(block_losses)==set(BLOCKS)
    return ce+sum(frozen['coefficients'][b]*block_losses[b] for b in BLOCKS)

def self_test():
    rows=[dict(dataset=d,seed=s,t=t,norm_squared=dict(CE=4.,degree=16.,closure=36.,propagation=100.))
          for d,ss in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]
          for s in ss for t in [1,75,250,550,800,1000]]
    f=freeze_coefficients(rows)
    for b,n in [('degree',4.),('closure',6.),('propagation',10.)]:
        assert math.isclose(f['coefficients'][b]*n,2./3.)
    # Changing units of one block changes its coefficient inversely.
    scaled=[dict(r,norm_squared={**r['norm_squared'],'closure':r['norm_squared']['closure']*49}) for r in rows]
    fs=freeze_coefficients(scaled)
    assert math.isclose(fs['coefficients']['closure']*7,f['coefficients']['closure'])
    assert math.isclose(compose(.3,dict(degree=2.,closure=3.,propagation=4.),f),
                        compose(.3,dict(degree=2.,closure=21.,propagation=4.),fs))
    for broken in [rows[:-1],rows+[rows[0]],
                   [dict(r,norm_squared={**r['norm_squared'],'degree':0.}) for r in rows]]:
        try:freeze_coefficients(broken)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('invalid panel accepted')
    return dict(status='passed',checks=['balanced panel','equal block RMS budget','unit rescaling','CE coefficient1','missing/duplicate/zero rejection'],
                model_calls=0,GPU_hours=0)

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
