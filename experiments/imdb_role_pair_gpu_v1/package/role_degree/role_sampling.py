"""Independent bounded-step adapter over unchanged CN v2 runtime.

There is intentionally NO full-chain or training entry in this package. A/C use
baseline-native weights, B/D CN-native weights; never disable a trained branch.
"""
import time
import numpy as np

ARMS={'A':('baseline',False),'B':('cn',False),'C':('baseline',True),'D':('cn',True)}


def validate_arm(model,arm,receipt=None,formal=False):
    if arm not in ARMS:raise ValueError('unknown four-arm selector')
    architecture,enabled=ARMS[arm]
    keys=model.state_dict()
    has_module=model.common_neighbor is not None
    has_weights=any(k.startswith('common_neighbor.') for k in keys)
    if has_module!=(architecture=='cn') or has_weights!=has_module:raise ValueError('arm/architecture mismatch; cannot disable trained CN')
    if receipt is not None:
        if receipt.get('experiment_arm')!=architecture:raise ValueError('weight provenance arm mismatch')
        if formal and (receipt.get('formal_training') is not True or receipt.get('completed_updates')!=10000):
            raise ValueError('formal four-arm protocol requires native final 10000-update weights')
    elif formal:raise ValueError('formal final weight receipt required')
    return enabled


def reverse_step(model,case,state,kernel,t,rounds,seed,rng,device,*,arm,guide=None,receipt=None,
                 formal=False,trace=False,interrupt_block=None):
    from hsgd.runtime import reverse_step as original,StepPredictor
    from hsgd.bw_preflight.common import plan_for,rng_save,rng_restore
    enabled=validate_arm(model,arm,receipt,formal)
    if not enabled or t!=1:
        return original(model,case,state,kernel,t,rounds,seed,rng,device,
            backend='upstream',trace=trace,interrupt_block=interrupt_block)
    if guide is None:raise ValueError('guidance configuration required')
    layout=guide.layout
    if (not np.array_equal(layout.types,case['space'].types) or
        not np.array_equal(layout.pairs,np.asarray(case['space'].pairs)) or
        not np.array_equal(layout.prior,kernel.prior)):
        raise ValueError('guide graph/schema/prior binding differs')
    state=np.asarray(state)
    if state.shape!=layout.families.shape or not np.isin(state,[0,1]).all():raise ValueError('invalid frozen state')
    if (state[layout.fixed0]!=0).any() or (state[layout.fixed1]!=1).any():raise ValueError('frozen state violates fixed support')
    # Complete canonical global slot mapping must agree, not just array lengths.
    f,local=case['space'].encode(layout.slots)
    if not np.array_equal(case['space'].offsets[f]+local,np.arange(case['space'].total)):
        raise ValueError('guide slots are not canonical global IDs')
    before=rng_save(rng,device);callback=None;started=time.perf_counter()
    try:
        plan=plan_for(case,[seed,t,61],rounds)
        uniform=rng.random(case['space'].total)  # exactly the original one draw
        callback=StepPredictor(model,case,state,kernel,t,'upstream',profile=False)
        p0=np.repeat(kernel.prior,case['space'].counts).copy()
        coverage=np.zeros(case['space'].total,dtype=np.uint8);tick=time.perf_counter()
        for r in range(rounds):
            ids,fam,query=plan.block(r)
            if not len(ids):continue
            if coverage[ids].any():raise AssertionError('duplicate query slot')
            p0[ids]=callback(state,ids,fam,query);coverage[ids]+=1
            if interrupt_block is not None and callback.blocks==interrupt_block:raise InterruptedError('injected incomplete guided step')
        expected=np.repeat(case['active'].astype(np.uint8),case['space'].counts)
        if not np.array_equal(coverage,expected):raise AssertionError('incomplete probability coverage')
        collection=time.perf_counter()-tick
        # No retained network activation and no neural backward, even if caller
        # accidentally enabled gradients. StepPredictor already uses no_grad.
        predicted_blocks=callback.blocks;callback.close();callback=None
        guided,report=guide.apply(p0)
        if report['status']=='numerical_failure':raise FloatingPointError(report['reason'])
        result=np.array(state,copy=True);q_all=np.repeat(kernel.prior,case['space'].counts).copy() if trace else None
        max_bridge_error=0.;tick=time.perf_counter()
        for r in range(rounds):
            ids,fam,_=plan.block(r)
            if not len(ids):continue
            q=kernel.reverse(guided[ids],np.asarray(state)[ids],fam,0,1)
            error=float(np.max(np.abs(q-guided[ids]),initial=0));max_bridge_error=max(max_bridge_error,error)
            if error>1e-12:raise AssertionError('t1 clean/sampling probability correspondence failed')
            result[ids]=uniform[ids]<q
            if trace:q_all[ids]=q
        record=dict(time=1,remaining_time=0,backend='upstream',arm=arm,blocks=predicted_blocks,query_rounds=rounds,
            covered_slots=int(coverage.sum()),exact_coverage=True,step_seconds=time.perf_counter()-started,
            probability_collection_seconds=collection,posterior_sampling_seconds=time.perf_counter()-tick,
            max_final_probability_bridge_error=max_bridge_error,guidance=report,
            conditional_contract='new guided product Bernoulli law given frozen Xt/query plan; no exact target conditioning')
        if trace:record['trace']=dict(p0=p0,p_guided=guided,q=q_all,uniform=uniform,coverage=coverage,state=result.copy())
        return result.astype(np.uint8),record
    except BaseException:
        rng_restore(before,rng,device)
        raise
    finally:
        if callback is not None:callback.close()
