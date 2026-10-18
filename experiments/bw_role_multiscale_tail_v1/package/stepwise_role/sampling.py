"""Atomic frozen-Xt sampling adapter. No full-chain scientific runner."""
import time
import numpy as np
from stepwise_role.core import signature

ARMS = {'A':(False,False),'B':(True,False),'C':(False,True),'D':(True,True)}


def validate_arm(model, arm, receipt=None, formal=False):
    """Architecture + caller receipt sanity, NOT an independent weight importer.

    This adapter receives an already loaded model. The external caller must
    verify the actual checkpoint hash/source/config/data before giving a receipt.
    No BW weight import has been executed or certified by this package.
    """
    if arm not in ARMS: raise ValueError('unknown arm')
    cn, enabled = ARMS[arm]
    actual = model.common_neighbor is not None
    weights = any(k.startswith('common_neighbor.') for k in model.state_dict())
    if actual != cn or actual != weights: raise ValueError('native architecture/arm mismatch')
    if formal:
        # The user reports two native final 6000-update runs. Do NOT silently
        # accept the historical terminal module's 10000-update protocol.
        if not receipt or receipt.get('formal_training') is not True or receipt.get('completed_updates') != 6000:
            raise ValueError('native final 6000-update provenance required')
        if receipt.get('experiment_arm') != ('cn' if cn else 'baseline') or receipt.get('final') is not True:
            raise ValueError('native final architecture receipt required')
        for field in ('checkpoint_sha256','source_sha256','config_sha256','data_sha256'):
            value=receipt.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('missing final weight provenance: '+field)
    return enabled


def reverse_step(model, case, state, kernel, t, rounds, seed, rng, device, *, arm,
                 guide=None, receipt=None, formal=False, trace=False, interrupt_block=None,
                 event=None, check_time=None):
    from hsgd.runtime import reverse_step as original, StepPredictor
    from hsgd.bw_preflight.common import plan_for, rng_save, rng_restore
    enabled = validate_arm(model,arm,receipt,formal)
    if not enabled:
        return original(model,case,state,kernel,t,rounds,seed,rng,device,backend='upstream',
                        trace=trace,interrupt_block=interrupt_block,event=event,check_time=check_time)
    if guide is None: raise ValueError('guidance artifact required')
    if model.training: raise ValueError('guided inference requires model.eval()')
    l = guide.layout; state = np.asarray(state)
    if not np.array_equal(l.types,case['space'].types) or not np.array_equal(l.pairs,np.asarray(case['space'].pairs)):
        raise ValueError('schema/inventory binding mismatch')
    if not np.array_equal(l.prior,case['prior']) or not np.array_equal(guide.alpha,kernel.alpha):
        raise ValueError('prior/schedule binding mismatch')
    l.probability(state) # binary and fixed support checked below
    if not np.isin(state,[0,1]).all(): raise ValueError('nonbinary frozen state')
    f,local=case['space'].encode(l.slots)
    if not np.array_equal(case['space'].offsets[f]+local,np.arange(case['space'].total)):
        raise ValueError('full canonical slot order required')
    before=rng_save(rng,device); callback=None; start=time.perf_counter()
    versions=tuple(p._version for p in model.parameters())
    def check():
        if tuple(p._version for p in model.parameters())!=versions: raise ValueError('weights changed during frozen step')
        if check_time: check_time()
    try:
        check()
        plan=plan_for(case,[seed,t,61],rounds)
        uniform=rng.random(case['space'].total) # identical original order/count
        tick=time.perf_counter()
        callback=StepPredictor(model,case,state,kernel,t,'upstream',event,check,profile=trace)
        p0=np.repeat(kernel.prior,case['space'].counts).copy()
        coverage=np.zeros(len(p0),dtype=np.uint8)
        for r in range(rounds):
            ids,fam,query=plan.block(r)
            if not len(ids): continue
            if coverage[ids].any(): raise AssertionError('duplicate slot coverage')
            p0[ids]=callback(state,ids,fam,query);coverage[ids]+=1
            if interrupt_block is not None and callback.blocks==interrupt_block:
                raise InterruptedError('injected partial step; state/RNG not committed')
        expected=np.repeat(case['active'].astype(np.uint8),case['space'].counts)
        if not np.array_equal(coverage,expected): raise AssertionError('incomplete slot coverage')
        check(); blocks=callback.blocks; timings=callback.times or {}
        callback.close();callback=None # release neural activations/context BEFORE statistics
        collection=time.perf_counter()-tick
        guided,q,report=guide.apply(p0,state,kernel,t,check_time=check)
        if report['status']=='numerical_failure': raise FloatingPointError(report['reason'])
        check();tick=time.perf_counter()
        result=(uniform<q).astype(np.uint8)
        l.probability(result)
        record=dict(time=int(t),remaining_time=int(t-1),backend='upstream',arm=arm,
                    blocks=blocks,query_rounds=rounds,covered_slots=int(coverage.sum()),exact_coverage=True,
                    probability_collection_seconds=collection,guidance=report,timings=timings,
                    sampling_seconds=time.perf_counter()-tick,step_seconds=time.perf_counter()-start,
                    raw_edges=int(result.sum()),new_sampling_law=True)
        if trace: record['trace']=dict(p0=p0,p_guided=guided,q=q,uniform=uniform,coverage=coverage,state=result.copy())
        return result,record
    except BaseException:
        rng_restore(before,rng,device)
        raise
    finally:
        if callback is not None: callback.close()


def binding(model, case, kernel, guide, arm, rounds, seed, receipt=None):
    """Checkpoints bind code/runtime/model/data/law/plan. No private upload."""
    import hashlib
    import platform
    import torch
    import torch_geometric
    from pathlib import Path
    validate_arm(model,arm,receipt,formal=receipt is not None)
    root=Path(__file__).resolve().parents[1]
    code={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for sub in ('hsgd','role_degree','stepwise_role') for p in sorted((root/sub).rglob('*.py'))}
    def digest(array):
        a=np.asarray(array)
        return signature(dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    weights={k:digest(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
    return dict(code_sha256=signature(code),model_sha256=signature(weights),
                architecture_sha256=signature(str(model)),backend='upstream',
                time_steps=len(kernel.alpha)-1,slots=int(case['space'].total),
                data_sha256=digest(case['clean']),types_sha256=digest(case['space'].types),
                guide=guide.binding(),alpha_sha256=digest(kernel.alpha),prior_sha256=digest(kernel.prior),
                arm=arm,rounds=int(rounds),seed=int(seed),receipt_sha256=signature(receipt),
                python=platform.python_version(),torch_runtime_version=torch.__version__,
                pyg=torch_geometric.__version__,numpy=np.__version__,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


def save_checkpoint(path, state, remaining_time, rng, device, bound):
    """Call ONLY after a whole step commits; reject invalid counters/state."""
    import hashlib
    import io
    import os
    import torch
    from pathlib import Path
    from hsgd.bw_preflight.common import rng_save
    a=np.asarray(state)
    if a.shape!=(bound['slots'],) or not np.isin(a,[0,1]).all() or not 0<=remaining_time<=bound['time_steps']:
        raise ValueError('invalid committed checkpoint')
    packet=dict(state=a.astype(np.uint8),remaining_time=int(remaining_time),binding=bound,
                rng=rng_save(rng,device),committed=True)
    stream=io.BytesIO();torch.save(packet,stream);raw=stream.getvalue()
    outer=dict(raw=raw,sha256=hashlib.sha256(raw).hexdigest())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(outer,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def load_checkpoint(path, rng, device, expected, slots, total_time, expected_remaining=None):
    import hashlib
    import io
    import torch
    from hsgd.bw_preflight.common import rng_restore
    def load(x):
        try: return torch.load(x,map_location='cpu',weights_only=False)
        except TypeError: return torch.load(x,map_location='cpu')
    # Trusted locally produced files only (Torch pickle), never arbitrary input.
    outer=load(path)
    if hashlib.sha256(outer['raw']).hexdigest()!=outer['sha256']: raise ValueError('checkpoint payload digest differs')
    packet=load(io.BytesIO(outer['raw']))
    if packet['binding']!=expected or packet.get('committed') is not True: raise ValueError('checkpoint binding/commit differs')
    state=np.asarray(packet['state']);s=packet['remaining_time']
    if slots!=expected['slots'] or total_time!=expected['time_steps'] or state.shape!=(slots,) or not np.isin(state,[0,1]).all() or not 0<=s<=total_time:
        raise ValueError('checkpoint state/count invalid')
    if expected_remaining is not None and s!=expected_remaining: raise ValueError('completed-step count differs')
    rng_restore(packet['rng'],rng,device)
    return state.copy(),s
