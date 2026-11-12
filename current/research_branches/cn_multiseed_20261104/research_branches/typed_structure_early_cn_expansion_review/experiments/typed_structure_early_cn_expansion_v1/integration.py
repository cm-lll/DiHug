"""Small hooks around existing normal2/checkpoint/sampling machinery.

No new scientific training loop, sampler or automatic execution entrypoint.
The reviewed launcher must supply the existing functions and authorization.
"""
from ema import ParameterEMA,cpu_copy

def complete_normal2(normal2_fn,model,opt,case,inputs,T,device,ema,completed_updates):
    if completed_updates!=ema.updates or not ema.at_update_boundary:raise ValueError('raw/EMA counter or boundary mismatch before update')
    # Existing function does zero_grad ONCE, two half-loss backwards, one step.
    # An exception after a raw step must not save an apparently complete old
    # counter with new raw parameters and old/partial EMA parameters.
    ema.at_update_boundary=False
    result=normal2_fn(model,opt,case,inputs,T,device,step=True)
    if result.get('optimizer_steps')!=1:raise ValueError('normal2 did not report one complete step')
    ema.update(model,completed_updates+1)
    ema.at_update_boundary=True
    return dict(result,ema_updates=ema.updates)

def checkpoint_payload(model,opt,ema,updates,rng,binding):
    if getattr(model,'_dihug_checkpoint_invalid',None) or getattr(model,'_dihug_validation_in_flight',False):raise RuntimeError('unverified/in-flight raw restore: checkpoint forbidden')
    if updates!=ema.updates:raise ValueError('save only at complete raw/EMA boundary')
    return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(opt.state_dict()),
                ema=ema.state_dict(),updates=updates,rng=cpu_copy(rng),binding=cpu_copy(binding))

def restore_payload(payload,model,opt,ema,binding,restore_rng,*,legacy_complete_boundary_verified=False):
    # The existing checkpoint loader still authenticates bytes and paths.
    recovery=payload.get('recovery')
    if recovery is None and not legacy_complete_boundary_verified:
        raise ValueError('legacy checkpoint requires independent complete-boundary verification; a stopped filename is not evidence')
    if recovery is not None and (recovery.get('format')!='dihug_complete_boundary_v2' or recovery.get('exact_resume_allowed') is not True or recovery.get('phase')!='committed' or recovery.get('completed_updates')!=payload['updates']):
        raise ValueError('checkpoint is not a complete resumable training boundary')
    if payload['binding']!=binding:raise ValueError('checkpoint binding mismatch')
    # Validate EMA fully before mutating model/optimizer/RNG.
    check=type(ema)(model)
    check.load_state_dict(payload['ema'],model,payload['updates'])
    model.load_state_dict(payload['model'],strict=True)
    opt.load_state_dict(payload['optimizer'])
    ema.load_state_dict(payload['ema'],model,payload['updates'])
    restore_rng(cpu_copy(payload['rng']))
    return payload['updates']

def final_readouts(model,ema,completed_updates):
    if completed_updates!=3000 or ema.updates!=3000:
        raise ValueError('only the final3000 readouts may enter the scientific sample plan')
    if hasattr(ema,'readouts'):return dict(raw=cpu_copy(model.state_dict()),**ema.readouts(model))
    return dict(raw=cpu_copy(model.state_dict()),ema=ema.readout_state_dict(model))

def ema_validation(validation_fn,model,ema,*args,boundary=None,**kwargs):
    """Call the existing RNG-preserving validation with EMA, restore raw in finally.

    The existing validation closes each FrozenTypedContext. No learned output
    cache may survive this weight swap. No second model construction is needed.
    """
    raw=cpu_copy(model.state_dict());mode=model.training
    prior_in_flight=getattr(model,'_dihug_validation_in_flight',False)
    model._dihug_validation_in_flight=True
    try:
        model.load_state_dict(ema.readout_state_dict(model),strict=True)
        return validation_fn(model,*args,**kwargs)
    finally:
        try:
            model.load_state_dict(raw,strict=True)
            model.train(mode)
            model._dihug_validation_in_flight=prior_in_flight
        except BaseException as exc:
            # A matching EMA counter cannot certify partially restored raw
            # parameters. Poison both direct and boundary-managed save paths.
            model._dihug_checkpoint_invalid=dict(reason='raw_restore_failed',error=repr(exc))
            ema.at_update_boundary=False
            if boundary is not None:boundary.invalidate(exc,'ema_validation_raw_restore_failed')
            raise
