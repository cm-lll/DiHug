"""Exception-safe execution boundaries, without changing successful math/RNG draws.

Training can undo input/forward/backward failures before AdamW starts. It does
NOT pretend to undo a partially executed optimizer/EMA update: that state is
explicitly non-resumable, and only an older complete checkpoint may be used.
"""
import contextlib
import copy
import random

import numpy as np
import torch

FORMAT = 'dihug_complete_boundary_v2'


def rng_capture(rng, device):
    """All streams used by these single-visible-device workers, plus observers."""
    return dict(numpy=copy.deepcopy(rng.bit_generator.state),
                global_numpy=copy.deepcopy(np.random.get_state()),
                python=copy.deepcopy(random.getstate()), torch=torch.get_rng_state().clone(),
                device=torch.cuda.get_rng_state(0).clone() if device == 'cuda' else None,
                cuda_all=[v.clone() for v in torch.cuda.get_rng_state_all()] if device == 'cuda' else None,
                observer={k: copy.deepcopy(getattr(rng, k)) for k in ('calls', 'last_sha') if hasattr(rng, k)})


def rng_restore(saved, rng, device):
    rng.bit_generator.state = copy.deepcopy(saved['numpy'])
    np.random.set_state(saved['global_numpy'])
    random.setstate(saved['python'])
    torch.set_rng_state(saved['torch'])
    if device == 'cuda':
        if saved.get('cuda_all') is None:
            raise ValueError('complete CUDA RNG inventory required by v2')
        torch.cuda.set_rng_state_all(saved['cuda_all'])
    for name, value in saved.get('observer', {}).items():
        if not hasattr(rng, name):
            raise ValueError('RNG observer layout changed: ' + name)
        setattr(rng, name, copy.deepcopy(value))


def _stamp(value):
    # Guards for the audited pre-optimizer path, not arbitrary third-party code
    # that writes through .data or external memory. Such mutations are unsupported.
    if torch.is_tensor(value):
        return ('tensor', id(value), value._version)
    if isinstance(value, dict):
        return tuple((k, _stamp(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_stamp(v) for v in value)
    return value


def _check_role_cache(model):
    # StructuralDenoiser defines _role_identity for EVERY arm. Only C/E
    # actually register these buffers, so method presence is not a role test.
    fields={'role_basis','role_offsets'}
    present=fields.intersection(dict(model.named_buffers()))
    if not present:
        if any(hasattr(model,k) for k in ('_basis_cpu','_offsets_cpu','_basis_tensor_identity')):
            raise RuntimeError('role cache exists without registered role buffers')
        return
    if present!=fields or not all(hasattr(model,k) for k in ('_basis_cpu','_offsets_cpu','_basis_tensor_identity','_basis_hash','_role_identity')):
        raise RuntimeError('incomplete role buffer/cache state')
    if model._role_identity()!=model._basis_tensor_identity:
        raise RuntimeError('frozen dictionary identity changed; cache rollback unverified')
    if (not np.array_equal(model._basis_cpu,model.role_basis.detach().cpu().numpy())
            or not np.array_equal(model._offsets_cpu,model.role_offsets.detach().cpu().numpy())):
        raise RuntimeError('role CPU cache differs from registered buffers')
    from features import digest
    if digest(model._basis_cpu,model._offsets_cpu)!=model._basis_hash:
        raise RuntimeError('role CPU cache digest changed')


def _optimizer_stamp(optimizer):
    # Loading EMA/raw bumps parameter versions even when AdamW is untouched.
    # Guard optimizer tensors normally, but refer to group parameters by identity.
    groups=tuple(tuple((k,tuple(id(p) for p in v) if k=='params' else _stamp(v))
                       for k,v in group.items()) for group in optimizer.param_groups)
    return (_stamp(optimizer.state),groups)


def _same_model_state(model,saved):
    actual=model.state_dict()
    return actual.keys()==saved.keys() and all(
        value.shape==saved[k].shape and value.dtype==saved[k].dtype
        and torch.equal(value.detach().cpu(),saved[k]) for k,value in actual.items())


class TrainingBoundary:
    def __init__(self, model, optimizer, ema, rng, device, completed):
        if completed != ema.updates or not ema.at_update_boundary or getattr(model,'_dihug_checkpoint_invalid',None) or getattr(model,'_dihug_validation_in_flight',False):
            raise ValueError('initial model/optimizer/EMA boundary required')
        self.model, self.optimizer, self.ema = model, optimizer, ema
        self.rng, self.device = rng, device
        self.completed = completed
        self.phase, self.resumable = 'idle', True
        self.boundary_rng = rng_capture(rng, device)
        self.last_failure = None

    def invalidate(self,error,phase):
        self.resumable,self.phase=False,'invalid'
        self.ema.at_update_boundary=False
        self.last_failure=dict(error=repr(error),phase=phase,exact_resume_allowed=False,
                               use_previous_complete_checkpoint_only=True)

    @contextlib.contextmanager
    def validation(self):
        """Do not certify any checkpoint while temporary readout weights are live.

        A validation error is recoverable only if full raw contents, optimizer,
        EMA, role caches, modes and RNG are certified/restored. An uncertain raw
        restore is sticky-invalid; no second blind load can declare it healthy.
        """
        from ema import cpu_copy
        if not self.resumable or self.phase!='idle' or getattr(self.model,'_dihug_checkpoint_invalid',None) or getattr(self.model,'_dihug_validation_in_flight',False):
            raise RuntimeError('validation requires a complete training boundary')
        raw=cpu_copy(self.model.state_dict());before=rng_capture(self.rng,self.device)
        optimizer=_optimizer_stamp(self.optimizer);shadow=_stamp(self.ema.shadow)
        updates,decay=self.ema.updates,self.ema.decay
        parameter_ids={k:id(p) for k,p in self.model.named_parameters()}
        modes=[(m,m.training) for m in self.model.modules()]
        self.phase='validation_in_flight'
        self.model._dihug_validation_in_flight=True
        def certify():
            if (not self.resumable or self.phase!='validation_in_flight'
                    or getattr(self.model,'_dihug_checkpoint_invalid',None)
                    or not self.ema.at_update_boundary or self.ema.updates!=updates
                    or self.completed!=updates or self.ema.decay!=decay
                    or parameter_ids!={k:id(p) for k,p in self.model.named_parameters()}
                    or optimizer!=_optimizer_stamp(self.optimizer) or shadow!=_stamp(self.ema.shadow)
                    or not _same_model_state(self.model,raw)):
                raise RuntimeError('raw/AdamW/EMA validation restore not certified')
            _check_role_cache(self.model)
            for module,mode in modes:module.training=mode
            rng_restore(before,self.rng,self.device)
            self.model._dihug_validation_in_flight=False
            self.boundary_rng,self.phase=before,'idle'
        try:
            yield
        except BaseException as exc:
            try:certify()
            except BaseException as check_error:
                if self.resumable:self.invalidate(check_error,'validation_restore_unverified')
            else:
                self.last_failure=dict(error=repr(exc),phase='validation',restored_raw=True,
                                       restored_real_rng=True,exact_resume_allowed=True)
            raise
        else:
            try:certify()
            except BaseException as exc:
                self.invalidate(exc,'validation_restore_unverified')
                raise

    def mark_optimizer_started(self):
        if self.phase != 'pre_optimizer':
            raise RuntimeError('optimizer outside input-to-update transaction')
        # Mark dirty BEFORE AdamW can mutate its first parameter/state tensor.
        self.phase = 'optimizer_or_ema_in_flight'
        self.ema.at_update_boundary = False

    @contextlib.contextmanager
    def step(self, position):
        if not self.resumable or self.phase != 'idle' or position != self.completed + 1:
            raise RuntimeError('invalid training transaction boundary')
        before = rng_capture(self.rng, self.device)  # BEFORE input sampling
        parameters = _stamp(dict(self.model.named_parameters()))
        optimizer = _stamp((self.optimizer.state, self.optimizer.param_groups))
        shadow = _stamp(self.ema.shadow)
        buffer_objects = dict(self.model.named_buffers())
        buffers = {k: v.detach().clone() for k, v in buffer_objects.items()}
        modes = [(m, m.training) for m in self.model.modules()]
        self.phase = 'pre_optimizer'
        try:
            yield self
            if self.phase != 'optimizer_or_ema_in_flight' or not self.ema.at_update_boundary or self.ema.updates != position:
                raise RuntimeError('optimizer/EMA did not complete exactly one update')
            after = rng_capture(self.rng, self.device)
            # Only this successful point publishes the new counter/RNG boundary.
            self.completed, self.boundary_rng, self.phase = position, after, 'idle'
        except BaseException as exc:
            failed_phase = self.phase
            safe = (failed_phase == 'pre_optimizer'
                    and parameters == _stamp(dict(self.model.named_parameters()))
                    and optimizer == _stamp((self.optimizer.state, self.optimizer.param_groups))
                    and shadow == _stamp(self.ema.shadow)
                    and self.ema.updates == self.completed and self.ema.at_update_boundary)
            try:
                if not safe:
                    raise RuntimeError('cannot roll back mutated or partial optimizer/EMA state')
                actual = dict(self.model.named_buffers())
                if actual.keys() != buffers.keys():
                    raise RuntimeError('buffer inventory changed')
                if any(actual[k] is not buffer_objects[k] for k in actual):
                    raise RuntimeError('buffer identity changed; alias/cache rollback unverified')
                with torch.no_grad():
                    for k, value in actual.items():
                        # Do not bump unchanged frozen-dictionary buffer versions.
                        if not torch.equal(value, buffers[k]):
                            value.copy_(buffers[k])
                _check_role_cache(self.model)
                self.optimizer.zero_grad(set_to_none=True)
                for module, mode in modes:
                    module.training = mode
                rng_restore(before, self.rng, self.device)
                self.boundary_rng, self.phase = before, 'idle'
            except BaseException as rollback_error:
                self.resumable, self.phase = False, 'invalid'
                self.ema.at_update_boundary = False
                self.last_failure = dict(error=repr(exc), phase=failed_phase,
                                         rollback_error=repr(rollback_error), exact_resume_allowed=False,
                                         use_previous_complete_checkpoint_only=True)
            else:
                self.last_failure = dict(error=repr(exc), phase=failed_phase, rolled_back_to_update=self.completed,
                                         exact_resume_allowed=True, restored_real_rng=True)
            raise

    def checkpoint(self, binding):
        from integration import checkpoint_payload
        if not self.resumable or self.phase != 'idle' or getattr(self.model,'_dihug_checkpoint_invalid',None):
            raise RuntimeError('in-flight/invalid training state must not be saved as resumable')
        payload = checkpoint_payload(self.model, self.optimizer, self.ema,
                                     self.completed, self.boundary_rng, binding)
        payload['recovery'] = dict(format=FORMAT, exact_resume_allowed=True,
                                   completed_updates=self.completed, phase='committed')
        return payload


class SamplingBoundary:
    def __init__(self, state, next_t, rng, device):
        self.rng, self.device = rng, device
        self.boundary = (state.copy(), int(next_t), rng_capture(rng, device))
        self.resumable, self.last_failure = True, None

    @property
    def state(self):
        return self.boundary[0]

    @property
    def next_t(self):
        return self.boundary[1]

    def advance(self, step, validate):
        if not self.resumable or self.next_t < 1:
            raise RuntimeError('invalid reverse boundary')
        state, t, before = self.boundary
        # Retain the committed graph even if a future backend writes its argument.
        try:
            candidate, record = step(state.copy(), t)
            validate(candidate, record)  # includes OUTER plan/draw/coverage checks
            after = rng_capture(self.rng, self.device)
            self.boundary = (candidate.copy(), t - 1, after)
            return record
        except BaseException as exc:
            try:
                rng_restore(before, self.rng, self.device)
            except BaseException as rollback_error:
                self.resumable = False
                self.last_failure = dict(error=repr(exc), rollback_error=repr(rollback_error), exact_resume_allowed=False)
            else:
                self.last_failure = dict(error=repr(exc), restored_real_rng=True, rolled_back_to_t=t,
                                         exact_resume_allowed=True)
            raise

    def checkpoint(self, binding):
        if not self.resumable:
            raise RuntimeError('failed RNG restore: chain state cannot be used for exact resume')
        state, t, saved = self.boundary
        return dict(state=state.copy(), next_t=t, rng=copy.deepcopy(saved),
                    draw_calls=saved['observer'].get('calls'), binding=copy.deepcopy(binding),
                    recovery=dict(format=FORMAT, exact_resume_allowed=True, phase='committed', committed_t=t))

    def restore(self, payload, binding):
        recovery = payload.get('recovery', {})
        if (payload['binding'] != binding or recovery.get('format') != FORMAT
                or recovery.get('exact_resume_allowed') is not True or recovery.get('phase') != 'committed'):
            raise ValueError('unverified/legacy stopped chain cannot be exactly resumed')
        state, t, saved = payload['state'], payload['next_t'], payload['rng']
        if (state.shape != self.state.shape or state.dtype != self.state.dtype or not np.isin(state, [0, 1]).all()
                or type(t) is not int or not 0 <= t <= self.next_t or recovery.get('committed_t') != t
                or payload.get('draw_calls') != saved['observer'].get('calls')):
            raise ValueError('chain state/time/RNG boundary mismatch')
        rng_restore(saved, self.rng, self.device)
        self.boundary = (state.copy(), t, copy.deepcopy(saved))
