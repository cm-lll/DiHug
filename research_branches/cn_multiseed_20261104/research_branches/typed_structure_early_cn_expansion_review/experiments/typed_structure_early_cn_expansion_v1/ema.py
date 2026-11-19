"""Fixed EMA readout. No optimizer, data, model factory or RNG calls.

theta_bar[0] = theta[0]; theta_bar[k] = beta*theta_bar[k-1]+(1-beta)*theta[k].
Call exactly once AFTER each completed normal2 optimizer update.
"""
import copy
import torch

DECAY = 0.99
FORMAT = 'pubmed_o_ema_from_theta0_v1'

def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)

def parameters(model):
    # Include serialization aliases, so tied parameters cannot be overwritten
    # by a raw alias when an exported EMA state_dict is loaded.
    return dict(model.named_parameters(remove_duplicate=False))

class ParameterEMA:
    def __init__(self, model, decay=DECAY):
        if decay not in (0.9,0.97,0.99):raise ValueError('only frozen EMA decays')
        self.decay=decay
        ps=parameters(model)
        if not ps:raise ValueError('empty parameter set')
        if any(not p.is_floating_point() or not torch.isfinite(p).all() for p in ps.values()):
            raise ValueError('EMA requires finite floating parameters')
        self.shadow={n:p.detach().clone() for n,p in ps.items()}
        self.updates=0
        self.at_update_boundary=True

    def _parameters(self,model):
        ps=parameters(model)
        if ps.keys()!=self.shadow.keys():raise ValueError('parameter names differ')
        for n,p in ps.items():
            q=self.shadow[n]
            if p.shape!=q.shape or p.dtype!=q.dtype or p.device!=q.device:
                raise ValueError('parameter shape/dtype/device differ: '+n)
        return ps

    @torch.no_grad()
    def update(self,model,completed_updates):
        if type(completed_updates) is not int or completed_updates!=self.updates+1:
            raise ValueError('EMA needs the next complete normal2 update, once')
        ps=self._parameters(model)
        if any(not torch.isfinite(p).all() for p in ps.values()):
            raise FloatingPointError('nonfinite raw parameters; EMA not updated')
        for n,p in ps.items():
            # Explicit two-operation FP32 implementation; no first-call copy,
            # warmup, bias correction, scheduler or conditional update.
            self.shadow[n].mul_(self.decay).add_(p.detach(),alpha=1.0-self.decay)
        if any(not torch.isfinite(p).all() for p in self.shadow.values()):
            raise FloatingPointError('nonfinite EMA; stop, do not use readout')
        self.updates=completed_updates

    def state_dict(self):
        if not self.at_update_boundary:raise RuntimeError('incomplete raw/EMA update cannot be a resumable checkpoint')
        return dict(format=FORMAT,decay=self.decay,updates=self.updates,shadow=cpu_copy(self.shadow))

    def load_state_dict(self,state,model,expected_updates):
        if state.get('format')!=FORMAT or state.get('decay')!=self.decay:
            raise ValueError('EMA scientific contract mismatch')
        if type(state.get('updates')) is not int or state['updates']!=expected_updates or expected_updates<0:
            raise ValueError('EMA update counter mismatch')
        ps=self._parameters(model);ss=state.get('shadow',{})
        if ss.keys()!=ps.keys():raise ValueError('EMA saved parameter names differ')
        for n,p in ps.items():
            q=ss[n]
            if not torch.is_tensor(q) or q.shape!=p.shape or q.dtype!=p.dtype or not torch.isfinite(q).all():
                raise ValueError('invalid saved EMA parameter: '+n)
        self.shadow={n:ss[n].detach().to(device=p.device).clone() for n,p in ps.items()}
        self.updates=expected_updates
        self.at_update_boundary=True

    def readout_state_dict(self,model):
        if not self.at_update_boundary:raise RuntimeError('incomplete raw/EMA update cannot be evaluated')
        self._parameters(model)
        # Buffers (including float priors and integer counters) are COPIED.
        # This does not modify the live training model, optimizer, mode or RNG.
        state=cpu_copy(model.state_dict())
        for n,p in self.shadow.items():state[n]=cpu_copy(p)
        return state
