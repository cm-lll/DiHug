"""Three independent recurrences, all reading the same live raw parameters."""
from ema import ParameterEMA
from runtime_contract import DECAYS,READOUTS
class MultiEMA:
    def __init__(self,model):
        self.members={name:ParameterEMA(model,decay) for name,decay in DECAYS.items()}
    @property
    def updates(self):
        counts={x.updates for x in self.members.values()}
        if len(counts)!=1:raise RuntimeError('partial multi EMA update')
        return next(iter(counts))
    @property
    def at_update_boundary(self):return all(x.at_update_boundary for x in self.members.values())
    @at_update_boundary.setter
    def at_update_boundary(self,value):
        for e in self.members.values():e.at_update_boundary=value
    def update(self,model,completed_updates):
        if self.updates+1!=completed_updates:raise ValueError('next complete raw update required')
        self.at_update_boundary=False
        for e in self.members.values():e.update(model,completed_updates)
        self.at_update_boundary=True
    def state_dict(self):
        if not self.at_update_boundary:raise RuntimeError('partial bank cannot be checkpointed')
        return dict(format='multi_ema_theta0_v1',updates=self.updates,members={k:e.state_dict() for k,e in self.members.items()})
    def load_state_dict(self,state,model,expected_updates):
        if state.get('format')!='multi_ema_theta0_v1' or state.get('updates')!=expected_updates or set(state.get('members',{}))!=set(DECAYS):raise ValueError('multi EMA identities/count')
        # All members validate into independent temporaries before any bank mutation.
        new={}
        for k,d in DECAYS.items():
            e=ParameterEMA(model,d);e.load_state_dict(state['members'][k],model,expected_updates);new[k]=e
        self.members=new
    def readouts(self,model):
        if not self.at_update_boundary:raise RuntimeError('partial bank cannot be evaluated')
        return {k:e.readout_state_dict(model) for k,e in self.members.items()}
