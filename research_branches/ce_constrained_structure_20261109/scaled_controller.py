"""Fixed positive objective rescaling with pre-update stochastic dual observations.

Multiplying the structural objective by a frozen positive scalar leaves its
constrained minimizers unchanged. It does not leave finite optimization paths
unchanged. Production calibration and AdamW integration still require checks.
"""
from dataclasses import asdict, dataclass
import math

@dataclass
class ScaledController:
    ce_reference: float
    structure_coefficient: float
    dual_rate: float = 0.05
    multiplier: float = 1.0
    updates: int = 0

    def __post_init__(self):
        if not all(math.isfinite(x) and x>0 for x in
                   [self.ce_reference,self.structure_coefficient,self.dual_rate]):
            raise ValueError('Invalid frozen scales; no silent clipping.')
        if not math.isfinite(self.multiplier) or self.multiplier<0 or self.updates<0:
            raise ValueError('Invalid dual state.')

    def coefficients(self):return self.multiplier,self.structure_coefficient

    def commit_observation(self, pre_update_ce):
        if not math.isfinite(pre_update_ce) or pre_update_ce<0:
            raise FloatingPointError('Bad observation; caller rolls back full transaction.')
        proposal=max(0.,self.multiplier+self.dual_rate*(pre_update_ce/self.ce_reference-1.))
        if not math.isfinite(proposal):raise FloatingPointError('Bad dual proposal.')
        self.multiplier=proposal;self.updates+=1

    def state_dict(self):return asdict(self)

    @classmethod
    def from_state_dict(cls,state):return cls(**state)
