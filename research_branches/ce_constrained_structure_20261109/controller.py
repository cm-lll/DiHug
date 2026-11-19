"""CPU prototype of a training-level CE-constrained structural objective.

No per-update accept/reject or parameter projection. This module is NOT yet wired
to a production model. Caller commits its state together with optimizer/RNG state.
"""
from dataclasses import asdict, dataclass
import math


@dataclass
class Controller:
    ce_reference: float
    structure_reference: float
    dual_rate: float = 0.05
    multiplier: float = 1.0
    updates: int = 0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   [self.ce_reference, self.structure_reference, self.dual_rate]):
            raise ValueError('References/rate must be positive and finite; no epsilon floor.')
        if not math.isfinite(self.multiplier) or self.multiplier < 0 or self.updates < 0:
            raise ValueError('Invalid dual state.')

    def coefficients(self):
        # L = S/S_ref + eta * (CE/CE_ref - 1).
        return self.multiplier / self.ce_reference, 1.0 / self.structure_reference

    def value(self, ce, structure):
        if not math.isfinite(ce) or not math.isfinite(structure):
            raise FloatingPointError('Nonfinite objective; do not commit state.')
        return structure / self.structure_reference + self.multiplier * (ce / self.ce_reference - 1.0)

    def commit(self, ce):
        """Projected dual ASCENT, called only at a successful update boundary.

        CPU exact-risk tests supply CE at the new parameter point. Production
        integration must explicitly choose and test its stochastic estimate;
        changing this to a pre-step value is not asserted equivalent here.
        """
        if not math.isfinite(ce) or ce < 0:
            raise FloatingPointError('Invalid CE; state unchanged.')
        proposed = max(0.0, self.multiplier + self.dual_rate * (ce / self.ce_reference - 1.0))
        if not math.isfinite(proposed):
            raise FloatingPointError('Nonfinite dual proposal; state unchanged.')
        self.multiplier = proposed
        self.updates += 1

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
