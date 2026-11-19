"""Remove output path coordinates without rebuilding scales or directions."""
import numpy as np
from score_candidate import CountShapeScore

class WithoutPaths(CountShapeScore):
    def __init__(self,core):
        super().__init__(core)
        ref=core.features(core.reference).numpy()
        for c in self.cache.values():
            keep=c['cols']<core.base_width
            # Scale/U/dimension remain exactly as computed from the ORIGINAL reference.
            z=ref[c['ids']][:,c['mask']]/c['scale']
            z[:,~keep]=0
            c['mu']=z.mean(0)
            centered=z-c['mu']
            c['target_axis']=np.sort(centered,axis=0)
            c['target_mixed']=np.sort(centered@c['u'],axis=0)
    def terms(self,raw):
        fixed=np.asarray(raw).copy()
        fixed[...,self.core.base_width:]=0
        return super().terms(fixed)
