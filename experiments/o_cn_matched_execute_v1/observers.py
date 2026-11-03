"""Read-only wrappers: same numpy Generator draws and same frozen query plan."""
import hashlib,numpy as np
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
class RecordedGenerator:
    def __init__(self,seed,total):self.base=np.random.default_rng(seed);self.total=total;self.calls=0;self.last_sha=None
    @property
    def bit_generator(self):return self.base.bit_generator
    def random(self,size):
        if size!=self.total:raise ValueError('full S uniform draw required')
        x=self.base.random(size);self.calls+=1;self.last_sha=digest(x);return x
class RecordedPlan:
    def __init__(self,plan):self.plan=plan;self.h=hashlib.sha256();self.blocks=0
    def __getattr__(self,k):return getattr(self.plan,k)
    def block(self,index):
        ids,f,q=self.plan.block(index);self.h.update(np.array([index,len(ids)],np.int64).tobytes())
        for x in [ids,f,q]:self.h.update(np.ascontiguousarray(x).tobytes())
        self.blocks+=1;return ids,f,q

