"""Model-independent, float64 joint-role last-step probability guidance.

No neural import, global RNG use, node assignment, epsilon or graph repair.
Analytic derivatives touch probabilities only; workspace O(S+N*K), not O(S*K).
"""
import hashlib
import json
import time
import numpy as np
from role_statistics import validate

SETTINGS=dict(version="role_joint_cf_v1",K=32,seed=2026101603,delta=.1,max_trials=8,
    descent_atol=1e-12,descent_rtol=1e-9,gradient_floor=1e-14,
    renumber_atol=2e-10,renumber_rtol=1e-9,finite_difference_atol=2e-8,
    finite_difference_rtol=2e-5,frequency_bound=.5)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Layout:
    def __init__(self,types,slots,families,pairs,prior):
        self.types,self.slots,self.families,self.pairs=(v.copy() for v in validate(types,slots,families,pairs))
        self.prior=np.asarray(prior,dtype=np.float64).copy()
        if self.prior.shape!=(len(self.pairs),) or not np.isfinite(self.prior).all() or ((self.prior<0)|(self.prior>1)).any():
            raise ValueError("invalid exact family prior")
        self.sizes={int(c):int((self.types==c).sum()) for c in np.unique(self.types)}
        count=[self.sizes.get(int(a),0)*(self.sizes.get(int(a),0)-1)//2 if a==b else
            self.sizes.get(int(a),0)*self.sizes.get(int(b),0) for a,b in self.pairs]
        if len(families)!=sum(count):raise ValueError("complete legal slots required")
        self.fixed0=self.prior[self.families]==0;self.fixed1=self.prior[self.families]==1
        self.endpoints={}
        # Exclude only exact-zero prior factors, which are identically one.
        for c in self.sizes:
            nodes=np.flatnonzero(self.types==c);local=np.full(len(self.types),-1,dtype=np.int64);local[nodes]=np.arange(len(nodes))
            ids=[];ends=[]
            for row in self.slots:
                selected=np.flatnonzero((self.types[row]==c)&~self.fixed0)
                ids.append(selected);ends.append(local[row[selected]])
            self.endpoints[str(c)]=(np.concatenate(ids),np.concatenate(ends))
        for arr in (self.types,self.slots,self.families,self.pairs,self.prior):arr.flags.writeable=False

    @classmethod
    def canonical(cls,types,pairs,prior):
        types=np.asarray(types,dtype=np.int64);arrays=[];families=[]
        for r,(a,b) in enumerate(pairs):
            u=np.flatnonzero(types==a);v=np.flatnonzero(types==b)
            if a==b:
                i,j=np.triu_indices(len(u),1);e=np.stack([u[i],u[j]])
            else:e=np.sort(np.stack([np.repeat(u,len(v)),np.tile(v,len(u))]),axis=0)
            arrays.append(e);families.append(np.full(e.shape[1],r,dtype=np.int64))
        return cls(types,np.concatenate(arrays,axis=1),np.concatenate(families),pairs,prior)

    def probability(self,p):
        p=np.asarray(p,dtype=np.float64)
        if p.shape!=self.families.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("invalid/nonfinite probability input; cannot sample")
        if (p[self.fixed0]!=0).any() or (p[self.fixed1]!=1).any():raise ValueError("exact family support violation")
        return p


def freeze_target(profile,layout):
    """Canonical degree rows only; same artifact must be used by C and D."""
    if profile.get('format')!='anonymous-joint-relation-degree-v1' or profile['family_pairs']!=layout.pairs.tolist():
        raise ValueError("target schema differs")
    if set(profile['groups'])!=set(layout.endpoints):raise ValueError("target type inventory differs")
    omega={};target={};scales={}
    for c,g in profile['groups'].items():
        rows=np.asarray(g['rows']);counts=np.asarray(g['counts'])
        if rows.ndim!=2 or rows.shape[1]!=len(layout.pairs) or counts.shape!=(len(rows),):raise ValueError("target dimensions")
        if not np.issubdtype(rows.dtype,np.integer) or not np.issubdtype(counts.dtype,np.integer) or (rows<0).any() or (counts<=0).any():raise ValueError("target nonnegative integer rows/counts required")
        if counts.sum()!=g['nodes'] or g['nodes']!=layout.sizes[int(c)]:raise ValueError("target node count differs")
        if len(np.unique(rows,axis=0))!=len(rows):raise ValueError("target rows must be compressed")
        cap=[]
        for r,(a,b) in enumerate(layout.pairs):
            n=layout.sizes.get(int(b if int(c)==a else a),0)-(a==b)
            cap.append(max(0,n) if int(c) in (a,b) else 0)
        cap=np.asarray(cap)
        if (rows>cap).any() or (rows[:,layout.prior==0]!=0).any():raise ValueError("target outside legal support/capacity")
        if (rows[:,layout.prior==1]!=cap[layout.prior==1]).any():raise ValueError("target conflicts with exact-one support")
        scale=1+np.sqrt((rows.astype(float)**2*counts[:,None]).sum(0)/g['nodes']);scales[c]=scale.tolist()
        incident=np.flatnonzero((cap>0)&(layout.prior>0))
        w=np.zeros((SETTINGS['K'],len(layout.pairs)))
        for k,r in enumerate(incident):w[k,r]=.5/scale[r]
        rng=np.random.default_rng(np.random.SeedSequence([SETTINGS['seed'],int(c)]))
        if len(incident):w[len(incident):,incident]=rng.uniform(-.5,.5,(SETTINGS['K']-len(incident),len(incident)))/scale[incident]
        cf=(counts[:,None]*np.exp(1j*(rows@w.T))).sum(0)/g['nodes']
        omega[c]=w.tolist();target[c]=dict(real=cf.real.tolist(),imag=cf.imag.tolist())
    # Endpoint first moments must agree, even though no node gets a target row.
    for r,(a,b) in enumerate(layout.pairs):
        def total(c):
            g=profile['groups'][str(c)];return int((np.asarray(g['rows'])[:,r]*g['counts']).sum())
        if a!=b and total(a)!=total(b):raise ValueError("target endpoint sums disagree")
        if a==b and total(a)%2:raise ValueError("target odd homogeneous degree sum")
    payload=dict(settings=SETTINGS.copy(),profile=profile,profile_sha256=signature(profile),scales=scales,omega=omega,target_cf=target,
        condition='known training-reference empirical joint degree distribution; no node assignments',
        frequency_rule='incident axes first, then per-type seeded uniform joint directions; scaled by 1+RMS degree')
    return dict(payload=payload,sha256=signature(payload))


class RoleGuide:
    def __init__(self,layout,frozen):
        self.layout=layout
        if signature(frozen['payload'])!=frozen['sha256']:raise ValueError("frozen guidance hash mismatch")
        expected=freeze_target(frozen['payload']['profile'],layout)
        if expected!=frozen:raise ValueError("settings/frequencies/target binding mismatch")
        self.frozen=frozen;self.settings=frozen['payload']['settings']
        self.omega={c:np.asarray(w) for c,w in frozen['payload']['omega'].items()}
        self.target={c:np.array(v['real'])+1j*np.array(v['imag']) for c,v in frozen['payload']['target_cf'].items()}
        for array in list(self.omega.values())+list(self.target.values()):array.flags.writeable=False

    def statistic(self,p,gradient=False):
        """Log-products avoid accumulation underflow; denominator bounded away from zero.

        abs(omega)<=.5 means real(1-p+p*exp(i*omega))>=cos(.5)>0.
        Underflow of exp(sum log) itself is recorded, never epsilon-repaired.
        """
        p=self.layout.probability(p);gp=np.zeros_like(p) if gradient else None
        loss=0.;underflow=0;small=0;min_log=0.;K=self.settings['K'];C=len(self.target)
        for c,(ids,node) in self.layout.endpoints.items():
            n=self.layout.sizes[int(c)];f=self.layout.families[ids];values=p[ids]
            for k in range(K):
                h=np.expm1(1j*self.omega[c][k])[f];factor=1+values*h
                logs=np.log1p(values*h)
                lr=np.bincount(node,weights=logs.real,minlength=n);li=np.bincount(node,weights=logs.imag,minlength=n)
                phi=np.exp(lr+1j*li);residual=phi.mean()-self.target[c][k]
                loss+=float(abs(residual)**2)/(C*K)
                underflow+=int((phi==0).sum());small+=int((np.abs(phi)<1e-12).sum());min_log=min(min_log,float(lr.min(initial=0)))
                if gradient:
                    contribution=(2/(C*K*n))*np.real(np.conj(residual)*phi[node]*h/factor)
                    gp+=np.bincount(ids,weights=contribution,minlength=len(p))
        if not np.isfinite(loss) or gp is not None and not np.isfinite(gp).all():raise FloatingPointError("nonfinite joint statistic/gradient")
        gl=gp*p*(1-p) if gradient else None
        return loss,gl,dict(underflow_node_frequencies=underflow,small_node_frequencies=small,min_log_modulus=min_log)

    def apply(self,p):
        p=self.layout.probability(p);original=p.copy();started=time.perf_counter();movable=(p>0)&(p<1)&~self.layout.fixed0&~self.layout.fixed1
        record=dict(status='skipped',accepted_delta=0.,trials=0,backtracks=0,effective_movable_slots=int(movable.sum()),
            exact_zero_slots=int((p==0).sum()),exact_one_slots=int((p==1).sum()),configuration_sha256=self.frozen['sha256'],
            cpu_only=True,new_sampling_law=True,probability_change_l1=0.,probability_change_max=0.,statistics_gradient_seconds=0.,backtracking_seconds=0.)
        try:
            tick=time.perf_counter();before,g,diagnostics=self.statistic(p,True)
            record.update(statistics_gradient_seconds=time.perf_counter()-tick,loss_before=before,loss_after=before,**diagnostics)
            g[~movable]=0.;norm=float(np.max(np.abs(g),initial=0))
            record.update(raw_logit_gradient_max=norm,raw_logit_gradient_l2=float(np.linalg.norm(g)),gradient_floor=self.settings['gradient_floor'])
            if not movable.any():record['reason']='no_movable_probabilities'
            elif norm<=self.settings['gradient_floor']:record['reason']='zero_or_roundoff_gradient'
            else:
                direction=g[movable]/norm;z=np.log(p[movable])-np.log1p(-p[movable]);tick=time.perf_counter()
                threshold=self.settings['descent_atol']+self.settings['descent_rtol']*abs(before)
                record['descent_threshold']=threshold
                for j in range(self.settings['max_trials']):
                    delta=self.settings['delta']/(2**j);candidate=p.copy()
                    candidate[movable]=np.exp(-np.logaddexp(0.,-(z-delta*direction)))
                    after,_,_=self.statistic(candidate,False)
                    record.update(trials=j+1,backtracks=j,trial_loss=after)
                    if after<before-threshold:
                        original=candidate;record.update(status='accepted',reason='verified_proxy_decrease',accepted_delta=delta,loss_after=after);break
                else:record['reason']='no_verified_descent'
                record['backtracking_seconds']=time.perf_counter()-tick
        except FloatingPointError as exc:
            record.update(status='numerical_failure',reason=str(exc))
        change=np.abs(original-p)
        record.update(probability_change_l1=float(change.sum()),probability_change_max=float(change.max(initial=0)),
            changed_slots=int(np.count_nonzero(change)),wall_seconds=time.perf_counter()-started)
        return original,record
