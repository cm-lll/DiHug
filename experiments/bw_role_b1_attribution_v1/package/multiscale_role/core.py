"""Reference-only frozen multiscale joint CF + analytic thinned tail target.

Reuses v1's original posterior derivative and identical step controller. No
neural changes, no dependence on six-state outputs or external-assistant code.
"""
import copy
import hashlib
from types import MappingProxyType
import numpy as np
from stepwise_role.core import StepwiseGuide,freeze as freeze_v1,signature
from multiscale_role.tail import combine,binomial_tail_states,empirical_tail

DESIGN=dict(version='multiscale_role_tail_v1',K_per_scale=32,scales=[.5,1.,2.],
            frequency_seed=2026101801,CF_weight=.5,tail_weight=.5,CF_divisor=4.,
            quantile=.9,thinning_max=.5,tail_threshold=3,
            mass_atol=5e-11,frequency_bound=2.)


def freeze(profile,layout,alpha):
    old=freeze_v1(profile,layout,alpha) # original target/support/capacity validation
    payload=old['payload'];source=payload['frequencies']['payload']
    omega={};channels={}
    for c,g in profile['groups'].items():
        cap=np.asarray(payload['capacities'][c]);rows=np.asarray(g['rows']);counts=np.asarray(g['counts'])
        incident=np.flatnonzero((cap>0)&(layout.prior>0))
        if len(incident)>32:raise ValueError('more than 32 incident families unsupported by frozen design')
        scale=1+np.sqrt((rows.astype(float)**2*counts[:,None]).sum(0)/g['nodes'])
        directions=np.zeros((32,len(layout.pairs)))
        for j,r in enumerate(incident):directions[j,r]=1
        rng=np.random.default_rng(np.random.SeedSequence([DESIGN['frequency_seed'],int(c)]))
        rest=32-len(incident)
        if len(incident):
            values=rng.uniform(.25,1.,(rest,len(incident)))*rng.choice([-1.,1.],(rest,len(incident)))
            directions[len(incident):,incident]=values/np.sqrt(len(incident))
        omega[c]=np.concatenate([x*directions/scale for x in DESIGN['scales']]).tolist()
        masks=[[int(r)] for r in incident]
        if len(incident)>1:masks.append(incident.tolist()) # total only when nonduplicate
        group=[]
        for members in masks:
            degrees=rows[:,members].sum(1);order=np.argsort(degrees,kind='stable')
            at=np.searchsorted(np.cumsum(counts[order]),.9*g['nodes'],side='left')
            d90=int(degrees[order[at]])
            group.append(dict(families=members,p=float(min(.5,3/max(1,d90))),reference_D90=d90,
                              kind='total' if len(members)>1 else 'relation'))
        channels[c]=group
    # Keep only old frequency source as validation provenance; never use it or
    # the external independent64 evaluation bank in the optimization objective.
    result=dict(design=copy.deepcopy(DESIGN),controller=copy.deepcopy(payload['settings']),
                profile=copy.deepcopy(profile),alpha=payload['alpha'],prior=payload['prior'],
                capacities=payload['capacities'],omega=omega,channels=channels,
                frequency_source='training profile only; axes + seeded nonzero joint directions, three scales',
                controller_base_sha256=old['sha256'],condition=payload['condition'])
    return dict(payload=result,sha256=signature(result))


class MultiScaleGuide(StepwiseGuide):
    def __init__(self,layout,frozen,check_time=None):
        p=frozen['payload']
        if signature(p)!=frozen['sha256'] or freeze(p['profile'],layout,p['alpha'])!=frozen:
            raise ValueError('multiscale profile/frequency/tail/controller binding differs')
        self.layout=layout;self.frozen=copy.deepcopy(frozen)
        self.settings=copy.deepcopy(p['controller']) # unchanged v1 delta/floor/tolerances
        self.alpha=np.asarray(p['alpha']).copy();self.alpha.flags.writeable=False
        self.omega={c:np.asarray(w) for c,w in p['omega'].items()}
        self.channels=copy.deepcopy(p['channels']);self.targets={};self._check=check_time
        self.tail_layout={};self.__target_cache={};self.__target_seals={};self._target_diagnostics={};self._last_components={}
        for c,(ids,nodes) in layout.endpoints.items():
            entries=[]
            for ch in self.channels[c]:
                take=np.isin(layout.families[ids],ch['families'])
                order=np.argsort(nodes[take],kind='stable')
                eids=ids[take][order].copy();enodes=nodes[take][order].copy()
                eids.flags.writeable=enodes.flags.writeable=False
                entries.append((eids,enodes))
            self.tail_layout[c]=entries
        for w in self.omega.values():w.flags.writeable=False

    def check(self):
        if self._check:self._check()

    @property
    def _target_cache(self):
        return MappingProxyType(self.__target_cache)

    @staticmethod
    def _target_seal(value):
        cf,tail=value
        return signature(dict(cf={c:dict(real=a.real.tolist(),imag=a.imag.tolist()) for c,a in cf.items()},
                              tail={c:list(v) for c,v in tail.items()}))

    def target(self,s):
        if not 0<=s<len(self.alpha):raise ValueError('target time outside schedule')
        if s in self._target_cache:return self._target_cache[s]
        self.check();a=self.alpha[s];pi=self.layout.prior
        b=a+(1-a)*pi;z=(1-a)*pi
        cf={};tails={};target_underflow=0
        for c,g in self.frozen['payload']['profile']['groups'].items():
            self.check();rows=np.asarray(g['rows']);weight=np.asarray(g['counts'])/g['nodes']
            cap=np.asarray(self.frozen['payload']['capacities'][c]);h=np.expm1(1j*self.omega[c])
            values=((1+b*h)[None,:,:]**rows[:,None,:]*(1+z*h)[None,:,:]**(cap-rows)[:,None,:]).prod(-1)
            if not np.isfinite(values).all():raise FloatingPointError('nonfinite multiscale target')
            target_underflow+=int((values==0).sum())
            cf[c]=(weight[:,None]*values).sum(0)
            tails[c]=[]
            for ch in self.channels[c]:
                self.check();dist=np.zeros((len(rows),4));dist[:,0]=1
                for r in ch['families']:
                    dist=combine(dist,binomial_tail_states(rows[:,r],ch['p']*b[r]))
                    dist=combine(dist,binomial_tail_states(cap[r]-rows[:,r],ch['p']*z[r]))
                if np.max(abs(dist.sum(-1)-1),initial=0)>DESIGN['mass_atol']:
                    raise FloatingPointError('forward tail target mass error')
                tails[c].append(float(weight@dist[:,3]))
            # Immutable bytes backing prevents even setflags(write=True), not
            # just element writes. Outer maps and tail sequences are immutable.
            cf[c]=np.frombuffer(cf[c].tobytes(),dtype=cf[c].dtype)
        value=(MappingProxyType(cf),MappingProxyType({c:tuple(v) for c,v in tails.items()}))
        self.__target_cache[s]=value
        self.__target_seals[s]=self._target_seal(value)
        self._target_diagnostics[s]=dict(target_underflow_role_frequencies=target_underflow)
        return value

    def binding(self):
        if self.settings!=self.frozen['payload']['controller'] or signature(self.frozen['payload'])!=self.frozen['sha256']:
            raise ValueError('multiscale controller/artifact mutated')
        if self.channels!=self.frozen['payload']['channels']:raise ValueError('tail channels mutated')
        for s,value in self.__target_cache.items():
            if self._target_seal(value)!=self.__target_seals[s]:raise ValueError('cached target changed')
        def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
        actual=dict(alpha=digest(self.alpha),omega={c:digest(w) for c,w in self.omega.items()},
                    layout={k:digest(getattr(self.layout,k)) for k in ('types','slots','families','pairs','prior')})
        return dict(frozen_sha256=self.frozen['sha256'],derived_sha256=signature(actual))

    def statistic(self,q,s,gradient=False):
        self.check();q=self.layout.probability(q);target_cf,target_tail=self.target(s)
        C=len(self.omega);cf_loss=0.;tail_loss=0.;underflow=0;tail_underflow=0;small=0;min_log=0.;mass_error=0.
        gp=np.zeros_like(q) if gradient else None;cf_means={};tail_means={}
        for c,(ids,node) in self.layout.endpoints.items():
            self.check();n=self.layout.sizes[int(c)];f=self.layout.families[ids];v=q[ids]
            K=len(self.omega[c]);cf_means[c]=[];tail_means[c]=[]
            for k,w in enumerate(self.omega[c]):
                self.check();h=np.expm1(1j*w)[f];factor=1+v*h
                if (abs(factor)<np.cos(1.)-1e-14).any() or not np.isfinite(factor).all():
                    raise FloatingPointError('multiscale nonzero factor bound failed')
                logs=np.log1p(v*h)
                lr=np.bincount(node,weights=logs.real,minlength=n);li=np.bincount(node,weights=logs.imag,minlength=n)
                phi=np.exp(lr+1j*li);mean=phi.mean();cf_means[c].append(mean)
                residual=mean-target_cf[c][k]
                cf_loss+=float(abs(residual)**2)/(4*C*K)
                underflow+=int((phi==0).sum());small+=int((abs(phi)<1e-12).sum());min_log=min(min_log,float(lr.min(initial=0)))
                if gradient:
                    derivative=.5*2/(4*C*K*n)*np.real(np.conj(residual)*phi[node]*h/factor)
                    gp+=np.bincount(ids,weights=derivative,minlength=len(q))
            H=len(self.channels[c])
            for j,ch in enumerate(self.channels[c]):
                self.check();eids,enodes=self.tail_layout[c][j]
                mean,derivative,detail=empirical_tail(q,eids,enodes,n,ch['p'],gradient,self.check)
                tail_means[c].append(mean);residual=mean-target_tail[c][j]
                tail_loss+=residual**2/(C*H)
                mass_error=max(mass_error,detail['mass_error']);tail_underflow+=detail['underflow']
                if gradient:gp+=np.bincount(eids,weights=.5*2*residual/(C*H*n)*derivative,minlength=len(q))
        loss=.5*cf_loss+.5*tail_loss
        if not np.isfinite(loss) or gp is not None and not np.isfinite(gp).all():
            raise FloatingPointError('nonfinite multiscale statistic/gradient')
        self._last_components=dict(CF_loss=cf_loss,tail_loss=tail_loss)
        return loss,gp,dict(loss_CF=cf_loss,loss_tail=tail_loss,underflow_node_frequencies=underflow+tail_underflow,
                           CF_underflow_node_frequencies=underflow,tail_low_count_underflow_nodes_channels=tail_underflow,
                           underflow_aggregate_scope='legacy controller trigger count; separate CF and tail counts also reported',
                           small_node_frequencies=small,min_log_modulus=min_log,tail_mass_error=mass_error,
                           mean_cf=cf_means,mean_tail=tail_means,**self._target_diagnostics[s])

    def posterior_gradient(self,*args,**kwargs):
        # Original v1 apply removes only mean_cf; no large per-node diagnostic
        # arrays escape the statistic. mean_tail is at most a few scalars/type.
        return super().posterior_gradient(*args,**kwargs)

    def apply(self,p0,state,kernel,t,check_time=None):
        old_check=self._check
        if check_time:self._check=check_time
        try:
            p,q,record=super().apply(p0,state,kernel,t,check_time=self.check)
            if 'loss_CF' in record:
                record['components_before']=dict(CF_loss=record.pop('loss_CF'),tail_loss=record.pop('loss_tail'))
                record['components_after']=copy.deepcopy(self._last_components if record['status']=='accepted' else record['components_before'])
            record['objective']='0.5*multiscale_CF/4 + 0.5*analytic_thinned_tail'
            return p,q,record
        finally:self._check=old_check
