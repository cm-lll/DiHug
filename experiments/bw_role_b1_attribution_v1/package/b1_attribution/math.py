"""Diagnostic formulas only. Never searches a step or changes the guide."""
import numpy as np
from multiscale_role.tail import empirical_tail,binomial_tail_states
from multiscale_role.evaluation import full_pmf_rows,genre64

TOL=dict(gradient_sum_rtol=5e-11,loss_atol=2e-14,loss_rtol=5e-11,
         metric_CF_atol=2e-10,metric_W1_atol=1e-8,metric_rtol=1e-9,
         finite_difference_atol=2e-8,finite_difference_rtol=2e-5,PMF_atol=5e-11)


def check_close(a,b,atol,rtol,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(label+' shape/nonfinite')
    error=float(np.max(abs(a-b),initial=0));scale=float(max(np.max(abs(a),initial=0),np.max(abs(b),initial=0)))
    bound=atol+rtol*scale
    if error>bound:raise ValueError(label+' differs: '+str(dict(error=error,tolerance=bound)))
    return dict(max_error=error,tolerance=bound)


def posterior_chain(p,state,kernel,guide,t):
    l=guide.layout;q=kernel.reverse(p,state,l.families,t-1,t)
    lo=np.zeros_like(p);hi=np.ones_like(p);lo[l.fixed1]=1;hi[l.fixed0]=0
    b0=kernel.reverse(lo,state,l.families,t-1,t);b1=kernel.reverse(hi,state,l.families,t-1,t)
    proof=check_close(q,b0+(b1-b0)*p,guide.settings['posterior_atol'],0.,'original affine posterior')
    factor=(b1-b0)*p*(1-p);factor[l.fixed0|l.fixed1]=0.
    return q,factor,proof


def components(q,guide,s,gradient=False,chain=None,check=lambda:None):
    """Weighted loss + derivative by type x {CF,tail}; original GLOBAL weights.

    .5/(4*C*K) CF and .5/(C*H_c) tail. No reweighting within a type.
    With chain, derivatives are in original collected-p0 logit coordinates.
    """
    q=guide.layout.probability(q);l=guide.layout;target_cf,target_tail=guide.target(s)
    C=len(guide.omega);losses={};grads={};values={}
    for c,(ids,node) in l.endpoints.items():
        check();n=l.sizes[int(c)];f=l.families[ids];v=q[ids];K=len(guide.omega[c])
        cf=0.;cg=np.zeros_like(q) if gradient else None
        for k,w in enumerate(guide.omega[c]):
            check();h=np.expm1(1j*w)[f];factor=1+v*h
            if (abs(factor)<np.cos(1.)-1e-14).any() or not np.isfinite(factor).all():raise FloatingPointError('CF factor bound')
            logs=np.log1p(v*h);phi=np.exp(np.bincount(node,weights=logs.real,minlength=n)+1j*np.bincount(node,weights=logs.imag,minlength=n))
            residual=phi.mean()-target_cf[c][k];cf+=float(abs(residual)**2)/(4*C*K)
            if gradient:
                derivative=.5*2/(4*C*K*n)*np.real(np.conj(residual)*phi[node]*h/factor)
                cg+=np.bincount(ids,weights=derivative,minlength=len(q))
        H=len(guide.channels[c]);tail=0.;tg=np.zeros_like(q) if gradient else None;probes=[]
        for j,ch in enumerate(guide.channels[c]):
            check();eids,enodes=guide.tail_layout[c][j]
            mean,derivative,detail=empirical_tail(q,eids,enodes,n,ch['p'],gradient,check)
            residual=mean-target_tail[c][j];tail+=residual**2/(C*H)
            probes.append(dict(families=ch['families'],p=ch['p'],target=target_tail[c][j],value=mean))
            if gradient:tg+=np.bincount(eids,weights=.5*2*residual/(C*H*n)*derivative,minlength=len(q))
        losses[c+'/CF']=.5*cf;losses[c+'/tail']=.5*tail
        values[c]=dict(raw_CF_global_normalized=cf,raw_tail_global_normalized=tail,probes=probes)
        if gradient:
            for name,g in ((c+'/CF',cg),(c+'/tail',tg)):
                if chain is not None:g*=chain
                if not np.isfinite(g).all():raise FloatingPointError('nonfinite component derivative')
                grads[name]=g
    return dict(losses=losses,total_loss=sum(losses.values()),details=values),grads


def sum_certificate(parts,total):
    summed=np.zeros_like(total);absolute=np.zeros_like(total)
    for g in parts.values():summed+=g;absolute+=abs(g)
    scale=float(max(abs(total).max(initial=0),absolute.max(initial=0)))
    error=float(abs(summed-total).max(initial=0));bound=TOL['gradient_sum_rtol']*scale
    if error>bound:raise ValueError('component logit gradients do not sum to original total')
    return dict(max_abs_error=error,scale=scale,tolerance=bound,relative_to_scale=error/scale if scale else None,
                exact_zero_case=scale==0,passed=True),summed


def norm(x):
    scale=float(abs(x).max(initial=0))
    return scale*float(np.linalg.norm(x/scale)) if scale else 0.


def alignment(a,b):
    na,nb=norm(a),norm(b);dot=float(np.dot(a,b))
    cosine=float(np.dot(a/na,b/nb)) if na and nb else None
    if cosine is not None and (not np.isfinite(cosine) or abs(cosine)>1+5e-12):raise FloatingPointError('invalid cosine')
    return dict(inner_product=dot,norm_left=na,norm_right=nb,cosine=cosine,
                zero_norm_left=na==0,zero_norm_right=nb==0,
                cosine_status='undefined_zero_norm' if not na or not nb else 'defined')


def replay(p,state,kernel,guide,t,old,total_gradient):
    """ONE old accepted step, or the recorded no-op. No guide.apply/backtracking."""
    if old['time']!=t or old['next_time']!=t-1 or old['configuration_sha256']!=guide.frozen['sha256']:
        raise ValueError('old step time/objective binding differs')
    if old['status'] not in ('accepted','skipped'):raise ValueError('cannot attribute a failed/unknown old step')
    maximum=float(abs(total_gradient).max(initial=0));step=float(old['accepted_delta'])
    for name,value in [('raw_logit_gradient_max',maximum),('delta_max',guide.settings['delta_max']),
                       ('proposed_delta',guide.settings['delta_max']*guide.alpha[t-1])]:
        check_close(value,old[name],0.,5e-11,'old '+name)
    if old['status']=='accepted':
        j=old['backtracks']
        if not isinstance(j,int) or not 0<=j<guide.settings['max_trials'] or old['trials']!=j+1 or maximum<=guide.settings['gradient_floor']:
            raise ValueError('invalid old accepted controller record')
        if step!=old['proposed_delta']/(2**j):raise ValueError('old step is not frozen backtracking schedule')
    elif step!=0:raise ValueError('skipped old step has nonzero update')
    free=(p>0)&(p<1)&~guide.layout.fixed0&~guide.layout.fixed1
    moved=p.copy();dz=np.zeros_like(p)
    if step:
        z=np.log(p[free])-np.log1p(-p[free]);dz[free]=-step*(total_gradient[free]/maximum)
        # Keep EXACT original ordering: z-step*(g/max), not z+(-step*g/max).
        moved[free]=np.exp(-np.logaddexp(0.,-(z-step*(total_gradient[free]/maximum))))
    q=kernel.reverse(moved,state,guide.layout.families,t-1,t)
    if not np.array_equal(moved[~free],p[~free]):raise ValueError('fixed support moved')
    return moved,q,dz,dict(accepted_delta=step,original_gradient_infinity_norm=maximum,
                           candidate_updates_replayed=int(step!=0),new_candidate_searches=0,free_slots=int(free.sum()),
                           numerical_saturated_before=int(((p==0)|(p==1)).sum()),
                           numerical_saturated_after=int(((moved==0)|(moved==1)).sum()))


def independent_cf_gradient(q,guide,s,spec,check=lambda:None):
    """Derivative of ORIGINAL independent Genre64 squared MEAN-CF error."""
    c=str(spec['Genre_type']);w=np.asarray(spec['omega64']);ids,node=guide.layout.endpoints[c]
    n=guide.layout.sizes[int(c)];f=guide.layout.families[ids];K=len(w);normalizer=spec['CF_normalizer']
    group=guide.frozen['payload']['profile']['groups'][c];rows=np.asarray(group['rows']);weights=np.asarray(group['counts'])/group['nodes']
    cap=np.asarray(guide.frozen['payload']['capacities'][c]);a=guide.alpha[s];pi=guide.layout.prior
    b=a+(1-a)*pi;z=(1-a)*pi;loss=0.;gradient=np.zeros_like(q)
    for k in range(K):
        check();h=np.expm1(1j*w[k]);factor=1+q[ids]*h[f]
        if (abs(factor)<.99).any():raise FloatingPointError('held-out Genre64 factor unsupported')
        phi=np.ones(n,complex);np.multiply.at(phi,node,factor)
        target=(weights*((1+b*h)[None,:]**rows*(1+z*h)[None,:]**(cap-rows)).prod(-1)).sum()
        residual=phi.mean()-target;loss+=float(abs(residual)**2)/(K*normalizer)
        d=2/(K*normalizer*n)*np.real(np.conj(residual)*phi[node]*h[f]/factor)
        gradient+=np.bincount(ids,weights=d,minlength=len(q))
    if not np.isfinite(gradient).all():raise FloatingPointError('nonfinite held-out derivative')
    proof=check_close(loss,genre64(q,guide,s,spec,check)['error'],TOL['metric_CF_atol'],TOL['metric_rtol'],'Genre64 value/gradient definition')
    return loss,gradient,proof


def first_order(parts,metric_gradient,step,denominator,actual_change,dz):
    rows={}
    for name,g in parts.items():
        rows[name]=dict(gradient_inner_product=float(np.dot(metric_gradient,g)),
            first_order_contribution=float(-step/denominator*np.dot(metric_gradient,g)) if denominator else 0.)
    total=sum(r['first_order_contribution'] for r in rows.values());direct=float(np.dot(metric_gradient,dz))
    bound=TOL['gradient_sum_rtol']*max(sum(abs(r['first_order_contribution']) for r in rows.values()),abs(direct))
    if abs(total-direct)>bound:raise FloatingPointError('first-order component sum differs from original total direction')
    return dict(components=rows,summed_first_order=total,direct_first_order=direct,additivity_error=total-direct,
                additivity_tolerance=bound,
                actual_metric_change=actual_change,actual_minus_first_order=actual_change-total,
                meaning='first-order fixed-original-denominator attribution, NOT exact causal decomposition',
                positive_contribution='increases independent metric error',
                limitation='other types can also affect the original global infinity normalization; denominator held fixed, no component-removal counterfactual')


def genre_laws(before,after,guide,s,genre='3',check=lambda:None):
    channels=guide.channels[genre]
    if len(channels)!=1:raise ValueError('Genre must have exactly one nonduplicate degree channel')
    ch=channels[0];ids,node=guide.tail_layout[genre][0];n=guide.layout.sizes[int(genre)]
    capacity=int(np.asarray(guide.frozen['payload']['capacities'][genre])[ch['families']].sum())
    counts=np.bincount(node,minlength=n)
    if not (counts==capacity).all():raise ValueError('nonuniform node capacities')
    start=np.r_[0,np.cumsum(counts)[:-1]];pos=np.arange(len(ids))-np.repeat(start,counts)
    pmfs=[];arrays=dict(current_Genre_node_ids=np.flatnonzero(guide.layout.types==int(genre)))
    for label,q in [('before',before),('after',after)]:
        check();mat=np.zeros((n,capacity));mat[node,pos]=q[ids]
        pmf,error=full_pmf_rows(mat,check);pmfs.append(pmf.mean(0))
        arrays[label+'_node_pmf']=pmf;arrays[label+'_node_cdf']=pmf.cumsum(1)
    group=guide.frozen['payload']['profile']['groups'][genre];roles=np.asarray(group['rows']);weights=np.asarray(group['counts'])/group['nodes']
    cap=np.asarray(guide.frozen['payload']['capacities'][genre]);a=guide.alpha[s];pi=guide.layout.prior;b=a+(1-a)*pi;z=(1-a)*pi
    mat=np.zeros((len(roles),capacity));offset=0
    for r in ch['families']:
        for j,d in enumerate(roles[:,r]):mat[j,offset:offset+d]=b[r];mat[j,offset+d:offset+cap[r]]=z[r]
        offset+=cap[r]
    target,error=full_pmf_rows(mat,check)
    arrays['target_anonymous_role_pmf']=target;arrays['target_anonymous_role_cdf']=target.cumsum(1)
    target=weights@target;pmfs.append(target)
    arrays['target_role_degrees']=roles;arrays['target_role_counts']=np.asarray(group['counts'])
    arrays['degree_support']=np.arange(capacity+1);D90=ch['reference_D90'];probe=binomial_tail_states(np.arange(capacity+1),ch['p'])[:,3]
    labels=['before','after','target'];out=dict(D90=D90,probe_p=ch['p'],families=ch['families'],capacity=capacity,
        interpretation='expected empirical integer degree PMF, NOT histogram of expected degrees nor expected graph-W1',
        intervals_use='CDF cutoff k, summands of unit-spacing W1; not PMF bin contributions',states={})
    for label,pmf in zip(labels,pmfs):
        arrays[label+'_pmf']=pmf;arrays[label+'_cdf']=pmf.cumsum()
        out['states'][label]=dict(mass_error=float(abs(pmf.sum()-1)),P_degree_gt_D90=float(pmf[D90+1:].sum()),
            original_tail_probe=float(np.dot(pmf,probe)))
    targetcdf=arrays['target_cdf']
    for label in ('before','after'):
        error=abs(arrays[label+'_cdf'][:-1]-targetcdf[:-1]);k=np.arange(capacity)
        ranges=[('k_lt_D90',k<D90),('k_eq_D90',k==D90),('k_gt_D90',k>D90)]
        pieces={name:float(error[mask].sum()) for name,mask in ranges}
        w1=float(error.sum());out['states'][label].update(W1=w1,W1_intervals=pieces,W1_additivity_error=sum(pieces.values())-w1)
        check_close(sum(pieces.values()),w1,TOL['PMF_atol'],TOL['metric_rtol'],'W1 interval sum')
        arrays[label+'_W1_summands']=error
    return out,arrays
