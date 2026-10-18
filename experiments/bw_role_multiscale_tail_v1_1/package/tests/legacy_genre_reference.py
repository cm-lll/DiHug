"""Pure NumPy conditional Genre diagnostics. No model, RNG or graph sampling.

Condition on frozen state AND query plan. Slot variables are independent under
sampler q. Exact-zero Genre-Genre support proves different Genre degrees use
disjoint random slots. This module fails closed if that condition is violated.
"""
import numpy as np

DIAGNOSTIC_FREQUENCIES = np.geomspace(0.0005, 0.1, 64)
TAIL_THRESHOLDS = (1, 50, 100, 150, 200, 300, 497)

def poisson_binomial(probabilities):
    """Full PMF by stable positive recurrence, O(m²) work/O(m) workspace."""
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probabilities')
    ones=int((p==1).sum()); free=p[(p>0)&(p<1)]
    out=np.zeros(len(free)+1);out[0]=1.
    for j,x in enumerate(free):
        # RHS arrays are evaluated before assignment, so no cascading updates.
        out[1:j+2]=out[1:j+2]*(1-x)+out[:j+1]*x
        out[0]*=1-x
    full=np.zeros(len(p)+1);full[ones:ones+len(out)]=out
    if abs(full.sum()-1)>1e-11 or (full<0).any():raise FloatingPointError('PMF normalization/nonnegativity failed')
    return full

def _cf_summary(prob_rows, target_prob_rows, omega):
    w=np.asarray(omega,dtype=np.float64);h=np.expm1(1j*w)
    if w.ndim!=1 or not np.isfinite(w).all() or abs(w).max(initial=0)>.5:raise ValueError('invalid diagnostic frequency')
    phi=np.exp(np.log1p(prob_rows[:,:,None]*h).sum(axis=1))
    target_phi=np.exp(np.log1p(target_prob_rows[:,:,None]*h).sum(axis=1))
    mean=phi.mean(axis=0); target=target_phi.mean(axis=0)
    # |exp(i*w*D)|²=1 and distinct Genre nodes depend on disjoint slots.
    variance=(1-abs(phi)**2).sum(axis=0)/len(phi)**2
    if variance.min(initial=0)<-1e-13:raise FloatingPointError('negative CF variance')
    bias2=abs(mean-target)**2
    return {'omega':w.tolist(),'conditional_mean_cf_real':mean.real.tolist(),'conditional_mean_cf_imag':mean.imag.tolist(),'noise_matched_target_cf_real':target.real.tolist(),'noise_matched_target_cf_imag':target.imag.tolist(),'squared_mean_cf_distance_by_frequency':bias2.tolist(),'conditional_empirical_cf_variance_by_frequency':variance.tolist(),'expected_squared_empirical_cf_distance_by_frequency':(bias2+variance).tolist(),'mean_squared_mean_cf_distance':float(bias2.mean()),'mean_conditional_empirical_cf_variance':float(variance.mean()),'mean_expected_squared_empirical_cf_distance':float((bias2+variance).mean())}

def _pmf_report(pmf):
    n,m=pmf.shape
    mass=pmf.sum(axis=1); raw_error=float(abs(mass-1).max())
    if not np.isfinite(pmf).all() or (pmf<0).any() or raw_error>1e-11:raise FloatingPointError('invalid PMF report mass')
    # Diagnostic-only roundoff normalization; q and the sampler are untouched.
    pmf=pmf/mass[:,None]
    raw_cdf=pmf.cumsum(axis=1)
    if raw_cdf.min()< -1e-13 or raw_cdf.max()>1+1e-13:raise FloatingPointError('invalid normalized CDF')
    cdf=np.clip(raw_cdf,0,1);tails={};tail_correction=0.
    for k in TAIL_THRESHOLDS:
        tail=pmf[:,k:].sum(axis=1) if k<m else np.zeros(n)
        if not np.isfinite(tail).all() or tail.min()< -1e-13 or tail.max()>1+1e-13:raise FloatingPointError('invalid normalized tail')
        bounded=np.clip(tail,0,1);tail_correction=max(tail_correction,float(abs(tail-bounded).max()));tail=bounded
        no_event=1-tail
        any_event=1. if (no_event==0).any() else float(-np.expm1(np.log1p(-tail).sum()))
        tails[str(k)]={'node_prob_degree_ge':tail.tolist(),'expected_count_degree_ge':float(tail.sum()),'probability_max_degree_ge':any_event}
    expected_pmf=pmf.mean(axis=0)
    # P(max >= k), k=1..capacity, summed is E[max].
    all_below=np.prod(cdf[:,:-1],axis=0)
    return {'node_mean_degree':(pmf@np.arange(m)).tolist(),'expected_empirical_degree_pmf':expected_pmf.tolist(),'expected_max_degree':float((1-all_below).sum()),'tails':tails,'pmf_max_mass_error_before_diagnostic_normalization':raw_error,'diagnostic_cdf_roundoff_bound_max':float(abs(cdf-raw_cdf).max()),'diagnostic_tail_roundoff_bound_max':tail_correction,'diagnostic_normalization_note':'PMF rows normalized and CDF/tails bounded only within explicit roundoff tolerances; input q never changed.'}

def diagnose_genre_arrays(q, types, slots, families, pairs, prior, alpha_s,
                   reference_genre_degrees, guide_omega_genre, genre_type=3, movie_type=2):
    """Return true conditional degree-law/CF diagnostics, never sampled graphs.

    q: full legal unordered-slot probabilities, shape S; slots: 2xS.
    reference_genre_degrees: anonymous clean MG degrees, one per Genre (23).
    guide_omega_genre: full KxR type-3 frequency table from frozen guide.
    alpha_s: cumulative retention for the NEXT state s=t-1, not t.
    All tails and variance condition on provided q/frozen state/query plan.
    """
    q=np.asarray(q,dtype=float);types=np.asarray(types);slots=np.asarray(slots);families=np.asarray(families);pairs=np.asarray(pairs);prior=np.asarray(prior,dtype=float)
    if slots.shape!=(2,len(q)) or families.shape!=q.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('invalid slot arrays')
    if not 0<=alpha_s<=1:raise ValueError('invalid next-state alpha')
    gids=np.flatnonzero(types==genre_type); mids=np.flatnonzero(types==movie_type)
    mg=np.flatnonzero(np.all(pairs==sorted((movie_type,genre_type)),axis=1))
    if len(mg)!=1:raise ValueError('unique Movie-Genre family required')
    r=int(mg[0]);genre_incident=np.flatnonzero(np.any(types[slots]==genre_type,axis=0))
    other=genre_incident[families[genre_incident]!=r]
    if (q[other]!=0).any() or (prior[families[other]]!=0).any():raise ValueError('non-MG Genre support breaks scalar/disjoint-node proof')
    active=np.flatnonzero(families==r); es=slots[:,active]
    if len(active)!=len(gids)*len(mids) or (types[es]==genre_type).sum()!=len(active):raise ValueError('MG capacity/inventory mismatch')
    ge=np.where(types[es[0]]==genre_type,es[0],es[1]);me=np.where(types[es[0]]==movie_type,es[0],es[1])
    if not np.isin(me,mids).all() or len(np.unique(ge*len(types)+me))!=len(active):raise ValueError('duplicate/wrong MG slots')
    rows=np.stack([q[active[ge==g]] for g in gids]);cap=len(mids)
    if rows.shape!=(len(gids),cap):raise ValueError('incomplete MG slots')
    deg=np.asarray(reference_genre_degrees)
    if deg.shape!=(len(gids),) or (deg!=deg.astype(int)).any() or ((deg<0)|(deg>cap)).any():raise ValueError('invalid anonymous reference degrees')
    b=alpha_s+(1-alpha_s)*prior[r];z=(1-alpha_s)*prior[r]
    target_rows=np.where(np.arange(cap)[None,:]<deg[:,None],b,z)
    omega=np.asarray(guide_omega_genre)
    if omega.ndim!=2 or omega.shape[1]!=len(pairs):raise ValueError('guide frequency shape')
    pmf=np.stack([poisson_binomial(p) for p in rows]);target_pmf=np.stack([poisson_binomial(p) for p in target_rows])
    report={'condition':'Independent Bernoulli slots conditional on frozen state, query plan, and supplied q; no graph samples.', 'genre_node_ids':gids.tolist(),'genre_nodes':len(gids),'movie_capacity':cap,'movie_genre_family':r,'next_state_alpha':float(alpha_s),'proof':{'all_random_genre_slots_movie_genre':True,'unique_complete_movie_genre_slots':True,'other_genre_families_exact_zero_prior_and_q':True,'different_genre_degrees_use_disjoint_slot_ids':True},'guide_frequencies':_cf_summary(rows,target_rows,omega[:,r]),'independent_diagnostic_frequencies':_cf_summary(rows,target_rows,DIAGNOSTIC_FREQUENCIES),'conditional_degree_law':_pmf_report(pmf),'noise_matched_reference_degree_law':_pmf_report(target_pmf),'notes':['Expected empirical PMF is the average of full Poisson-binomial node laws, NOT a histogram of expected degrees.','CF squared-distance target is the deterministic noise-matched expected reference CF; reference sampling variance is NOT added.','Tail max probabilities and expected maxima use proven conditional independence across Genre nodes.','Diagnostic grid is fixed before q inspection: 64 geometric frequencies 0.0005..0.1, unrelated to original random-frequency seed.']}
    # Equal unit spacing: W1 between two expected empirical probability laws.
    report['w1_between_expected_empirical_laws']=float(abs(np.cumsum(pmf.mean(0)-target_pmf.mean(0))[:-1]).sum())
    report['notes'].append('w1_between_expected_empirical_laws is NOT expected W1 of a sampled graph; these differ by nonlinearity.')
    return report


def diagnose_genre(q, layout, guide, s, check_time=lambda: None):
    check_time()
    if not 0 <= s < len(guide.alpha):raise ValueError('invalid next-state time')
    profile=guide.frozen['payload']['frequencies']['payload']['profile']
    group=profile['groups']['3']
    pairs=np.asarray(layout.pairs)
    r=int(np.flatnonzero(np.all(pairs==[2,3],axis=1))[0])
    degrees=np.repeat(np.asarray(group['rows'])[:,r],group['counts'])
    result=diagnose_genre_arrays(q,layout.types,layout.slots,layout.families,layout.pairs,layout.prior,guide.alpha[s],degrees,guide.omega['3'])
    check_time()
    result['next_state_time']=int(s)
    # Source target validation prevents using a clean target at noisy times.
    block=result['guide_frequencies']
    reconstructed=np.array(block['noise_matched_target_cf_real'])+1j*np.array(block['noise_matched_target_cf_imag'])
    error=float(abs(reconstructed-guide.targets['3'][s]).max())
    if error>2e-12:raise FloatingPointError('noise-matched reference differs from frozen target')
    result['frozen_target_cf_max_abs_error']=error
    return result

