"""Sparse base measure and unbiased sparse-supervision algebra; archived data only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,itertools,json,resource,signal,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structure_sparse_target_20261109'
FULL=ROOT/'results/joint_degree_full_signal_20261109';COST=ROOT/'results/structure_posterior_real_cost_20261109'
START=time.monotonic()
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sigmoid(z):return 1/(1+np.exp(-z))
def ce(m,z):return np.logaddexp(0,z)-m*z
def conditional(base,delta):
    # Fixed/observed supports are exact; no epsilon clipping to create support.
    result=np.asarray(base,dtype=float).copy();mask=(result>0)&(result<1)
    result[mask]=sigmoid(np.log(base[mask])-np.log1p(-base[mask])-delta[mask])
    return result
def algebra():
    pi=np.array([0.,1.,.03,.2,.7]);bayes_max=0.;valid=invalid=0
    for a in [0.,.5,.95,1.]:
        for x in [0,1]:
            q0=np.where(x,a*0+(1-a)*pi,1-(1-a)*pi)
            q1=np.where(x,a+(1-a)*pi,1-a-(1-a)*pi)
            denom=(1-pi)*q0+pi*q1;keep=denom>0
            b=np.divide(pi*q1,denom,out=np.zeros_like(pi),where=keep)
            bayes_max=max(bayes_max,float(abs(b[keep]-(a*x+(1-a)*pi[keep])).max()))
            valid+=int(keep.sum());invalid+=int((~keep).sum())
    assert bayes_max<1e-13
    base=np.array([0.,1.,.03,.2,.7]);d=np.array([100.,-100.,.5,-.8,.3]);h=conditional(base,d)
    assert h[0]==0 and h[1]==1
    # Exhaust all subsets for an independent finite unbiasedness check, no sampling.
    z=np.linspace(-3,2,8);b=np.linspace(.05,.8,8);delta=np.linspace(-.4,.3,8);h=conditional(b,delta)
    losses=[];grads=[];direct=[];correction=(b-h)*z;baseline=ce(b,z).mean()
    for subset in itertools.combinations(range(8),2):
        ids=np.array(subset);losses.append(baseline+correction[ids].mean());direct.append(ce(h,z)[ids].mean())
        grad=(sigmoid(z)-b)/8;grad[ids]+=(b-h)[ids]/2;grads.append(grad)
    loss_error=abs(np.mean(losses)-ce(h,z).mean());grad_error=abs(np.mean(grads,axis=0)-(sigmoid(z)-h)/8).max()
    variance_pred=(1-2/8)*np.var(correction,ddof=1)/2
    var_error=abs(np.var(losses)-variance_pred)
    assert max(loss_error,grad_error,var_error)<1e-13
    fd=[]
    for e in range(8):
        zp=z.copy();zm=z.copy();zp[e]+=1e-6;zm[e]-=1e-6
        fd.append((ce(h,zp).mean()-ce(h,zm).mean())/2e-6)
    fd_error=float(abs(np.array(fd)-(sigmoid(z)-h)/8).max());assert fd_error<1e-8
    # Exact finite posterior identity with a nonuniform sparse base, not a new chain.
    u=np.load(ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz');bits=u['bits'];E=u['energy'];fam=u['families']
    pi=np.array([.25,.4,.35])[fam];mu=np.prod(np.where(bits,pi,1-pi),axis=1)
    prior=mu*np.exp(-E);prior/=prior.sum();masks=1<<np.arange(10);zero=np.arange(1024)[:,None]&~masks;one=zero|masks
    identities=[]
    for a,xcode in itertools.product([0.,.5,.95,1.],[659,421,0,1023]):
        x=bits[xcode];q1=a*bits+(1-a)*pi;like=np.prod(np.where(x,q1,1-q1),axis=1)
        post=prior*like;post/=post.sum();base=a*x+(1-a)*pi
        htab=np.array([conditional(base,E[one[c]]-E[zero[c]]) for c in range(1024)])
        den=post[zero]+post[one];keep=den>0
        independent=np.divide(post[one],den,out=np.zeros_like(den),where=keep)
        err=float(abs(htab[keep]-independent[keep]).max());meanerr=float(abs(post@htab-post@bits).max())
        assert max(err,meanerr)<1e-12
        identities.append(dict(retention=a,observation=xcode,conditional_error=err,expectation_error=meanerr,undefined_backgrounds=int((~keep).sum())))
    save('finite_posterior_checks',identities)
    return dict(bayes_error=bayes_max,valid_events=valid,zero_evidence_events_retained=invalid,
        subset_enumerations=28,loss_unbiased_error=float(loss_error),gradient_unbiased_error=float(grad_error),
        sampling_variance_error=float(var_error),finite_difference_error=fd_error,fixed_support_exact=True)

def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU algebra and 4 archived PubMed0422 raw2040 terminal-noise hard backgrounds; full saved deltas; no score recomputation or network',
        sparse_base='Independent Bernoulli per legal slot using fixed reference relation frequency; exact zero/one support',
        benchmark='rho0 proportional mu*exp(-F), lambda1 fixed diagnostic only. No temperature/grid/candidate adoption.',
        scoring='unchanged real six-term F; not previous toy four-term E',
        supervision='hypothetical uniform without replacement 128 slots; exact analytical variance of full soft CE vs analytic-base plus sampled structural residual',
        no_new_samples=True,no_hyperparameter_search=True,
        limits=dict(threads=1,seconds=120,memory_MiB=1024,GPU=0)))
    hashes={}
    def bind(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None:assert expected[str(p)]==h,str(p)
        hashes[str(p)]=h;return p
    done=json.loads(bind(FULL/'completion.json').read_text())
    support=json.loads(bind(COST/'support.json').read_text());budget=json.loads(bind(COST/'budget_extrapolation.json').read_text())
    bind(ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz')
    pi=np.array([s['prior'] for s in support]);checks=algebra();save('algebra_checks',checks)
    summaries=[];groups=[];sampling=[];arrays={};K=128
    for parent in ['CE','PS']:
        for draw in [0,1]:
            tag=f'PubMed_{parent}_CE_t1000_draw{draw}'
            with np.load(bind(FULL/(tag+'.npz'),done['output_hashes'])) as z:
                bits=z['bits'].copy();f=z['relation'].copy();p=z['probability'].copy();delta=z['new_delta'].copy();labels=z['reference'].copy()
            M=len(bits);assert M==18336 and np.isfinite(delta).all() and np.all((p>0)&(p<1))
            assert np.isin(bits,[0,1]).all() and labels.sum()==926
            base=pi[f];d01=(1-2*bits)*delta;h=conditional(base,d01);shift=h-base;z=np.log(p)-np.log1p(-p)
            assert np.max(abs(shift)-abs(d01)/4)<1e-13
            logodds_change=np.log(h)-np.log1p(-h)-np.log(base)+np.log1p(-base)
            assert np.max(abs(logodds_change+d01))<1e-12
            full=ce(h,z);cheap=ce(base,z);res=(base-h)*z
            assert np.max(abs(full-cheap-res))<1e-13
            # These are CONDITIONAl probabilities on archived G_-e, not posterior averages m_e.
            summaries.append(dict(state=tag,slots=M,base_probability_min=float(base.min()),base_probability_max=float(base.max()),
                conditional_min=float(h.min()),conditional_max=float(h.max()),
                structural_shift_mean=float(shift.mean()),structural_shift_MAE=float(abs(shift).mean()),structural_shift_max=float(abs(shift).max()),
                relative_probability_change_MAE=float(np.mean(abs(shift)/base)),
                logodds_change_abs_mean=float(abs(d01).mean()),logodds_change_abs_max=float(abs(d01).max()),
                logodds_correction_to_base_norm=float(np.linalg.norm(d01)/np.linalg.norm(np.log(base)-np.log1p(-base))),
                logit_gradient_residual_to_base_norm=float(np.linalg.norm(base-h)/np.linalg.norm(p-base)),
                local_conditional_sum=float(h.sum()),base_sum=float(base.sum()),
                sum_warning='sum of differing-background full conditionals, NOT posterior expected edge count or final graph density'))
            for relation in range(10):
                for present in [-1,0,1]:
                    mask=f==relation
                    if present>=0:mask&=bits==present
                    if not mask.any():continue
                    g=shift[mask]
                    groups.append(dict(state=tag,relation=relation,background_edge=present,support=int(mask.sum()),
                        base=float(pi[relation]),mean_shift=float(g.mean()),mean_abs_shift=float(abs(g).mean()),
                        min_shift=float(g.min()),max_shift=float(g.max()),std_shift=float(g.std()),
                        conditional_min=float(h[mask].min()),conditional_max=float(h[mask].max())))
            var_full=(1-K/M)*np.var(full,ddof=1)/K;var_res=(1-K/M)*np.var(res,ddof=1)/K
            total_grad=p-h;res_grad=base-h
            sampling.append(dict(state=tag,slots=M,selected=K,inclusion_probability=K/M,
                full_CE=float(full.mean()),base_CE=float(cheap.mean()),correction_mean=float(res.mean()),
                direct_CE_estimator_SD=float(np.sqrt(var_full)),base_plus_residual_CE_estimator_SD=float(np.sqrt(var_res)),
                variance_ratio=float(var_res/var_full),
                residual_estimator_SD=float(np.sqrt(var_res)),residual_absmean=float(np.mean(abs(res))),
                direct_logit_vector_relative_RMS=float(np.sqrt(M/K-1)),
                residual_logit_vector_RMS_over_total=float(np.sqrt(M/K-1)*np.linalg.norm(res_grad)/np.linalg.norm(total_grad)),
                residual_logit_vector_relative_RMS=float(np.sqrt(M/K-1)),
                caveat='Analytic coordinate sampling error at fixed saved background and fixed logits only; not parameter gradient, teacher bias/mixing or generation effect'))
            arrays[tag+'_conditional']=h;arrays[tag+'_base']=base;arrays[tag+'_d01']=d01;arrays[tag+'_p']=p;arrays[tag+'_family']=f
    save('backgrounds',summaries);save('relations',groups);save('sampling_variance',sampling)
    np.savez_compressed(OUT/'full_slot_arrays.npz',**arrays)
    s=budget['incremental_seconds_per_flip']
    save('cost_scenario',dict(measured_seconds_per_scoring_query=s,selected_slots=K,
        one_background_selected_scoring_seconds=s*K,hundred_backgrounds_selected_scoring_seconds=s*K*100,
        updates3000_hundred_backgrounds_selected_scoring_CPU_hours=s*K*100*3000/3600,
        previous_full_slot_hundred_background_seconds=s*18336*100,
        caveat='Arithmetic only, balanced old scoring timing, excludes teacher chain, initialization/commits, neural full-query forward/backward and I/O. No performance claim for 128 slots.'))
    for p,h in hashes.items():assert sha(p)==h
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets=hashes))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        archived_slots_analyzed=4*18336,new_structural_score_calls=0,new_random_samples=0,
        model_calls=0,training_updates=0,GPU_hours=0,new_graphs=0,formal_training_authorized=False,old_assets_unchanged=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
