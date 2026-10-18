"""Bounded SYNTHETIC engineering cost and reference-only operation counts.

No frozen scientific state, model forward, graph draw or quality comparison.
"""
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np
from role_degree.role_statistics import joint_profile
from stepwise_role.core import Layout
from multiscale_role.core import MultiScaleGuide,freeze
from multiscale_role.evaluation import degree_w1,ForwardPMFCache
from multiscale_role.runtime_guard import Budget,atomic_json,checked_sha


def operation_counts(profile,pairs,prior,sizes):
    rows=[];prediction=0;target=0;workspace=0
    for c,g in profile['groups'].items():
        ty=int(c);incident=[];caps={}
        for r,(a,b) in enumerate(pairs):
            if ty not in (a,b) or prior[r]==0:continue
            cap=sizes[int(b if ty==a else a)]-(a==b)
            if cap>0:incident.append(r);caps[r]=cap
        channels=[[r] for r in incident]+([incident] if len(incident)>1 else [])
        for members in channels:
            m=sum(caps[r] for r in members);n=g['nodes'];roles=len(g['rows'])
            # count pmf[1:k+2] updates; p0 updates reported separately
            triangle=m*(m+1)//2
            prediction+=n*triangle;target+=roles*triangle
            workspace=max(workspace,8*n*(m+1))
            rows.append(dict(type=c,families=members,capacity=m,nodes=n,role_rows=roles,
                             predicted_coefficient_updates=n*triangle,target_coefficient_updates=roles*triangle))
    return dict(channels=rows,one_evaluation_predicted_updates=prediction,one_evaluation_target_updates=target,
                six_states_before_after_uncached_updates=12*(prediction+target),
                with_shared_target_cache_updates=12*prediction+3*target,
                largest_single_PMF_array_bytes=workspace,
                excludes_p0_updates=True,meaning='static arithmetic count, NOT measured seconds or proof of timeout')


def run_cost(out,deadline,cleanup_deadline):
    out=Path(out);b=Budget(deadline,out/'STOP',cleanup_deadline);b.install();check=b.check
    start,cpu=time.monotonic(),time.process_time();records=[]
    result=dict(status='incomplete',synthetic_engineering_only=True,model_calls=0,GPU_use=0,scientific_graphs=0,
                input_hash_verification_complete=False,passes=records)
    try:
        check();tick=time.monotonic()
        from hsgd.bw_preflight import common as c
        from hsgd.kernel import MarginalKernel
        result['cold_import_seconds']=time.monotonic()-tick
        reference=c.PACKAGE/'data/IMDB_movie_reference.npz';refsha=checked_sha(reference,check)
        real=c.imdb_case(reference,c.config());space=real['space']
        pf=joint_profile(space.types,real['clean'],space.encode(real['clean'])[0],space.pairs)
        result['reference_static_work']=operation_counts(pf,space.pairs,real['prior'],{i:int((space.types==i).sum()) for i in np.unique(space.types)})
        # Twelve nodes, deterministic toy edges, not an IMDB generation sample.
        types=np.repeat(np.arange(4),3);pairs=tuple(itertools.combinations_with_replacement(range(4),2))
        first=Layout.canonical(types,pairs,np.full(10,.5))
        selected=np.array([np.flatnonzero(first.families==r)[0] for r in range(10)])
        case=c.make_case(types,first.slots[:,selected],pairs);l=Layout.canonical(types,pairs,case['prior'])
        profile=joint_profile(types,case['clean'],case['space'].encode(case['clean'])[0],pairs)
        tick=time.monotonic();guide=MultiScaleGuide(l,freeze(profile,l,c.schedule(1000)),check)
        result['fixture_and_objective_prepare_seconds']=time.monotonic()-tick
        result['fixture']=dict(nodes=12,slots=len(l.families),reference_edges=10,not_real_IMDB_state=True)
        state=np.zeros(len(l.families),dtype=np.uint8);state[case['clean_ids']]=1
        p0=np.linspace(.13,.77,len(l.families));kernel=MarginalKernel(l.prior,guide.alpha);cache=ForwardPMFCache()
        expected=None
        for name in ('cold','warm_reuse'):
            check();phase_start=time.monotonic();timings={};tick=time.monotonic();guide.target(19)
            timings['next_time_target_seconds']=time.monotonic()-tick
            class Metered:
                prior=kernel.prior;alpha=kernel.alpha
                seconds=0.;calls=0
                def reverse(self,*args):
                    tick=time.monotonic();value=kernel.reverse(*args);self.seconds+=time.monotonic()-tick;self.calls+=1;return value
            meter=Metered();tick=time.monotonic();q=kernel.reverse(p0,state,l.families,19,20)
            timings['baseline_posterior_seconds']=time.monotonic()-tick
            pg,qg,report=guide.apply(p0,state,meter,20,check)
            timings.update(apply_total_seconds=report['wall_seconds'],all_apply_posterior_seconds=meter.seconds,
                           statistics_gradient_controller_seconds=report['statistics_gradient_seconds'],
                           backtracking_total_seconds=report['backtracking_seconds'])
            tick=time.monotonic();before=degree_w1(q,guide,19,check,cache);after=degree_w1(qg,guide,19,check,cache)
            timings['exact_all_channel_W1_evaluation_seconds']=time.monotonic()-tick
            check();fingerprint=(pg.tobytes(),qg.tobytes(),json.dumps([before,after],sort_keys=True))
            if expected is None:expected=fingerprint
            elif fingerprint!=expected:raise AssertionError('cold/warm exact result changed')
            row=dict(pass_name=name,timings=timings,posterior_calls=meter.calls,target_cache_hits=cache.hits,target_cache_misses=cache.misses,
                     total_pass_seconds=time.monotonic()-phase_start,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     memory_scope='cumulative process RSS high-water; NOT per-phase allocation or device memory')
            tick=time.monotonic();atomic_json(out/(name+'.json'),row,check);row['IO_seconds']=time.monotonic()-tick;records.append(row)
        if checked_sha(reference,check)!=refsha:raise ValueError('reference changed')
        result.update(status='completed_synthetic_engineering_cost',input_hash_verification_complete=True,reference_sha256=refsha,
                      cold_warm_results_exact=True,independent_Genre64_cost='not timed: no real saved-report input',
                      actual_six_state_cost='NOT measured; do not extrapolate toy timing')
    except BaseException as exc:
        result.update(status='incomplete' if isinstance(exc,(InterruptedError,TimeoutError,KeyboardInterrupt)) else 'failed',reason=type(exc).__name__+': '+str(exc))
    finally:
        result.update(worker_wall_seconds=time.monotonic()-start,worker_CPU_seconds=time.process_time()-cpu,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        try:atomic_json(out/'summary.json',result,b.cleanup_check)
        finally:b.restore()
    return 0 if result['status']=='completed_synthetic_engineering_cost' else 2
