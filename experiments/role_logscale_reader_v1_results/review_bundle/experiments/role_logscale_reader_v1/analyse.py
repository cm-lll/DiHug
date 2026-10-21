"""Independent paired-array checks and descriptive summaries; no model call."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from diagnose import c,OUT,CFG,check,ORIGINAL
from scaling import scale

def summary(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)),max=float(x.max())) if len(x) else None

def main():
    tick=time.perf_counter();rows=[];failure=[];alloc=[];audited=0
    conditions=c.read(c.RESULT/'conditions.json');records,targets=c.load_data();case=c.metadata_case(c.base,c.PANEL,targets)
    kernel=c.base.MarginalKernel(case['prior'],c.base.c.schedule(64));fam=np.repeat(np.arange(10),case['space'].counts)
    for seed in (2026102011,2026102012,2026102013):
        for name in conditions['final_targets']:
            for t in CFG['diagnostic_times']:
                check();p=OUT/f'seed{seed}/{name}_t{t}.json'
                if not p.exists():failure.append(dict(seed=seed,name=name,t=t,status='missing'));continue
                r=c.read(p)
                if r['status']!='completed':failure.append(r);continue
                with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
                    assert c.base.sha(z['state'].tolist())==r['state_sha256']
                    for label in ('exact','scaled'):
                        q=kernel.reverse(z['p0_'+label],z['state'],fam,t-1,t)
                        np.testing.assert_array_equal(q,z['q_'+label])
                    if t==1:
                        np.testing.assert_allclose(z['q_exact'],z['p0_exact'],atol=2e-12,rtol=0)
                        np.testing.assert_allclose(z['q_scaled'],z['p0_scaled'],atol=2e-12,rtol=0)
                    for a in r['allocation']:
                        typ=a['type'];x=z[f'type{typ}_exact'];y=z[f'type{typ}_scaled'];ll=z[f'type{typ}_LL'];counts=z[f'type{typ}_counts']
                        np.testing.assert_allclose(y.sum(1),1,rtol=0,atol=1e-8)
                        assert np.max(abs(y.sum(0)-counts)/np.maximum(1,counts))<=1e-8
                        assert np.all(y[np.isneginf(ll)]==0)
                        assert abs(a['row_TV_mean']-abs(x-y).sum(1).mean()/2)<1e-14
                        alloc.append(dict(seed=seed,condition=name,t=t,**a))
                    for a in r['relations']:
                        ix=fam==a['family'];qe=z['q_exact'][ix];qs=z['q_scaled'][ix];st=z['state'][ix]
                        for label,q in [('exact',qe),('scaled',qs)]:
                            assert abs(a['expected_'+label]-q.sum())<1e-12
                            assert abs(a['net_'+label]-(q[st==0].sum()-(1-q[st==1]).sum()))<1e-12
                        rows.append(dict(seed=seed,condition=name,t=t,family=a['family'],p0_max_abs=a['p0_max_abs'],p0_mean_abs=a['p0_mean_abs'],
                            q_max_abs=a['q_max_abs'],expected_edge_delta=a['expected_scaled']-a['expected_exact'],net_delta=a['net_scaled']-a['net_exact'],
                            birth_delta=a['births_scaled']-a['births_exact'],death_delta=a['deaths_scaled']-a['deaths_exact']))
                audited+=1
    groups={}
    for key in ('seed','condition','t','family'):
        groups[key]={}
        for v in sorted(set(x[key] for x in rows)):
            rr=[x for x in rows if x[key]==v]
            groups[key][str(v)]={metric:summary([x[metric] for x in rr if x[metric] is not None]) for metric in ('p0_max_abs','p0_mean_abs','q_max_abs','expected_edge_delta')}
    bytime={str(t):dict(completed=sum(1 for r in rows if r['t']==t and r['family']==8),
               MG_expected_shift=summary([r['expected_edge_delta'] for r in rows if r['t']==t and r['family']==8]),
               MG_p0_max=summary([r['p0_max_abs'] for r in rows if r['t']==t and r['family']==8]),
               max_allocation_TV=max([x['row_TV_max'] for x in alloc if x['t']==t],default=None)) for t in CFG['diagnostic_times']}
    result=dict(expected_states=132,completed_states=audited,failed_states=failure,by_time=bytime,groups=groups,
        allocation_TV=summary([x['row_TV_mean'] for x in alloc]),allocation_max=summary([x['max_abs'] for x in alloc]),
        soft_degree_max=summary([x['soft_target_degree_max_abs'] for x in alloc]),context_max=summary([x['role_context_max_abs'] for x in alloc]),
        p0_max=summary([r['p0_max_abs'] for r in rows]),q_max=summary([r['q_max_abs'] for r in rows]),
        seconds=time.perf_counter()-tick,independent_audit=True,new_independent_graphs=0)
    c.save(OUT/'summary.json',result);c.save(OUT/'all_family_differences.json',rows);c.save(OUT/'all_allocation_differences.json',alloc)
    print(json.dumps({k:result[k] for k in ('completed_states','by_time','allocation_TV','p0_max','q_max')},indent=2))
if __name__=='__main__':main()
