"""Bounded CPU-only stored-asset audit; no new real graph draws/model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,itertools,hashlib,csv,resource,signal
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from feedback import topology,differences,estimate

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/hard_clean_zero_safe_20261110'
OUT=ROOT/'results/component_feedback_cpu_20261110'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def count(n,q,b):
    a=np.zeros((n,n),dtype=bool);i,j=q[:,b.astype(bool)];a[i,j]=True;a[j,i]=True
    return int(connected_components(csr_matrix(a),directed=False,return_labels=False))


def tests():
    n=4;q=np.array(list(itertools.combinations(range(n),2))).T
    all_bits=np.array(list(itertools.product([0,1],repeat=q.shape[1])),dtype=np.uint8)
    p=np.array([.13,.27,.41,.59,.73,.87]);mass=np.prod(np.where(all_bits,p,1-p),axis=1)
    checks=0;gradient_error=0.;variance=[]
    for target in (1,3):
        risks=[];deltas=[]
        for b in all_bits:
            d,r=differences(n,q,b,target);assert r['components']==count(n,q,b)
            risk=lambda z:(count(n,q,z)-target)**2/(n-1)**2
            for e in range(len(p)):
                b0=b.copy();b0[e]=0;b1=b.copy();b1[e]=1
                assert abs(d[e]-(risk(b1)-risk(b0)))<1e-14;checks+=1
            risks.append(risk(b));deltas.append(d)
        risks=np.array(risks);deltas=np.array(deltas)
        sf=risks[:,None]*(all_bits-p);rb=deltas*p*(1-p)
        exact=mass@sf;err=float(np.max(abs(mass@rb-exact)));gradient_error=max(gradient_error,err)
        assert err<1e-14
        vs=mass@(sf-exact)**2;vr=mass@(rb-exact)**2
        assert np.all(vr<=vs+1e-14)
        variance.append(dict(target=target,unbaselined_score_function_variance=vs.tolist(),conditional_variance=vr.tolist()))
    # Dense graph really has zero one-edge connectivity feedback; do not force it.
    d,_=differences(n,q,np.ones(len(p),dtype=np.uint8),3);assert np.count_nonzero(d)==0
    return dict(passed=True,hard_graphs=64,edge_toggle_checks=checks,exact_gradient_error=gradient_error,
        variance=variance,variance_claim='per coordinate versus unbaselined single-sample score function; not a universal comparison with LOO or parameter-space variance')


def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    def alarm(*_):raise TimeoutError('180 second CPU execution bound')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    receipt=dict(start_epoch=start,GPU_hours=0,model_calls=0,updates=0,new_real_graph_draws=0)
    try:
        original=json.loads((SOURCE/'protocol.json').read_text());bindings={};tasks=original['tasks']
        paths=[]
        for row in tasks:
            folder=SOURCE/row['task'];paths+=list(folder.glob('*.npz'))
            paths += [Path(row['evaluation_file']),folder/'result.json',
                      ROOT/'results/hard_clean_pilot_v2_20261110'/row['task']/'score_definition.json']
        paths+=[SOURCE/'protocol.json',Path(__file__),Path(__file__).with_name('feedback.py')]
        for p in paths:bindings[str(p)]=sha(p)
        save(OUT/'protocol.json',dict(track='CORE_SUPPORT',source=str(SOURCE),bindings=bindings,
            conditions=4,failed_batches=4,evaluation_batches=48,total_saved_hard_graphs=1600,
            time_points=original['times'],CPU_threads=1,CPU_wall_seconds_limit=180,memory_limit_GiB=2,GPU=0,
            target='unchanged (component_count-reference_component_count)^2/(N-1)^2',
            analysis='conditional per-edge finite difference via bridges/components; paired before/after probability-coordinate linear prediction',
            no_threshold_fit=True,no_model=True,no_new_real_sampling=True))
        save(OUT/'cpu_tests.json',tests())
        # Load CPU tensor containers only. No model/runtime module is imported.
        import torch
        torch.set_num_threads(1)
        rows=[];relations=[];pairs=[];gradient_dir=OUT/'gradients';gradient_dir.mkdir()
        compute_tick=time.perf_counter()
        for row in tasks:
            fixed=torch.load(row['evaluation_file'],map_location='cpu')['inputs']
            inp=fixed[0];order=np.argsort(inp['ids']);q=inp['query'][:,order];families=inp['f'][order]
            labels=inp['labels'][order];ids=inp['ids'][order]
            definition=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110'/row['task']/'score_definition.json').read_text())
            target=definition['reference']['components'];n=len(definition['reference']['degree'])
            folder=SOURCE/row['task'];rinfo=json.loads((folder/'result.json').read_text())
            pending=[('failure',rinfo['t'],folder/'failed_batch_replay.npz')]
            pending += [(label,t,folder/f'{label}_t{t}.npz') for label in ('before','after') for t in original['times']]
            cache={}
            for label,t,path in pending:
                saved=np.load(path);p=saved['probability'];bits=saved['bits'];assert np.array_equal(saved['ids'],ids)
                tick=time.perf_counter();v=estimate(n,q,p,bits,target)
                np.testing.assert_allclose(v['costs'],saved['costs'][:,4],atol=1e-15,rtol=1e-13)
                norm=lambda a:float(np.linalg.norm(a))
                gr=v['logit_gradient'];lo=v['loo_gradient'];dd=v['diffs'];cost=v['costs']
                result=dict(task=row['task'],dataset=row['dataset'],seed=row['seed'],readout='raw',state=label,t=t,
                    samples=len(bits),n_nodes=n,reference_components=target,mean_components=float(np.mean([x['components'] for x in v['records']])),
                    cost_mean=float(cost.mean()),cost_std=float(cost.std(ddof=1)),
                    bridges_per_graph=float(np.mean([x['bridges'] for x in v['records']])),
                    cross_component_slots_per_graph=float(np.mean([x['cross_slots'] for x in v['records']])),
                    nonzero_difference_occurrences=int(np.count_nonzero(dd)),
                    conditional_logit_gradient_norm=norm(gr),LOO_logit_gradient_norm=norm(lo),
                    conditional_nonzero_slots=int(np.count_nonzero(gr)),LOO_nonzero_slots=int(np.count_nonzero(lo)),
                    encourage_add_slots=int(np.sum(gr<0)),encourage_remove_slots=int(np.sum(gr>0)),
                    conditional_gradient_standard_error_norm=norm(v['logit_standard_error']),seconds=time.perf_counter()-tick)
                rows.append(result)
                for f in np.unique(families):
                    for positive in (0,1):
                        ix=(families==f)&(labels==positive)
                        relations.append(dict(task=row['task'],state=label,t=t,family=int(f),reference_positive=int(positive),slots=int(ix.sum()),
                            conditional_gradient_norm=norm(gr[ix]),LOO_gradient_norm=norm(lo[ix]),
                            encourage_add=int(np.sum(gr[ix]<0)),encourage_remove=int(np.sum(gr[ix]>0)),
                            reference_used_for_group_reporting_only=True))
                np.savez_compressed(gradient_dir/f'{row["task"]}_{label}_{t}.npz',prob_gradient=v['prob_gradient'],
                    logit_gradient=gr,loo_gradient=lo,logit_standard_error=v['logit_standard_error'],
                    ids=ids,components=np.array([x['components'] for x in v['records']]),
                    bridges=np.array([x['bridges'] for x in v['records']]))
                cache[(label,t)]=(v,p,result)
            for t in original['times']:
                vb,pb,rb=cache[('before',t)];va,pa,ra=cache[('after',t)]
                shift=pa-pb
                samples=vb['diffs']@shift
                risk_delta=va['costs']-vb['costs']
                first=float(samples.mean())
                pairs.append(dict(task=row['task'],dataset=row['dataset'],seed=row['seed'],t=t,
                    before_cost=rb['cost_mean'],after_cost=ra['cost_mean'],paired_cost_delta=float(risk_delta.mean()),
                    paired_delta_standard_error=float(risk_delta.std(ddof=1)/np.sqrt(len(risk_delta))),
                    probability_linear_prediction=first,
                    linear_prediction_standard_error=float(samples.std(ddof=1)/np.sqrt(len(samples))),
                    after_gradient_dot_same_probability_shift=float(va['prob_gradient']@shift),
                    probability_L1_shift=float(abs(shift).sum()),max_probability_shift=float(abs(shift).max()),
                    warning='first order in probability coordinates, not parameter gradient or causal attribution; MC estimates share saved paired uniforms'))
            save(OUT/'progress.json',dict(tasks_done=len(rows)//13,seconds=time.time()-start))
        for name,table in [('states',rows),('relations',relations),('paired_probability_response',pairs)]:
            with (OUT/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        for p,h in bindings.items():assert sha(p)==h
        summary=dict(failures=[r for r in rows if r['state']=='failure'],states=rows,pairs=pairs,
            theory_tests_passed=True,bound_files_unchanged=len(bindings),feature_compute_seconds=time.perf_counter()-compute_tick,
            limitations=['no parameter Jacobian/model call','no new training or reverse generation',
                'conditional estimator remains zero if all sampled background graphs have no pivotal edge',
                'no variance guarantee relative to LOO after sharing model parameters',
                'finite probability displacement is not controlled by first-order estimates'])
        save(OUT/'summary.json',summary)
        receipt.update(status='complete',state_batches=len(rows),saved_graphs=sum(r['samples'] for r in rows),
            feature_compute_seconds=summary['feature_compute_seconds'])
        print(json.dumps(summary['failures'],ensure_ascii=False,indent=2))
    except BaseException as exc:
        import traceback
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save(OUT/'receipt.json',receipt)
        print(json.dumps(receipt))

if __name__=='__main__':main()
