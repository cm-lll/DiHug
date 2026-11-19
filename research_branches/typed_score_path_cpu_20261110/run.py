"""Fixed probability-path audit, CPU only. Not a parameter line search."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,csv,signal,resource,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/typed_score_cpu_20261110'))
from audit import Descriptor,sha,dump,table,graph
SOURCE=ROOT/'results/hard_clean_zero_safe_20261110'
PREVIOUS=ROOT/'results/typed_score_cpu_20261110_attempt2'
OUT=ROOT/'results/typed_score_path_cpu_20261110'
ALPHAS=np.array([0.,.25,.5,.75,1.])
NAMES=['degree_at_joint','closed','eligible_open','unclosable','component','joint']

def evaluate(d,bits):
    ft=d.features(bits); rows=[]
    for typ in d.type_ids:
        ix=d.types==typ; mask=d.masks[int(typ)]; scale=d.scales[int(typ)]
        x=ft['local'][ix][:,mask]/scale; y=d.ref['local'][ix][:,mask]/scale
        deg=cdist(ft['degree'][ix],d.ref['degree'][ix],'cityblock')/max(1,d.n-1)
        loc=cdist(x,y,'cityblock')/max(1,mask.sum());i,j=linear_sum_assignment(deg+loc)
        dif=abs(x[i]-y[j]); modes=np.flatnonzero(mask)//d.r**2
        vals=[float(deg[i,j].mean())]+[float(dif[:,modes==m].sum()/max(1,mask.sum())/len(i)) for m in range(3)]
        cm=d.component_masks[int(typ)]
        c=cdist(ft['component'][ix][:,cm],d.ref['component'][ix][:,cm],'cityblock')/max(1,cm.sum())
        a,b=linear_sum_assignment(c);vals+=[float(c[a,b].mean()),sum(vals)]
        rows.append(vals)
    return np.mean(rows,axis=0), np.array([ft[k] for k in ('edges','triangles','components','LCC','isolates')])

def covariance_estimate(costs,score):
    # Exactly the LOO score-function directional estimator; no clipping or std normalization.
    n=len(score);c=costs-costs[0];z=score-score[0]
    return (np.sum(c*z[:,None],axis=0)-c.sum(0)*z.sum()/n)/(n-1)

def derivative(costs,bits,p,dp):
    singular=((p==0)|(p==1))&(dp!=0)
    if singular.any():return None,None,int(singular.sum())
    active=(p>0)&(p<1)
    score=((bits[:,active]-p[active])*(dp[active]/(p[active]*(1-p[active])))).sum(1)
    if not np.isfinite(score).all():raise FloatingPointError('path score nonfinite')
    value=covariance_estimate(costs,score)
    jack=np.array([covariance_estimate(np.delete(costs,i,axis=0),np.delete(score,i)) for i in range(len(score))])
    se=np.sqrt((len(score)-1)/len(score)*((jack-jack.mean(0))**2).sum(0))
    return value,se,0

def tests():
    typ,q,f,ref=graph(4,[(0,1),(1,2),(2,3)])
    d=Descriptor(typ,q,f,ref);bits=np.array(list(itertools.product([0,1],repeat=6)),dtype=float)
    costs=np.array([evaluate(d,b)[0] for b in bits]);p=np.array([.18,.27,.41,.53,.68,.81]);dp=np.array([.2,-.1,.08,-.2,.1,-.1])
    def risk(prob):return np.prod(np.where(bits,prob,1-prob),axis=1)@costs
    score=((bits-p)*dp/(p*(1-p))).sum(1);mass=np.prod(np.where(bits,p,1-p),axis=1)
    exact=(mass*score)@costs;finite=(risk(p+1e-6*dp)-risk(p-1e-6*dp))/2e-6
    error=float(abs(exact-finite).max());assert error<1e-8
    # Exact independent K=2 LOO expectation over all pairs.
    pair=(costs[:,None,:]-costs[None,:,:])*(score[:,None,None]-score[None,:,None])/2
    loo=(mass[:,None,None]*mass[None,:,None]*pair).sum((0,1))
    assert abs(loo-exact).max()<1e-12
    a,b,s=derivative(costs,bits,np.zeros(6),dp);assert a is None and s==6
    a,b,s=derivative(costs,bits,p,np.zeros(6));assert np.max(abs(a))==np.max(abs(b))==s==0
    return dict(passed=True,graphs=64,independent_pair_expectation_terms=4096,max_finite_difference_error=error,
                zero_direction_and_support_boundary_checked=True)

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    def stop(*_):raise RuntimeError('CPU wall/memory supervisor stop')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(240)
    receipt=dict(start_epoch=start,model_calls=0,GPU_hours=0,training_updates=0,reverse_chains=0,new_CPU_hard_draws=0)
    summaries=[];intervals=[];pairs=[];done=0
    try:
        protocol=json.loads((SOURCE/'protocol.json').read_text())
        bindings=json.loads((PREVIOUS/'protocol.json').read_text())['bindings'].copy()
        for fn in ('graphs.csv','joint_attribution_types.csv','REPORT_ZH.md'):
            p=PREVIOUS/fn;bindings[str(p)]=sha(p)
        bindings[str(Path(__file__))]=sha(__file__)
        for p,h in bindings.items():assert sha(p)==h
        dump(OUT/'protocol.json',dict(bindings=bindings,alphas=ALPHAS.tolist(),times=protocol['times'],models=4,
            conditions=24,K=32,new_CPU_hard_draws=2304,reused_endpoint_graphs=1536,CPU_threads=1,memory_GiB=2,wall_seconds=240,
            uniform_rule='same verified original default_rng(training_seed+810001+100+time_index), shared across all alpha',
            probability_path='p(alpha)=(1-alpha)*p_before+alpha*p_after; NOT actual parameter-scaled trajectory',
            score_names=NAMES,score_definitions='unchanged typed_score_cpu_20261110; closed/open/unclosable contributions under SAME joint assignment',
            feedback='LOO score-function derivative of expected per-graph cost w.r.t. alpha; jackknife SE, no clipping; nonregular support endpoint = unavailable',
            decision='descriptive paired finite risk changes versus local estimates, no alpha selection or optimization',
            observed_previous_inputs=True,no_independent_confirmation_claim=True))
        dump(OUT/'tests.json',tests())
        oldgraphs={};oldparts={}
        for row in csv.DictReader((PREVIOUS/'graphs.csv').open()):
            if row['state']!='failure':oldgraphs[(row['task'],row['state'],int(row['t']),int(row['sample']))]=row
        for row in csv.DictReader((PREVIOUS/'joint_attribution_types.csv').open()):
            key=(row['task'],row['state'],int(row['t']),int(row['sample']));oldparts.setdefault(key,[]).append(row)
        import torch
        torch.set_num_threads(1)
        for task in protocol['tasks']:
            inputs=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs']
            inp=inputs[0];order=np.argsort(inp['ids']);ids=inp['ids'][order]
            d=Descriptor(np.load(task['data'])['types'],inp['query'][:,order],inp['f'][order],inp['labels'][order])
            for j,t in enumerate(protocol['times']):
                before=np.load(SOURCE/task['task']/f'before_t{t}.npz');after=np.load(SOURCE/task['task']/f'after_t{t}.npz')
                np.testing.assert_array_equal(before['ids'],ids);np.testing.assert_array_equal(after['ids'],ids)
                p0=before['probability'];p1=after['probability'];dp=p1-p0
                uniforms=np.random.default_rng(task['seed']+810001+100+j).random(before['bits'].shape)
                assert np.array_equal(uniforms<p0,before['bits']) and np.array_equal(uniforms<p1,after['bits'])
                allcost=[];allstructure=[];allg=[];allse=[];meta=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],t=t)
                for ai,alpha in enumerate(ALPHAS):
                    p=(1-alpha)*p0+alpha*p1;bits=(uniforms<p).astype(np.uint8)
                    if ai in (0,4):
                        state='before' if ai==0 else 'after';cc=[];ss=[]
                        for k in range(32):
                            key=(task['task'],state,t,k);g=oldgraphs[key];pt=oldparts[key]
                            vals=[float(np.mean([float(x[name]) for x in pt])) for name in ('J_at_joint','closed','eligible_open','unclosable')]
                            cc.append(vals+[float(g['C']),float(g['JL'])]);ss.append([float(g[name]) for name in ('edges','triangles','components','LCC','isolates')])
                        c,s=np.array(cc),np.array(ss)
                        # One graph at both endpoints independently checks reused score identity.
                        check_c,check_s=evaluate(d,bits[0]);np.testing.assert_allclose(c[0],check_c,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(s[0],check_s,atol=1e-12)
                    else:
                        values=[evaluate(d,b) for b in bits];c=np.stack([v[0] for v in values]);s=np.stack([v[1] for v in values]);receipt['new_CPU_hard_draws']+=32
                    assert np.isfinite(c).all();grad,se,singular=derivative(c,bits,p,dp)
                    allcost.append(c);allstructure.append(s);allg.append(grad);allse.append(se)
                    base=dict(**meta,alpha=float(alpha),samples=32,singular_boundary_slots=singular,min_probability=float(p.min()),max_probability=float(p.max()),max_probability_shift=float(abs(dp).max()))
                    for k,name in enumerate(NAMES):
                        summaries.append(dict(**base,metric=name,mean=float(c[:,k].mean()),sample_std=float(c[:,k].std(ddof=1)),
                            delta_from_before=float((c[:,k]-allcost[0][:,k]).mean()),paired_delta_SE=float((c[:,k]-allcost[0][:,k]).std(ddof=1)/np.sqrt(32)),
                            path_derivative=None if grad is None else float(grad[k]),derivative_jackknife_SE=None if se is None else float(se[k]),unique_costs=int(len(np.unique(c[:,k])))))
                for ai in range(4):
                    diff=allcost[ai+1]-allcost[ai]
                    for k,name in enumerate(NAMES):
                        predicted=None if allg[ai] is None else .25*float(allg[ai][k])
                        intervals.append(dict(**meta,alpha_start=float(ALPHAS[ai]),alpha_end=float(ALPHAS[ai+1]),metric=name,
                            paired_cost_delta=float(diff[:,k].mean()),paired_delta_SE=float(diff[:,k].std(ddof=1)/np.sqrt(32)),
                            first_order_prediction=predicted,prediction_jackknife_SE=None if allse[ai] is None else .25*float(allse[ai][k]),
                            sign_disagreement=None if predicted is None else bool(predicted*diff[:,k].mean()<0)))
                np.savez_compressed(OUT/f'{task["task"]}_t{t}.npz',alphas=ALPHAS,costs=np.array(allcost),structure=np.array(allstructure),
                    p_before=p0,p_after=p1,ids=ids,uniform_seed=task['seed']+810001+100+j,
                    graph_bit_sha256=np.array([__import__('hashlib').sha256((uniforms<((1-a)*p0+a*p1)).astype(np.uint8).tobytes()).hexdigest() for a in ALPHAS]))
                done+=1;table(OUT/'risk_path.csv',summaries);table(OUT/'intervals.csv',intervals)
                dump(OUT/'progress.json',dict(conditions=done,seconds=time.time()-start,new_CPU_hard_draws=receipt['new_CPU_hard_draws']))
            print(json.dumps(dict(task=task['task'],conditions=done,seconds=time.time()-start)),flush=True)
        for p,h in bindings.items():assert sha(p)==h
        receipt.update(status='complete',unchanged_bindings=len(bindings),endpoint_random_reproductions=48)
    except BaseException as exc:
        import traceback
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        table(OUT/'risk_path.csv',summaries);table(OUT/'intervals.csv',intervals)
        receipt.update(conditions=done,seconds=time.time()-start,end_epoch=time.time(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump(OUT/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
