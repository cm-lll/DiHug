"""CPU-only count-conditioned diagnostic on saved probabilities; no model import."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,time,sys,signal,resource,hashlib,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_pilot_v2_20261110'))
from objective import HardScore
from sampler import prepare,draw,tests
OUT=ROOT/'results/fixed_count_diagnostic_20261110'
NEW=ROOT/'results/exact_count_step_20261110_attempt2'
OLD=ROOT/'results/hard_clean_zero_safe_20261110'
K=32
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate(core,b):
    costs,v=core.costs(b);degree=v['degree'];ref=core.ref
    totals=degree.sum(1);rd=ref['degree'].sum(1)
    row=dict(edges=int(b.sum()),triangles=int(v['triangles'].sum()),triangle_abs_error=float(abs(v['triangles'].sum()-ref['triangles'].sum())),
        components=v['components'],LCC=v['lcc']/core.n,isolates=v['isolated'],hub=v['max_degree'],
        joint_degree=float(costs[1]),typed_triangle_present=float(costs[2]),typed_triangle_absent=float(costs[3]),component_cost=float(costs[4]),
        low_degree_fraction=float(np.mean(totals<=1)),degree_W1=float(np.mean([np.mean(abs(np.sort(totals[ix])-np.sort(rd[ix]))) for ix in core.nodes])))
    detail=[]
    for typ,ix in zip(np.unique(core.types),core.nodes):
        for r in range(core.r):
            if not np.any((core.f==r)&(np.isin(core.q[0],ix)|np.isin(core.q[1],ix))):continue
            x=degree[ix,r];y=ref['degree'][ix,r]
            detail.append(dict(node_type=int(typ),relation=r,nodes=len(ix),reference_edges=int(ref['counts'][r]),edges=int(v['counts'][r]),
                relation_degree_W1=float(np.mean(abs(np.sort(x)-np.sort(y)))),mean_degree=float(x.mean()),
                zero_fraction=float(np.mean(x==0)),low_le1_fraction=float(np.mean(x<=1)),hub=int(x.max()),
                reference_hub=int(y.max()),reference_low_le1_fraction=float(np.mean(y<=1))))
    tr=[dict(types='-'.join(map(str,key)),triangles=int(x),reference=int(y),absolute_error=int(abs(x-y))) for key,x,y in zip(core.triples,v['triangles'],ref['triangles'])]
    return row,detail,tr

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    def stop(*_):raise TimeoutError('CPU wall or memory bound')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(540)
    graphs=[];relations=[];triangles=[];batches=[];sampling=[];failures=[]
    receipt=dict(start_epoch=start,GPU_hours=0,model_calls=0,updates=0,reverse_chains=0,new_CPU_hard_draws=0)
    bindings={};tasks=[]
    try:
        parent=json.loads((NEW/'protocol.json').read_text());tasks=parent['tasks'];times=parent['times']
        paths=[Path(__file__),Path(__file__).with_name('sampler.py'),ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py',NEW/'protocol.json']
        for task in tasks:
            paths +=[Path(task['data']),Path(task['evaluation_file']),ROOT/'results/hard_clean_pilot_v2_20261110'/task['task']/'score_definition.json']
            for label,folder,stage in [('before',NEW/task['task'],'before'),('old_update',OLD/task['task'],'after'),('exact_update',NEW/task['task'],'after')]:
                paths += [folder/f'{stage}_t{t}.npz' for t in times]
        bindings={str(p):sha(p) for p in paths}
        dump('protocol.json',dict(bindings=bindings,tasks=tasks,times=times,K=K,CPU_threads=1,wall_seconds=600,memory_GiB=2,GPU=0,
            panel='4 prior models x 6 fixed times x 3 saved predictor states x 32 samples; all retained',
            states=['before','old_update','exact_update'],new_draws=2304+64,reused_raw_draws=2304,
            conditional_law='independent Bernoulli conditioned separately on each reference relation edge count; exact log-DP; p=0/1 preserved',
            invariants='schema, node inventory, per-relation counts, simplicity and no self-loops; node degrees/connectivity/triangles NOT fixed',
            coupling='same uniforms within task/time/relation across before/old/exact; raw original samples retained separately, NOT sample-wise paired to conditional draws',
            random_seeds='SeedSequence([2026111018,training_seed,time,relation]) for conditional draws; [2026111019,dataset_index,relation] for uniform baseline',
            uniform_baseline='32 per dataset once; exactly same counts and schema; no claim of learned generation',
            zero_probability_event='record as unavailable; never clip probabilities or replace support',
            no_training=True,no_formal_sampler_change=True,interpretation='conditional clean predictions on previously observed inputs; not reverse-chain results or independent confirmation'))
        dump('tests.json',tests())
        import torch
        torch.set_num_threads(1)
        uniform_done=set()
        for task in tasks:
            inputs=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs'];inp=inputs[0]
            order=np.argsort(inp['ids']);q=inp['query'][:,order];f=inp['f'][order];ids=inp['ids'][order];ref=inp['labels'][order]
            data=np.load(task['data']);definition=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110'/task['task']/'score_definition.json').read_text())
            core=HardScore(data['types'],q,f,ref,len(definition['reference']['counts']))
            np.testing.assert_array_equal(core.ref['counts'],definition['reference']['counts'])
            np.testing.assert_array_equal(core.ref['degree'],definition['reference']['degree'])
            if task['dataset'] not in uniform_done:
                ub=np.zeros((K,len(f)),dtype=np.uint8)
                for r,m in enumerate(core.ref['counts']):
                    ix=np.flatnonzero(f==r);rng=np.random.default_rng(np.random.SeedSequence([2026111019,0 if task['dataset']=='IMDB' else 1,r]))
                    for b in ub:b[rng.choice(ix,size=int(m),replace=False)]=1
                np.savez_compressed(OUT/f'{task["dataset"]}_uniform.npz',bits=ub,ids=ids)
                for k,b in enumerate(ub):
                    value,rr,tr=evaluate(core,b);meta=dict(task='uniform',dataset=task['dataset'],seed=0,t=0,state='uniform',sampling='fixed_count',sample=k)
                    graphs.append(dict(**meta,**value));relations.extend(dict(**meta,**x) for x in rr);triangles.extend(dict(**meta,**x) for x in tr)
                receipt['new_CPU_hard_draws']+=K;uniform_done.add(task['dataset'])
                dump(f'{task["dataset"]}_reference.json',dict(counts=core.ref['counts'].tolist(),triangles=core.ref['triangles'].tolist(),types=data['types'].tolist(),
                    metrics=evaluate(core,ref)[0],typed_triples=core.triples))
            for t in times:
                for label,folder,stage in [('before',NEW/task['task'],'before'),('old_update',OLD/task['task'],'after'),('exact_update',NEW/task['task'],'after')]:
                    tic=time.time();path=folder/f'{stage}_t{t}.npz';saved=np.load(path);np.testing.assert_array_equal(ids,saved['ids']);p=saved['probability']
                    conditioned=np.zeros((K,len(f)),dtype=np.uint8);valid=True
                    for r,m in enumerate(core.ref['counts']):
                        ix=np.flatnonzero(f==r);a=time.time()
                        try:s=prepare(p[ix],m)
                        except ValueError as exc:
                            failures.append(dict(task=task['task'],state=label,t=t,relation=r,error=str(exc)));valid=False;break
                        rng=np.random.default_rng(np.random.SeedSequence([2026111018,task['seed'],t,r]))
                        conditioned[:,ix]=draw(s,rng.random((K,len(ix))))
                        sampling.append(dict(task=task['task'],state=label,t=t,relation=r,slots=len(ix),target=int(m),log_conditioning_probability=s['log_event'],
                            DP_MiB=s['dp'].nbytes/2**20,seconds=time.time()-a,zero_slots=int((p[ix]==0).sum()),one_slots=int((p[ix]==1).sum())))
                        assert np.all(conditioned[:,ix].sum(1)==m)
                        del s
                    if valid:
                        np.savez_compressed(OUT/f'{task["task"]}_{label}_t{t}.npz',bits=conditioned,ids=ids)
                        receipt['new_CPU_hard_draws']+=K
                    for kind,bs in [('raw',saved['bits'])]+([('fixed_count',conditioned)] if valid else []):
                        local=[]
                        for k,b in enumerate(bs):
                            value,rr,tr=evaluate(core,b);meta=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],t=t,state=label,sampling=kind,sample=k)
                            graphs.append(dict(**meta,**value));relations.extend(dict(**meta,**x) for x in rr);triangles.extend(dict(**meta,**x) for x in tr);local.append(value)
                        br=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],t=t,state=label,sampling=kind,n=K)
                        for key in local[0]:
                            vals=np.array([x[key] for x in local]);br[key]=float(vals.mean());br[key+'_std']=float(vals.std(ddof=1))
                        batches.append(br)
                    table('batches.csv',batches);dump('progress.json',dict(completed=len(batches),seconds=time.time()-start,new_draws=receipt['new_CPU_hard_draws']))
                    print(json.dumps(dict(task=task['task'],state=label,t=t,seconds=time.time()-tic,total_seconds=time.time()-start)),flush=True)
        for p,h in bindings.items():assert sha(p)==h
        receipt.update(status='complete' if not failures else 'complete_with_unavailable_conditions',bindings_unchanged=len(bindings))
    except BaseException as exc:
        receipt.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
    finally:
        for name,rows in [('graphs.csv',graphs),('relations.csv',relations),('triangles.csv',triangles),('batches.csv',batches),('sampling.csv',sampling)]:table(name,rows)
        dump('failures.json',failures)
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,graphs=len(graphs),batches=len(batches),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump('receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status'].startswith('complete') else 1
if __name__=='__main__':raise SystemExit(main())
