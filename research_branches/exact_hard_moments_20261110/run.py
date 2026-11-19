import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,csv,itertools,hashlib,resource,traceback,importlib.util
from pathlib import Path
import numpy as np
import torch
from moments import ExactHardMoments
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_hard_moments_20261110';CPU=ROOT/'results/real_hard_feedback_cost_20261110'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
HardScore=module('original_hard',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py').HardScore
OldAbsent=module('original_exact',ROOT/'research_branches/hard_clean_pilot_v2_20261110/exact_absent.py').ExactAbsentTriangles
def dump(n,z):(OUT/n).write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def write(n,rows):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cos(a,b):
    n=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/n) if n else None
def finite_gate():
    rng=np.random.default_rng(2026111058);records=[];fd=[]
    types=np.array([0,0,1,1,2]);schema=list(itertools.combinations_with_replacement(range(3),2))
    for restricted in (False,True):
        q=np.array([e for e in itertools.combinations(range(5),2) if not restricted or tuple(sorted(types[list(e)]))!=(1,2)]).T
        fam=np.array([schema.index(tuple(sorted(types[e]))) for e in q.T]);ref=np.array([int(k%3!=0) for k in range(q.shape[1])],dtype=np.uint8);core=HardScore(types,q,fam,ref,len(schema));exact=ExactHardMoments(core)
        bits=np.array(list(itertools.product([0,1],repeat=len(ref))),dtype=np.int64)
        # Independent node-triple enumeration, not the vectorized moment index builder.
        edge_index={tuple(e):i for i,e in enumerate(q.T)};tr=np.zeros((len(bits),len(core.triples)),dtype=np.int64)
        for ns in itertools.combinations(range(5),3):
            es=list(itertools.combinations(ns,2))
            if all(e in edge_index for e in es):tr[:,core.tindex[tuple(sorted(types[list(ns)]))]]+=bits[:,[edge_index[e] for e in es]].prod(1)
        terr=((tr-core.ref['triangles'])/np.maximum(1,core.ref['triangles']))**2
        cc=sum(((bits[:,fam==f].sum(1)-core.ref['counts'][f])/max(1,core.ref['counts'][f]))**2/core.r for f in range(core.r))
        costs=np.stack([cc,terr[:,core.present].mean(1) if core.present.any() else np.zeros(len(bits)),terr[:,core.absent].mean(1) if core.absent.any() else np.zeros(len(bits))],1)
        for k in range(8):
            p=rng.uniform(.03,.97,len(ref)) if k<6 else (np.linspace(0,1,len(ref)) if k==6 else ref.astype(float))
            mass=np.prod(np.where(bits,p,1-p),axis=1);V,G,stats=exact.value_gradient(p);v=mass@costs;g=(costs*mass[:,None]).T@(bits-p)
            np.testing.assert_allclose(V,v,atol=1e-11,rtol=1e-10);np.testing.assert_allclose(G,g,atol=1e-11,rtol=1e-10)
            for j,st in enumerate(stats['triangles']):
                mean=float(mass@tr[:,j]);var=float(mass@((tr[:,j]-mean)**2));np.testing.assert_allclose([st['mean'],st['variance']],[mean,var],atol=1e-11,rtol=1e-10)
            records.append(dict(restricted=restricted,case=k,states=len(bits),value_max_error=float(abs(V-v).max()),gradient_max_error=float(abs(G-g).max()),boundary=bool(k>=6)))
            if k==0:
                logits=np.log(p/(1-p));h=1e-5
                for ix in (0,len(ref)//2,len(ref)-1):
                    a=logits.copy();b=logits.copy();a[ix]+=h;b[ix]-=h
                    va=exact.value_gradient(1/(1+np.exp(-a)))[0];vb=exact.value_gradient(1/(1+np.exp(-b)))[0];num=(va-vb)/(2*h)
                    np.testing.assert_allclose(num,G[:,ix],atol=2e-9,rtol=1e-7);fd.append(dict(restricted=restricted,edge=ix,max_error=float(abs(num-G[:,ix]).max())))
    return dict(passed=True,enumeration=records,finite_difference=fd)
def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rec=dict(start_epoch=start,status='running',GPU_hours=0,model_calls=0,updates=0,new_real_samples=0,real_conditions=0)
    try:
        assets=json.loads((CPU/'assets.json').read_text());bindings={}
        for p in [Path(__file__),Path(__file__).with_name('moments.py'),CPU/'assets.json',CPU/'audit.json',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py',ROOT/'research_branches/hard_clean_pilot_v2_20261110/exact_absent.py']:bindings[str(p)]=sha(p)
        assert json.loads((CPU/'audit.json').read_text())['passed']
        dump('protocol.json',dict(track='CORE objective estimator engineering, not new loss',conditions=24,times=[1,75,250,550,800,1000],seeds=[2026110411,2026110412,2026110421,2026110422],reference=assets,
            exact_components=['relation_count','present_triangles','absent_triangles'],unchanged_sampled_components=['joint_degree','components'],target='original HardScore expected costs, unchanged group means and denominators',
            law='independent Bernoulli clean predictions on fixed legal slots',tests='16 five-node enumerated probability/support cases, 6 coordinate finite differences; shared-edge covariance and zero/one support; 24 legacy absent-risk comparisons',
            comparisons='cached K64 four repeats versus exact logit gradients; replace three rows only for read-only hybrid consistency, no model VJP or projection/update',
            resources=dict(CPU_threads=1,GPU=0,artificial_RSS_stop=None,scope='fixed panel only; log actual memory/time, stop on numerical or implementation errors'),prohibition='no weights/thresholds/targets/model changes; no new real hard graphs or chains'))
        dump('freeze.json',dict(epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json')))
        gate=finite_gate();dump('finite_gate.json',gate);print('finite gate passed',flush=True)
        caches={};rows=[];comparisons=[];pairs=[];typed=[];relations=[];oldchecks=[]
        for asset in assets:
            ds,seed,t=asset['dataset'],asset['seed'],asset['t'];folder=Path(asset['prediction']).parent;bindpaths=[Path(asset['data']),Path(asset['prediction']),folder/'evaluation_inputs.pt',folder/'score_definition.json',CPU/f'{ds}_{seed}_t{t}.npz']
            for p in bindpaths:bindings[str(p)]=sha(p)
            assert sha(asset['prediction'])==asset['prediction_sha256']
            saved=np.load(CPU/f'{ds}_{seed}_t{t}.npz');prob=saved['probability'];original=np.load(asset['prediction']);np.testing.assert_array_equal(prob,original['probability'].astype(float))
            v=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False);inp=next(x for x in v['inputs'] if x['t']==t);order=np.argsort(inp['ids']);np.testing.assert_array_equal(saved['ids'],inp['ids'][order])
            if ds not in caches:
                ts=time.perf_counter();d=np.load(asset['data']);core=HardScore(d['types'],inp['query'][:,order],inp['f'][order],inp['labels'][order],len(d['schema']));calc=ExactHardMoments(core);old=OldAbsent(core);caches[ds]=(core,calc,old)
                dump(ds+'_setup.json',dict(seconds=time.perf_counter()-ts,triangle_capacity=sum(len(x) for x in calc.triangle_edges),group_capacities=[len(x) for x in calc.triangle_edges],triples=core.triples,reference_triangles=core.ref['triangles'].tolist()))
            core,calc,old=caches[ds]
            for x,y in ((core.q,inp['query'][:,order]),(core.f,inp['f'][order]),(core.reference,inp['labels'][order])):np.testing.assert_array_equal(x,y)
            ts=time.perf_counter();V,G,stats=calc.value_gradient(prob);seconds=time.perf_counter()-ts
            vo,go=old.value_gradient(prob);np.testing.assert_allclose(V[2],vo,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(G[2],go,rtol=1e-9,atol=1e-11)
            oldchecks.append(dict(dataset=ds,seed=seed,t=t,value_error=float(abs(V[2]-vo)),gradient_error=float(abs(G[2]-go).max())))
            meta=dict(dataset=ds,seed=seed,t=t);hybrid=saved['logit_gradients'].copy();hybrid[:,[0,2,3]]=G[None]
            for rep in range(4):
                for k,j in enumerate((0,2,3)):
                    a=saved['logit_gradients'][rep,j];b=G[k];den=float(np.linalg.norm(b));varpart=0.
                    if j in (2,3):
                        ids=np.flatnonzero(core.present if j==2 else core.absent)
                        varpart=float(np.mean([stats['triangles'][i]['variance']/max(1,stats['triangles'][i]['target'])**2 for i in ids])) if len(ids) else 0.
                    comparisons.append(dict(**meta,repeat=rep,group=core.names[j],sample_cost_mean=float(saved['costs'][rep,:,j].mean()),exact_risk=float(V[k]),exact_gradient_norm=den,estimated_gradient_norm=float(np.linalg.norm(a)),cosine=cos(a,b),relative_error=float(np.linalg.norm(a-b)/den) if den else None,variance_risk=varpart if j!=0 else None))
            for i,j in itertools.combinations(range(4),2):pairs.append(dict(**meta,repeat_a=i,repeat_b=j,old_mean_cosine=cos(saved['logit_gradients'][i].mean(0),saved['logit_gradients'][j].mean(0)),hybrid_mean_cosine=cos(hybrid[i].mean(0),hybrid[j].mean(0))))
            for s in stats['triangles']:typed.append(dict(**meta,types='-'.join(map(str,core.triples[s['index']])),**s))
            for s in stats['relations']:relations.append(dict(**meta,**s))
            rows.append(dict(**meta,exact_seconds=seconds,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**dict(zip(['relation_risk','present_triangle_risk','absent_triangle_risk'],map(float,V)))))
            np.savez_compressed(OUT/f'{ds}_{seed}_t{t}.npz',values=V,logit_gradients=G,hybrid_logit_gradients=hybrid)
            rec['real_conditions']+=1;dump('receipt.json',dict(rec,elapsed_seconds=time.time()-start));print('done',ds,seed,t,round(seconds,3),flush=True)
        for n,r in [('costs.csv',rows),('estimator_comparison.csv',comparisons),('repeat_consistency.csv',pairs),('typed_moments.csv',typed),('relation_moments.csv',relations)]:write(n,r)
        for p,h in bindings.items():assert sha(p)==h,'bound file changed '+p
        dump('bindings.json',bindings);dump('legacy_absent_checks.json',dict(passed=True,conditions=oldchecks));rec['status']='completed'
    except BaseException as exc:rec['status']='failed';rec['error']=repr(exc);(OUT/'failure.txt').write_text(traceback.format_exc());raise
    finally:rec.update(end_epoch=time.time(),elapsed_seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);dump('receipt.json',rec)
if __name__=='__main__':main()
