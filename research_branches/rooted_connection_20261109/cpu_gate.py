"""Bounded finite checks and two registered real cost panels, CPU only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,itertools,json,resource,time,sys,traceback
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from score import RootedScore,RootedIncremental,JointLocalScore
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/rooted_connection_20261109T015142'
OUT.mkdir(parents=True,exist_ok=True)
START=__import__('datetime').datetime.fromisoformat('2026-11-09T01:51:42+08:00').timestamp()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False))
def independent(types,q,f,b,r):
    n=len(types);a=np.zeros((n,n),dtype=int);a[q[0],q[1]]=b;a+=a.T
    degree=np.zeros((n,r),dtype=int)
    for e,(u,v) in enumerate(q.T):degree[u,f[e]]+=int(b[e]);degree[v,f[e]]+=int(b[e])
    pairs=list(itertools.combinations_with_replacement(sorted(set(types.tolist())),2));tri=np.zeros((n,len(pairs)),dtype=int)
    for i,j,k in itertools.combinations(range(n),3):
        if a[i,j] and a[i,k] and a[j,k]:
            for root,u,v in [(i,j,k),(j,i,k),(k,i,j)]:tri[root,pairs.index(tuple(sorted((types[u],types[v]))))]+=1
    family={(min(u,v),max(u,v)):int(ff) for (u,v),ff in zip(q.T,f)};paths=np.zeros((n,r,r),dtype=int)
    for i in range(n):
        for j in np.flatnonzero(a[i]):
            for k in np.flatnonzero(a[j]):
                if k!=i:paths[i,family[tuple(sorted((i,j)))],family[tuple(sorted((j,k)))]]+=1
    return np.concatenate([degree,tri,paths.reshape(n,-1)],axis=1)
def main():
    began=time.time();rng=np.random.default_rng(2026110909)
    save('ledger.json',dict(start_epoch=START,science_stop_epoch=START+450*60,deadline_epoch=START+480*60,GPU_hours_limit=24,engineering_GPU_hours_limit=.5,CPUmemory_process_MiB=2048,max_GPU=8))
    save('protocol.json',dict(candidate='RootedScore v1',scope='conditional authorization: theory, CPU, engineering, complete matched training and generation if fit',
        main_CE='unaltered sumCE/full schema',architecture='O/A late CN, unchanged inputs',objective='joint type-wise MMD over relation degree, typed triangle participation, non-backtracking relation-pair walk counts at SAME root',
        kernel='Gaussian, equal active blocks, reference log1p RMS scale; no fit or search',
        gradient='hard Bernoulli one auxiliary graph; exact conditional sum per edge; no soft substitute',
        auxiliary_weight=.1,auxiliary_frequency='each update',initializations='existing sealed ordinary CE parents',
        train_seeds={'IMDB':[2026110411,2026110412],'PubMed':[2026110421,2026110422]},
        sample_seeds=[2026102821,2026102822,2026110231,2026110232],updates=3000,
        proposed_new_models=8,proposed_new_full_chains=32,baseline_models_reused=8,
        gates={'CPU':'exact enumeration feature and edge gradient; type-permutation invariance; nontrivial old-role tie witness; complete scope cost fits',
               'GPU':'no finite/state/input mismatch; exact load including auxiliary RNG; compare new estimator to CPU; ordinary recalculation variance disclosed, no tolerance relaxation',
               'formal':'all 8 models and 32 chains fit remaining 24 GPUh/450min with reserve; otherwise do not silently drop arms/seeds'},
        scientific_success='per dataset/seed compare all metrics; useful joint connection improvement must not be bought by >10% worsening degree W1, triangle absolute error, joint TV or >5 percentage point LCC loss on either seed; engineering thresholds not statistical significance',
        no_guarantee='role neighbor histogram and triangle summary do not specify closing partner roles, graph isomorphism or connectivity; not a solved CE/anonymous conditional target conflict',
        source='isolated new candidate, not revival of unchanged MMD/transport or RolX'))
    checks=0;maxerr=0.;types=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T;f=types[q[0]]+types[q[1]];y=np.array([1,1,0,1,0,1.])
    core=RootedScore(types,q,f,y,3);states=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    with torch.no_grad():cost=core(torch.from_numpy(states)).numpy()
    for b,cost0 in zip(states,cost):
        inc=RootedIncremental(core,b);assert abs(inc.base_score-cost0)<1e-12
        assert np.array_equal(inc.raw,independent(types,q,f,b,3))
        raw,changed=inc.raw_flips(np.arange(6));diff=inc.differences()
        for e in range(6):
            other=b.copy();other[e]=1-other[e]
            expected=independent(types,q,f,other,3);assert np.array_equal(raw[e],expected)
            assert np.all((expected==inc.raw).all(1)|changed[e])
            with torch.no_grad():err=abs((float(core(torch.from_numpy(other)))-cost0)-(1-2*b[e])*diff[e])
            assert err<1e-11;maxerr=max(maxerr,err);checks+=1
    p=np.array([.11,.26,.39,.61,.77,.91]);law=np.prod(np.where(states==1,p,1-p),axis=1)
    exact=(law[:,None]*cost[:,None]*(states-p)).sum(0);est=np.zeros(6)
    for b,w in zip(states,law):est+=w*RootedIncremental(core,b).logit_gradient(p)
    assert np.max(abs(exact-est))<1e-12
    perm=np.array([1,0,3,2]);adj=np.zeros((4,4));adj[q[0],q[1]]=y;adj+=adj.T
    with torch.no_grad():assert abs(float(core(torch.from_numpy(adj[np.ix_(perm,perm)][q[0],q[1]]))))<1e-12
    # Same node degree/triangle distributions; different hub-to-hub connection.
    ty=np.array([0,0,0,1,1,1]);qq=np.array([(i,j) for i in range(3) for j in range(3,6)]).T;ff=np.zeros(9,dtype=int)
    def graph(edges):return np.array([tuple(e) in edges for e in qq.T],dtype=float)
    ref=graph({(0,3),(0,4),(1,3),(2,5)});alt=graph({(0,4),(0,5),(1,3),(2,3)})
    old=JointLocalScore(ty,qq,ff,ref,1);new=RootedScore(ty,qq,ff,ref,1)
    with torch.no_grad():ov=float(old(torch.from_numpy(alt)));nv=float(new(torch.from_numpy(alt)))
    assert abs(ov)<1e-12 and nv>1e-6
    save('finite_checks.json',dict(passed=True,graphs=64,edge_checks=checks,max_difference_error=maxerr,gradient_error=float(np.max(abs(exact-est))),permutation=True,witness_old_score=ov,witness_new_score=nv))
    tasks=json.loads((ROOT/'results/output_boundary_replay_20261108T081654/tasks.json').read_text());real=[];bindings={}
    def bind(p):bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    for ds in ['IMDB','PubMed']:
        entry=next(x for x in tasks if x['dataset']==ds)['row']['config']['datasets'][ds]
        data=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'/entry['file']
        bind(data);assert bindings[str(data)]==entry['sha256']
        with np.load(data) as z:ty=z['types'].copy()
        fixed=torch.load(bind(ROOT/'results/hard_structure_response_20261108T182154'/f'{ds}_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
        inp=next(inp for t,seed,inp in fixed if t==500);qq,ff,yy=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
        with np.load(bind(ROOT/f'results/hard_structure_edge_direction_cpu_20261108/{ds}_CE_CE_t500_draw0_flips.npz')) as z:bb=(1-z['flip_sign'])/2
        core=RootedScore(ty,qq,ff,yy,int(ff.max())+1);inc=RootedIncremental(core,bb)
        indices=np.unique(np.linspace(0,len(bb)-1,32,dtype=int));start=time.time();d=inc.differences(indices);small=time.time()-start
        for e,delta in zip(indices[:8],d[:8]):
            other=bb.copy();other[e]=1-other[e]
            with torch.no_grad():assert abs(float(core(torch.from_numpy(other)))-inc.base_score-(1-2*bb[e])*delta)<1e-10
        full_seconds=None
        if small*len(bb)/len(indices)<120:
            start=time.time();full=inc.differences();full_seconds=time.time()-start
            assert np.max(abs(full[indices]-d))<1e-12
            np.savez_compressed(OUT/f'{ds}_cpu_cost_panel.npz',types=ty,query=qq,families=ff,reference=yy,bits=bb,conditional_difference=full)
        real.append(dict(dataset=ds,slots=len(bb),selected32_seconds=small,full_seconds=full_seconds,projected_full_seconds=small*len(bb)/len(indices),feature_columns=inc.raw.shape[1],score=inc.base_score))
        assert time.time()-began<900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<2048
    save('cpu_cost.json',real);save('CPU_asset_binding.json',bindings)
    save('cpu_receipt.json',dict(passed=True,seconds=time.time()-began,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU=0))
    assert not torch.cuda.is_initialized();print(json.dumps(real,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('cpu_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
