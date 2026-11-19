import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import csv,hashlib,itertools,json,resource,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'results/hard_structure_rebar_cpu_20261108'
sys.path.insert(0,str(ROOT/'research_branches/hard_structure_rebar_cpu_20261108'))
from objective import JointLocalScore,edge_conditional
from incremental import IncrementalScore

OUT=ROOT/'results/hard_structure_incremental_cpu_20261108'
CFG=json.loads((HERE/'protocol.json').read_text())
OLD_CFG=json.loads((PREVIOUS/'protocol.json').read_text())
START=time.time()
COUNTS=dict(model_calls=0,GPU_hours=0,optimizer_updates=0,diffusion_chains=0,
            new_auxiliary_random_sequences=0,reused_hard_graph_draws=0,
            full_edge_maps=0,synthetic_graphs=0,synthetic_edge_checks=0,
            real_reference_edge_checks=0,archived_coordinate_checks=0,panels_complete=0)
BIND={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))


def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def check_time():
    if time.time()-START>CFG['worker_wall_seconds']-10:
        raise TimeoutError('Bounded CPU window exhausted; keep partial results')


def fixture(n,types,edges,query=None):
    q=np.array(np.triu_indices(n,1)) if query is None else np.asarray(query)
    types=np.asarray(types)
    pairs=sorted(set(tuple(sorted((int(types[i]),int(types[j])))) for i,j in q.T))
    fam=np.array([pairs.index(tuple(sorted((int(types[i]),int(types[j]))))) for i,j in q.T])
    edges={tuple(sorted(e)) for e in edges}
    y=np.array([tuple(x) in edges for x in q.T],float)
    return JointLocalScore(types,q,fam,y,len(pairs)),y


def reference_difference(core,bits,indices):
    out=[]
    with torch.no_grad():
        b=torch.as_tensor(bits)
        for k in range(0,len(indices),16):
            ix=torch.tensor(indices[k:k+16],dtype=torch.long)
            lo=b.expand(len(ix),-1).clone();hi=lo.clone()
            lo[torch.arange(len(ix)),ix]=0;hi[torch.arange(len(ix)),ix]=1
            out.append((core(hi)-core(lo)).numpy())
    return np.concatenate(out)


def synthetic():
    rng=np.random.default_rng(CFG['synthetic_seed']);rows=[]
    specifications=[]
    for k in range(CFG['random_small_graphs']):
        n=4+k%7;types=np.arange(n)%3
        q=np.array(np.triu_indices(n,1))
        # Include missing legal families in half the cases.
        if k%2:q=q[:,types[q[0]]!=types[q[1]]]
        reference=[tuple(e) for e in q.T if rng.random()<.4]
        core,y=fixture(n,types,reference,q)
        bits=(rng.random(len(y))<(.1+.8*(k%5)/4)).astype(float)
        specifications.append((f'random_{k}',core,bits))
    n=8;types=np.arange(n)%2;q=np.array(np.triu_indices(n,1))
    for name in CFG['explicit_graphs']:
        if name=='empty':e=[]
        elif name=='complete':e=list(map(tuple,q.T))
        elif name=='bipartite':e=[tuple(x) for x in q.T if types[x[0]]!=types[x[1]]]
        else:e=[(0,1)]
        core,y=fixture(n,types,[(0,1),(1,2),(0,2),(2,3)])
        bits=np.array([tuple(x) in e for x in q.T],float)
        specifications.append((name,core,bits))
    for name,core,bits in specifications:
        check_time();before=bits.copy();inc=IncrementalScore(core,bits)
        ix=np.arange(len(bits));ref=reference_difference(core,bits,ix)
        for batch in [1,7,128]:
            fast=inc.differences(ix,batch)
            err=float(np.max(abs(fast-ref)))
            assert err<CFG['absolute_difference_tolerance'],(name,batch,err)
            rows.append(dict(test=name,batch=batch,edges=len(ix),max_abs_difference_error=err))
        assert np.array_equal(bits,before) and np.array_equal(inc.bits,before)
        assert abs(inc.base_score-float(core(torch.tensor(bits))))<1e-12
        # Boundaries are exact-zero logit derivatives, not numerical clipping.
        p=np.linspace(0,1,len(bits));g=inc.logit_gradient(p)
        assert g[0]==0 and g[-1]==0 and np.isfinite(g).all()
        # Same edge and other graph edges: flip base bit, conditional difference unchanged.
        j=len(bits)//2;b2=bits.copy();b2[j]=1-b2[j]
        alt=IncrementalScore(core,b2).differences([j])[0]
        assert abs(alt-fast[j])<1e-12
        COUNTS['synthetic_graphs']+=1;COUNTS['synthetic_edge_checks']+=len(ix)*3
    # Fully enumerated distribution: preserve exact objective gradient.
    core,y=fixture(4,[0,0,1,1],[(0,1),(0,2),(1,2),(2,3)])
    states=np.array(list(itertools.product([0.,1.],repeat=len(y))))
    p=torch.tensor([.1,.26,.42,.58,.74,.9]);logits=torch.logit(p).requires_grad_(True)
    pp=logits.sigmoid();s=torch.tensor(states)
    mass=torch.where(s.bool(),pp,1-pp).prod(-1)
    loss=(mass*core(s)).sum();exact=torch.autograd.grad(loss,logits)[0].numpy()
    estimates=np.array([IncrementalScore(core,b).logit_gradient(p.numpy()) for b in states])
    mean=(mass.detach().numpy()[:,None]*estimates).sum(0)
    err=float(abs(mean-exact).max());assert err<1e-12
    rows.append(dict(test='64_state_exact_expected_gradient',max_abs_difference_error=err,edges=len(y),batch=None))
    dump('synthetic_tests.json',rows)
    return rows


def real():
    oldbind=json.loads((PREVIOUS/'asset_binding.json').read_text())
    for path,h in oldbind.items():
        assert sha(path)==h,path;BIND[path]=h
    for name in ['panels.json','protocol.json','selected_edges.json']:
        BIND[str(PREVIOUS/name)]=sha(PREVIOUS/name)
    panels=json.loads((PREVIOUS/'panels.json').read_text())
    coords=json.loads((PREVIOUS/'selected_edges.json').read_text())
    results=[];timings=[];byrel=[];sample_checks=[]
    for panel_id,panel in enumerate(panels):
        check_time();name=panel['panel'];ds=panel['dataset']
        data=next(Path(p) for p in oldbind if p.endswith(f'/assets/{ds}.npz'))
        inputfile=next(Path(p) for p in oldbind if p.endswith(f'/{ds}_fixed_inputs.pt'))
        with np.load(data) as a:
            types=a['types'];schema=a['schema'];clean={tuple(sorted(x)) for x in a['edges'].tolist()}
        inputs=torch.load(inputfile,map_location='cpu')['inputs']
        inp=next(x for phase,t,seed,x in inputs if phase=='new_diagnostic' and t==panel['t'])
        q,f,y=np.asarray(inp['query']),np.asarray(inp['f']),np.asarray(inp['labels'])
        assert np.array_equal(y,np.array([tuple(sorted(x)) in clean for x in q.T],int))
        with np.load(panel['probability_file']) as a:p=a['probability'].astype(float)
        # Previous real panel had no 0/1 outputs. Refuse a silently changed support.
        assert ((p>0)&(p<1)).all() and len(p)==panel['mutable_slots']
        core=JointLocalScore(types,q,f,y,len(schema))
        cc=[x for x in coords if x['panel']==name]
        selected=np.array([x['active_index'] for x in cc],int)
        oldfile=PREVIOUS/f'{name}_samples.npz';BIND[str(oldfile)]=sha(oldfile)
        with np.load(oldfile) as a:old_grad=a['edge_conditional'].copy()
        reference_rng=np.random.default_rng(CFG['reference_edge_seed']+panel_id)
        reference_ix=np.unique(np.r_[selected,reference_rng.choice(len(p),128,replace=False)])
        gen=torch.Generator().manual_seed(OLD_CFG['evaluation_seed']+panel_id)
        pt=torch.tensor(p)
        fullmaps=[];old_errors=[];cache_total=0.;selected_total=0.;full_total=0.
        for draw in range(OLD_CFG['evaluation_samples']):
            check_time()
            u=torch.rand(len(p),generator=gen);torch.rand(len(p),generator=gen)
            bits=(torch.logit(pt)+u.log()-torch.log1p(-u)>=0).double().numpy()
            tick=time.perf_counter();inc=IncrementalScore(core,bits);cache_seconds=time.perf_counter()-tick
            tick=time.perf_counter();gsel=inc.logit_gradient(p,selected,CFG['batch_size'])
            selected_seconds=time.perf_counter()-tick
            err=float(abs(gsel-old_grad[draw]).max());old_errors.append(err)
            assert err<CFG['absolute_logit_gradient_tolerance'],(name,draw,err)
            cache_total+=cache_seconds;selected_total+=selected_seconds
            COUNTS['reused_hard_graph_draws']+=1;COUNTS['archived_coordinate_checks']+=len(selected)
            if draw in CFG['full_edge_draws']:
                tick=time.perf_counter();differences=inc.differences(batch_size=CFG['batch_size'])
                all_seconds=time.perf_counter()-tick
                g=p*(1-p)*differences;fullmaps.append(g)
                assert abs(g[selected]-gsel).max()<1e-12
                full_total+=cache_seconds+all_seconds
                timings.append(dict(panel=name,dataset=ds,draw=draw,slots=len(p),
                    cache_seconds=cache_seconds,full_difference_seconds=all_seconds,
                    full_seconds_including_cache=cache_seconds+all_seconds,
                    selected_seconds_including_cache=cache_seconds+selected_seconds,
                    gradient_L2=float(np.linalg.norm(g)),max_abs_gradient=float(abs(g).max())))
                COUNTS['full_edge_maps']+=1
                if draw==0:
                    tick=time.perf_counter();ref=reference_difference(core,bits,reference_ix)
                    reference_seconds=time.perf_counter()-tick
                    tick=time.perf_counter();fast=inc.differences(reference_ix,CFG['batch_size'])
                    fast_reference_seconds=time.perf_counter()-tick
                    differr=float(abs(ref-fast).max());assert differr<CFG['absolute_difference_tolerance']
                    graderr=float(abs(p[reference_ix]*(1-p[reference_ix])*(ref-fast)).max())
                    assert graderr<CFG['absolute_logit_gradient_tolerance']
                    sample_checks.append(dict(panel=name,reference_edges=len(reference_ix),
                        max_abs_D_difference_error=differr,max_abs_gradient_error=graderr,
                        reference_seconds=reference_seconds,incremental_seconds=fast_reference_seconds,
                        cache_seconds=cache_seconds,same_slots_speedup=reference_seconds/(cache_seconds+fast_reference_seconds),
                        reference_full_linear_extrapolation_seconds=reference_seconds*len(p)/len(reference_ix)))
                    COUNTS['real_reference_edge_checks']+=len(reference_ix)
                for r in np.unique(f):
                    for label in [0,1]:
                        mask=(f==r)&(y==label)
                        if mask.any():
                            byrel.append(dict(panel=name,draw=draw,relation=int(r),label=label,
                                support=int(mask.sum()),gradient_L1=float(abs(g[mask]).sum()),
                                gradient_sum=float(g[mask].sum()),
                                nonzero_gradient=int(np.count_nonzero(g[mask])),
                                note='Cost/equivalence draw, not a training or quality decision'))
        np.savez_compressed(OUT/f'{name}_full_maps.npz',gradients=np.array(fullmaps),
                            probability=p,query=q,families=f,labels=y)
        newrow=dict(panel=name,dataset=ds,arm=panel['arm'],t=panel['t'],slots=len(p),
            selected=len(selected),reused_draws=64,full_edge_maps=len(fullmaps),
            max_archived_gradient_error=max(old_errors),
            cache_seconds_per_draw=cache_total/64,selected_seconds_per_draw=selected_total/64,
            full_seconds_per_draw_including_cache=full_total/len(fullmaps),
            reference_same_slots=sample_checks[-1])
        results.append(newrow);COUNTS['panels_complete']+=1
        dump('results.json',results);dump('progress.json',dict(counts=COUNTS,seconds=time.time()-START))
        print(json.dumps(newrow),flush=True)
    csvout('timings.csv',timings);csvout('reference_comparison.csv',sample_checks)
    csvout('relation_gradient_descriptors.csv',byrel)
    csvout('results.csv',[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in results])
    return results


def main():
    OUT.mkdir(exist_ok=False);dump('protocol.json',CFG)
    dump('start.json',dict(start=datetime.now().astimezone().isoformat(),CPU_threads=1,
        worker_wall_seconds=CFG['worker_wall_seconds'],python=sys.version,torch=torch.__version__,numpy=np.__version__))
    sources={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    sources[str(ROOT/CFG['source_objective'])]=sha(ROOT/CFG['source_objective'])
    assert sources[str(ROOT/CFG['source_objective'])]==json.loads((PREVIOUS/'source_binding.json').read_text())[str(ROOT/CFG['source_objective'])]
    dump('source_binding.json',sources)
    synthetic();real()
    assert all(sha(p)==h for p,h in BIND.items())
    assert all(sha(p)==h for p,h in sources.items())
    production=json.loads((ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json').read_text())
    assert all(sha(p)==h for p,h in production.items())
    assert not torch.cuda.is_initialized()
    dump('asset_binding.json',BIND)
    dump('receipt.json',dict(status='complete',counts=COUNTS,seconds=time.time()-START,
         max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         production_sources_unchanged=len(production),input_files_unchanged=len(BIND),
         cuda_initialized=False,finish=datetime.now().astimezone().isoformat()))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():dump('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),
            counts=COUNTS,seconds=time.time()-START))
        raise
