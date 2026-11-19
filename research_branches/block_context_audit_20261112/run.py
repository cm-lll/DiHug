"""Bounded read-only frozen-Xt R1/R10/R20 audit. No training or full chains."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,hashlib,traceback,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as support
w,np,torch=support.w,support.np,support.torch
from hsgd.kernel import MarginalKernel

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
    p=Path(p); q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def main(out):
    out=Path(out); protocol=json.loads((out/'protocol.json').read_text());alloc=time.time()
    write(out/'allocation.json',dict(start_epoch=alloc,pid=os.getpid(),device=os.environ['CUDA_VISIBLE_DEVICES']))
    records=[]; calls=0; status='failed'; error=None
    def check():
        if time.time()>=min(protocol['science_stop_epoch'],alloc+1800):raise TimeoutError('budget stop')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('independent watchdog')))
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        write(out/'environment.json',w.common.environment('cuda'))
        for path,h in protocol['bindings'].items():assert sha(path)==h,path
        for di,ds in enumerate(['IMDB','PubMed']):
            check();row=protocol['models'][ds];folder=out/ds;folder.mkdir()
            parent=torch.load(row['parent_readout'],map_location='cpu',weights_only=False)
            z=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
            assert z['updates']==2 and z['recovery']['exact_resume_allowed']
            data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
            case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
            m=w.StructuralDenoiser(case,parent['binding']['config'],'A').cuda().eval()
            m.load_state_dict(z['model'],strict=True)
            before=support.state_digest(m.state_dict());assert before==support.state_digest(z['model'])
            del parent,z
            write(folder/'binding.json',dict(**row,readout='raw at terminal update2',model_digest=before,
                n_nodes=len(data['types']),reference_edges=edges.shape[1],source_checkpoint_sha256=sha(row['checkpoint'])))
            ids,ff,query=w.common.plan_for(case,2026111201,1).block(0)
            fullf=np.repeat(np.arange(case['n_families']),case['space'].counts)
            active=np.repeat(case['active'].astype(bool),case['space'].counts)
            clean=np.zeros(case['space'].total,dtype=np.uint8);clean[case['clean_ids']]=1
            kernel=MarginalKernel(case['prior'],w.common.schedule(1000))
            types=case['space'].types;n=len(types)
            for ti,t in enumerate(protocol['times']):
                check();noise_seed=[protocol['noise_seed_root'],di,t]
                rng=np.random.default_rng(np.random.SeedSequence(noise_seed))
                xt=(rng.random(len(clean)) < kernel.alpha[t]*clean+(1-kernel.alpha[t])*case['prior'][fullf]).astype(np.uint8)
                uniform_seed=[protocol['uniform_seed_root'],di,t]
                np.savez_compressed(folder/f'input_t{t}.npz',xt=xt,clean=clean,types=types,prior=case['prior'],families=fullf)
                reference=None;traces={}
                variants=[(1,0),(1,1),(10,0),(10,1),(20,0),(20,1)]
                for rounds,rep in variants:
                    check();tag=f't{t}_R{rounds}_v{rep}'
                    plan_seed=protocol['partition_seeds'][0 if rounds==1 else rep]
                    plan=w.common.plan_for(case,[plan_seed,t,61],rounds)
                    inv=w.common.query_inventory(case,plan)
                    query_hash=hashlib.sha256()
                    for b in range(rounds):query_hash.update(plan.block(b)[0].tobytes())
                    rs=np.random.default_rng(np.random.SeedSequence(uniform_seed))
                    global_rng=support.state_digest(support.base.rng())
                    old=xt.copy();torch.cuda.reset_peak_memory_stats();tick=time.time()
                    nxt,rec=w.runtime.reverse_step(m,case,xt,kernel,t,rounds,plan_seed,rs,'cuda',trace=True,check_time=check)
                    calls+=rec['blocks'];tr=rec['trace'];traces[(rounds,rep)]=tr
                    assert np.array_equal(xt,old) and rec['exact_coverage'] and rec['blocks']==rounds
                    assert global_rng==support.state_digest(support.base.rng())
                    expected=np.random.default_rng(np.random.SeedSequence(uniform_seed)).random(len(xt))
                    assert np.array_equal(tr['uniform'],expected)
                    assert np.array_equal(nxt[active],(expected<tr['q'])[active])
                    if reference is None:reference=tr
                    relations=[]
                    for f,pair in enumerate(case['space'].pairs if hasattr(case['space'],'pairs') else data['schema']):
                        mask=(fullf==f)&active; nslots=int(mask.sum())
                        if not nslots:
                            relations.append(dict(family=f,endpoint_types=list(map(int,pair)),slots=0,fixed=True));continue
                        p=tr['p0'][mask];q=tr['q'][mask];pbase=reference['p0'][mask];qb=reference['q'][mask]
                        yy=clean[mask].astype(bool); xx=xt[mask].astype(bool)
                        qm=ff==f;deg=np.zeros(n);target=np.zeros(n)
                        np.add.at(deg,query[:,qm].ravel(),np.tile(tr['p0'][ids[qm]],2))
                        np.add.at(target,query[:,qm].ravel(),np.tile(clean[ids[qm]],2))
                        endpoint=(types==int(pair[0]))|(types==int(pair[1]))
                        relations.append(dict(family=f,endpoint_types=list(map(int,pair)),slots=nslots,positive_n=int(yy.sum()),
                            p0_sum=float(p.sum()),expected_births=float(q[~xx].sum()),expected_deaths=float((1-q[xx]).sum()),
                            expected_next_edges=float(q.sum()),expected_net=float(q.sum()-xx.sum()),
                            p0_abs_diff_mean=float(abs(p-pbase).mean()),p0_abs_diff_max=float(abs(p-pbase).max()),
                            q_abs_diff_mean=float(abs(q-qb).mean()),q_abs_diff_max=float(abs(q-qb).max()),
                            clean_false_negative_mass=float((1-p[yy]).sum()),clean_false_positive_mass=float(p[~yy].sum()),
                            expected_relation_degree_MAE=float(abs(deg[endpoint]-target[endpoint]).mean())))
                    rowout=dict(dataset=ds,t=t,rounds=rounds,repeat=rep,plan_seed=plan_seed,noise_seed=noise_seed,
                        uniform_seed=uniform_seed,xt_sha256=digest(xt),uniform_sha256=digest(expected),query_sha256=query_hash.hexdigest(),
                        active_slots=int(active.sum()),calls=rec['blocks'],seconds=time.time()-tick,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                        p0_abs_diff_mean=float(abs(tr['p0'][active]-reference['p0'][active]).mean()),
                        p0_abs_diff_max=float(abs(tr['p0'][active]-reference['p0'][active]).max()),
                        q_abs_diff_mean=float(abs(tr['q'][active]-reference['q'][active]).mean()),
                        expected_next_edges=float(tr['q'][active].sum()),
                        diagnostic_next_state_disagreements=int((nxt!=reference['state']).sum()),
                        coverage=True,input_immutable=True,global_RNG_unchanged=True,relations=relations)
                    np.savez_compressed(folder/(tag+'.npz'),**tr)
                    records.append(rowout);write(out/'panels.json',records)
                    write(out/'progress.json',dict(last=tag,dataset=ds,calls=calls,elapsed_seconds=time.time()-alloc))
                    print(ds,tag,'seconds',round(rowout['seconds'],2),'p0_diff',rowout['p0_abs_diff_mean'],flush=True)
                partition=[]
                for r in (10,20):
                    a,b=traces[(r,0)],traces[(r,1)]
                    partition.append(dict(rounds=r,p0_mean=float(abs(a['p0'][active]-b['p0'][active]).mean()),
                        q_mean=float(abs(a['q'][active]-b['q'][active]).mean()),next_state_disagreements=int((a['state']!=b['state']).sum())))
                write(folder/f'partition_t{t}.json',partition)
            assert before==support.state_digest(m.state_dict())
            write(folder/'unchanged.json',dict(model_unchanged=True,updates=0,backwards=0,full_chains=0))
            del m;torch.cuda.empty_cache()
        assert calls==372 and len(records)==36
        status='completed'
    except BaseException:
        error=traceback.format_exc();(out/'failure.txt').write_text(error)
    finally:
        write(out/'receipt.json',dict(status=status,calls=calls,panels=len(records),start_epoch=alloc,end_epoch=time.time(),
            worker_seconds=time.time()-alloc,error=error,formal_updates=0,backwards=0,full_chains=0))
    if status!='completed':raise SystemExit(1)

if __name__=='__main__':main(sys.argv[1])
