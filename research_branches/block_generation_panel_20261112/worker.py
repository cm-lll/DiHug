"""One immutable-policy T1000 chain, R1/R10/R20; no optimization."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,time,json,signal,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as s
w,np,torch=s.w,s.np,s.torch

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=Path(sys.argv[1]);p=read(out/'protocol.json');job=next(x for x in p['new_jobs'] if x['id']==sys.argv[2])
    folder=out/'chains'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    start=time.time();chain=None;identity=None;original_plan=None;status='failed'
    def check():
        if time.time()>=p['science_stop_epoch']:raise TimeoutError('scientific deadline')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('own queue watchdog')))
    save(folder/'allocation.json',dict(pid=os.getpid(),start_epoch=start,device=os.environ['CUDA_VISIBLE_DEVICES'],job=job))
    try:
        check();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        for path,h in p['bindings'].items():assert sha(path)==h,path
        ds=job['dataset'];row=p['models'][ds];rounds=job['rounds'];ss=job['sample_seed']
        parent=torch.load(row['parent_readout'],map_location='cpu',weights_only=False)
        z=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert z['updates']==2 and z['recovery']['exact_resume_allowed']
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        assert np.array_equal(case['space'].encode(edges)[0]+1,data['relations'])
        m=w.StructuralDenoiser(case,parent['binding']['config'],'A').cuda().eval();m.load_state_dict(z['model'],strict=True)
        model_hash=s.state_digest(m.state_dict());assert model_hash==row['model_digest']==s.state_digest(z['model'])
        del parent,z
        save(folder/'environment.json',w.common.environment('cuda'))
        global_rng=s.state_digest(s.base.rng())
        generator=s.RecordedGenerator(ss,case['space'].total)
        families=np.repeat(np.arange(case['n_families']),case['space'].counts)
        active=np.repeat(case['active'].astype(bool),case['space'].counts)
        state=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
        expected=read(out/'random_contract.json')[ds][str(ss)]
        assert s.digest(state)==expected['initial_hash'] and generator.last_sha==expected['initial_uniform_hash']
        chain=w.SamplingBoundary(state,1000,generator,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
        identity=dict(dataset=ds,train_seed=row['train_seed'],sample_seed=ss,model_hash=model_hash,updates=2,
            readout='fixed raw',T=1000,R=rounds,initial_hash=s.digest(state),checkpoint_sha256=p['bindings'][row['checkpoint']],
            protocol_sha256=sha(out/'protocol.json'),query_plan_seed_rule='[sample_seed,t,61]',uniform_contract='one full S draw per step')
        save(folder/'identity.json',identity)
        def checkpoint(name):
            tmp=folder/(name+'.tmp');torch.save(chain.checkpoint(identity),tmp);tmp.replace(folder/name)
        checkpoint('initial.pt');original_plan=w.common.plan_for;plans=[]
        def record_plan(*args,**kw):
            plan=s.RecordedPlan(original_plan(*args,**kw));plans.append(plan);return plan
        w.common.plan_for=record_plan
        for t in range(1000,0,-1):
            check();plans.clear();draw_calls=generator.calls
            if t in p['snapshot_times']:np.savez_compressed(folder/f'Xt_{t:04d}.npz',xt=chain.state,t=t)
            def validate(candidate,record):
                assert generator.calls==draw_calls+1 and len(plans)==1 and plans[0].blocks==rounds
                assert record['exact_coverage'] and record['covered_slots']==int(active.sum())
                fixed=~active;assert np.array_equal(candidate[fixed],case['prior'][families[fixed]].astype(np.uint8))
                assert generator.last_sha==expected['step_uniform_hashes'][1000-t]
            rec=chain.advance(lambda x,at:w.runtime.reverse_step(m,case,x,kernel,at,rounds,ss,generator,'cuda',check_time=check),validate)
            rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest(),state_hash=s.digest(chain.state),
                edges=int(chain.state.sum()),elapsed_seconds=time.time()-start)
            with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if t%100==0 or t==1:
                checkpoint('latest_complete.pt');save(folder/'progress.json',dict(next_t=chain.next_t,elapsed_seconds=time.time()-start,edges=int(chain.state.sum())))
            if t==1000:save(folder/'startup.json',dict(model_hash_matched=True,initial_and_uniform_contract_passed=True,exact_coverage=True,blocks=rounds,first_step_seconds=rec['step_seconds'],at_epoch=time.time()))
        assert chain.next_t==0 and generator.calls==1001
        assert model_hash==s.state_digest(m.state_dict()) and global_rng==s.state_digest(s.base.rng())
        ee,ff=w.common.state_edges(case,chain.state)
        np.savez_compressed(folder/'graph.npz',types=case['space'].types,edges=ee,families=ff,schema=np.array(case['space'].pairs))
        save(folder/'summary.json',dict(status='complete',seconds=time.time()-start,steps=1000,graph_sha256=sha(folder/'graph.npz'),
            model_unchanged=True,global_RNG_unchanged=True,full_uniform_pairing=True,blocks_per_step=rounds,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
        status='complete'
    except BaseException:
        (folder/'failure.txt').write_text(traceback.format_exc())
        if chain is not None and identity is not None:
            try:checkpoint('stopped_complete.pt')
            except BaseException:(folder/'recovery_failure.txt').write_text(traceback.format_exc())
    finally:
        if original_plan is not None:w.common.plan_for=original_plan
        save(folder/'receipt.json',dict(status=status,worker_seconds=time.time()-start,at_epoch=time.time(),next_t=chain.next_t if chain else None))
    if status!='complete':raise SystemExit(1)

if __name__=='__main__':main()
