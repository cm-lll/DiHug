"""Isolated single-step hard-clean continuation and unchanged R1 sampling."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,copy,traceback,signal,importlib.util,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_20261111'
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=mod('ssf_old',ROOT/'research_branches/exact_feedback_updates_20261110/worker.py');rt=old.rt;g=rt.g;torch=rt.torch;np=rt.np
from observers import RecordedGenerator,RecordedPlan,digest as array_digest
def read(p):return json.loads(Path(p).read_text())
def check():
    if time.time()>=read(OUT/'window.json')['science_stop_epoch']:raise TimeoutError('scientific deadline')
def inputs(case,cfg,rng,row,u,B):
    result=[g.one_input(case,cfg,u,rng)]
    for i in range(1,B):result.append(g.one_input(case,cfg,4*u+i,np.random.default_rng(np.random.SeedSequence([2026111131,row['seed'],u,i]))))
    return result
def setup(row):
    z,cfg,case,m,o,e,rng,b=rt.load(row);fixed=g.one_input(case,cfg,900000,np.random.default_rng(row['seed']+2026111133),force_t=75);ix=np.argsort(fixed['ids'])
    core=old.HardScore(case['space'].types,fixed['query'][:,ix],fixed['f'][ix],fixed['labels'][ix],case['n_families']);exact=old.Exact(core)
    live=b.checkpoint(z['binding'])
    assert all(g.digest(live[k])==g.digest(z[k]) for k in ('model','optimizer','ema','rng','updates'))
    return z,cfg,case,m,o,e,rng,b,core,exact,fixed['ids'][ix]
def update(row,B,K,u,z,cfg,case,m,o,e,rng,b,core,exact,canonical,history):
    check();torch.cuda.synchronize();tick=time.time();records=[];score_seconds=0
    with b.step(u):
        batch=inputs(case,cfg,rng,row,u,B);main_hash=rt.input_hash([batch[0]])
        assert main_hash==history[u-1]['input_sha256'],'historical main input mismatch'
        after_input=g.digest(g.rng_capture(rng,'cuda'));o.zero_grad(set_to_none=True);m.train()
        for i,inp in enumerate(batch):
            check();assert inp['ce_weight']==1/case['space'].total
            tok=('ssf',u,i);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],tok)
            try:
                graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,tok,details=True);g.common.check_prob_device(p,inp['f'],case,m.prior)
                ix=np.argsort(inp['ids']);assert np.array_equal(inp['ids'][ix],canonical) and np.array_equal(inp['labels'][ix],core.reference)
                prob=p.detach().cpu().double().numpy()[ix];margin=(logits[:,1]-logits[:,0])[torch.as_tensor(ix,device='cuda')]
                assert float(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()-prob).max())<2e-6
                aux=np.random.default_rng(np.random.SeedSequence([2026111132,row['seed'],u,i]));draw=aux.random((K,len(prob)));bits=(draw<prob).astype(np.uint8);st=time.time()
                values,grad,_=exact.value_gradient(prob);cost=old.remaining(core,bits);feedback=np.zeros((5,len(prob)));feedback[[0,2,3]]=grad;feedback[[1,4]]=old.loo(prob,bits,cost)
                v=np.zeros(5);v[[0,2,3]]=values;v[[1,4]]=cost.mean(0);score_seconds+=time.time()-st
                if not np.isfinite(feedback).all():raise FloatingPointError('nonfinite structural feedback')
                ce=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/case['space'].total
                margin.backward(torch.as_tensor(feedback.mean(0)/B,device='cuda',dtype=margin.dtype))
                records.append(dict(index=i,t=inp['t'],input_hash=rt.input_hash([inp]),query_hash=array_digest(inp['query']),hard_uniform_hash=array_digest(draw),hard_prefix16_hash=array_digest(draw[:16]),slots=len(prob),CE=float(ce.detach()),risks=v.tolist()))
            finally:ctx.close()
        assert after_input==g.digest(g.rng_capture(rng,'cuda')),'auxiliary or network contaminated main RNG'
        grads=[p.grad for p in m.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(v).all() for v in grads):raise FloatingPointError('nonfinite gradient')
        norm=float(torch.linalg.vector_norm(torch.stack([v.double().norm() for v in grads])))
        b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
    torch.cuda.synchronize()
    return dict(update=u,B=B,K=K,records=records,gradient_norm=norm,score_seconds=score_seconds,seconds=time.time()-tick,main_rng_hash=after_input,EMA_updates=e.updates)
def snapshot(folder,name,z,b,row,B,K):
    p=folder/name;state=b.checkpoint(z['binding']);rt.checkpoint(p,dict(state=state,run=dict(protocol_sha256=rt.sha(OUT/'protocol.json'),dataset=row['dataset'],seed=row['seed'],B=B,K=K,objective='original five hard risks arithmetic mean',aux_input_root=2026111131,hard_root=2026111132,stateless_auxiliary=True)))
    rt.save(p.with_suffix('.sha256.json'),dict(sha256=rt.sha(p),updates=b.completed));return state
@contextlib.contextmanager
def restoring_weights(m):
    raw=rt.cpu_copy(m.state_dict())
    try:yield
    finally:m.load_state_dict(raw,strict=True)
def evaluation(folder,label,z,cfg,case,m,o,e,rng,b,row):
    before=g.digest(b.checkpoint(z['binding']));rows=[]
    with b.validation(),restoring_weights(m):
        for readout in ('raw','EMA'):
            if readout=='EMA':m.load_state_dict(e.readout_state_dict(m),strict=True)
            m.eval()
            for j,t in enumerate((1,75,250,550,800,1000)):
                check();inp=g.one_input(case,cfg,910000+j,np.random.default_rng(row['seed']+2026111133+j),force_t=t);tok=('eval',label,readout,j);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],tok)
                try:
                    with torch.no_grad():
                        graph,logits,p=ctx.predict(inp['query'],inp['f'],t/1000,tok,details=True);g.common.check_prob_device(p,inp['f'],case,m.prior)
                        terms=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                    ix=np.argsort(inp['ids']);np.savez_compressed(folder/f'validation_{label}_{readout}_t{t}.npz',probability=p.cpu().double().numpy()[ix],labels=inp['labels'][ix],families=inp['f'][ix],ids=inp['ids'][ix],query=inp['query'][:,ix])
                    rel=[]
                    for f in range(case['n_families']):
                        masks=[(inp['f']==f)&(inp['labels']==v) for v in (0,1)]
                        rel.append(dict(family=f,negative_n=int(masks[0].sum()),positive_n=int(masks[1].sum()),negative_CE=float(terms[masks[0]].mean()) if masks[0].any() else None,positive_CE=float(terms[masks[1]].mean()) if masks[1].any() else None))
                    rows.append(dict(update=b.completed,readout=readout,t=t,input_hash=rt.input_hash([inp]),CE=float(terms.sum()/case['space'].total),relations=rel))
                finally:ctx.close()
    assert before==g.digest(b.checkpoint(z['binding'])),'validation state mismatch';rt.save(folder/f'validation_{label}.json',rows)
def sample(model,case,row,ss,folder,steps=1000,contract=None):
    folder.mkdir(exist_ok=False);m=model;m.eval();rng=RecordedGenerator(ss,case['space'].total);fam=np.repeat(np.arange(case['n_families']),case['space'].counts);state=(rng.random(case['space'].total)<case['prior'][fam]).astype(np.uint8)
    ident=dict(dataset=row['dataset'],train_seed=row['seed'],sample_seed=ss,T=1000,R=1,initial_state_sha256=array_digest(state),initial_uniform_sha256=rng.last_sha)
    if contract:assert ident['initial_state_sha256']==contract['initial_state_sha256'] and rng.last_sha==contract['initial_uniform_sha256']
    rt.save(folder/'identity.json',ident);chain=g.SamplingBoundary(state,1000,rng,'cuda');kernel=g.MarginalKernel(case['prior'],g.common.schedule(1000));original=g.common.plan_for;plans=[];tick=time.time();ct={a['t']:a for a in contract['steps']} if contract else {}
    def plan(*a,**kw):p=RecordedPlan(original(*a,**kw));plans.append(p);return p
    g.common.plan_for=plan;before=g.digest(m.state_dict())
    try:
        for t in range(1000,1000-steps,-1):
            check();plans.clear();calls=rng.calls
            def validate(candidate,record):
                assert len(plans)==1 and plans[0].blocks==1 and rng.calls==calls+1 and record['exact_coverage'] and record['covered_slots']==int(np.repeat(case['active'],case['space'].counts).sum())
                if contract:assert rng.last_sha==ct[t]['uniform_sha256'] and plans[0].h.hexdigest()==ct[t]['query_plan_sha256']
            rec=chain.advance(lambda x,at:g.reverse_step(m,case,x,kernel,at,1,ss,rng,'cuda',check_time=check,trace=False),validate)
            rec.update(uniform_sha256=rng.last_sha,query_plan_sha256=plans[0].h.hexdigest())
            with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if t%100==0 or t==1:rt.checkpoint(folder/'latest_complete.pt',chain.checkpoint(ident))
            if t in (1000,750,500,250,100,50,10,1):
                with (folder/'trajectory.jsonl').open('a') as f:f.write(json.dumps(dict(input_t=t,committed_t=chain.next_t,total_edges=int(chain.state.sum()),relation_counts=np.bincount(fam,weights=chain.state,minlength=case['n_families']).tolist()))+'\n')
        assert before==g.digest(m.state_dict())
        if steps==1000:
            edges,ef=g.common.state_edges(case,chain.state);np.savez_compressed(folder/'graph.npz',types=case['space'].types,edges=edges,families=ef,schema=np.array(case['space'].pairs))
        result=dict(status='complete' if steps==1000 else 'smoke_only',steps=steps,seconds=time.time()-tick,model_state_unchanged=True,paired_contract=contract is not None);rt.save(folder/'summary.json',result);return result
    except BaseException:
        try:rt.checkpoint(folder/'stopped_complete.pt',chain.checkpoint(ident))
        except BaseException as ex:rt.save(folder/'save_failure.json',dict(error=repr(ex)))
        raise
    finally:g.common.plan_for=original
def main():
    mode=sys.argv[1];idx=int(sys.argv[2]);proto=read(OUT/'protocol.json');task=proto['tasks'][idx];row=task['parent'];B,K=task['B'],task['K'];folder=OUT/(('preflight_' if mode=='preflight' else '')+task['id']);folder.mkdir(exist_ok=False);start=time.time();receipt=dict(status='running',start_epoch=start,mode=mode,task=task['id'],GPU=os.environ.get('CUDA_VISIBLE_DEVICES'))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('budget supervisor stop')))
    b=None;z=None
    try:
        rt.setup();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        for p,h in read(OUT/'bindings.json').items():assert rt.sha(p)==h,'source/asset changed '+p
        z,cfg,case,m,o,e,rng,b,core,exact,canonical=setup(row);history=[json.loads(v) for v in Path(row['history']).read_text().splitlines()];rt.save(folder/'environment.json',g.common.environment('cuda'));rt.save(folder/'identity.json',dict(task=task,config=cfg,protocol_sha256=rt.sha(OUT/'protocol.json')))
        if mode=='preflight':
            records=[]
            for u in (2001,2002):
                rec=update(row,B,K,u,z,cfg,case,m,o,e,rng,b,core,exact,canonical,history);records.append(rec)
                if u==2001:
                    saved=snapshot(folder,'complete.pt',z,b,row,B,K);on_disk=torch.load(folder/'complete.pt',map_location='cpu',weights_only=False);assert g.digest(saved)==g.digest(on_disk['state'])
                    rr=copy.deepcopy(rng);planned=inputs(case,cfg,rr,row,2002,B);b=rt.restore(on_disk['state'],m,o,e,rng);rr=copy.deepcopy(rng);reloaded=inputs(case,cfg,rr,row,2002,B);assert rt.input_hash(planned)==rt.input_hash(reloaded)
            snapshot(folder,'final_temporary.pt',z,b,row,B,K);evaluation(folder,'u2002',z,cfg,case,m,o,e,rng,b,row)
            contract=read(proto['sampling_contracts'])[row['dataset']+'_'+str(proto['sample_seeds'][0])]
            sr=sample(m,case,row,proto['sample_seeds'][0],folder/'smoke_sampling',steps=5,contract=contract)
            rt.save(folder/'gates.json',dict(passed=True,updates=records,sampling=sr,saved_state_exact=True,next_input_exact=True,evaluation_restores_raw=True,ordinary_GPU_recomputation_not_bitwise_certified=True,temporary_states_not_used_for_formal=True));receipt['status']='completed'
        else:
            evaluation(folder,'u2000',z,cfg,case,m,o,e,rng,b,row)
            for u in range(2001,3001):
                rec=update(row,B,K,u,z,cfg,case,m,o,e,rng,b,core,exact,canonical,history)
                with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if u%100==0:snapshot(folder,'latest_complete.pt',z,b,row,B,K)
                if u in (2500,3000):snapshot(folder/f'checkpoint_u{u}.pt',z,b,row,B,K);evaluation(folder,f'u{u}',z,cfg,case,m,o,e,rng,b,row)
                if u%10==0:rt.save(folder/'progress.json',dict(phase='training',completed=u,seconds=time.time()-start,last_update_seconds=rec['seconds']))
            readout=e.readout_state_dict(m);rt.checkpoint(folder/'EMA3000.pt',dict(model=readout,updates=3000,decay=e.decay,parent_checkpoint=row['checkpoint_sha256'],protocol_sha256=rt.sha(OUT/'protocol.json')));m.load_state_dict(readout,strict=True);m.eval();readhash=rt.sha(folder/'EMA3000.pt');del readout
            contracts=read(proto['sampling_contracts'])
            for ss in proto['sample_seeds']:
                rt.save(folder/'progress.json',dict(phase='sampling',sample_seed=ss,completed=3000,seconds=time.time()-start));sf=folder/f'sample_{ss}';sample(m,case,row,ss,sf,contract=contracts[row['dataset']+'_'+str(ss)]);rt.save(sf/'binding.json',dict(readout_sha256=readhash,protocol_sha256=rt.sha(OUT/'protocol.json'),graph_sha256=rt.sha(sf/'graph.npz'),updates=3000,readout='EMA0.99'))
            receipt.update(status='completed',updates=1000,complete_chains=4)
    except BaseException as ex:
        receipt.update(status='failed',error=repr(ex),traceback=traceback.format_exc())
        if b is not None and z is not None and mode=='formal':
            # Raw training snapshots only; after loading EMA, use saved u3000.
            if b.completed<3000:
                try:snapshot(folder,'stopped_complete.pt',z,b,row,B,K)
                except BaseException as se:receipt['snapshot_error']=repr(se)
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0);rt.save(folder/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
