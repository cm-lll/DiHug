"""Bounded three-arm learning pilot. Reuses audited production maths.

S16 uses per-draw leave-one-out cost advantages, never winner selection.
T_eager collects new trajectories at each round's frozen current raw policy.
Its clean-head terminal update is a biased surrogate, not transition REINFORCE.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,time,json,random,signal,traceback,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('pilot_support',Path(__file__).resolve().parents[2]/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
base,w,np,torch=s.base,s.w,s.np,s.torch
from support import Score
base.Score=Score

def main():
    out=Path(sys.argv[1]); ds,arm=sys.argv[2:4]; assert arm in ('baseline','T_eager'); folder=out/(ds+'_'+arm);folder.mkdir(exist_ok=False)
    protocol=json.loads((out/'protocol.json').read_text()); start=time.time(); updates=[]; calls=backwards=0; boundary=None
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned watchdog stop')))
    def save(name,x):base.write(folder/name,x)
    def check():
        if time.time()>=protocol['science_stop_epoch']:raise TimeoutError('science deadline')
    def progress(phase,**kw):save('progress.json',dict(phase=phase,elapsed=time.time()-start,at_epoch=time.time(),**kw))
    receipt=dict(status='running',dataset=ds,arm=arm,start_epoch=start,device=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        for path,h in protocol['bindings'].items():assert base.sha(path)==h,path
        row=protocol['parents'][ds];z=torch.load(row['readout'],map_location='cpu');cfg=z['binding']['config']
        assert z['updates']==3000 and z['readout']=='ema_0.99' and z['train_seed']==row['train_seed']
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        m=w.StructuralDenoiser(case,cfg,'A').cuda();m.load_state_dict(z['model'],strict=True);m.eval();del z
        binding=dict(dataset=ds,train_seed=row['train_seed'],parent_readout_sha256=protocol['bindings'][row['readout']],
            protocol_sha256=base.sha(out/'protocol.json'),new_optimizer=True,new_EMA=True,parent_updates=3000)
        optim_cfg={k:v for k,v in cfg['optimizer'].items() if k!='type'};optim_cfg['lr']=1e-5
        o=torch.optim.AdamW(m.parameters(),**optim_cfg);e=s.ParameterEMA(m,.99)
        random.seed(row['train_seed']);np.random.seed(row['train_seed']);torch.manual_seed(row['train_seed']);torch.cuda.manual_seed_all(row['train_seed'])
        train_rng=np.random.default_rng(np.random.SeedSequence([2026111401,row['train_seed']]))
        boundary=s.TrainingBoundary(m,o,e,train_rng,'cuda',0);initial=boundary.checkpoint(binding)
        torch.save(initial,folder/'initial_complete.pt')
        # Exact serialized-boundary check, without claiming deterministic CUDA re-computation.
        loaded=torch.load(folder/'initial_complete.pt',map_location='cpu');assert s.state_digest(initial)==s.state_digest(loaded)
        s.restore_payload(loaded,m,o,e,binding,lambda x:s.rng_restore(x,train_rng,'cuda'))
        boundary=s.TrainingBoundary(m,o,e,train_rng,'cuda',0)
        assert s.state_digest(boundary.checkpoint(binding))==s.state_digest(initial)
        ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
        labels=np.isin(ids,case['clean_ids']).astype(np.uint8);scorer=base.Score(ds,case,query,fam,labels)
        assert sum(scorer(labels)[:2])<1e-12
        save('binding.json',dict(**binding,optimizer=optim_cfg,config=cfg,initial_state_hash=s.state_digest(initial),
            model_hash=s.state_digest(m.state_dict()),active_slots=len(ids),reference_edges=int(labels.sum()),
            loaded_complete_state_exact=True,recomputed_CUDA_update_bitwise_not_certified=True))
        save('environment.json',w.common.environment('cuda'));del initial,loaded
        params=list(m.parameters())
        def checkpoint(name):
            state=boundary.checkpoint(binding)
            state['pilot_progress']=dict(arm=arm,completed_updates=boundary.completed,
                main_input_root=2026111401,hard_root=2026111404,terminal_plan=protocol['terminal_plan'][ds],
                elapsed_from_worker=time.time()-start,absolute_deadline=protocol['science_stop_epoch'])
            tmp=folder/(name+'.tmp');torch.save(state,tmp);tmp.replace(folder/name)
        def make_input(position,rng_,force_t=None):
            return w.common.train_input(case,1000,1,row['train_seed'],position,rng_,force_t=force_t)
        def forward(inp):
            nonlocal calls
            check();logits,p=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
            w.common.check_prob_device(p,inp['f'],case,m.prior);return logits,p
        def sample(ss,directory,selected=()):
            directory.mkdir(exist_ok=False);before=s.state_digest(m.state_dict());rng_before=s.state_digest(s.rng_capture(train_rng,'cuda'))
            generator=s.RecordedGenerator(ss,case['space'].total)
            families=np.repeat(np.arange(case['n_families']),case['space'].counts)
            state=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
            chain=w.SamplingBoundary(state,1000,generator,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
            ident=dict(dataset=ds,arm=arm,train_seed=row['train_seed'],sample_seed=ss,readout='fixed raw',
                model_hash=before,updates=boundary.completed,T=1000,R=1,initial_hash=s.digest(state),protocol_sha256=base.sha(out/'protocol.json'))
            base.write(directory/'identity.json',ident);tick=time.time();original_plan=w.common.plan_for;plans=[];snapshots=[]
            def plan(*args,**kwargs):
                p=s.RecordedPlan(original_plan(*args,**kwargs));plans.append(p);return p
            w.common.plan_for=plan
            try:
                for t in range(1000,0,-1):
                    check();plans.clear();draw_calls=generator.calls
                    if t in selected:
                        assert chain.next_t==t
                        path=directory/f'Xt_{t:04d}.npz';np.savez_compressed(path,xt=chain.state,t=t);snapshots.append(path)
                    def validate(candidate,record):
                        assert generator.calls==draw_calls+1 and len(plans)==1 and plans[0].blocks==1
                        assert record['exact_coverage'] and record['covered_slots']==len(ids)
                        assert not np.any(candidate[~np.repeat(case['active'],case['space'].counts)])
                    rec=chain.advance(lambda x,at:w.runtime.reverse_step(m,case,x,kernel,at,1,ss,generator,'cuda',check_time=check),validate)
                    rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest())
                    with (directory/'steps.jsonl').open('a') as log:log.write(json.dumps(rec)+'\n')
                    if t%100==0 or t==1:
                        torch.save(chain.checkpoint(ident),directory/'latest_complete.pt')
                        progress('sampling',chain=directory.name,next_t=chain.next_t,updates=boundary.completed)
                ee,ff=w.common.state_edges(case,chain.state)
                np.savez_compressed(directory/'graph.npz',types=case['space'].types,edges=ee,families=ff,schema=np.array(case['space'].pairs))
                assert before==s.state_digest(m.state_dict())
                assert rng_before==s.state_digest(s.rng_capture(train_rng,'cuda'))
                assert chain.next_t==0 and generator.calls==1001 and len(snapshots)==len(selected)
                base.write(directory/'summary.json',dict(status='complete',seconds=time.time()-tick,graph_sha256=base.sha(directory/'graph.npz'),steps=1000,model_unchanged=True))
                return chain.state[ids].copy(),snapshots
            except BaseException:
                torch.save(chain.checkpoint(ident),directory/'stopped_complete.pt');raise
            finally:w.common.plan_for=original_plan
        def commit_update(records,tick):
            nonlocal backwards
            grads=[p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True))
            boundary.mark_optimizer_started();o.step();e.update(m,boundary.completed+1);e.at_update_boundary=True
            assert all(torch.isfinite(p).all() for p in params)
            return dict(update=boundary.completed+1,seconds=time.time()-tick,gradient_norm_before_clip=norm,
                clip_scale=min(1.,1/(norm+1e-6)),records=records)
        training_start=time.time()
        if arm in ('S1','S16'):
            # Allocation includes initialization, checkpointing and scoring, not only kernels.
            stop=start+protocol['training_seconds'][ds]
            while time.time()+4<stop:
                check();u=boundary.completed+1;tick=time.time();o.zero_grad(set_to_none=True)
                with boundary.step(u):
                    inp=make_input(u,train_rng);rng_before=s.state_digest(s.rng_capture(train_rng,'cuda'))
                    logits,p=forward(inp);ix=np.argsort(inp['ids']);baseix=np.argsort(ids)
                    assert np.array_equal(inp['ids'][ix],ids[baseix]);order=ix[np.argsort(baseix)]
                    lp=torch.log_softmax(logits.double(),dim=-1)[torch.as_tensor(order,device='cuda')]
                    prob=lp[:,1].exp().detach().cpu().numpy();K=1 if arm=='S1' else 16
                    aux=np.random.default_rng(np.random.SeedSequence([2026111404,row['train_seed'],u]));uniform=aux.random((K,len(ids)))
                    bits=(uniform<prob).astype(np.uint8);parts=[scorer(b)[:2] for b in bits];costs=np.array([sum(v) for v in parts])
                    a=costs if K==1 else (K*costs-costs.sum())/(K-1)
                    ll=torch.where(torch.as_tensor(bits,device='cuda').bool(),lp[:,1],lp[:,0]).sum(1)
                    loss=.01*(torch.as_tensor(a,device='cuda')*ll).mean();loss.backward();backwards+=1
                    assert rng_before==s.state_digest(s.rng_capture(train_rng,'cuda'))
                    rec=commit_update([dict(t=inp['t'],input_hash=s.input_hash([inp]),prefix_uniform_hash=s.digest(uniform[0]),
                        query_hash=s.digest(inp['query']),costs=costs.tolist(),components=parts,advantages=a.tolist(),surrogate_loss=float(loss.detach()))],tick)
                rec.update(main_rng_hash=s.state_digest(s.rng_capture(train_rng,'cuda')),EMA_updates=e.updates)
                with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                if u%50==0 or u==1:
                    checkpoint('latest_complete.pt');progress('training',updates=u,target_seconds=protocol['training_seconds'][ds])
        elif arm=='T_eager':
            for u in range(1,protocol['terminal_updates']+1):
                batch=[];policy_hash=s.state_digest(m.state_dict());checkpoint('latest_complete.pt')
                for plan in protocol['terminal_plan'][ds][str(u)]:
                    bits,paths=sample(plan['seed'],folder/f'train_round{u}_seed{plan["seed"]}',plan['times'])
                    assert s.state_digest(m.state_dict())==policy_hash
                    batch.append((bits,paths,scorer(bits)[:2]))
                costs=np.array([sum(v[2]) for v in batch]);a=2*costs-costs.sum()
                base.write(folder/f'round{u}_batch.json',dict(policy_hash=policy_hash,costs=costs.tolist(),advantages=a.tolist(),
                    new_on_policy_trajectories=True,clean_head_estimator_is_biased=True))
                tick=time.time();o.zero_grad(set_to_none=True)
                with boundary.step(u):
                    before=s.state_digest(s.rng_capture(train_rng,'cuda'))
                    for k,(bits,paths,_) in enumerate(batch):
                        for path in paths:
                            z=np.load(path);ee,ff=w.common.state_edges(case,z['xt']);t=int(z['t'])
                            logits,p=forward(dict(edges=ee,edge_families=ff,query=query,f=fam,t=t))
                            lp=torch.log_softmax(logits.double(),dim=-1);target=torch.as_tensor(bits,device='cuda').bool()
                            loss=.01*float(a[k])*torch.where(target,lp[:,1],lp[:,0]).sum()/16
                            loss.backward();backwards+=1
                    assert before==s.state_digest(s.rng_capture(train_rng,'cuda'))
                    rec=commit_update([dict(terminal_costs=costs.tolist(),advantages=a.tolist(),trajectories=2,selected_times=16,
                        on_policy_collection=True,policy_hash=policy_hash)],tick)
                rec.update(main_rng_hash=s.state_digest(s.rng_capture(train_rng,'cuda')),EMA_updates=e.updates)
                with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                checkpoint('latest_complete.pt');progress('terminal_update_complete',updates=u)
        else:assert arm=='baseline'
        training_seconds=time.time()-training_start
        checkpoint('final_complete.pt')
        save('training_summary.json',dict(updates=boundary.completed,training_seconds=training_seconds,
            allocation_before_final_evaluation_seconds=time.time()-start,forwards=calls,VJPs=backwards,raw_model_hash=s.state_digest(m.state_dict()),
            primary_readout='fixed final raw',EMA_updates=e.updates))
        # Fixed endpoint validation; inputs never used for updates or model selection.
        fixed=[make_input(910000+j,np.random.default_rng(np.random.SeedSequence([2026111511,row['train_seed'],j])),t)
               for j,t in enumerate([1,75,250,550,800,1000])]
        def evaluate():
            rows=[]
            with torch.no_grad():
                for inp in fixed:
                    logits,p=forward(inp);terms=s.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy()
                    pp=p.cpu().double().numpy();groups=[]
                    for f in range(case['n_families']):
                        pos=(inp['f']==f)&(inp['labels']==1);neg=(inp['f']==f)&(inp['labels']==0)
                        groups.append(dict(family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(terms[pos].mean()) if pos.any() else None,
                            negative_CE=float(terms[neg].mean()) if neg.any() else None,expected_edges=float(pp[inp['f']==f].sum())))
                    rows.append(dict(t=inp['t'],CE=float(terms.sum()/case['space'].total),input_hash=s.input_hash([inp]),relations=groups))
            return rows
        before=s.state_digest(boundary.checkpoint(binding))
        with boundary.validation():
            save('validation_raw.json',evaluate())
            save('validation_EMA.json',s.ema_validation(lambda model:evaluate(),m,e,boundary=boundary))
        assert before==s.state_digest(boundary.checkpoint(binding))
        for ss in protocol['sample_seeds']:
            sample(ss,folder/f'final_sample_{ss}')
        for path,h in protocol['bindings'].items():assert base.sha(path)==h,path
        receipt.update(status='complete',updates=boundary.completed,forwards=calls,VJPs=backwards)
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc());save('failure.json',receipt)
        if boundary is not None:
            try:checkpoint('stopped_complete.pt')
            except BaseException:save('nonresumable_failure.json',dict(error=traceback.format_exc(),use_previous_complete_checkpoint=True))
        raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        save('receipt.json',receipt)

if __name__=='__main__':main()

