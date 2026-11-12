"""Matched temporary optimizer/recovery checks + one fresh baseline chain.

Two temporary updates per arm: the second T update REPLAYS a fixed old batch
solely to exercise restoration; it is not an on-policy learning experiment.
All temporary models are discarded before untouched-parent sampling.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,copy,random,signal,traceback,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('threearm_base',Path(__file__).with_name('gpu_preflight.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
w,np,torch=base.w,base.np,base.torch
from ema import ParameterEMA,cpu_copy
from boundary_transactions import TrainingBoundary,rng_capture,rng_restore
from integration import restore_payload,ema_validation
from probe import input_hash
from hsgd.losses import supported_binary_ce
from observers import RecordedGenerator,RecordedPlan,digest

def state_digest(x):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for a in v:add(a)
        else:h.update(repr(v).encode())
    add(x);return h.hexdigest()

def main():
    out=Path(sys.argv[1]); ds=sys.argv[2]; folder=out/ds;folder.mkdir(exist_ok=False)
    protocol=json.loads((out/'protocol.json').read_text());start=time.time();results=[];calls=backwards=0
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned watchdog stop')))
    def save(name,x):base.write(folder/name,x)
    def check():
        if time.time()>=protocol['science_stop_epoch']:raise TimeoutError('science deadline')
    receipt=dict(status='running',dataset=ds,start_epoch=start,device=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        for p,h in protocol['bindings'].items():assert base.sha(p)==h,p
        old=json.loads((base.OLD/'protocol.json').read_text()); row=old['models'][ds]
        z=torch.load(row['readout'],map_location='cpu');cfg=z['binding']['config']
        assert z['updates']==3000 and z['readout']=='ema_0.99' and z['train_seed']==row['train_seed']
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        m=w.StructuralDenoiser(case,cfg,'A').cuda();m.load_state_dict(z['model'],strict=True);m.eval()
        parent=cpu_copy(m.state_dict());del z
        binding=dict(dataset=ds,train_seed=row['train_seed'],parent_readout_sha256=protocol['bindings'][row['readout']],
            protocol_sha256=base.sha(out/'protocol.json'),new_optimizer=True,new_EMA=True,parent_updates=3000)
        optim_cfg={k:v for k,v in cfg['optimizer'].items() if k!='type'};optim_cfg['lr']=1e-5
        o=torch.optim.AdamW(m.parameters(),**optim_cfg);e=ParameterEMA(m,.99)
        random.seed(row['train_seed']);np.random.seed(row['train_seed']);torch.manual_seed(row['train_seed']);torch.cuda.manual_seed_all(row['train_seed'])
        train_rng=np.random.default_rng(np.random.SeedSequence([2026111401,row['train_seed']]))
        boundary=TrainingBoundary(m,o,e,train_rng,'cuda',0);initial=boundary.checkpoint(binding)
        assert initial['updates']==0 and not initial['optimizer']['state']
        torch.save(initial,folder/'initial_complete.pt')
        ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
        labels=np.isin(ids,case['clean_ids']).astype(np.uint8)
        scorer=base.Score(ds,case,query,fam,labels);selfcost=scorer(labels)
        assert abs(selfcost[0])+abs(selfcost[1])<1e-12
        save('binding.json',dict(**binding,config=cfg,optimizer=optim_cfg,initial_complete_sha256=base.sha(folder/'initial_complete.pt'),
            active_slots=len(ids),reference_edges=int(labels.sum()),score_reference=selfcost[:2],source_model=base.sha(w.SCI/'package/hsgd/model.py')))
        save('environment.json',w.common.environment('cuda'))
        def reset(saved):
            nonlocal boundary
            restore_payload(cpu_copy(saved),m,o,e,binding,lambda x:rng_restore(x,train_rng,'cuda'))
            o.zero_grad(set_to_none=True);m.eval();boundary=TrainingBoundary(m,o,e,train_rng,'cuda',saved['updates'])
            assert state_digest(boundary.checkpoint(binding))==state_digest(saved)
        def make_input(position,rng_,force_t=None):
            return w.common.train_input(case,1000,1,row['train_seed'],position,rng_,force_t=force_t)
        def forward(inp):
            nonlocal calls
            check();logits,p=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
            w.common.check_prob_device(p,inp['f'],case,m.prior)
            return logits,p
        fixed=[make_input(900000+j,np.random.default_rng(np.random.SeedSequence([2026111403,row['train_seed'],j])),t)
               for j,t in enumerate([1,75,250,550,800,1000])]
        def evaluate(tag):
            before=state_digest(boundary.checkpoint(binding));ev=[]
            with boundary.validation():
                with torch.no_grad():
                    for inp in fixed:
                        logits,p=forward(inp);terms=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy()
                        pp=p.cpu().double().numpy();rr=[]
                        for f in range(case['n_families']):
                            pos=(inp['f']==f)&(inp['labels']==1);neg=(inp['f']==f)&(inp['labels']==0)
                            rr.append(dict(family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),
                                positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None,
                                expected_edges=float(pp[inp['f']==f].sum())))
                        ev.append(dict(t=inp['t'],CE=float(terms.sum()/case['space'].total),input_hash=input_hash([inp]),relations=rr))
            assert before==state_digest(boundary.checkpoint(binding))
            save('validation_'+tag+'.json',ev)
        evaluate('parent')
        # Frozen old trajectories are on-policy only at this unchanged parent.
        terminal=[];cost=[];trajectory=[]
        for ss in old['sample_seeds']:
            g=np.load(base.OLD/ds/str(ss)/'graph.npz');ee=g['edges'];ee=ee.T if ee.shape[0]!=2 else ee
            ff,ll=case['space'].encode(ee);bits=np.zeros(case['space'].total,dtype=np.uint8);bits[case['space'].offsets[ff]+ll]=1
            terminal.append(bits[ids]);c=scorer(bits[ids]);cost.append(c[0]+c[1])
            trajectory.append(sorted((base.OLD/ds/str(ss)).glob('step_*.npz')))
        c=np.asarray(cost);a=(4*c-c.sum())/3
        save('terminal_batch.json',dict(costs=c.tolist(),advantages=a.tolist(),on_policy_only_at_initial_state=True,
            second_update='fixed-batch engineering replay only, not on-policy training'))
        params=list(m.parameters())
        def update(arm):
            nonlocal backwards
            u=boundary.completed+1;tick=time.time();o.zero_grad(set_to_none=True);records=[]
            with boundary.step(u):
                if arm in ('S1','S16'):
                    inp=make_input(u,train_rng);before=state_digest(rng_capture(train_rng,'cuda'))
                    logits,p=forward(inp);ix=np.argsort(inp['ids']);baseix=np.argsort(ids)
                    assert np.array_equal(inp['ids'][ix],ids[baseix])
                    inv=np.argsort(baseix);order=ix[inv];lp=torch.log_softmax(logits.double(),dim=-1)[torch.as_tensor(order,device='cuda')]
                    prob=lp[:,1].exp().detach().cpu().numpy();K=1 if arm=='S1' else 16
                    aux=np.random.default_rng(np.random.SeedSequence([2026111404,row['train_seed'],u]));uniform=aux.random((K,len(ids)));bits=(uniform<prob).astype(np.uint8)
                    costs=np.array([sum(scorer(b)[:2]) for b in bits])
                    advantage=costs if K==1 else (K*costs-costs.sum())/(K-1)
                    loglik=torch.where(torch.as_tensor(bits,device='cuda').bool(),lp[:,1],lp[:,0]).sum(1)
                    loss=.01*(torch.as_tensor(advantage,device='cuda')*loglik).mean();loss.backward();backwards+=1
                    records.append(dict(t=inp['t'],input_hash=input_hash([inp]),prefix_uniform_hash=digest(uniform[0]),
                        costs=costs.tolist(),advantages=advantage.tolist(),surrogate_loss=float(loss.detach())))
                    assert before==state_digest(rng_capture(train_rng,'cuda'))
                else:
                    before=state_digest(rng_capture(train_rng,'cuda'))
                    for k,paths in enumerate(trajectory):
                        for path in paths:
                            zz=np.load(path);ee,ff=w.common.state_edges(case,zz['xt']);t=int(path.stem.split('_')[1])
                            inp=dict(edges=ee,edge_families=ff,query=query,f=fam,t=t)
                            logits,p=forward(inp);lp=torch.log_softmax(logits.double(),dim=-1)
                            target=torch.as_tensor(terminal[k],device='cuda').bool()
                            loss=.01*float(a[k])*torch.where(target,lp[:,1],lp[:,0]).sum()/32
                            loss.backward();backwards+=1
                    records.append(dict(trajectories=4,selected_times=32,uniform_time_mean=True,
                        terminal_costs=cost,advantages=a.tolist(),on_policy_at_first_update_only=u==1))
                    assert before==state_digest(rng_capture(train_rng,'cuda'))
                grads=[p.grad for p in params if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));theta=[p.detach().clone() for p in params]
                boundary.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
                displacement=float(torch.sqrt(sum((p.detach().double()-v.double()).square().sum() for p,v in zip(params,theta))))
            torch.cuda.synchronize()
            result=dict(arm=arm,update=u,seconds=time.time()-tick,gradient_norm_before_clip=norm,clip_scale=min(1.,1/(norm+1e-6)),displacement_norm=displacement,
                EMA_updates=e.updates,main_rng_hash=state_digest(rng_capture(train_rng,'cuda')),records=records,
                scope='temporary recovery test; not a trained candidate')
            assert np.isfinite(displacement) and displacement>0
            results.append(result);save('updates.json',results);return result
        for arm in ['S1','S16','T_eager']:
            reset(initial);update(arm)
            saved=boundary.checkpoint(binding);torch.save(saved,folder/(arm+'_u1_complete.pt'))
            disk=torch.load(folder/(arm+'_u1_complete.pt'),map_location='cpu')
            assert state_digest(saved)==state_digest(disk)
            expected=make_input(2,copy.deepcopy(train_rng));reset(disk);actual=make_input(2,copy.deepcopy(train_rng))
            assert input_hash([expected])==input_hash([actual])
            # Test ordinary EMA validation restores raw without changing state.
            before=state_digest(boundary.checkpoint(binding))
            def ema_check():
                with torch.no_grad(): return forward(fixed[0])[1].detach().cpu().numpy()
            # integration helper passes model to validation_fn; wrapper is explicit.
            with boundary.validation():ema_validation(lambda model:ema_check(),m,e,boundary=boundary)
            assert before==state_digest(boundary.checkpoint(binding))
            update(arm);evaluate(arm+'_temporary_u2')
            save(arm+'_checks.json',dict(initial_boundary_identical=True,saved_state_bitwise_equal=True,next_input_exact=True,
                EMA_validation_restores_raw=True,update_after_reload_finite=True,recomputed_update_bitwise_not_certified=True))
        first={r['arm']:r for r in results if r['update']==1}
        for field in ('input_hash','prefix_uniform_hash'):
            assert first['S1']['records'][0][field]==first['S16']['records'][0][field]
        assert first['S1']['main_rng_hash']==first['S16']['main_rng_hash']
        save('update_gates.json',dict(passed=True,temporary_updates=6,model_calls=calls,VJPs=backwards,
            not_quality_gate=True,second_terminal_update_is_engineering_replay=True))
        # Discard every temporary update before calibrating unchanged parent inference.
        reset(initial);assert base.equal(m.state_dict(),{k:v.cuda() for k,v in parent.items()})
        ss=2026111341;directory=folder/f'baseline_sample_{ss}';directory.mkdir()
        generator=RecordedGenerator(ss,case['space'].total);families=np.repeat(np.arange(case['n_families']),case['space'].counts)
        state=(generator.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
        chain=w.SamplingBoundary(state,1000,generator,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
        ident=dict(dataset=ds,train_seed=row['train_seed'],sample_seed=ss,source='unchanged parent EMA3000',
            parent_sha256=binding['parent_readout_sha256'],T=1000,R=1,initial_hash=digest(state))
        base.write(directory/'identity.json',ident);tick=time.time();original_plan=w.common.plan_for;plans=[]
        def plan(*args,**kwargs):p=RecordedPlan(original_plan(*args,**kwargs));plans.append(p);return p
        w.common.plan_for=plan
        try:
            for t in range(1000,0,-1):
                check();plans.clear();draw_calls=generator.calls
                def validate(candidate,record):
                    assert generator.calls==draw_calls+1 and len(plans)==1 and plans[0].blocks==1
                    assert record['exact_coverage'] and record['covered_slots']==len(ids)
                    assert not np.any(candidate[~np.repeat(case['active'],case['space'].counts)])
                rec=chain.advance(lambda x,at:w.runtime.reverse_step(m,case,x,kernel,at,1,ss,generator,'cuda',check_time=check),validate)
                rec.update(uniform_hash=generator.last_sha,query_hash=plans[0].h.hexdigest())
                with (directory/'steps.jsonl').open('a') as log:log.write(json.dumps(rec)+'\n')
                if t%100==0 or t==1:
                    torch.save(chain.checkpoint(ident),directory/'latest_complete.pt')
                    save('progress.json',dict(phase='baseline_chain',next_t=chain.next_t,seconds=time.time()-start))
                    print(json.dumps(dict(dataset=ds,next_t=chain.next_t,elapsed=time.time()-start)),flush=True)
                if t==950 and time.time()+(time.time()-tick)/50*950+30>protocol['science_stop_epoch']:
                    raise TimeoutError('measured complete chain does not fit frozen budget')
            ee,ff=w.common.state_edges(case,chain.state)
            np.savez_compressed(directory/'graph.npz',types=case['space'].types,edges=ee,families=ff,schema=np.array(case['space'].pairs))
            assert all(torch.equal(v.cpu(),parent[k]) for k,v in m.state_dict().items())
            base.write(directory/'summary.json',dict(status='complete',seconds=time.time()-tick,graph_sha256=base.sha(directory/'graph.npz'),
                steps=1000,model_unchanged=True,not_candidate_effect=True))
        except BaseException:
            torch.save(chain.checkpoint(ident),directory/'stopped_complete.pt');raise
        finally:w.common.plan_for=original_plan
        for p,h in protocol['bindings'].items():assert base.sha(p)==h,p
        receipt.update(status='complete',temporary_updates=6,forwards=calls,VJPs=backwards,new_complete_baseline_chains=1)
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc());save('failure.json',receipt);raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0)
        save('receipt.json',receipt)
if __name__=='__main__':main()
