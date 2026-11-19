"""Four disposable updates, two reverse timesteps; no quality gate."""
from .formal_runtime import *
from .objective import make_objective
from .slot_update import update
from observers import RecordedGenerator,RecordedPlan,digest
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
import gc,traceback

def main():
    setup();authenticate_sources();folder=OUT/'engineering'
    from .prepare_confirmation import seal_cuda
    seal_cuda()
    receipt=dict(passed=False,rows=[],ordinary_recompute_tolerance_passed=False)
    started=time.time()
    try:
        for task in (TASKS[0],TASKS[3]):
            check();row,case,cfg,m,o,e,r,b,binding=load(task)
            torch.cuda.reset_peak_memory_stats()
            objective=make_objective(case,cfg,row['dataset'])
            contracts=read(OUT/'prepared'/f"input_contract_{row['dataset']}_{row['train_seed']}.json")
            records=[];inputs=[]
            def perform(pos):
                nonlocal b
                tick=time.perf_counter()
                with b.step(pos):
                    inp=g.one_input(case,cfg,pos,r);ih=input_hash([inp])
                    assert ih==contracts[pos-1]['sha256']
                    before=g.digest(g.rng_capture(r,'cuda'))
                    if row['arm']=='PB':
                        rec=update(m,o,case,inp,1000,'cuda',e,b.completed,transaction=b,objective=objective,diagnostic=True)
                    else:
                        rec=g.update(m,o,case,inp,1000,'cuda',e,b.completed,transaction=b)
                    assert before==g.digest(g.rng_capture(r,'cuda'))
                rec.update(total_boundary_seconds=time.perf_counter()-tick,input_sha256=ih)
                assert rec['optimizer_steps']==1 and e.updates==pos
                if row['arm']=='PB':assert rec['aux_nonzero']>0
                records.append(rec);inputs.append(ih)
            path=folder/(task+'_u0.pt');saved=b.checkpoint(binding);atomic_checkpoint(path,saved)
            perform(1);continuous=b.checkpoint(binding)
            gradients={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
            z=torch.load(path,map_location='cpu')
            if row['arm']=='PB':make_objective(case,cfg,row['dataset']).assert_binding(z['binding']['output_objective']['target'])
            g.restore_payload(z,m,o,e,binding,lambda st:g.rng_restore(st,r,'cuda'))
            b=g.TrainingBoundary(m,o,e,r,'cuda',0)
            restored=b.checkpoint(binding)
            assert all(g.digest(saved[k])==g.digest(restored[k]) for k in ('model','optimizer','ema','rng','updates'))
            perform(1);repeat=b.checkpoint(binding)
            assert inputs[0]==inputs[1] and g.digest(continuous['rng'])==g.digest(repeat['rng']) and continuous['updates']==repeat['updates']
            comparison=g.compare_state(continuous,repeat,cfg['tolerances']['fp32_atol'],cfg['tolerances']['fp32_rtol'])
            grad_max=max(float((p.grad.detach().cpu()-gradients[n]).abs().max()) for n,p in m.named_parameters() if p.grad is not None)
            with b.validation():
                ema_validation(validation,m,e,case,cfg,row['train_seed'],r,boundary=b)
            assert all(g.digest(b.checkpoint(binding)[k])==g.digest(repeat[k]) for k in ('model','optimizer','ema','rng','updates'))
            m.eval();family=np.repeat(np.arange(case['n_families']),case['space'].counts)
            ss=SAMPLES[0];sr=RecordedGenerator(ss,case['space'].total)
            state=(sr.random(case['space'].total)<case['prior'][family]).astype(np.uint8)
            con=read(OUT/'prepared/sampling_contracts.json')[row['dataset']+'_'+str(ss)]
            assert digest(state)==con['initial_state_sha256']
            chain=g.SamplingBoundary(state,1000,sr,'cuda');kernel=MarginalKernel(case['prior'],g.common.schedule(1000))
            oldplan=g.common.plan_for;plans=[]
            def planned(*args,**kw):
                p=RecordedPlan(oldplan(*args,**kw));plans.append(p);return p
            g.common.plan_for=planned
            try:
                def contract(candidate,rec):
                    target=con['steps'][0]
                    assert plans[0].h.hexdigest()==target['query_plan_sha256'] and sr.last_sha==target['uniform_sha256'] and rec['exact_coverage']
                torch.cuda.synchronize();tick=time.perf_counter()
                chain.advance(lambda st,t:reverse_step(m,case,st,kernel,t,1,ss,sr,'cuda',check_time=check),contract)
                torch.cuda.synchronize();reverse_seconds=time.perf_counter()-tick
                snap=chain.checkpoint({'engineering':True});cp=folder/(task+'_reverse.pt');atomic_checkpoint(cp,snap)
                chain.restore(torch.load(cp,map_location='cpu'),{'engineering':True})
                assert g.digest(chain.checkpoint({'engineering':True}))==g.digest(snap)
            finally:g.common.plan_for=oldplan
            receipt['rows'].append(dict(task=task,dataset=row['dataset'],arm=row['arm'],updates=records,exact_load=True,exact_RNG_and_inputs=True,continuation_comparison=comparison,gradient_max_difference=grad_max,
                reverse_step_seconds=[reverse_seconds],exact_sampling_load=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,objective=objective.objective_binding()))
            save(folder/'receipt.json',receipt)
            del m,o,e,r,b,chain,saved,z,continuous,repeat,restored,gradients,objective;gc.collect();torch.cuda.empty_cache()
        receipt['passed']=True
        receipt['gate_note']='Exact state load/RNG/contracts required; ordinary GPU recomputation variation reported, old tolerance not relaxed, no bitwise continuation claim.'
    except BaseException:
        receipt['error']=traceback.format_exc();raise
    finally:
        receipt['seconds']=time.time()-started;save(folder/'receipt.json',receipt)
if __name__=='__main__':main()
