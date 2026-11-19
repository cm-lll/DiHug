"""Six disposable updates; unchanged sampler reused; no quality gate."""
from .formal_runtime import *
from .objective import make_objective, checkpoint_with_aux
from .slot_update import update
from observers import RecordedGenerator,RecordedPlan,digest
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
import gc,traceback

def main():
    setup();authenticate_sources();folder=OUT/'engineering'
    receipt=dict(passed=False,rows=[],ordinary_recompute_tolerance_passed=False)
    started=time.time()
    try:
        for task in [x['task_id'] for x in read(OUT/'tasks.json') if x['engineering']]:
            check();row,case,cfg,m,o,e,r,b,binding=load(task)
            torch.cuda.reset_peak_memory_stats()
            objective=make_objective(case,cfg,row['dataset'],row['objective_kind'])
            contracts=read(OUT/'prepared'/f"input_contract_{row['dataset']}_{row['train_seed']}.json")
            records=[];inputs=[]
            def perform(pos):
                nonlocal b
                tick=time.perf_counter()
                with objective.transaction(), b.step(pos):
                    inp=g.one_input(case,cfg,pos,r);ih=input_hash([inp])
                    assert ih==contracts[pos-1]['sha256']
                    before=g.digest(g.rng_capture(r,'cuda'))
                    rec=update(m,o,case,inp,1000,'cuda',e,b.completed,transaction=b,objective=objective,diagnostic=(pos==1))
                    assert before==g.digest(g.rng_capture(r,'cuda'))
                rec.update(total_boundary_seconds=time.perf_counter()-tick,input_sha256=ih)
                assert rec['optimizer_steps']==1 and e.updates==pos
                if pos==1:assert rec['weight_mean_error']==0 and rec['weight_min']==rec['weight_max']==1 and rec['weighted_CE']==rec['unweighted_CE']
                records.append(rec);inputs.append(ih)
            perform(1)
            path=folder/(task+'_u1.pt');saved=checkpoint_with_aux(b,binding,objective);atomic_checkpoint(path,saved)
            perform(2);continuous=checkpoint_with_aux(b,binding,objective)
            gradients={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
            z=torch.load(path,map_location='cpu')
            make_objective(case,cfg,row['dataset'],row['objective_kind']).assert_binding(z['binding']['output_objective']['target'])
            g.restore_payload(z,m,o,e,binding,lambda st:g.rng_restore(st,r,'cuda'));objective.load_state_dict(z['auxiliary_state'])
            b=g.TrainingBoundary(m,o,e,r,'cuda',1)
            restored=checkpoint_with_aux(b,binding,objective)
            assert all(g.digest(saved[k])==g.digest(restored[k]) for k in ('model','optimizer','ema','rng','updates','auxiliary_state'))
            perform(2);repeat=checkpoint_with_aux(b,binding,objective)
            assert g.digest(continuous['auxiliary_state'])==g.digest(repeat['auxiliary_state'])
            assert inputs[1]==inputs[2] and g.digest(continuous['rng'])==g.digest(repeat['rng']) and continuous['updates']==repeat['updates']
            comparison=g.compare_state(continuous,repeat,cfg['tolerances']['fp32_atol'],cfg['tolerances']['fp32_rtol'])
            grad_max=max(float((p.grad.detach().cpu()-gradients[n]).abs().max()) for n,p in m.named_parameters() if p.grad is not None)
            with b.validation():
                ema_validation(validation,m,e,case,cfg,row['train_seed'],r,boundary=b)
            assert all(g.digest(checkpoint_with_aux(b,binding,objective)[k])==g.digest(repeat[k]) for k in ('model','optimizer','ema','rng','updates','auxiliary_state'))
            # One complete reverse timestep, not a generation chain. Use an
            # isolated explicit sampler stream and verify training state intact.
            before_sample=checkpoint_with_aux(b,binding,objective)
            sr=np.random.default_rng(2026110807)
            family=np.repeat(np.arange(case['n_families']),case['space'].counts)
            state=(sr.random(case['space'].total)<case['prior'][family]).astype(np.uint8)
            kernel=MarginalKernel(case['prior'],g.common.schedule(1000))
            m.eval();tick=time.perf_counter()
            nxt,sampling_record=reverse_step(m,case,state.copy(),kernel,500,1,2026110807,sr,'cuda',check_time=check,trace=False)
            torch.cuda.synchronize();sampling_seconds=time.perf_counter()-tick
            assert sampling_record['exact_coverage'] and sampling_record['blocks']==1
            assert np.isin(nxt,[0,1]).all()
            after_sample=checkpoint_with_aux(b,binding,objective)
            assert all(g.digest(before_sample[k])==g.digest(after_sample[k]) for k in ('model','optimizer','ema','rng','updates','auxiliary_state'))
            receipt['rows'].append(dict(task=task,updates=records,exact_load=True,exact_RNG_and_inputs=True,continuation_comparison=comparison,gradient_max_difference=grad_max,
                sampling_formula_reused_unchanged=True,sampling_one_step_seconds=sampling_seconds,sampling_record=sampling_record,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,objective=objective.objective_binding()))
            save(folder/'receipt.json',receipt)
            del m,o,e,r,b,saved,z,continuous,repeat,restored,gradients,objective;gc.collect();torch.cuda.empty_cache()
        receipt['passed']=True
        receipt['gate_note']='Exact state load/RNG/contracts required; ordinary GPU recomputation variation reported, old tolerance not relaxed, no bitwise continuation claim.'
    except BaseException:
        receipt['error']=traceback.format_exc();raise
    finally:
        receipt['seconds']=time.time()-started;save(folder/'receipt.json',receipt)
if __name__=='__main__':main()


