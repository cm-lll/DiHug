"""Two datasets, two disposable updates and two reverse steps each; no quality gate."""
from .formal_runtime import *
from observers import RecordedGenerator,RecordedPlan,digest
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
import gc,traceback
def main():
    setup();folder=OUT/'engineering';receipt=dict(passed=False,rows=[],ordinary_recompute_tolerance_passed=False);began=time.time()
    original_extract=g.candidate_module.extract;expected=None;observed=[]
    def extract(types,edges,families,pairs,nt):
        if expected is not None and not np.array_equal(np.asarray(edges).reshape(-1,2).T,expected):raise ValueError('feature frozen Xt mismatch')
        tick=time.perf_counter();x=original_extract(types,edges,families,pairs,nt);observed.append(time.perf_counter()-tick);return x
    g.candidate_module.extract=extract
    try:
        for task in (TASKS[0],TASKS[2]):
            check();row,case,cfg,m,o,e,r,b,binding=load(task);records=[]
            contracts=read(OUT/'prepared'/f"input_contract_{row['dataset']}_{row['train_seed']}.json")
            for pos in (1,2):
                with b.step(pos):
                    inp=g.one_input(case,cfg,pos,r);expected=inp['edges'];assert input_hash([inp])==contracts[pos-1]['sha256']
                    rec=g.update(m,o,case,inp,1000,'cuda',e,b.completed,transaction=b)
                records.append(rec)
                if pos==1:
                    path=folder/(task+'_temporary.pt');saved=b.checkpoint(binding);atomic_checkpoint(path,saved)
                    z=torch.load(path,map_location='cpu');g.restore_payload(z,m,o,e,binding,lambda st:g.rng_restore(st,r,'cuda'))
                    restored=b.checkpoint(binding)
                    assert all(g.digest(saved[k])==g.digest(restored[k]) for k in ('model','optimizer','ema','rng','updates'))
            expected=None
            with b.validation():
                ema_validation(validation,m,e,case,cfg,row['train_seed'],r,boundary=b)
            m.eval();family=np.repeat(np.arange(case['n_families']),case['space'].counts)
            ss=SAMPLES[0];rng=RecordedGenerator(ss,case['space'].total);state=(rng.random(case['space'].total)<case['prior'][family]).astype(np.uint8)
            con=read(OUT/'prepared/sampling_contracts.json')[row['dataset']+'_'+str(ss)];assert digest(state)==con['initial_state_sha256']
            chain=g.SamplingBoundary(state,1000,rng,'cuda');kernel=MarginalKernel(case['prior'],g.common.schedule(1000));times=[]
            oldplan=g.common.plan_for;plans=[]
            def planned(*args,**kw):
                p=RecordedPlan(oldplan(*args,**kw));plans.append(p);return p
            g.common.plan_for=planned
            try:
                for index in range(2):
                    plans.clear();at=chain.next_t;expected,_=g.common.state_edges(case,chain.state)
                    def check_contract(s,rec):
                        target=con['steps'][index]
                        assert plans[0].h.hexdigest()==target['query_plan_sha256'] and rng.last_sha==target['uniform_sha256'] and rec['exact_coverage']
                    torch.cuda.synchronize();tick=time.perf_counter()
                    chain.advance(lambda st,t:reverse_step(m,case,st,kernel,t,1,ss,rng,'cuda',check_time=check),check_contract)
                    torch.cuda.synchronize();times.append(time.perf_counter()-tick)
                    snap=chain.checkpoint({'engineering':True});p=folder/(task+f'_reverse_{index}.pt');atomic_checkpoint(p,snap)
                    chain.restore(torch.load(p,map_location='cpu'),{'engineering':True});assert g.digest(chain.checkpoint({'engineering':True}))==g.digest(snap)
            finally:g.common.plan_for=oldplan
            receipt['rows'].append(dict(task=task,updates=records,reverse_step_seconds=times,exact_load=True,exact_sampling_load=True,
                frozen_Xt_feature_calls=len(observed),feature_seconds=observed[:],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            observed.clear();expected=None;del m,o,e,r,b,chain;gc.collect();torch.cuda.empty_cache();save(folder/'receipt.json',receipt)
        receipt['passed']=True
    except BaseException:
        receipt['error']=traceback.format_exc();raise
    finally:receipt['seconds']=time.time()-began;save(folder/'receipt.json',receipt)
if __name__=='__main__':main()
