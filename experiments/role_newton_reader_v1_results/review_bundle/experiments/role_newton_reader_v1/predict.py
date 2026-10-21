"""Optional same Xt fixed-weight prediction using validated saved Newton allocations."""
import argparse,sys,time,resource,traceback
import numpy as np
from prepare import OUT,HERE,SOURCE,read,sha,save,check,old

def run(seed):
    out=OUT/f'prediction_seed{seed}';out.mkdir();started=time.perf_counter();status='failed';reason=None;rows=[]
    try:
        assert read(OUT/'audit.json')['status']=='passed'
        records,targets=old.c.load_data();case=old.c.metadata_case(old.c.base,old.c.PANEL,targets);model,binding=old.get_model(seed,case)
        save(out/'binding.json',binding)
        allrows=read(OUT/'small_partial.json');oldcheck=old.check;oldrequired=old.required
        for name in old.c.read(old.c.RESULT/'conditions.json')['final_targets']:
            for t in (1,8,32,64):
                check();group=[x for x in allrows if x['input']['seed']==seed and x['input']['condition']==name and x['input']['t']==t]
                assert len(group)==4;group.sort(key=lambda x:x['input']['type'])
                if not all(x['result'].get('new_contract_pass') for x in group):
                    rows.append(dict(seed=seed,condition=name,t=t,status='not_run_reader_rejected'));continue
                meta=group[0]['input'];path=HERE.parent/meta['source_payload'];assert sha(path)==meta['source_sha256']
                with np.load(path) as z:data={k:z[k].copy() for k in z.files}
                allocations=[];counter=[0];values=[]
                for x in group:
                    assert sha(OUT/x['output_path'])==x['output_sha256']
                    with np.load(OUT/x['output_path']) as z:allocations.append(z['P'].copy())
                def cached(L,c,ignored_check):
                    j=counter[0];counter[0]+=1
                    np.testing.assert_array_equal(L,data[f'type{j}_LL']);np.testing.assert_array_equal(c,data[f'type{j}_counts'])
                    return allocations[j].copy(),group[j]['result'].copy()
                def value_hook(module,args,v):values.append(v.detach().cpu().numpy().copy())
                torch_rng=old.c.torch.get_rng_state().clone();numpy_rng=np.random.get_state()
                old.required=cached;old.check=check;hook=model.branch.value.register_forward_hook(value_hook)
                try:prediction=old.predict(model,case,data['state'],targets[name],t,2026102031,'scaled')
                finally:hook.remove();old.required=oldrequired;old.check=oldcheck
                assert old.c.torch.equal(torch_rng,old.c.torch.get_rng_state())
                after=np.random.get_state();assert numpy_rng[0]==after[0] and np.array_equal(numpy_rng[1],after[1]) and numpy_rng[2:]==after[2:]
                steps=read(old.c.RESULT/f'jobs/broad_{seed}/generated/{name}_2026102031_steps.json')
                assert prediction['query_sha256']==next(x['query_sha256'] for x in steps if x['t']==t)
                assert counter[0]==4 and len(values)==4
                checks=[]
                for typ,x in enumerate(group):
                    P=allocations[typ];V=values[typ];ix=case['space'].types==typ
                    mass=np.array(x['result']['row_omitted_mass_upper'])
                    tail=mass*np.max(abs(V));f32=P.astype(np.float32).astype(float)
                    conversion=(abs(f32-P)@abs(V.astype(float))).max(1)
                    k=P.shape[1];u=2**-24;gamma=k*u/(1-k*u)
                    multiplication=(gamma*(abs(f32)@abs(V.astype(float)))).max(1)+k*np.finfo(np.float32).tiny
                    actual=abs(prediction['context'][ix]-(P@V.astype(float))).max(1)
                    assert np.all(actual<=conversion+multiplication+1e-12)
                    checks.append(dict(type=typ,tail_context_bound=float(tail.max(initial=0)),
                       fp32_conversion_bound=float(conversion.max()),fp32_product_bound=float(multiplication.max()),observed_context_error=float(actual.max())))
                families=np.repeat(np.arange(10),case['space'].counts);delta=[]
                for rel in range(10):
                    ix=families==rel;new=prediction['q'][ix];reference=data['q_exact'][ix];st=data['state'][ix]
                    delta.append(dict(family=rel,p0_max_abs=float(abs(prediction['p0'][ix]-data['p0_exact'][ix]).max(initial=0)),
                        q_max_abs=float(abs(new-reference).max(initial=0)),expected_edge_delta=float(new.sum()-reference.sum()),
                        birth_delta=float(new[st==0].sum()-reference[st==0].sum()),death_delta=float((1-new[st==1]).sum()-(1-reference[st==1]).sum())))
                target=out/f'{name}_t{t}.npz';np.savez_compressed(target,p0=prediction['p0'],q=prediction['q'],role_context=prediction['context'])
                rows.append(dict(seed=seed,condition=name,t=t,status='completed',query_sha256=prediction['query_sha256'],
                    source_sha256=meta['source_sha256'],output_sha256=sha(target),relations=delta,context_error_bounds=checks,
                    neural_seconds_with_cached_certified_allocation=prediction['seconds'],mode='Newton allocation; original network/kernel; no draw'))
                save(out/'rows.json',rows)
        status='completed'
    except BaseException as exc:reason=repr(exc);traceback.print_exc()
    finally:save(out/'receipt.json',dict(status=status,reason=reason,rows=len(rows),seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,training=0,new_chains=0))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);sys.exit(run(p.parse_args().seed))
