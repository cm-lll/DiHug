"""Supplement failed-state arrays, using verified original replay, never partial reader output in NN."""
import argparse,time,resource,sys,traceback
from diagnose import *

def main(seed):
    started=time.perf_counter();out=OUT/f'seed{seed}';receipts=[];records,targets=c.load_data()
    case=c.metadata_case(c.base,c.PANEL,targets);m,bind=get_model(seed,case)
    for name in c.read(c.RESULT/'conditions.json')['final_targets']:
        missing=[]
        for t in CFG['diagnostic_times']:
            p=out/f'{name}_t{t}.json'
            if p.exists() and c.read(p)['status']!='completed':missing.append(t)
        if not missing:continue
        source=c.RESULT/f'jobs/broad_{seed}/generated';expected=c.read(source/f'{name}_2026102031_steps.json')
        saved={64:(c.np.random.default_rng(2026102031).random(case['space'].total)<case['prior'].repeat(case['space'].counts)).astype(c.np.uint8)}
        def progress(state,steps,rng):
            x=steps[-1];old=expected[len(steps)-1]
            for k in ('t','query_sha256','uniform_sha256','rng_after','state_sha256'):assert x[k]==old[k]
            if x['t']-1 in missing:saved[x['t']-1]=state.copy()
        c.original_chain(m,case,targets[name],2026102031,c.base,check,progress)
        for t in missing:
            check();state=saved[t];exact=predict(m,case,state,targets[name],t,2026102031,'exact')
            arrays=dict(state=state,p0_exact=exact['p0'],q_exact=exact['q'],context_exact=exact['context']);rows=[]
            for typ,item in enumerate(exact['records']):
                L=item['ll'];counts=item['counts'];p,r=scale(L,counts,check)
                rows.append(dict(type=typ,scaled=r,exact=item['receipt'],
                    diagnostic_only_not_used_in_network=True,
                    row_TV_mean_to_DP=float(abs(p-item['attention']).sum(1).mean()/2) if p is not None else None))
                arrays[f'type{typ}_LL']=L;arrays[f'type{typ}_counts']=counts;arrays[f'type{typ}_exact']=item['attention']
                if p is not None:arrays[f'type{typ}_unaccepted_scaled']=p
            p=out/f'{name}_t{t}_failure_evidence.npz';assert not p.exists();c.np.savez_compressed(p,**arrays)
            receipt=dict(seed=seed,condition=name,t=t,state_sha256=c.base.sha(state.tolist()),
                all_original_step_hashes_verified=True,source_state='second read-only replay for missing failure payload; no new sample',
                allocations=rows,downstream_scaled_prediction_not_executed=True,original_failure_preserved=True)
            c.save(out/f'{name}_t{t}_failure_evidence.json',receipt);receipts.append(receipt)
        print(seed,name,'failure evidence saved',flush=True)
    c.save(out/'failure_payload_receipt.json',dict(status='completed',count=len(receipts),seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,additional_independent_samples=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)
