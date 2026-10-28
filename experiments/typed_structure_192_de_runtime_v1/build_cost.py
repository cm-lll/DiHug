"""Proposal-only cost calculation. Never invokes training or sampling."""
import argparse,csv,json,math,pathlib

def estimate(proposal,engineering,old):
    rows={}
    for dataset in ['IMDB','PubMed']:
        baseline=[r for r in old if r['cell']==dataset+'_A']
        for arm in 'DE':
            label=dataset+'_'+arm;r=engineering[dataset]['cells'][label]
            if not r['passed'] or r['completed_steps']!=2:raise ValueError('incomplete combination short check')
            extra=max(x['refex_seconds']+x['NNLS_seconds'] for x in r['full_query_components'])
            train=max(r['normal2_seconds'],max(float(x['normal2_postwarmup_mean_seconds']) for x in baseline)+2*extra)
            # Historical chain time contains RNG/commit/I/O; component path does not.
            oldstep=max(float(x[k])/1000 for x in baseline for k in ['sample_1_seconds','sample_2_seconds'])
            step=max(r['sampling_component_seconds']+.01,oldstep+extra)
            rows[label]=dict(measured_two_update_max=r['normal2_seconds'],measured_forward_posterior_max=r['sampling_component_seconds'],
                measured_added_feature_encoding_max=extra,estimated_update_seconds=train,estimated_full_step_seconds=step,
                train_1000_seconds=1000*train,two_T1000_chains_seconds=2000*step,
                initialization_validation_save_allowance_seconds=120,estimated_worker_seconds=1000*train+2000*step+120,
                peak_allocated_mib=r['peak_allocated_mib'],peak_reserved_mib=r['peak_reserved_mib'],peak_rss_mib=r['peak_rss_mib'])
    queue=[dict(task,**rows[task['dataset']+'_'+task['arm']]) for task in proposal['queue']]
    if len(queue)!=8 or len({r['slot'] for r in queue})!=8:raise ValueError('eight devices/one task each required')
    critical=max(r['estimated_worker_seconds'] for r in queue)
    planned=critical*1.25+1200+900+300
    return dict(proposal_only=True,formal_authorized=False,queue=queue,cells=rows,
        estimated_sum_task_seconds=sum(r['estimated_worker_seconds'] for r in queue),
        estimated_eight_card_critical_path_seconds=critical,concurrency_planning_factor=1.25,
        measured_eight_card_DE_throughput=False,evaluation_delivery_reserve_seconds=1200,
        failure_reserve_seconds=900,unmeasured_state_reserve_seconds=300,
        planned_seconds_before_rounding=planned,suggested_budget_minutes=math.ceil(planned/300)*5,
        limits=['two engineering updates are not stable throughput',
                'single-step inputs are forward-noised states, not full free-generation trajectories',
                'new timing excludes sampling/commit; 0.01s component allowance and historical full-chain timings used',
                '25 percent factor is planning margin, not measured concurrent DE performance',
                'same queue must be used at execution; resource or budget changes require a new decision'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True);p.add_argument('--old-costs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    proposal=json.loads((pathlib.Path(__file__).parent/'proposal.json').read_text())
    engineering={d:json.loads((pathlib.Path(a.evidence)/('gpu_'+d+'_original_env')/'summary.json').read_text()) for d in ['IMDB','PubMed']}
    with open(a.old_costs) as f:old=list(csv.DictReader(f))
    result=estimate(proposal,engineering,old)
    with open(a.output,'x') as f:json.dump(result,f,indent=2,allow_nan=False)
