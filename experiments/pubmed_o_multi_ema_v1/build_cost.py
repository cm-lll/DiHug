"""Measured incremental EMA cost + sealed O costs; never activates a budget."""
import argparse,pathlib,json
from runtime_contract import ROOT,read,save,sha,jobs,source_binding,BUDGET,RESERVE
def estimate(preflight):
    history=read(ROOT/'initial_assets.json')['records'];p=preflight
    # Use second same-input update, not cold update1. Retain both raw timings.
    row=p['paired_updates'][1];ema_overhead=row['ema_wrapper_seconds']
    component=max(row['unchanged_normal2_component_seconds'],row['original_normal2_seconds'])
    # Engineering GPU was exclusive; old8-task measurements include real contention.
    per_update=max(component,max(x['training_seconds']/1000 for x in history))+ema_overhead
    chain=max(max(x['historical_two_chains_seconds']) for x in history)
    extra_validation=3*sum(x['seconds'] for x in p['validation']['ema'].values())
    save_load=3*(p['checkpoint_save_seconds']+p['restore_seconds'])
    historic_other=max(x['historical_total_elapsed_seconds']-x['training_seconds']-sum(x['historical_two_chains_seconds']) for x in history)
    critical=1000*per_update+8*chain+historic_other+extra_validation+save_load
    # Explicit small concurrency/state-cost planning allowance, not a measurement.
    allowance=.10*critical;planned=critical+allowance+RESERVE
    return dict(status='passed' if p['status']=='passed' and planned<=BUDGET else 'not_admitted',
                schedule=jobs(),per_update_seconds=per_update,measured_incremental_ema_seconds=ema_overhead,
                measured_warm_normal2_seconds=component,historical_max_chain_seconds=chain,
                extra_three_ema_validations_seconds=extra_validation,checkpoint_extra_conservative_seconds=save_load,
                historical_initialization_validation_save_seconds=historic_other,critical_compute_seconds=critical,
                unmeasured_planning_allowance_seconds=allowance,delivery_seconds=RESERVE,planned_seconds=planned,
                proposed_budget_seconds=BUDGET,source_sha256=source_binding()[1],
                limits='five engineering updates; no new chain measured. State-dependent cost and eight-task contention remain estimates; no completion guarantee.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    report=read(a.preflight)
    if report['source_sha256']!=source_binding()[1]:raise ValueError('preflight/source binding differs')
    c=estimate(report);c.update(preflight_path=str(pathlib.Path(a.preflight).resolve()),preflight_sha256=sha(a.preflight))
    save(out,c);print(json.dumps(c,indent=2))
if __name__=='__main__':main()
