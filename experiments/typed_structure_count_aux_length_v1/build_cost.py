"""Read-only cost extrapolation, never activates a run or grants approval."""
import argparse,pathlib,json,datetime,math
from runtime_contract import ROOT,PROTOCOL,jobs,source_binding,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--evidence',required=True);a=ap.parse_args()
    project=pathlib.Path(a.project);ev=pathlib.Path(a.evidence)
    history=project/'results/typed_structure_early_cn_expansion_formal_v1_20261029T1242/run_retry1'
    rates={};components={}
    for d,seed in [('IMDB',2026102931),('PubMed',2026102932)]:
        folder=history/f'{d}_s{seed}_E_earlyCN';receipt=json.loads((folder/'summary.json').read_text())
        hist=(receipt['finished_epoch']-receipt['started_epoch'])/3000
        probe=json.loads((ev/f'engineering_{d}.json').read_text())
        # Initial loss form used these updates; final formula has same graph and
        # adds only a per-family denominator. No final-form updated GPU timing.
        new=max(x['wall_seconds'] for x in probe['updates'] if x['arm']=='CE_COUNT')
        rate=max(hist,new)*1.02
        chains=[json.loads(p.read_text())['seconds'] for p in folder.glob('ema_0.99/sample_*/binding.json')]
        rates[d]=(rate,max(chains));components[d]=dict(historical_update_with_validation_seconds=hist,engineering_aux_update_max_seconds=new,additional_unmeasured_formula_allowance=1.02,planned_update_seconds=rate,historical_chain_seconds=chains,planned_chain_seconds=max(chains),evidence=str(folder))
    alternatives={}
    for steps in [9000,6000]:
        rows=[]
        for task in PROTOCOL['tasks']:
            rate,chain=rates[task['dataset']];stages=steps//3000
            # Per trajectory extra validations, CPU snapshots, readout swaps and IO.
            extra=240 if steps==9000 else 180
            rows.append(dict(**task,updates=steps,chains=stages*2,training_seconds=rate*steps,sampling_seconds=chain*stages*2,extra_validation_save_io_seconds=extra,serial_seconds=rate*steps+chain*stages*2+extra))
        compute=max(x['serial_seconds'] for x in rows)
        planned=compute*1.20+15*60+20*60
        proposed=780*60 if steps==9000 else 540*60
        alternatives[str(steps)]=dict(queue=rows,total_updates=8*steps,total_chains=8*(steps//3000)*2,compute_wall_seconds=compute,concurrency_planning_multiplier=1.20,concurrency_multiplier_measured=False,fault_reserve_seconds=900,evaluation_audit_publication_reserve_seconds=1200,planned_seconds=planned,requested_budget_seconds=proposed,fits_proposal=planned<proposed,task_compute_seconds_sum=sum(x['serial_seconds'] for x in rows))
    uuids=['GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7','GPU-cf73267c-774a-dada-fc26-ada91ccc27c1','GPU-dda1721a-d053-7371-3434-4c3c8502a1d4','GPU-994be766-a8bd-b90c-bf87-c667dfa8ab4d','GPU-e2284395-dd00-1e74-3eed-7f406fa69d5c','GPU-2db6cf6a-a8b5-58af-4e89-22bfc863a0fb','GPU-cddbf2e5-1af7-7219-ff58-38e6f4c44f69','GPU-360156ae-7a75-37f4-eaed-9a5500275409']
    value=dict(status='passed' if all(x['fits_proposal'] for x in alternatives.values()) else 'insufficient',formal_authorized=False,source_sha256=source_binding()[1],schedule=jobs(),proposed_gpu_uuids=uuids,components=components,alternatives=alternatives,**{k:alternatives['9000'][k] for k in ('planned_seconds','requested_budget_seconds')},required_free_bytes=32*2**30,
        disk_estimate=dict(checkpoints_80_at_111MB=80*111000000,readouts_24_at_28MB=24*28000000,logs_graphs_audit_allowance_bytes=4*2**30,required_free_with_reserve_bytes=32*2**30),
        limitations=['historical full chains only; new auxiliary-driven states may differ in cost','no 9000-step or eight-card new-candidate throughput measured','8 engineer optimizer calls used rejected initial loss normalization; final scale/backward no steps','formal start/deadline not yet activated'])
    zone=datetime.timezone(datetime.timedelta(hours=8));example=datetime.datetime(2026,10,29,20,0,tzinfo=zone)
    value['illustrative_timetable']={k:dict(start=example.isoformat(),science_stop=(example+datetime.timedelta(seconds=v['requested_budget_seconds']-1200)).isoformat(),delivery=(example+datetime.timedelta(seconds=v['requested_budget_seconds'])).isoformat()) for k,v in alternatives.items()}
    value['latest_start_for_09_delivery_9000']='2026-10-29T20:00:00+08:00 (planning only, no guarantee)'
    save(ROOT/'cost.json',value);print(json.dumps({k:{q:v[q] for q in ['total_updates','total_chains','compute_wall_seconds','planned_seconds','requested_budget_seconds']} for k,v in alternatives.items()},indent=2))
if __name__=='__main__':main()
