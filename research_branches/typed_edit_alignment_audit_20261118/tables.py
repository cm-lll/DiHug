"""Export all seeds and descriptive mean/sample-SD; starts are not fit repeats."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/typed_edit_alignment_audit_20261118'
PILOT=OUT/'progress_filter_pilot_128'


def write(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert json.loads((PILOT/'acceptance.json').read_text())['passed']
    raw=json.loads((PILOT/'per_output.json').read_text())
    rows=[]
    for ds in ['freebase','delicious']:
        for seed in [100,101]:
            group=[r for r in raw if r['dataset']==ds and r['start_seed']==seed]
            source=group[0]
            zero=source['initial_metrics']
            rows.append(dict(dataset=ds,start_seed=seed,arm='zero',steps=0,
                primary_error=zero['primary_error'],degree_residual=source['initial_error_per_node'],
                lcc_fraction=zero['lcc_fraction'],connected=zero['connected'],arrived=False,
                fitting_seed=None,new_training_fits=0,condition='saved_10k_prefix_first_stage_only'))
            for r in group:
                rows.append(dict(dataset=ds,start_seed=seed,arm=r['arm'],steps=r['steps'],
                    primary_error=r['primary_error'],degree_residual=r['final_error_per_node'],
                    lcc_fraction=r['lcc_fraction'],connected=r['connected'],arrived=r['arrived'],
                    fitting_seed=(810 if ds=='freebase' else 710) if r['arm']!='filtered_N' else None,
                    new_training_fits=0,condition='saved_10k_prefix_first_stage_only'))
    write('pilot_per_seed.csv',rows)
    summary=[]
    for ds in ['freebase','delicious']:
        for arm in ['zero','original_M','filtered_M','filtered_N']:
            rr=[r for r in rows if r['dataset']==ds and r['arm']==arm]
            r=dict(dataset=ds,arm=arm,starting_states=2,new_training_fits=0,
                sd_unit='two_starting_states_not_independent_training_fits',
                connected=sum(r['connected'] for r in rr),arrived=sum(r['arrived'] for r in rr))
            for k in ['primary_error','degree_residual','lcc_fraction']:
                r[k+'_mean']=statistics.mean(x[k] for x in rr)
                r[k+'_sample_sd']=statistics.stdev(x[k] for x in rr)
            summary.append(r)
    write('pilot_mean_sd.csv',summary)
    (OUT/'delivery.json').write_text(json.dumps(dict(
        fixed_state_audit='cpu_v2',short_pilot='progress_filter_pilot_128',
        completed=True,production_method_changed=False,adopted_candidate=False,
        scientific_new_training_fits=0,scientific_new_short_suffix_graphs=12,
        scientific_model_calls=2672,engineering_model_calls=70,total_model_calls=2742,
        gpu_calls=0,formal_training_started=False,
        failed_attempts=['failure.json','progress_filter_pilot/failure.json'],
        independent_scorer_increment_established=False,
        next_decision='Second-stage deployment-matched inverse-ranking design; no automatic long training',
        tables=['pilot_per_seed.csv','pilot_mean_sd.csv']),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
