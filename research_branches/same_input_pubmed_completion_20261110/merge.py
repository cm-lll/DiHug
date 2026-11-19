"""Merge prior IMDB and new PubMed records; no model calls or rescoring."""
import csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/same_input_update_check_20261110';OUT=ROOT/'results/same_input_pubmed_completion_20261110'
def main():
    assert json.loads((OLD/'audit.json').read_text())['passed'];assert json.loads((OUT/'audit.json').read_text())['passed']
    for name in ('risk_changes.csv','evaluation_summary.csv','relations.csv','node_groups.csv'):
        rows=[]
        for path,ds in [(OLD,'IMDB'),(OUT,'PubMed')]:rows.extend(x for x in csv.DictReader((path/name).open()) if x['dataset']==ds)
        with (OUT/('combined_'+name)).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    models=[]
    for path,ds in [(OLD,'IMDB'),(OUT,'PubMed')]:models.extend(x for x in json.loads((path/'decision_summary.json').read_text())['models'] if x['dataset']==ds)
    assert len(models)==4
    for m in models:
        m['same_change_percent']=100*(m['same_after']/m['same_before']-1)
        m['other_mean_before']=sum(x['before'] for x in m['other_times'])/6;m['other_mean_after']=sum(x['after'] for x in m['other_times'])/6
        m['other_harmed_mean_times']=[x['t'] for x in m['other_times'] if x['after']-x['before']>1e-7]
    result=dict(models=models,model_calls=0,old_IMDB_reused=True,total_panel_updates=4,total_panel_forwards=32,total_panel_VJPs=20,new_updates=2,new_forwards=16,new_VJPs=10,previous_GPU_accounting_remains_incomplete=True,scope='Conditional clean-prediction single-step diagnostic; not generation or root-cause proof for later updates')
    (OUT/'combined_summary.json').write_text(json.dumps(result,indent=2))
    for m in models:print(json.dumps({k:v for k,v in m.items() if k!='same_details'},indent=2))
if __name__=='__main__':main()
