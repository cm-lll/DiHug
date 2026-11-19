"""Summarize saved response tables only; no model calls."""
import csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/displacement_response_20261110'
def main():
    rows=list(csv.DictReader((OUT/'evaluation_summary.csv').open()));risks=list(csv.DictReader((OUT/'risk_changes.csv').open()));models=[]
    for ds,seed in sorted(set((r['dataset'],r['seed']) for r in rows)):
        rr=[r for r in rows if r['dataset']==ds and r['seed']==seed];ss=[r for r in rr if r['panel']=='same'];rs=[r for r in risks if r['dataset']==ds and r['seed']==seed]
        item=dict(dataset=ds,seed=seed,current=ss,other_time_means=[],exact_costs=[],per_time=[])
        for a in (0.,.25,.5,1.):
            selected=[r for r in rr if r['panel']!='same' and float(r['alpha'])==a]
            item['other_time_means'].append(dict(alpha=a,mean=sum(float(r['mean_five']) for r in selected)/len(selected)))
            item['exact_costs'].append(dict(alpha=a,items=[dict(panel=r['panel'],metric=r['metric'],change=float(r['change'])) for r in rs if float(r['alpha'])==a and r['analytic']=='True' and float(r['change'])>1e-10]))
        for label in ('t1','t75','t250','t550','t800','t1000'):
            selected=[r for r in rr if r['panel']==label];item['per_time'].append(dict(panel=label,values={r['alpha']:float(r['mean_five']) for r in selected}))
        models.append(item)
    (OUT/'summary.json').write_text(json.dumps(models,indent=2))
    print(json.dumps(models,indent=2))
if __name__=='__main__':main()
