"""Tabulate the completed, frozen CPU audit; no model or new graph computation."""
import csv,json,time
from pathlib import Path
from audit import ROOT,dump,sha

p=ROOT/'results/typed_score_cpu_20261110_attempt2'
rows=list(csv.DictReader((p/'batches.csv').open()))
parts=list(csv.DictReader((p/'joint_attribution_batches.csv').open()))
tasks=[]
for task in dict.fromkeys(r['task'] for r in rows):
    item=dict(task=task)
    for state in ('before','after'):
        selected=[r for r in rows if r['task']==task and r['state']==state]
        selected_parts=[r for r in parts if r['task']==task and r['state']==state]
        item[state]={k:sum(float(r[k]) for r in selected)/len(selected) for k in (
            'J','L','JL','assignment_gap','C','edges','triangles','components','LCC','isolates')}
        item[state]['joint_parts']={k:sum(float(r[k]) for r in selected_parts)/len(selected_parts) for k in (
            'J_at_joint','closed','eligible_open','unclosable')}
    pairs=[]
    for t in sorted(set(int(r['t']) for r in rows if r['task']==task and r['state']=='before')):
        a=next(r for r in rows if r['task']==task and r['state']=='after' and int(r['t'])==t)
        b=next(r for r in rows if r['task']==task and r['state']=='before' and int(r['t'])==t)
        pairs.append({k:float(a[k])-float(b[k]) for k in ('J','L','JL','C')})
    item['worse_times']={k:sum(x[k]>1e-12 for x in pairs) for k in ('J','L','JL','C')}
    tasks.append(item)
receipts={name:json.loads(path.read_text()) for name,path in (
    ('failed_loader',ROOT/'results/typed_score_cpu_20261110/receipt.json'),
    ('audit',p/'receipt.json'),('attribution',p/'attribution_receipt.json'),('watchdog',p/'watchdog.json'))}
dump(p/'summary.json',dict(tasks=tasks,receipts=receipts,
    failures=[r for r in rows if r['state']=='failure'],
    sample_unit='conditional hard clean predictions: 6 fixed times x 32 graphs per before/after model, not diffusion chains or independent trained models',
    old_assets_unchanged=68,production_changes=False,scientific_conclusion='more discriminative descriptions; no training efficacy evidence; local aggregation can hide closure cost and component reward remains flat on connected sample batches',
    next_only='CPU fixed old probability-displacement audit of separate path/closure/component risks and feedback; no new optimization or model calls without further authorization'))
dump(p/'delivery_manifest.json',{str(f):sha(f) for f in sorted(p.iterdir()) if f.is_file() and f.name!='delivery_manifest.json'})
print(json.dumps(dict(tasks=len(tasks),all_saved_graphs=receipts['audit']['graphs'],recorded_scientific_seconds=sum(receipts[k]['seconds'] for k in ('failed_loader','audit','attribution')),GPU_hours=0)))
