"""CPU-only extraction of all interventions, time points and relations."""
import json,csv,sys
from pathlib import Path
OUT=Path(sys.argv[1]);results=json.loads((OUT/'results.json').read_text());evaluations=json.loads((OUT/'evaluations.json').read_text())
def export(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows(rows)
export('evaluations_total.csv',[{k:v for k,v in z.items() if k!='relations'} for z in evaluations])
export('evaluations_relations.csv',[dict(**{k:z[k] for k in ['condition','dataset','parent_branch','arm','t','panel','input_sha256','completed']},**r) for z in evaluations for r in z['relations']])
paired=[];geometry=[];response=[];training=[]
for r in results:
    for t in [1,500,1000]:
        ce=r['outcomes']['CE']['responses'][str(t)];ps=r['outcomes']['PS']['responses'][str(t)];base=r['base'][str(t)]
        row=dict(condition=r['condition'],dataset=r['dataset'],parent_branch=r['parent_branch'],training_t=r['training_t'],evaluation_t=t)
        for metric in ['CE','positive_CE','negative_CE','expected_edges']:
            row[metric+'_before']=base[metric];row[metric+'_CE']=ce[metric];row[metric+'_PS']=ps[metric];row[metric+'_PS_minus_CE']=ps[metric]-ce[metric]
        paired.append(row)
    for group,metrics in r['gradient_geometry'].items():geometry.append(dict(condition=r['condition'],group=group,**metrics))
    for metric,branches in r['local_response'].items():
        for branch,values in branches.items():response.append(dict(condition=r['condition'],metric=metric,branch=branch,**values))
    for arm,outcome in r['outcomes'].items():
        row=dict(condition=r['condition'],arm=arm,t=r['training_t'],seconds=outcome['seconds'],gradient_sum_max_abs=outcome['gradient_sum_max_abs'])
        for metric in ['CE','positive_CE','negative_CE']:
            row[metric+'_before']=r['train_before'][metric];row[metric+'_after']=outcome['training_after'][metric];row[metric+'_change']=row[metric+'_after']-row[metric+'_before']
        training.append(row)
export('paired_response.csv',paired);export('parameter_group_geometry.csv',geometry);export('linear_response.csv',response);export('current_training_response.csv',training)
summary=[]
for r in results:
    summary.append(dict(condition=r['condition'],training_t=r['training_t'],current_CE_aux_cosine=r['current_CE_aux_cosine'],aux_over_CE_gradient_L2=r['aux_over_CE_gradient_L2'],
                        total=r['gradient_geometry']['all'],response=r['local_response'],moments=r['moment_differences']))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
