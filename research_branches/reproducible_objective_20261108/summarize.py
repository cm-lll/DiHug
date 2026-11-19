"""CPU-only paired table export; no model loading or inference."""
import csv, json, sys
from pathlib import Path
OUT=Path(sys.argv[1])
rows=json.loads((OUT/'probes.json').read_text())
updates=json.loads((OUT/'updates.json').read_text())
def dump(name,items):
    keys=list(dict.fromkeys(k for row in items for k in row))
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(items)
totals=[{k:v for k,v in row.items() if k!='relations'} for row in rows]
relations=[dict(**{k:row[k] for k in ['dataset','seed','arm','offset','completed','readout','panel','t']},**{k:v for k,v in rel.items() if k!='types'},endpoint_type_ids=json.dumps(rel['types'])) for row in rows for rel in row['relations']]
dump('validation_total.csv',totals);dump('validation_relations.csv',relations);dump('training_updates.csv',updates)
def compare(items,group_keys,metrics):
    index={tuple(row[k] for k in group_keys)+(row['arm'],):row for row in items}
    output=[]
    for row in items:
        if row['arm']!='PS':continue
        key=tuple(row[k] for k in group_keys)
        control=index.get(key+('CE',))
        if control is None:continue
        result={k:row[k] for k in group_keys}
        for metric in metrics:
            a,b=row[metric],control[metric]
            result[metric+'_PS']=a;result[metric+'_CE']=b
            result[metric+'_PS_minus_CE']=None if a is None or b is None else a-b
            result[metric+'_relative']=None if a is None or b is None or abs(b)<1e-12 else (a-b)/abs(b)
        output.append(result)
    return output
total_diff=compare(totals,['dataset','seed','offset','readout','panel','t'],['CE','positive_CE','negative_CE','expected_edges'])
rel_diff=compare(relations,['dataset','seed','offset','readout','panel','t','family'],['positive_CE','negative_CE','expected_edges','missed_positive_mass','false_positive_mass','expected_relation_degree_MAE'])
dump('paired_total.csv',total_diff);dump('paired_relations.csv',rel_diff)
summary=[]
for ds in ['IMDB','PubMed']:
    for readout in ['raw','EMA0.99']:
        rr=[r for r in total_diff if r['dataset']==ds and r['offset']==32 and r['readout']==readout]
        for panel in ['historical','new_diagnostic']:
            subset=[r for r in rr if r['panel']==panel]
            for r in subset:
                rel=[q for q in rel_diff if all(q[k]==r[k] for k in ['dataset','seed','offset','readout','panel','t'])]
                summary.append(dict(**r,relation_degree_MAE_improved=sum(q['expected_relation_degree_MAE_PS_minus_CE']<0 for q in rel if q['expected_relation_degree_MAE_PS_minus_CE'] is not None),
                                    relation_degree_MAE_worsened=sum(q['expected_relation_degree_MAE_PS_minus_CE']>0 for q in rel if q['expected_relation_degree_MAE_PS_minus_CE'] is not None)))
(OUT/'endpoint_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
dump('endpoint_summary.csv',summary)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for col,ds in enumerate(['IMDB','PubMed']):
        for rr,metric in enumerate(['CE','positive_CE','negative_CE']):
            ax=axes[rr,col]
            for t,color in [(1,'#2185d0'),(500,'#db6d28'),(1000,'#29854b')]:
                for arm,style in [('CE','-'),('PS','--')]:
                    selected=[x for x in totals if x['dataset']==ds and x['readout']=='raw' and x['panel']=='historical' and x['t']==t and x['arm'] in ['shared_start',arm]]
                    selected.sort(key=lambda x:x['offset'])
                    ax.plot([x['offset'] for x in selected],[x[metric] for x in selected],style,marker='o',color=color,label=f'{arm} t={t}')
            ax.set_title(ds+' '+metric);ax.set_xlabel('temporary updates from shared PS parent');ax.set_yscale('symlog',linthresh=1e-6);ax.grid(alpha=.2)
            if rr==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Diagnostic raw CE; existing fixed noisy inputs; not generation quality')
    fig.tight_layout();fig.savefig(OUT/'raw_probe_trends.png',dpi=160);plt.close(fig)
except ImportError as exc:
    (OUT/'plot_unavailable.json').write_text(json.dumps(dict(error=str(exc))))
print(json.dumps(summary,ensure_ascii=False))
