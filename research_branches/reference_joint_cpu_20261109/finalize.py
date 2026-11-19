"""Read-only component summary and final provenance; no scoring/model work."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reference_joint_cpu_20261109'
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bindings=read('binding.json');changed=[p for p,h in bindings.items() if sha(Path(p))!=h]
assert not changed,changed
rows=read('final_edges.json');metrics=['total_degree_W1','relation_degree_W1','joint_degree_TV','node_triangle_W1','root_path_W1','hub_error','triangle_error','edge_error'];eps=1e-12
summary=[]
for ds in ['IMDB','PubMed']:
    allrows=[r for r in rows if r['dataset']==ds]
    bad=[r for r in allrows if r['fixed_change']<-eps and all(r[m+'_change']>=-eps for m in metrics) and any(r[m+'_change']>eps for m in metrics)]
    keys=['fixed_degree_change','fixed_shape_change','fixed_occupation_change']
    summary.append(dict(dataset=ds,adverse_count=len(bad),
        negative_components={k:sum(r[k]<-eps for r in bad) for k in keys},
        largest_negative_component={k:sum(min(keys,key=lambda z:r[z])==k for r in bad) for k in keys},
        occupation_only_gain=sum(r[keys[2]]<-eps and r[keys[0]]>=-eps and r[keys[1]]>=-eps for r in bad),
        degree_CDF_improves_while_degree_W1_worsens=sum(r['fixed_degree_change']<-eps and r['total_degree_W1_change']>eps for r in allrows)))
(OUT/'component_summary.json').write_text(json.dumps(summary,indent=2))
decision=dict(status='complete',candidate_adopted=False,formal_training_authorized=False,
    conclusion='reference-fixed mass-preserving score resolves some old cases but not full-panel structural costs; PubMed low/mid-noise costs worsen',
    next='no further candidate automatically scheduled; do not repeat reference-fixing as an unexecuted next step or tune group weights',
    models=0,GPU_hours=0,updates=0,new_graphs=0)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
files=list((ROOT/'research_branches/reference_joint_cpu_20261109').glob('*.py'))
files += [OUT/n for n in ['REPORT_ZH.md','THEORY_ZH.md','protocol.json','decision.json','component_summary.json','tests.json']]
receipt=read('receipt.json')
completion=dict(status='complete',start_BJT=read('protocol.json')['start_BJT'],end_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    measured_CPU_body_seconds=receipt['wall_seconds'],peak_RSS_MiB=receipt['peak_RSS_MiB'],GPU_hours=0,
    workers_finished=True,changed_assets=changed,bindings_verified=len(bindings),failures=[],
    artifact_sha256={str(p):sha(p) for p in files})
(OUT/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2))
print(json.dumps(completion,ensure_ascii=False,indent=2))
