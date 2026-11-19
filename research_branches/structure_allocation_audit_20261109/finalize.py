"""Saved-table interpretation and binding checks only; no models or feature rerun."""
import json,hashlib,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/structure_allocation_audit_20261109'
DATA=OUT/'complete'
def main():
    start=time.monotonic();binding=json.loads((DATA/'binding.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
    source=ROOT/'results/rooted_connection_postmortem_20261109/graphs.json'
    graphs=json.loads(source.read_text());rows=json.loads((DATA/'final_edges.json').read_text());ex=json.loads((DATA/'exchanges.json').read_text())
    keys=['dataset','train_seed','arm','sample_seed'];key=lambda r:tuple(r[k] for k in keys)
    index={key(g):g for g in graphs};assert len(index)==64
    assert all(index[key(r)]['edges']==r['graph_edges'] for r in rows)
    output=[]
    for ds in ['IMDB','PubMed']:
        gg=[g for g in graphs if g['dataset']==ds]
        out=dict(dataset=ds,graphs=len(gg),over_triangle=sum(g['triangles']>g['reference_triangles'] for g in gg),
            exact_triangle=sum(g['triangles']==g['reference_triangles'] for g in gg),under_triangle=sum(g['triangles']<g['reference_triangles'] for g in gg))
        for op in ['add','delete']:
            rr=[r for r in rows if r['dataset']==ds and r['operation']==op and r['explicit_change'] < -1e-12 and r['triangle_error_change']>1e-12]
            out[op+'_triangle_cost']=len(rr)
            for label,sgn in [('over',1),('under',-1)]:
                out[op+'_triangle_cost_from_'+label]=sum((index[key(r)]['triangles']-index[key(r)]['reference_triangles'])*sgn>0 for r in rr)
        output.append(out)
    good=[r for r in ex if r['explicit_change'] < -1e-12]
    interpretation=dict(triangle_direction=output,
        exchange_case=dict(dataset='PubMed',train_seed=2026110421,arm='A',sample_seed=2026110232,relation=1,added_slot=3489,
            exchanges=len(ex),rewarded=len(good),TV_worse_but_relationW1_triangle_hub_nonworse=sum(r['joint_degree_TV_change']>1e-12 and all(r[m+'_change']<=1e-12 for m in ['relation_degree_W1','triangle_error','hub_error']) for r in good),
            limitation='single previously selected GD case, no IMDB or multi-seed exchange coverage'),
        gaps=['No model gradient or new generation test','No calibrated alternative objective','Small relation strata, 171 nonempty graph/relation/direction groups unselected','Positive associations do not prove double counting','No new independent states or references'],
        decision='No automatic square-loss or weight change. Prioritize whether joint-degree TV reflects geometric allocation distance on existing exchanges; keep prior shared-matching failures as controls.',
        resource=json.loads((DATA/'receipt.json').read_text()),
        errors='Three pre-completion engineering errors retained separately; science succeeded in complete/; official assets unchanged',
        graph_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),bound_files_reverified=len(binding))
    (OUT/'interpretation.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2)+'\n')
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete_stopped',models=0,GPU_hours=0,
        formal_training_authorized=False,finalize_seconds=time.monotonic()-start,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'REPORT_ZH.md',OUT/'interpretation.json',OUT/'engineering_notes.json',DATA/'receipt.json']}),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(interpretation,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
