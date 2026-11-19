"""Summaries of completed CPU decomposition; no new scientific samples."""
import csv,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/density_arrangement_audit_20261110'
def main():
    rows=list(csv.DictReader((OUT/'decomposition.csv').open()));results=[]
    for ds,seed in sorted(set((r['dataset'],r['seed']) for r in rows)):
        for alpha in (.25,.5,1.):
            for metric in ('relation_count','present_triangles','absent_triangles'):
                selected=[r for r in rows if r['dataset']==ds and r['seed']==seed and float(r['alpha'])==alpha and r['metric']==metric and r['panel']!='same']
                results.append(dict(dataset=ds,seed=int(seed),alpha=alpha,metric=metric,times=len(selected),**{k:sum(float(r[k]) for r in selected)/len(selected) for k in ('total_change','density_part','arrangement_part','interaction')},harmful_times=[r['panel'] for r in selected if float(r['total_change'])>1e-10],arrangement_sign_flips_between_orders=[r['panel'] for r in selected if float(r['arrangement_at_before_count'])*float(r['arrangement_at_after_count'])<0]))
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))
    binding=json.loads((OUT/'bindings.json').read_text());prior=json.loads((ROOT/'results/displacement_response_20261110/bindings.json').read_text());old=json.loads((ROOT/'results/same_input_pubmed_completion_20261110/bindings.json').read_text())
    compared=0
    for p,h in binding.items():
        if p in prior:assert h==prior[p];compared+=1
        if p in old:assert h==old[p];compared+=1
    audit=json.loads((OUT/'audit.json').read_text());audit['historical_matching_hash_entries']=compared;(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    paths=list(OUT.glob('*'))+list((ROOT/'research_branches/density_arrangement_audit_20261110').glob('*.py'))+list((ROOT/'research_branches/density_arrangement_audit_20261110').glob('*.md'))
    manifest={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths if p.is_file() and p.name!='artifact_manifest.json'}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(summary_rows=len(results),historical_matching_hash_entries=compared,artifacts=len(manifest))))
if __name__=='__main__':main()
