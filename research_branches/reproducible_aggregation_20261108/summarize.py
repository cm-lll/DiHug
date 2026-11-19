"""CPU-only receipt/table and artifact-binding export; no model imports."""
import json,csv,hashlib,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/reproducible_aggregation_20261108T113027'
run=OUT/'attempt3';z=json.loads((run/'receipt.json').read_text())
leases=[json.loads(p.read_text()) for p in [OUT/'lease.json',OUT/'attempt2/lease.json',run/'lease.json']]
summary=dict(status=z['status'],successful_panel_updates=z['updates'],failed_attempt_updates=3,total_actual_optimizer_updates=z['updates']+3,
             gpu_hours=sum(p['gpu_hours'] for p in leases),start_epoch=1794108627,compute_end_epoch=max(p['end_epoch'] for p in leases),
             elapsed_until_summary_minutes=(time.time()-1794108627)/60,timing=[],repeats=[],short=[],failures=[])
for ds in ['IMDB','PubMed']:
    for kind in ['CE','PS']:
        for mode in ['ordinary','strict']:
            rows=[r for r in z['rows'] if (r['dataset'],r['kind'],r['mode'])==(ds,kind,mode)]
            if rows:summary['timing'].append(dict(dataset=ds,kind=kind,mode=mode,n=len(rows),first_seconds=rows[0]['seconds'],later_mean_seconds=statistics.mean(r['seconds'] for r in rows[1:]) if len(rows)>1 else None,max_reserved_MiB=max(r['peak_reserved_MiB'] for r in rows)))
    for mode in ['ordinary','strict']:
        rows=[r for r in z['short'] if (r['dataset'],r['mode'])==(ds,mode)]
        comparisons=[r for r in rows if r['repeat']==1]
        summary['short'].append(dict(dataset=ds,mode=mode,updates=len(rows),repeat_steps=len(comparisons),exact_all_state_steps=sum(all(r['exact_vs_continuous'].values()) for r in comparisons),mean_seconds=statistics.mean(r['seconds'] for r in rows) if rows else None,
                                    exact_rng_steps=sum(r['exact_vs_continuous']['rng'] for r in comparisons)))
for row in z['comparisons']:
    summary['repeats'].append(row)
for path in [OUT/'receipt.json',OUT/'attempt2/receipt.json']:
    q=json.loads(path.read_text());summary['failures'].append(dict(path=str(path),error=q['error'],recorded_completed_updates=q['updates']))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for name,rows in [('timing',summary['timing']),('short_tracks',z['short']),('updates',z['rows'])]:
    if not rows:continue
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/(name+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()})
files=list(Path(__file__).parent.glob('*.py'))+list(OUT.rglob('*.pt'))
manifest={}
for p in files:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    manifest[str(p)]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
(OUT/'artifact_binding.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='repeats'},ensure_ascii=False,indent=2))
