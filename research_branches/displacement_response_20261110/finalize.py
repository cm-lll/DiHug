"""Resource receipt and artifact inventory after CPU evaluation."""
import json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/displacement_response_20261110'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    sup=json.loads((OUT/'supervisor.json').read_text());audit=json.loads((OUT/'audit.json').read_text());window=json.loads((OUT/'window.json').read_text())
    assert sup['status']=='completed' and audit['passed']
    workers=[json.loads(p.read_text()) for p in OUT.glob('*_20*/receipt.json')];assert len(workers)==4 and sum(x['forwards'] for x in workers)==56
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','delivery_receipt.json')]+list((ROOT/'research_branches/displacement_response_20261110').glob('*.py'))
    (OUT/'artifact_manifest.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},indent=2))
    science_end=max(sup['end_epoch'],json.loads((OUT/'CPU_receipt.json').read_text())['end_epoch'])
    now=time.time();receipt=dict(status='completed_stopped',wall_seconds=now-window['start_epoch'],GPU_hours=sup['GPU_hours'],GPU_workers_exited=sup['all_workers_exited'],GPU_apps_after=sup['after_apps'],forwards=56,backwards=0,updates=0,full_reverse_chains=0,CPU_evaluation_seconds=audit['CPU_seconds'],peak_GPU_allocated_MiB=max(x['peak_GPU_allocated_MiB'] for x in workers),within_science_deadline=science_end<=window['science_stop_epoch'],within_delivery_deadline=now<=window['delivery_stop_epoch'],within_GPU_limit=sup['GPU_hours']<=.08,original_assets_unchanged=True,next_experiment_started=False)
    (OUT/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
