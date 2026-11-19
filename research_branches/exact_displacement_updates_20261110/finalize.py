"""Archive receipts and hashes only; never runs a model."""
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_displacement_updates_20261110'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    supervisor=json.loads((OUT/'supervisor.json').read_text());audit=json.loads((OUT/'audit.json').read_text());assert supervisor['status']=='completed' and audit['passed']
    manifests=[]
    for root in (OUT,ROOT/'research_branches/exact_displacement_updates_20261110'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in str(p) and p.name not in ('artifact_manifest.json','delivery_receipt.json'):manifests.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifests,indent=2))
    receipts=[json.loads(p.read_text()) for p in OUT.glob('*/receipt.json')];window=json.loads((OUT/'window.json').read_text());end=time.time()
    result=dict(status='completed_stopped',archive_epoch=end,wall_seconds_from_GPU_supervisor=end-supervisor['start_epoch'],GPU_hours=supervisor['GPU_hours'],CPU_evaluation_audit_seconds=audit['seconds'],temporary_updates=sum(x['updates'] for x in receipts),forwards=sum(x['forwards'] for x in receipts),VJPs=sum(x['backwards'] for x in receipts),peak_GPU_allocated_MiB=max(x['peak_GPU_allocated_MiB'] for x in receipts),peak_RSS_MiB=max(x['peak_RSS_MiB'] for x in supervisor['tasks']),all_GPU_workers_exited=supervisor['all_workers_exited'],GPU_apps_after=supervisor['after_apps'],within_science_deadline=supervisor['end_epoch']<window['science_stop_epoch'],within_delivery_deadline=end<window['delivery_stop_epoch'],GPU_limit_met=supervisor['GPU_hours']<=.25,formal_training=False,full_reverse_chains=0,baseline_commit='df5c8898d91422b5fced712e718fc8f32290de73',new_sources='isolated uncommitted sources identified by SHA256; no official source replacement',ordinary_CUDA_recomputation_bitwise_certified=False,next_experiment_started=False)
    assert result['within_science_deadline'] and result['within_delivery_deadline'] and result['GPU_limit_met']
    (OUT/'delivery_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
