"""Archive partial completion without fabricating a supervisor receipt."""
import json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/same_input_update_check_20261110'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());progress=json.loads((OUT/'supervisor_progress.json').read_text());window=json.loads((OUT/'window.json').read_text());audit=json.loads((OUT/'audit.json').read_text());assert audit['passed']
    done=[];missing=[];lower=progress['GPU_seconds'];complete_indices={x['index'] for x in progress['tasks']}
    for i,row in enumerate(protocol['tasks']):
        folder=OUT/f"{row['dataset']}_{row['seed']}";p=folder/'receipt.json'
        if not p.exists():missing.append(dict(dataset=row['dataset'],seed=row['seed'],status='not_started'));continue
        z=json.loads(p.read_text());assert z['status']=='completed';done.append(dict(dataset=row['dataset'],seed=row['seed'],**z))
        if i not in complete_indices:lower+=z['seconds']
    result=dict(status='partial_stopped',completed=done,missing=missing,updates=sum(x['updates'] for x in done),forwards=sum(x['forwards'] for x in done),VJPs=sum(x['backwards'] for x in done),GPU_hours_lower_bound=lower/3600,GPU_hours_exact=None,GPU_limit_certified=False,supervisor_final_receipt_missing=True,interruption_cause='unknown; tool session no longer available on continuation; no scientific exception in completed workers',host_readonly_check='no matching ticket Python processes and no GPU compute apps at continuation',original_window=window,archive_epoch=time.time(),on_time_delivery=False,window_not_reset=True,CPU_evaluation_audit_seconds=audit['seconds'],formal_training=False,full_reverse_chains=0,new_GPU_started_after_expiry=False,next_step_not_started=True)
    (OUT/'delivery_receipt.json').write_text(json.dumps(result,indent=2));manifest=[]
    for root in (OUT,ROOT/'research_branches/same_input_update_check_20261110'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_manifest.json':manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
