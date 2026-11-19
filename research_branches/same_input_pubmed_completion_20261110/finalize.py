"""Record new-window costs separately; preserve the old missing receipt."""
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/same_input_pubmed_completion_20261110'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    supervisor=json.loads((OUT/'supervisor.json').read_text());audit=json.loads((OUT/'audit.json').read_text());window=json.loads((OUT/'window.json').read_text());combined=json.loads((OUT/'combined_summary.json').read_text());assert supervisor['status']=='completed' and audit['passed'] and len(combined['models'])==4
    for p,h in json.loads((OUT/'bindings.json').read_text()).items():assert sha(Path(p))==h,p
    receipts=[json.loads(p.read_text()) for p in OUT.glob('*/receipt.json')];manifest=[]
    for root in (OUT,ROOT/'research_branches/same_input_pubmed_completion_20261110'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in str(p) and p.name not in ('artifact_manifest.json','delivery_receipt.json'):manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));end=time.time()
    result=dict(status='completed_stopped',archive_epoch=end,new_window_wall_seconds=end-window['start_epoch'],new_GPU_hours=supervisor['GPU_hours'],new_GPU_allocation_accounting_complete=True,previous_IMDB_GPU_accounting_still_incomplete=True,total_combined_GPU_hours_exact=None,new_updates=sum(x['updates'] for x in receipts),new_forwards=sum(x['forwards'] for x in receipts),new_VJPs=sum(x['backwards'] for x in receipts),total_panel_updates=4,total_panel_forwards=32,total_panel_VJPs=20,CPU_evaluation_seconds=audit['seconds'],peak_GPU_allocated_MiB=max(x['peak_GPU_allocated_MiB'] for x in receipts),peak_RSS_MiB=max(x['peak_RSS_MiB'] for x in supervisor['tasks']),all_GPU_workers_exited=supervisor['all_workers_exited'],after_GPU_apps=supervisor['after_apps'],within_science_deadline=supervisor['end_epoch']<window['science_stop_epoch'],within_delivery_deadline=end<window['delivery_stop_epoch'],GPU_limit_met=supervisor['GPU_hours']<=.05,formal_training=False,full_reverse_chains=0,old_results_unchanged=True,next_experiment_started=False)
    (OUT/'delivery_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
