"""Archive the completed diagnostic without tensors or any new computation."""
import hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/real192_update_audit_20261111'
def read(n):return json.loads((OUT/n).read_text())
protocol=read('protocol.json');window=read('window.json');resource=read('same_input_resource_receipt.json')
receipts=[read(t['id']+'/receipt.json') for t in protocol['tasks']]
assert all(r['status']=='completed' and r['updates']==50 for r in receipts)
assert read('same_input_step/receipt.json')['status']=='completed'
assert all(r['saved_displacement_exact'] for r in read('same_input_step/results.json'))
assert resource['cumulative_GPU_hours']<1 and resource['after_apps']==''
now=time.time()
completion=dict(status='completed',start_epoch=window['start_epoch'],end_epoch=now,
  total_wall_minutes=(now-window['start_epoch'])/60,GPU_hours=resource['cumulative_GPU_hours'],
  temporary_training_updates=200,saved_gradient_optimizer_reconstructions=4,
  model_forwards=1024,main_backwards=800,additional_component_VJPs=560,
  reverse_generation_chains=0,new_formal_models=0,new_repair_implemented=False,
  original_assets_unchanged=True,all_ticket_GPU_processes_exited=True,
  recommendation='Develop one justified large-error risk-growth correction before another bounded real192 paired test; no formal training authorized.',
  limitations=['Ordinary CUDA recomputation diverged from historical numerical trajectories despite exact input/RNG pairing.',
               'Conditional clean predictions are not complete generated graphs.',
               'Observed post-spike momentum persistence does not identify the earliest cause of over-dense predictions.'])
(OUT/'completion.json').write_text(json.dumps(completion,indent=2))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
(OUT/'diagnostic_source_hashes.json').write_text(json.dumps(manifest,indent=2))
archive=OUT.with_name(OUT.name+'_review.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in HERE.glob('*.py'):z.write(p,p.relative_to(ROOT))
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.py','.log'):
            z.write(p,p.relative_to(ROOT))
print(json.dumps(completion));print(archive)
