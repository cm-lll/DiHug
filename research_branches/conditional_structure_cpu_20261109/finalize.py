"""Close CPU investigation without additional scientific computation."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/conditional_structure_cpu_20261109'
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bindings={}
for name in ['binding.json','attribution_binding.json']:bindings.update(read(name))
changed=[p for p,h in bindings.items() if sha(Path(p))!=h]
assert not changed,changed
receipt=read('receipt.json'); attribution=read('attribution_receipt.json')
decision=dict(status='complete',candidate_adopted=False,formal_training_authorized=False,
    finding='finite multi-mode witnesses pass; full fixed-edge audit shows substantial tradeoffs in both datasets',
    evidence='189 eight-summary adverse rewarded edits; 90 change reward sign only after residual/membership coordinates move',
    restriction='value attribution not production training causality; finite projections and smoothed conditionals do not identify arbitrary joint distributions',
    next_priority='define a reference-fixed joint-distribution objective before another candidate; no formula already approved or tested',
    new_GPU_model_calls=0,new_training_updates=0,new_generated_graphs=0)
(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
files=list((ROOT/'research_branches/conditional_structure_cpu_20261109').glob('*.py'))
files += [OUT/n for n in ['REPORT_ZH.md','THEORY_ZH.md','protocol.json','decision.json','tests.json']]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
completion=dict(status='complete',start_BJT=read('protocol.json')['start_BJT'],end_BJT=now.isoformat(),
    measured_CPU_body_seconds=receipt['wall_seconds']+attribution['wall_seconds'],
    peak_RSS_MiB=max(receipt['peak_RSS_MiB'],attribution['peak_RSS_MiB']),GPU_hours=0,
    changed_assets=changed,bindings_verified=len(bindings),failures=[],workers_finished=True,
    artifact_sha256={str(p):sha(p) for p in files})
(OUT/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2))
print(json.dumps(completion,ensure_ascii=False,indent=2))
