"""Bind frozen CPU proof, original science, four states, and finite budget."""
import json,hashlib,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_count_step_20261110'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
OUT.mkdir(exist_ok=False)
p=json.loads((ROOT/'results/hard_clean_zero_safe_20261110/protocol.json').read_text())
assert json.loads((ROOT/'results/count_adam_audit_cpu_20261110/tests.json').read_text())['passed']
for path,h in p['bindings'].items():assert sha(path)==h,path
for r in p['tasks']:r['diagnostic_variant']='exact_count_gradient_only'
for path in [*Path(__file__).parent.glob('*.py'),ROOT/'research_branches/count_adam_audit_cpu_20261110/run.py',
             ROOT/'results/count_adam_audit_cpu_20261110/tests.json']:
    p['bindings'][str(path)]=sha(path)
p.update(estimator='exact expected HARD count risk plus original exact absent triangles; unchanged LOO other terms',
    parent_protocol=str(ROOT/'results/hard_clean_zero_safe_20261110/protocol.json'),
    planned_calls=dict(forwards=52,backwards=20,temporary_updates=4,reverse_chains=0),
    expected_runtime_seconds=80,GPU_seconds_limit=540,wall_minutes_limit=15,
    purpose='local estimator substitution only; no quality gate, no long training or reverse chain',
    device_request='GPU-Node01 physical GPU0, RTX2080Ti; recheck empty at launch')
(OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
# This conservatively debits three minutes of earlier preflight as of the recorded
# 06:21:54 UTC clock reading; it is an accounting start, not a fabricated launch.
start=datetime.datetime.fromisoformat('2026-11-10T06:18:54+00:00').timestamp()
budget=dict(accounting_start_epoch=start,final_deadline_epoch=start+900,
    clock_reference_UTC='2026-11-10T06:21:54+00:00',preflight_debit_seconds=180,
    accounting_note='conservative preflight allowance; actual GPU launch separately recorded',
    GPU_seconds_max=540,science_reserve_for_reporting_seconds=90)
(OUT/'budget.json').write_text(json.dumps(budget,indent=2))
print(json.dumps(dict(bound_files=len(p['bindings']),tasks=len(p['tasks']),budget=budget)))
