"""Freeze CPU results and a small review archive; no launcher actions."""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/ce_constrained_structure_20261109'
def load(name):return json.loads((OUT/name).read_text())
assert all(x['passed'] for x in load('checks.json'))
assert all(x['passed'] for x in load('active_constraint_checks.json'))
assert all(x['restore'] and x['scale_identity'] for x in load('scaled_audit.json')['checks'])
assert all(x['matches'] for x in load('parents.json'))
duration=sum(load(n)['wall_seconds'] for n in ['cpu_summary.json','component_replay_audit.json','scaled_audit.json'])
record=dict(status='CPU_complete_production_not_started',finished_at=datetime.now().astimezone().isoformat(),
    user_authorization='conditional formal validation after theory/simulation/integration/resource gates',
    mathematical_identity_checks='passed',CPU_controller_restore='passed',
    quality_claim='not established; structural tradeoffs and CE-budget limitations observed',
    production_gate='not passed',
    blockers=['GPU unavailable in current execution context.',
              'Production stochastic primal-dual/AdamW/EMA integration not validated.',
              'Initial total-CE budget does not protect class-specific errors or structural components.'],
    computation_wall_seconds=duration,
    timing_scope='Three logged CPU numeric runs including exact toy replay; excludes preparation and report writing.',
    primary_CPU_run_peak_RSS_MiB=load('cpu_summary.json')['peak_RSS_MiB'],
    GPU_hours=0,production_model_calls=0,production_training_updates=0,full_generation_chains=0,
    source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    source_note='New isolated uncommitted prototype; exact files bound below. HEAD alone does not contain this code.',
    next='Resolve fidelity-budget definition and production integration/resource gates; do not auto-launch this prototype.',
    original_assets_modified=False,background_jobs_started=False)
files=sorted(SRC.glob('*.py'))+sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ['completion.json','manifest.json'])
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'completion.json').write_text(json.dumps(record,indent=2))
archive=ROOT/'results/ce_constrained_structure_20261109_review.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files+[OUT/'manifest.json',OUT/'completion.json']:
        z.write(p,str(p.relative_to(ROOT)))
print(json.dumps(dict(**record,review_archive=str(archive),archive_bytes=archive.stat().st_size),indent=2))
