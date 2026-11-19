"""Package only this CPU audit's scripts, tables and records; no model assets."""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/role_score_allocation_audit_20261109'
SRC=Path(__file__).parent
LOG=ROOT/'results/role_score_allocation_audit_20261109_logs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
first=json.loads((OUT/'receipt.json').read_text())
second=json.loads((OUT/'direct64_role_readout_v2/receipt.json').read_text())
assert first['status']==second['status']=='complete'
for p in SRC.glob('*.py'):compile(p.read_text(),str(p),'exec')
for p in (OUT/'asset_binding.json',OUT/'direct64_role_readout_v2/asset_binding.json'):
    for source,digest in json.loads(p.read_text()).items():assert sha(Path(source))==digest,source
failure=dict(status='preserved',path=str(LOG/'direct_readout.log'),
    error='KeyError: historical manifest used repository-relative paths',
    stage='before loading fixed inputs or reconstructing samples',
    fix='Canonicalize relative paths against repository root, reject aliases, retain all hashes',
    failed_attempt_elapsed_seconds=None)
(OUT/'failure_record.json').write_text(json.dumps(failure,indent=2))
sources=list(SRC.glob('*.py'))+[
    ROOT/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',
    ROOT/'research_branches/joint_transport_cpu_20261108/transport.py']
(OUT/'source_binding.json').write_text(json.dumps({str(p):sha(p) for p in sources},indent=2))
receipt=dict(status='complete',track='CORE_support_CPU_output_objective_audit',
    finalized_UTC=datetime.now(timezone.utc).isoformat(),
    repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    source_status='isolated uncommitted CPU scripts; exact hashes in source_binding.json',
    graphs=first['graphs'],selected_edges=first['selected_edges'],historical_case_rows=first['historical_case_rows'],
    reconstructed_old_DIRECT_samples=second['samples'],
    successful_worker_seconds=first['seconds']+second['seconds'],
    timing_excludes='preparation, failed path-binding attempt, summaries, reporting and packaging',
    peak_worker_RSS_MiB=max(first['peak_RSS_MiB'],second['peak_RSS_MiB']),
    GPU_hours=0,model_calls=0,training_updates=0,new_sample_seeds=0,T1000_chains=0,
    resources_released='CPU commands exited; no GPU allocated; no watchdog or background job created',
    decision='Do not revive unchanged MMD/JOINT_W1 or claim role matching invalid. Retain DIRECT reference; separate quantity and allocation evidence before defining another scalar loss.',
    formal_training_started=False,quality_gate_passed=False,
    limitations=['two historical training seeds with correlated parent branches',
        'posthoc metrics and already viewed input states','conditional clean-graph samples, not full diffusion generation',
        'no universal inference that all unmeasured structure worsened',
        'new synthetic/CPU checks do not establish model quality'])
(OUT/'completion.json').write_text(json.dumps(receipt,indent=2))
files=[p for folder in (OUT,SRC,LOG) for p in folder.rglob('*')
       if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.csv','.json','.md','.log')]
archive=ROOT/'results/role_score_allocation_audit_20261109_review.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),completion=receipt),indent=2))
