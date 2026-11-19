"""Verify preserved evidence and close the CPU-only ticket; no model calls."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/output_path_ablation_cpu_20261109'
def read(name):
    return json.loads((OUT / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
bindings = read('binding.json')
changed = [p for p, h in bindings.items() if sha(Path(p)) != h]
assert not changed, changed
identity = read('identity_summary.json')
receipt = read('receipt.json')
decision = {
    'status': 'complete', 'adopt_uniform_path_removal': False,
    'formal_training_authorized_by_this_ticket': False,
    'reason': 'IMDB gains coexist with PubMed costs and increased path costs in both final-graph panels',
    'next_recommendation': 'design degree-determined quantity / neighbor-arrangement residual joint score on CPU',
    'next_score_implemented_or_evaluated': False,
    'identity_is_not_training_causality_or_quality_proof': True,
}
(OUT / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))
files = list((ROOT / 'research_branches/output_path_ablation_cpu_20261109').glob('*.py'))
files += [OUT / n for n in ['REPORT_ZH.md', 'THEORY_ZH.md', 'protocol.json', 'decision.json', 'identity_summary.json']]
end = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
start = datetime.datetime.fromisoformat(read('protocol.json')['start_BJT'])
completion = {
    'status': 'complete', 'protocol_start_BJT': start.isoformat(), 'end_BJT': end.isoformat(),
    'protocol_to_delivery_wall_seconds': (end-start).total_seconds(),
    'note': 'Protocol timestamp follows initial read-only preparation; wall interval includes report work, not just CPU computation',
    'measured_worker_body_seconds': receipt['wall_seconds'] + identity['CPU_seconds'],
    'peak_RSS_MiB': max(receipt['peak_RSS_MiB'], identity['peak_RSS_MiB']),
    'GPU_hours': 0, 'model_calls': 0, 'training_updates': 0, 'new_random_graphs': 0,
    'old_bindings_verified': len(bindings), 'changed_old_assets': changed,
    'failures': [], 'workers_finished': True, 'followup_started': False,
    'artifact_sha256': {str(p): sha(p) for p in files},
}
(OUT / 'completion.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2))
print(json.dumps(completion, ensure_ascii=False, indent=2))
