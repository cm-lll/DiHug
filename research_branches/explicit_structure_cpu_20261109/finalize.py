"""Read-only evidence verification and machine-readable completion; no model work."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/explicit_structure_cpu_20261109'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    bindings = json.loads((OUT / 'binding.json').read_text())
    differences = [p for p, h in bindings.items() if not Path(p).is_file() or sha(Path(p)) != h]
    if differences:
        raise RuntimeError({'changed_assets': differences})
    receipt = json.loads((OUT / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['GPU_hours'] == 0
    decision = dict(
        status='complete_stopped', formal_training_authorized=False,
        adopt=False, reason='Repeated local hub/connectivity benefit, with joint-degree/triangle/path costs; no model or generation evidence.',
        interpretation='Explicit penalties, not hard quotas. Weighted sum guarantees only that some optimized component improves.',
        counts_are='fixed selected overlapping edits, not failure rates or a training trajectory',
        next='Decide acceptable tradeoff before any revised objective or model computation; no automatic coefficient search.',
        resource=receipt, bound_files_checked=len(bindings), assets_unchanged=True,
    )
    (OUT / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n')
    completion = dict(
        completed_BJT=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        files={str(p.relative_to(ROOT)): sha(p) for p in [
            OUT/'REPORT_ZH.md', OUT/'THEORY_ZH.md', OUT/'decision.json',
            OUT/'protocol.json', OUT/'receipt.json', Path(__file__),
        ]}, assets_unchanged=True, additional_scientific_computation=False,
    )
    (OUT / 'completion.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({**receipt, 'status':'complete_stopped', 'bound_files_checked':len(bindings)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
