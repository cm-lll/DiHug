"""Read and verify completed CPU accounting; no model operations."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exchange_closure_accounting_20261109';V=OUT/'verified'
def main():
    b=json.loads((V/'binding.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in b.items())
    cases=json.loads((V/'cases.json').read_text());summary=json.loads((V/'summary.json').read_text())
    rows=json.loads((V/'all_exchanges.json').read_text())
    assert len(cases)==7 and all(r['dominant_all_reported']>0 for r in cases)
    assert all(r['rank']>=151 for r in cases)
    decision=dict(status='complete_stopped',training_authorized=False,
        conclusion='Existing score already ranks componentwise-better alternatives higher; seven negative-delta edits do not justify blanket rejection by count alone.',
        residual_risk='Five cases create a new isolated node; actual probability/gradient weights and training effect unknown.',
        new_isolate_cases=[r['removed_slot'] for r in rows if r['case'] and r['isolates_error_change']>0],
        next='Bind same-state probabilities/gradients if already available; otherwise budget a bounded real signal check. No new formula or training automatically.',
        all_assets_unchanged=True,bound_files=len(b),resource=json.loads((V/'receipt.json').read_text()))
    (OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    files=[OUT/'REPORT_ZH.md',OUT/'engineering_notes.json',OUT/'decision.json',V/'receipt.json',Path(__file__)]
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete_stopped',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
