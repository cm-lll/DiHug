"""Verify saved results and write explicit stop decision; no numerical experiment."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_distance_20261109'
def main():
    b=json.loads((OUT/'binding.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in b.items())
    rows=json.loads((OUT/'exchanges.json').read_text())
    selected=[r for r in rows if r['explicit_change'] < -1e-12 and r['joint_degree_TV_change']>1e-12]
    assert len(selected)==146 and all(r['joint_distance_change'] < -1e-12 for r in selected)
    decision=dict(status='complete_stopped',adopt=False,formal_training_authorized=False,
        verified_TV_worse_joint_distance_better=146,
        interpretation='Distance diagnostic explains exact-match versus geometric disagreement; substitution alone does not resolve closure costs.',
        next='Keep diagnostic; inspect paired degree benefit/closure opportunity of newly admitted exchanges before proposing another rule; no automatic tuning or training.',
        failures=[],bound_files_checked=len(b),assets_unchanged=True,
        resource=json.loads((OUT/'receipt.json').read_text()))
    (OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    files=[OUT/'REPORT_ZH.md',OUT/'THEORY_ZH.md',OUT/'decision.json',OUT/'receipt.json',Path(__file__)]
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete_stopped',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
