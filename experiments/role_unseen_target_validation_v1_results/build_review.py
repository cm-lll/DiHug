"""Seal complete read-only evidence and frozen scientific source; no model calls."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BASE='bdd30efba7353dea9bb789a56f313718a4ead584'
SOURCE='f538c7463d13a7e0c699206ba898262fe36f6261'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);a=ap.parse_args()
    ledger=read(a.results/'run01/activation.json')
    def check():
        if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original45min ledger')
    check()
    assert ledger['source_commit']==SOURCE and read(a.results/'audit_repair_v1/receipt.json')['status']=='completed'
    assert read(a.results/'post_audit.json')['status']=='passed'
    assert len(list((a.results/'run01').glob('seed*/*_result.json')))==120
    source_manifest=read(a.results/'run01/source_manifest.json')
    for name,h in source_manifest.items():assert sha(REPO/'experiments'/name)==h,name
    # Original failed statuses are artifacts, never rewritten as passed.
    assert read(a.results/'run01/controller.json')['status']=='failed'
    assert read(a.results/'audit_repair_v1/attempt001_failed_receipt.json')['status']=='failed'
    ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc','*.pt')
    evidence=ROOT/'evidence';shutil.copytree(a.results,evidence,ignore=ignore)
    patch=subprocess.check_output(['git','diff',BASE,SOURCE,'--','experiments/role_unseen_target_validation_v1','docs/PROJECT_STATUS.md','docs/NEXT_EXECUTION_TASK.md'],cwd=REPO)
    (ROOT/'source_vs_bdd30ef.patch').write_bytes(patch)
    bundle=ROOT/'review_bundle';bundle.mkdir()
    for name,h in source_manifest.items():
        src=REPO/'experiments'/name;dst=bundle/'experiments'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    pinned=REPO/'experiments/role_robust_training_6000_v1_results/evidence/learning_run01'
    dst=bundle/'experiments/role_robust_training_6000_v1_results/evidence/learning_run01';dst.mkdir(parents=True)
    for name in ('resolved_config.json','source_manifest.json','input_bindings.json','weight_hashes.json'):
        shutil.copyfile(pinned/name,dst/name)
    # No private weights; public archived training metadata is sufficient for the loader gate
    # when caller provides the original local read-only weight files.
    shutil.copytree(evidence,bundle/'results')
    for name in ('RESULTS.md','README.md','source_vs_bdd30ef.patch'):
        shutil.copyfile(ROOT/name,bundle/name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    assert not any(n.endswith('.pt') for n in manifest)
    (bundle/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
    path=ROOT/'role_unseen_target_validation_v1_review.zip'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.rglob('*')):
            if f.is_file():z.write(f,'role_unseen_target_validation_v1_review/'+str(f.relative_to(bundle)))
    write(ROOT/'release_receipt.json',dict(sha256=sha(path),bytes=path.stat().st_size,payload_files=len(manifest),
        scientific_source_commit=SOURCE,protocol_commit=BASE,new_scientific_graphs=120,training_updates=0,
        original_controller='failed-retained',post_evaluation='completed',
        elapsed_from_original_activation=time.time()-ledger['activation_epoch']))
    (ROOT/'SHA256SUMS').write_text(f'{sha(path)}  {path.name}\n{sha(ROOT/"source_vs_bdd30ef.patch")}  source_vs_bdd30ef.patch\n')
    check();print((ROOT/'release_receipt.json').read_text())

if __name__=='__main__':main()
