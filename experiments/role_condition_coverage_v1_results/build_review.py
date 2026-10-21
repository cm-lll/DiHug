"""Package frozen source and complete evidence; private weights/checkpoints stay local."""
import argparse,hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
BASE='fdc446ca2a3f33ec5ae89b167d659559d2662641'
SOURCE='795b4bafb3259dc00f67cfb008991fb1fe5a1124'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);a=parser.parse_args()
    ledger=read(a.results/'activation.json')
    if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original3h budget')
    receipt=read(a.results/'controller.json')
    if receipt['status']=='completed':assert read(a.results/'audit.json')['status']=='passed'
    assert read(a.results/'formal_launch.json')['source_commit']==SOURCE
    frozen=read(a.results/'freeze.json')
    for n,h in frozen['source'].items():assert sha(REPO/'experiments'/n)==h,n
    for n,h in frozen['inputs'].items():assert sha(a.results/n)==h,n
    inventory={}
    for p in sorted(a.results.rglob('*.pt')):
        inventory[str(p.relative_to(a.results))]=dict(sha256=sha(p),bytes=p.stat().st_size,
            public_payload=False,purpose='disposable engineering fixture' if 'cost' in p.parts else 'scientific initialization/checkpoint/final or sampling-state',
            retained_local=True)
    save(a.results/'checkpoint_inventory.json',inventory)
    ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc','*.pt','*.tmp')
    evidence=ROOT/'evidence';shutil.copytree(a.results,evidence,ignore=ignore)
    patch=subprocess.check_output(['git','diff',BASE,SOURCE,'--','experiments/role_condition_coverage_v1'],cwd=REPO)
    (ROOT/'source_vs_fdc446c.patch').write_bytes(patch)
    bundle=ROOT/'review_bundle';bundle.mkdir()
    for n,h in frozen['source'].items():
        dst=bundle/'experiments'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/'experiments'/n,dst)
    # Actual public immutable metadata used by read-only original initialization gates.
    pinned=REPO/'experiments/role_specialization_v1_results/evidence/learning_run01'
    dst=bundle/'experiments/role_specialization_v1_results/evidence/learning_run01';dst.mkdir(parents=True)
    for name in ('weight_hashes.json','resolved_config.json','environment.json','source_manifest.json'):
        shutil.copyfile(pinned/name,dst/name)
    archived_panel='experiments/role_error_recovery_v1_results/evidence/archived_panel/panel.json'
    dst=bundle/archived_panel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/archived_panel,dst)
    shutil.copytree(evidence,bundle/'results')
    for name in ('RESULTS.md','README.md','source_vs_fdc446c.patch'):shutil.copyfile(ROOT/name,bundle/name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    assert not any(n.endswith('.pt') for n in manifest)
    (bundle/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
    archive=ROOT/'role_condition_coverage_v1_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,'role_condition_coverage_v1_review/'+str(p.relative_to(bundle)))
    save(ROOT/'release_receipt.json',dict(status=receipt['status'],sha256=sha(archive),bytes=archive.stat().st_size,
        payload_files=len(manifest),scientific_source_commit=SOURCE,baseline=BASE,
        updates=receipt['scientific_updates_recorded'],graphs=receipt['scientific_graphs_recorded'],
        private_weight_files_retained_local=len(inventory),elapsed_from_original_activation=time.time()-ledger['activation_epoch']))
    (ROOT/'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n{sha(ROOT/"source_vs_fdc446c.patch")}  source_vs_fdc446c.patch\n')
    if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original3h budget after save')
    print((ROOT/'release_receipt.json').read_text())
if __name__=='__main__':main()
