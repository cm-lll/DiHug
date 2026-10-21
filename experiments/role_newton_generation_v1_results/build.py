"""Archive complete new evidence and unchanged scientific dependencies; no weights."""
import json,hashlib,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
R=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_generation_20261021_v1')
BASE='4061ccbb9d97c58272d174be80aa14b339d10b16';SOURCE='e6f0cb15bf17e0de0a02ff5980dfd1e1abc6f645'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert time.time()<read(R/'activation.json')['deadline_epoch']-90
    assert read(R/'audit.json')['status']=='passed' and read(R/'controller.json')['status']=='completed'
    f=read(R/'freeze.json')
    for n,h in f['source'].items():assert sha(REPO/'experiments'/n)==h,n
    evidence=ROOT/'evidence';shutil.copytree(R,evidence,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','*.tmp'))
    (ROOT/'source_vs_4061ccbb.patch').write_bytes(subprocess.check_output(['git','diff',BASE,'--','experiments/role_newton_generation_v1'],cwd=REPO))
    bundle=ROOT/'review_bundle';bundle.mkdir()
    def copy(p,n):
        d=bundle/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    for n,h in f['source'].items():copy(REPO/'experiments'/n,'experiments/'+n)
    for directory in ('role_newton_generation_v1','role_newton_reader_v1','role_logscale_reader_v1'):
        for p in (REPO/'experiments'/directory).iterdir():
            if p.is_file() and p.suffix in ('.py','.md','.json'):copy(p,f'experiments/{directory}/'+p.name)
    shutil.copytree(evidence,bundle/'results')
    for n in ('conditions.json','resolved_config.json','sealed_final_weights.json','freeze.json'):
        copy(Path('/data2/lyh/gnn_project/DiHug/results/role_coverage_20261021_v1')/n,'history/coverage_'+n)
    for n in ('RESULTS.md','README.md','source_vs_4061ccbb.patch'):copy(ROOT/n,n)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    assert not any(n.endswith(('.pt','.pth','.pem')) for n in manifest)
    (bundle/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
    archive=ROOT/'role_newton_generation_v1_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,'role_newton_generation_v1_review/'+str(p.relative_to(bundle)))
    receipt=dict(source_commit=SOURCE,baseline=BASE,ZIP_sha256=sha(archive),ZIP_bytes=archive.stat().st_size,payload_files=len(manifest),
                 graphs=528,steps=33792,Newton_reads=67584,tests=39,private_weights=False,elapsed_seconds=time.time()-read(R/'activation.json')['activation_epoch'])
    (ROOT/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n{sha(ROOT/"source_vs_4061ccbb.patch")}  source_vs_4061ccbb.patch\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
