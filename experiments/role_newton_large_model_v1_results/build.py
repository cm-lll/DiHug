"""Content-addressed ZIP avoids duplicating identical cold/repeat numeric bytes."""
import json,hashlib,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
R=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_large_model_20261021_v1')
BASE='8855b78b8b5b79f0a1671c2c3427647754291f21';SOURCE='1e7f8832e2343cd287471e9104d4d37658d32ed4'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(R/'audit.json')['status']=='passed_for_available_evidence'
    assert time.time()<read(R/'activation.json')['deadline_epoch']-420
    f=read(R/'freeze.json')
    for n,h in f['source'].items():assert sha(REPO/'experiments'/n)==h,n
    evidence=ROOT/'evidence';shutil.copytree(R,evidence,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','*.tmp'))
    patch=ROOT/'source_vs_8855.patch';patch.write_bytes(subprocess.check_output(['git','diff',BASE,'--','experiments/role_newton_large_model_v1'],cwd=REPO))
    bundle=ROOT/'review_bundle';bundle.mkdir()
    def copy(p,n):
        d=bundle/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    for n,h in f['source'].items():copy(REPO/'experiments'/n,'experiments/'+n)
    for p in (REPO/'experiments/role_newton_large_model_v1').iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.md'):copy(p,'experiments/role_newton_large_model_v1/'+p.name)
    for name in ('IMDB','PubMed'):
        n=f'experiments/joint_cn_dual_rounds_v1_1/package/data/{name}.npz';copy(REPO/n,n)
    n='experiments/role_target_scalability_v1/evidence/large_graph_inventory.json';copy(REPO/n,n)
    shutil.copytree(evidence,bundle/'results')
    for n in ('RESULTS.md','README.md','source_vs_8855.patch'):copy(ROOT/n,n)
    manifest={str(p.relative_to(bundle)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(bundle.rglob('*')) if p.is_file()}
    assert not any(n.endswith(('.pt','.pth','.pem')) for n in manifest)
    (bundle/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {n}\n" for n,x in manifest.items()))
    # Mapping preserves EVERY logical path; one immutable object per identical content hash.
    manifest['SHA256SUMS']=dict(sha256=sha(bundle/'SHA256SUMS'),bytes=(bundle/'SHA256SUMS').stat().st_size)
    archive=ROOT/'role_newton_large_model_v1_review.zip';seen=set()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('logical_manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.write(ROOT/'materialize.py','materialize.py')
        for n,x in manifest.items():
            if x['sha256'] not in seen:z.write(bundle/n,'objects/'+x['sha256']);seen.add(x['sha256'])
    receipt=dict(sampling_probe_source_commit=SOURCE,baseline=BASE,ZIP_sha256=sha(archive),ZIP_bytes=archive.stat().st_size,logical_files=len(manifest),unique_content_objects=len(seen),
        packaging='content-addressed objects plus lossless logical manifest; materialize.py reconstructs every original path after SHA verification',
        complete_items=read(R/'audit.json')['complete_items'],complete_paths=read(R/'audit.json')['complete_paths'],tests=43,trained_weights_in_archive=False,
        elapsed_seconds=time.time()-read(R/'activation.json')['activation_epoch'])
    (ROOT/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(ROOT/'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n{sha(patch)}  source_vs_8855.patch\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
