import hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
R=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_20261021_v1')
BASE='9b010a6b41b39755a9bb36f34fa210b32b8fd689'
SOURCE='ad562a70c9f3e5d8fc1801afbd8d0d80772fcb48'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert time.time()<read(R/'activation.json')['deadline_epoch']-90
    assert read(R/'final_audit.json')['status']=='passed'
    f=read(R/'freeze.json')
    for n,h in f['source'].items():assert sha(REPO/'experiments/role_newton_reader_v1'/n)==h,n
    evidence=ROOT/'evidence';shutil.copytree(R,evidence,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','*.tmp'))
    patch=subprocess.check_output(['git','diff',BASE,SOURCE,'--','experiments/role_newton_reader_v1'],cwd=REPO)
    (ROOT/'source_vs_9b010.patch').write_bytes(patch)
    bundle=ROOT/'review_bundle';bundle.mkdir()
    def copy(p,n):
        dst=bundle/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    old=Path('/data2/lyh/gnn_project/DiHug/results/role_coverage_20261021_v1')
    for n,h in read(old/'freeze.json')['source'].items():
        p=REPO/'experiments'/n;assert sha(p)==h;copy(p,'experiments/'+n)
    for name in ('role_logscale_reader_v1','role_newton_reader_v1'):
        for p in (REPO/'experiments'/name).iterdir():
            if p.is_file() and p.suffix in ('.py','.json','.md'):copy(p,f'experiments/{name}/'+p.name)
    shutil.copytree(evidence,bundle/'results')
    # All original small state/LL arrays, with exact path/hash mapping.
    done=set()
    for x in read(R/'input_manifest.json'):
        if x['scope']=='small' and x['source_payload'] not in done:
            p=REPO/'experiments'/x['source_payload'];assert sha(p)==x['source_sha256']
            copy(p,'history/source_arrays/'+x['source_payload']);done.add(x['source_payload'])
    copy(REPO/'experiments/role_logscale_reader_v1_results/review_bundle/SHA256SUMS','history/9b010_SHA256SUMS')
    for n in ('large_input_identity.json','guard_cost/IMDB_reader.json','guard_cost/PubMed_reader.json'):
        copy(REPO/'experiments/role_logscale_reader_v1_results/evidence'/n,'history/'+n)
    for n in ('conditions.json','resolved_config.json','sealed_final_weights.json'):
        copy(old/n,'history/coverage_'+n)
    for n in ('RESULTS.md','README.md','source_vs_9b010.patch'):copy(ROOT/n,n)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    assert not any(n.endswith('.pt') for n in manifest)
    (bundle/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
    archive=ROOT/'role_newton_reader_v1_review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,'role_newton_reader_v1_review/'+str(p.relative_to(bundle)))
    receipt=dict(source_commit=SOURCE,baseline=BASE,ZIP_sha256=sha(archive),ZIP_bytes=archive.stat().st_size,
       payload_files=len(manifest),matrix_count=544,original_contract=538,bounded_tail_only=6,tests=35,
       prediction_pairs=132,large_LL_historical_hash_unavailable=True,private_weights=False,
       elapsed_seconds=time.time()-read(R/'activation.json')['activation_epoch'])
    (ROOT/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n{sha(ROOT/"source_vs_9b010.patch")}  source_vs_9b010.patch\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
