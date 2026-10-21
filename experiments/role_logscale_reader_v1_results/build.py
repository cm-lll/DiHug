"""Source + all diagnostics release; no weights or execution activation."""
import hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
R=Path('/data2/lyh/gnn_project/DiHug/results/role_logscale_20261021_v1')
BASE='7b45d502e34b283192e19212c343b79a5910298a'
SOURCE='5fed0b89560ff718be912672c2c63e536452452e'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert time.time()<read(R/'activation.json')['deadline_epoch']-60
    assert read(R/'audit_guard.json')['status']=='passed'
    assert read(R/'numerical_guard_equivalence.json')['all_probability_arrays_bitwise_equal']
    assert read(R/'guard_cost/PubMed_reader.json')['status']=='completed'
    old=Path('/data2/lyh/gnn_project/DiHug/results/role_coverage_20261021_v1')
    evidence=ROOT/'evidence';shutil.copytree(R,evidence,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.tmp','*.pt'))
    patch=subprocess.check_output(['git','diff',BASE,SOURCE,'--','experiments/role_logscale_reader_v1'],cwd=REPO)
    (ROOT/'source_vs_7b45d502.patch').write_bytes(patch)
    bundle=ROOT/'review_bundle';bundle.mkdir()
    def copy(p,n):
        dst=bundle/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for n,h in read(old/'freeze.json')['source'].items():
        p=REPO/'experiments'/n;assert sha(p)==h;copy(p,'experiments/'+n)
    for p in (REPO/'experiments/role_logscale_reader_v1').iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.md'):copy(p,'experiments/role_logscale_reader_v1/'+p.name)
    for n in ('freeze.json','conditions.json','panel.json','resolved_config.json','sealed_final_weights.json'):
        copy(old/n,'history/'+n)
    for seed in (2026102011,2026102012,2026102013):
        copy(old/f'jobs/broad_{seed}/validation_u006000.json',f'history/seed{seed}/validation_u006000.json')
        for p in (R/f'seed{seed}').glob('*_replay.json'):
            name=p.stem[:-7]
            for suffix in ('.npz','_steps.json','_result.json'):
                src=old/f'jobs/broad_{seed}/generated/{name}_2026102031{suffix}'
                copy(src,f'history/seed{seed}/'+src.name)
    inv=REPO/'experiments/role_target_scalability_v1/evidence/large_graph_inventory.json'
    copy(inv,'experiments/role_target_scalability_v1/evidence/large_graph_inventory.json')
    for name in ('IMDB','PubMed'):
        copy(REPO/f'experiments/joint_cn_dual_rounds_v1_1/package/data/{name}.npz',f'experiments/joint_cn_dual_rounds_v1_1/package/data/{name}.npz')
    shutil.copytree(evidence,bundle/'results')
    for name in ('RESULTS.md','README.md','source_vs_7b45d502.patch'):copy(ROOT/name,name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    (bundle/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
    zpath=ROOT/'role_logscale_reader_v1_review.zip'
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,'role_logscale_reader_v1_review/'+str(p.relative_to(bundle)))
    receipt=dict(source_commit=SOURCE,baseline=BASE,ZIP_sha256=sha(zpath),ZIP_bytes=zpath.stat().st_size,
        payload_files=len(manifest),passed_prediction_pairs=77,rejected_states=55,tests=21,
        weights_included=False,elapsed_seconds=time.time()-read(R/'activation.json')['activation_epoch'])
    (ROOT/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'SHA256SUMS').write_text(f'{sha(zpath)}  {zpath.name}\n{sha(ROOT/"source_vs_7b45d502.patch")}  source_vs_7b45d502.patch\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
