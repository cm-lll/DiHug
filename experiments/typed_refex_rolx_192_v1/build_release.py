"""Build review bundle; engineering checkpoint bytes stay local, hashes included."""
import hashlib,json,pathlib,shutil,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parents[1]
EVIDENCE=REPO/'results/typed_refex_rolx_192_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    release=ROOT/'release';release.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'package',release/'package',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'received',release/'received')
    shutil.copytree(ROOT/'archive',release/'archive')
    dest=release/'evidence';dest.mkdir()
    omitted={}
    for p in sorted(EVIDENCE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(EVIDENCE)
        if p.suffix=='.pt':
            omitted[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size,'purpose':'local disposable engineering weights, not formal initialization'};continue
        if p.suffix in ('.zip','.pyc') or '__pycache__' in str(p):continue
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    (dest/'local_engineering_checkpoints.json').write_text(json.dumps(omitted,indent=2))
    shutil.copy2(ROOT/'build_release.py',release/'build_release.py')
    manifest={str(p.relative_to(release)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(release.rglob('*')) if p.is_file()}
    (release/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (release/'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k,v in manifest.items()))
    out=EVIDENCE.parent/'typed_refex_rolx_192_v1_review.zip'
    if out.exists():raise FileExistsError(out)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(release.rglob('*')):
            if p.is_file():z.write(p,'typed_refex_rolx_192_v1/'+str(p.relative_to(release)))
    receipt={'zip':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'payload_files':len(manifest),'excluded_checkpoint_count':len(omitted)}
    (EVIDENCE/'release_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
