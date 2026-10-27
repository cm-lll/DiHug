"""Publish source and actual engineering evidence; no checkpoint bytes."""
import hashlib,json,pathlib,shutil,zipfile,difflib
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parents[1];EVIDENCE=REPO/'results/typed_refex_rolx_192_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    release=HERE/'release';release.mkdir(exist_ok=False)
    shutil.copytree(HERE/'package',release/'package',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
    shutil.copytree(HERE/'historical_v1',release/'historical_v1')
    for name in ['README_zh.md','REPORT_zh.md','REVISION.md','test_revision_extra.py','audit_v2.py','summarize.py','build_release.py']:
        shutil.copy2(HERE/name,release/name)
    old=HERE.parent/'typed_refex_rolx_192_v1/package';patch=[]
    for p in sorted((HERE/'package').rglob('*')):
        if not p.is_file() or p.suffix not in ('.py','.json') or '__pycache__' in str(p):continue
        rel=p.relative_to(HERE/'package');q=old/rel
        left=q.read_text().splitlines(True) if q.exists() else []
        patch.extend(difflib.unified_diff(left,p.read_text().splitlines(True),fromfile='v1/package/'+str(rel),tofile='v2/package/'+str(rel)))
    (release/'CHANGES_FROM_V1.patch').write_text(''.join(patch))
    dest=release/'evidence';dest.mkdir();omitted={}
    for p in sorted(EVIDENCE.rglob('*')):
        if not p.is_file() or p.suffix in ('.zip','.pyc') or '__pycache__' in str(p):continue
        rel=p.relative_to(EVIDENCE)
        if p.suffix=='.pt':omitted[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size,'purpose':'disposable engineering; kept locally'};continue
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    (dest/'local_engineering_checkpoints.json').write_text(json.dumps(omitted,indent=2))
    manifest={str(p.relative_to(release)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(release.rglob('*')) if p.is_file()}
    (release/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (release/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in manifest.items()))
    archive=EVIDENCE.parent/'typed_refex_rolx_192_v2_review.zip'
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(release.rglob('*')):
            if p.is_file():z.write(p,'typed_refex_rolx_192_v2/'+str(p.relative_to(release)))
    receipt=dict(zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,payload_files=len(manifest),local_checkpoints_omitted=len(omitted))
    (EVIDENCE/'release_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
