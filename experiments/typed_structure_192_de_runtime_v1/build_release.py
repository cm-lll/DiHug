"""Deterministic source/evidence ZIP, excluding worktrees, caches and private weights."""
import argparse
import hashlib
import json
import pathlib
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent;dest=pathlib.Path(a.output).resolve()
    if dest.exists() or root in dest.parents:raise ValueError('new external ZIP path required')
    def included(f):
        return f.is_file() and not any(x in ('__pycache__','.pytest_cache','.git','review_worktree','postpack_verification') for x in f.relative_to(root).parts) and f.suffix not in ('.pyc','.pt','.zip') and f.name not in ('MANIFEST.json','SHA256SUMS')
    paths=sorted(p for p in root.rglob('*') if included(p))
    if any(p.is_symlink() for p in paths):raise ValueError('symlinks excluded')
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    (root/'MANIFEST.json').write_text(json.dumps(dict(version='typed_structure_192_de_runtime_v1',baseline_commit='89922be56927412dbdb2b5f9e6f36f8264f027d4',files=files),indent=2))
    checks=paths+[root/'MANIFEST.json']
    (root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in checks))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in checks+[root/'SHA256SUMS']:
            name='typed_structure_192_de_runtime_v1/'+p.relative_to(root).as_posix()
            info=zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    digest=sha(dest);dest.with_suffix('.zip.sha256').write_text(digest+'  '+dest.name+'\n')
    print(json.dumps(dict(zip=str(dest),sha256=digest,bytes=dest.stat().st_size,manifest_payloads=len(files),archive_files=len(checks)+1)))
if __name__=='__main__':main()
