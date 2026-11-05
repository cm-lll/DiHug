"""Scan this isolated review tree and its public-base history; no upload.

Heuristic checks are not a proof of absence of secrets. Findings contain only
paths and categories, never matching secret strings. Review all pending files.
"""
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = '175373c83202b6c8b80aa570e0092c4652031ce6'
REPORT = ROOT/'review/structure_input_v1'
patterns = {
    'private_key': re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'aws_access_key': re.compile(rb'AKIA[0-9A-Z]{16}'),
    'personal_absolute_path': re.compile(rb'/(?:home|Users|data[0-9]+)/[A-Za-z0-9_.-]+/'),
}
forbidden = {'.pt','.pth','.ckpt','.npz','.npy','.zip','.gz','.tar','.pickle','.pkl'}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def scan(name, content):
    issues = [dict(path=name, category=k) for k,p in patterns.items() if p.search(content)]
    if pathlib.Path(name).suffix in forbidden or len(content)>1024*1024:
        issues.append(dict(path=name,category='binary_asset_or_file_over_1MiB'))
    if pathlib.Path(name).name.startswith('.env'):
        issues.append(dict(path=name,category='environment_file'))
    return issues


def main():
    paths = sorted(set(git('ls-files','--cached','--others','--exclude-standard').decode().splitlines()))
    findings = []
    source_hashes = {}
    for name in paths:
        path=ROOT/name
        if not path.is_file():
            findings.append(dict(path=name,category='missing_file')); continue
        data=path.read_bytes()
        findings.extend(scan(name,data))
        if path.suffix in {'.py','.json','.yaml','.yml','.txt'} and not name.startswith('review/'):
            source_hashes[name]=hashlib.sha256(data).hexdigest()
    history=[]; seen=set()
    for commit in git('rev-list',BASE).decode().splitlines():
        for row in git('ls-tree','-r',commit).decode().splitlines():
            metadata,name=row.split('\t',1); _,kind,oid=metadata.split()
            if kind!='blob' or (name,oid) in seen: continue
            seen.add((name,oid))
            history.extend(scan(name,git('cat-file','blob',oid)))
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'SOURCE_HASHES.json').write_text(json.dumps(dict(
        base_commit=BASE,scope='review checkout source/config/dependencies; excludes reports',
        sha256=source_hashes),indent=2)+'\n')
    receipt=dict(base_commit=BASE,tree_files=len(paths),history_unique_path_blobs=len(seen),
        pending_tree_findings=findings,public_base_history_findings=history,
        status='passed' if not findings and not history else 'needs_review',
        limitations='Pattern and extension scan plus manual diff review; not a guarantee. Public base ancestry preserved, unrelated local repository history is excluded.')
    (REPORT/'PUBLICATION_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='limitations'}))
    return 0 if receipt['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
