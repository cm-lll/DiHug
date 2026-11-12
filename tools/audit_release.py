"""Inspect the staged tree, including decompressed result tables; print no secrets."""
import ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    forbidden_suffix={'.pt','.pth','.ckpt','.npz','.npy','.zip','.log','.key','.pem'}
    patterns={
      'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
      'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
      'aws_key':re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
      'credential_url':re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
      'personal_absolute_path':re.compile(r'/(?:home|Users|data2)/[A-Za-z0-9_.-]+/'),
    }
    problems=[];manifest=[];parsed=0
    for name in names:
        if not name:continue
        path=ROOT/name
        if path.is_symlink():problems.append((name,'symlink'));continue
        raw=path.read_bytes()
        if path.suffix in forbidden_suffix or path.name.startswith('.env'):problems.append((name,'forbidden artifact'))
        if len(raw)>5*1024**2:problems.append((name,'oversized file'))
        content=gzip.decompress(raw) if path.suffix=='.gz' else raw
        try:text=content.decode('utf8')
        except UnicodeDecodeError:problems.append((name,'unexpected binary'));continue
        for key,pattern in patterns.items():
            if pattern.search(text):problems.append((name,key))
        if path.suffix=='.py':ast.parse(text,filename=name);parsed+=1
        manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    result=dict(status='passed' if not problems else 'failed',files=len(manifest),python_parsed=parsed,total_bytes=sum(x['bytes'] for x in manifest),problems=problems,scope='complete tracked publication tree; compressed CSV inspected as text',original_worktree_not_staged=True)
    print(json.dumps(result,indent=2))
    if problems:raise SystemExit(1)
if __name__=='__main__':main()
