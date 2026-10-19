"""Deterministic ZIP; no weights or scientific outputs. Run from this directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'package'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def files():
    return sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and '.pytest_cache' not in p.parts and p.suffix!='.pyc')

def main():
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    (PACKAGE/'source_revision.json').write_text(json.dumps(dict(source_commit=source,
        base_commit='0cdd8274d7359c4cc189685735fbe77c60c712c7',
        scope='source commit; release manifest/ZIP are generated afterwards and committed separately'),indent=2)+'\n')
    # Collection must be frozen before build; evidence uses exact IDs, zero skips.
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'.');from dual.data import load;[load(n) for n in ['IMDB','PubMed']]"],cwd=PACKAGE,check=True)
    manifest={str(p.relative_to(PACKAGE)):sha(p) for p in files() if p!=PACKAGE/'release_manifest.json'}
    (PACKAGE/'release_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    target=HERE/'dihug_joint_cn_dual_rounds_v1_1.zip'
    if target.exists():raise ValueError('release ZIP exists: never overwrite an old delivery')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files():
            info=zipfile.ZipInfo('dihug_joint_cn_dual_rounds_v1_1/'+str(p.relative_to(PACKAGE)),date_time=(2026,10,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    (HERE/'SHA256SUMS').write_text(sha(target)+'  '+target.name+'\n')
    print((HERE/'SHA256SUMS').read_text())

if __name__=='__main__':main()
