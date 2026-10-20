"""Deterministic source-only archive; never loads weights or starts experiments."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pt'))

def main():
    package=ROOT/'package'
    manifest={str(p.relative_to(package)):digest(p) for p in files(package) if p.name!='manifest.json'}
    (package/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    items=[p for p in files(ROOT) if 'dist' not in p.relative_to(ROOT).parts
           and p.name not in ('SHA256SUMS','release_receipt.json','unpack_tests.xml')]
    sums=''.join(digest(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in items)
    (ROOT/'SHA256SUMS').write_text(sums)
    archive=ROOT/'dist/dihug_role_network_condition_v1_1.zip';archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in items+[ROOT/'SHA256SUMS']:
            info=zipfile.ZipInfo('dihug_role_network_condition_v1_1/'+str(p.relative_to(ROOT)),(2026,10,20,0,0,0))
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    (archive.with_suffix('.zip.sha256')).write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(zip=str(archive),sha256=digest(archive),files=len(items)+1)))

if __name__=='__main__':main()
