"""Explicit content-hashed dependencies; never cache SHA values or timestamps."""
import sys
from pathlib import Path
from round01 import b,json,ROOT

def loaded_sources():
    paths=set(b.FROZEN.rglob('*.py'))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        p=Path(name).resolve()
        if p.suffix=='.py' and p.is_file() and p.is_relative_to(ROOT.parent):paths.add(p)
    return paths

def put(out,name,value):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def bind(out,files,**metadata):
    paths=loaded_sources()|{Path(p).resolve() for p in files}
    put(out,'manifest.json',dict(files={str(p):b.sha(p) for p in sorted(paths)},legacy_lineage_verified_before_freeze=True,**metadata))

def verify(out):
    files=b.read(out/'manifest.json')['files'];assert loaded_sources()<={Path(p) for p in files},'unbound project source loaded after freeze'
    for p,h in files.items():assert b.sha(p)==h,p
