"""Reconstruct byte-identical logical paths from deduplicated review ZIP; no model."""
import argparse,json,hashlib,zipfile
from pathlib import Path
def materialize(archive,out):
    out=Path(out);assert not out.exists(),'new output directory required'
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('logical_manifest.json'))
        for n,x in m.items():
            p=Path(n);assert not p.is_absolute() and '..' not in p.parts
            data=z.read('objects/'+x['sha256']);assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'],n
            d=out/p;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(data)
    return len(m)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(materialize(a.archive,a.output),'verified files restored')
