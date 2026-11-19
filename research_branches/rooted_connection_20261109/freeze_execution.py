"""CPU final launch binding; keep prior engineering source identity, never reset ledger."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent;OUT=ROOT/'results/rooted_connection_20261109T015142'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (OUT/'formal1_started.json').exists():raise RuntimeError('formal run already exists')
    old=json.loads((OUT/'source_binding.json').read_text())
    changed={p:dict(old=h,new=sha(p)) for p,h in old.items() if sha(p)!=h}
    allowed={str(SRC/'runtime/supervise_formal.py'),str(SRC/'runtime/finalize_cpu.py')}
    if not set(changed)<=allowed:raise ValueError(('unexpected post-engineering source change',changed))
    (OUT/'engineering/source_binding.json').write_text(json.dumps(old,indent=2))
    for p in SRC.rglob('*.py'):old[str(p)]=sha(p)
    (OUT/'source_binding.json').write_text(json.dumps(old,indent=2))
    (OUT/'execution_source_amendment.json').write_text(json.dumps(dict(epoch=time.time(),changes=changed,note='completed engineering runtime untouched; resource watchdog/CPU delivery only; source freeze before training'),indent=2))
    with tarfile.open(OUT/'frozen_execution_source.tar.gz','w:gz') as tar:
        for p in old:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    version=json.loads((OUT/'source_version.json').read_text());version['execution_manifest_sha256']=sha(OUT/'source_binding.json');(OUT/'source_version.json').write_text(json.dumps(version,indent=2))
if __name__=='__main__':main()
