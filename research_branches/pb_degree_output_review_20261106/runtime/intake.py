"""Rehash prepared assets and freeze actual runtime sources; no model calls."""
import json,os,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parents[1];OUT=Path(os.environ['STRUCTURE_RUN'])
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    expected=read(OUT/'asset_manifest_expected.json');rows=[]
    for r in expected['tasks']:
        for path,key in [('initial','initial_sha256'),('readout','readout_sha256'),('training_log','training_log_sha256')]:
            assert sha(r[path])==r[key],r[path];rows.append(r[path])
    for r in expected['graphs']:assert sha(r['file'])==r['sha256'];rows.append(r['file'])
    sources=read(P/'results/cn_multiseed_20261104T042546/source_manifest.json')
    for p,h in sources.items():assert sha(p)==h,p
    # Include all local executable modules and baseline modules actually importable.
    source_root=P/'research_branches/cn_multiseed_20261104'
    for tree in [source_root/'experiments/o_cn_matched_execute_v1',source_root/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1']:
        for p in tree.rglob('*.py'):sources[str(p)]=sha(p)
    for p in ROOT.rglob('*.py'):sources[str(p)]=sha(p)
    (OUT/'source_binding.json').write_text(json.dumps(sources,indent=2)+'\n')
    (OUT/'asset_audit.json').write_text(json.dumps(dict(passed=True,files_rehashed=rows,source_files=len(sources),disk_free_GiB=shutil.disk_usage(P).free/2**30,epoch=time.time()),indent=2)+'\n')
    print('intake passed',len(rows),len(sources))
if __name__=='__main__':main()
