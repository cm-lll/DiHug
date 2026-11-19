"""Seal finalized, never-started extension source; preserve the draft manifest."""
import ast,json,hashlib,shutil,tarfile,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/structure_count_unit_extension_20261107T173846'
PARENT=ROOT/'results/structure_count_unit_formal_20261107T163906'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,r):p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
def main():
    assert not list((OUT/'leases').glob('*.json')) and not (OUT/'formal1_started.json').exists()
    sources=read(PARENT/'source_binding.json')
    for p,h in sources.items():assert sha(Path(p))==h,p
    for name in ['source_binding.json','source_version.json','runtime_diff.patch','frozen_source.tar.gz']:
        if (OUT/name).exists():shutil.copy2(OUT/name,OUT/('draft_'+name))
    for p in SRC.rglob('*.py'):ast.parse(p.read_text());sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources)
    version=read(OUT/'source_version.json');version['source_manifest_sha256']=sha(OUT/'source_binding.json');version['revision']='prepared-final-before-any-GPU'
    save(OUT/'source_version.json',version)
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    before=ROOT/'research_branches/structure_count_unit_formal_review_20261107';diff=[]
    for p in SRC.rglob('*.py'):
        old=before/p.relative_to(SRC)
        diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile=str(old),tofile=str(p)))
    (OUT/'runtime_diff.patch').write_text(''.join(diff))
    tasks=read(OUT/'tasks.json');checks=[]
    for ds in ['IMDB','PubMed']:
        pair=[r for r in tasks if r['dataset']==ds];a,b=pair
        assert a['config']==b['config'] and a['initial']==b['initial'] and a['initial_sha256']==b['initial_sha256'] and a['CUDA_stream_seed']==b['CUDA_stream_seed']
        assert sha(Path(a['initial']))==a['initial_sha256']
        assert len(read(OUT/'prepared'/f"input_contract_{ds}_{a['train_seed']}.json"))==3000
        checks.append(dict(dataset=ds,paired_config_initial_CUDA_seed=True,initial_sha256=a['initial_sha256']))
    for name in ['pb_degree.py','triangle_risk.py','count_deviance.py','shifted_count_deviance.py','runtime/slot_update.py','runtime/science.py']:
        assert sha(SRC/name)==sha(before/name),name
    save(OUT/'source_audit.json',dict(passed=True,source_files=len(sources),syntax=True,math_and_update_sources_unchanged=True,parent_source_hashes_unchanged=True,paired=checks,GPU_calls=0,source_manifest_sha256=sha(OUT/'source_binding.json')))
    print(json.dumps(read(OUT/'source_audit.json')))
if __name__=='__main__':main()
