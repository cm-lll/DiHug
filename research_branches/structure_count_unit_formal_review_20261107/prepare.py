"""Register the user-authorized fixed ticket; audit files only, no model calls."""
import json,hashlib,shutil,datetime,subprocess,difflib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/structure_count_unit_formal_20261107T163906'
PB=ROOT/'results/pb_degree_formal_20261106T170600'
PLAN=ROOT/'results/structure_count_unit_plan_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
def main():
    OUT.mkdir(exist_ok=False)
    for n in ['engineering','prepared','leases','logs']:(OUT/n).mkdir()
    start=1794040746
    save(OUT/'ledger.json',dict(start_epoch=start,science_stop_epoch=start+130*60,deadline_epoch=start+160*60,
        GPU_hours_limit=7,engineering_GPU_hours_limit=.25,ordinary_recompute_tolerance_passed=False))
    save(OUT/'approval.json',dict(authorized=True,user_instruction='开始吧',scope='frozen four models3000 and16T1000;160min/7GPUh',
        launcher_pid=None,structure_weight=1,count_shift=1,automatic_followup=False))
    for p,h in read(PLAN/'hashes.json').items():assert sha(p)==h,p
    tasks=read(PB/'tasks.json')
    for t in tasks:t['arm']='A_PB_UNIT'
    save(OUT/'tasks.json',tasks)
    for p in (PB/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name)
    sources=read(PB/'source_binding.json')
    for p,h in sources.items():assert sha(p)==h,p
    for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
    for n in ['count_deviance.py','shifted_count_deviance.py']:
        assert sha(SRC/n)==sha(ROOT/'research_branches/structure_moment_review_20261107'/n)
    save(OUT/'source_binding.json',sources)
    save(OUT/'source_version.json',dict(parent_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated source SHA manifest, not claiming new files belong to parent Git commit'))
    save(OUT/'asset_audit.json',dict(passed=True,source_files=len(sources),plan_hash=sha(PLAN/'asset_audit.json'),
        controls=read(PLAN/'graphs.json'),initials=read(PLAN/'models.json'),disk_free_GiB=shutil.disk_usage(ROOT).free/2**30))
    diff=[];old=ROOT/'research_branches/coupled_reverse_formal_review_20261107'
    for p in SRC.rglob('*.py'):
        rel=p.relative_to(SRC);before=(old/rel).read_text().splitlines(True) if (old/rel).exists() else []
        diff.extend(difflib.unified_diff(before,p.read_text().splitlines(True),fromfile=str(old/rel),tofile=str(p)))
    (OUT/'runtime_diff.patch').write_text(''.join(diff))
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    print(str(OUT))
if __name__=='__main__':main()
