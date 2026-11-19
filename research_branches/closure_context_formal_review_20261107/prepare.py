"""Register the user-authorized fixed ticket; audit files only, no model calls."""
import json,hashlib,shutil,datetime,subprocess,difflib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/closure_context_formal_20261107T231108'
PB=ROOT/'results/pb_degree_formal_20261106T170600'
PLAN=ROOT/'results/closure_context_plan_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
def main():
    if OUT.exists():
        assert not (OUT/'tasks.json').exists() and not list((OUT/'leases').glob('*.json')), 'registration already completed or computation started'
        save(OUT/'preparation_failure1.json',dict(error='plan hash paths are repository-relative; initial read from source cwd failed before GPU or asset modification',fix='resolve each recorded path against repository ROOT',original_start_preserved=True))
    OUT.mkdir(exist_ok=True)
    for n in ['engineering','prepared','leases','logs']:(OUT/n).mkdir(exist_ok=True)
    start=datetime.datetime.fromisoformat('2026-11-07T23:11:08+08:00').timestamp()
    save(OUT/'ledger.json',dict(start_epoch=start,science_stop_epoch=start+120*60,deadline_epoch=start+150*60,
        GPU_hours_limit=7,engineering_GPU_hours_limit=.25,ordinary_recompute_tolerance_passed=False))
    save(OUT/'approval.json',dict(authorized=True,user_instruction='下一步',scope='frozen four models3000 and16T1000;150min/7GPUh',
        launcher_pid=None,PB_weight=1,CE_weight='(1+hit_Xt)/(1+mean_relation_hit_Xt)',automatic_followup=False))
    for p,h in read(PLAN/'hashes.json').items():assert sha(ROOT/p)==h,p
    tasks=read(PB/'tasks.json')
    for t in tasks:t['arm']='A_PB_CONTEXT'
    save(OUT/'tasks.json',tasks)
    for p in (PB/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name)
    sources=read(PB/'source_binding.json')
    for p,h in sources.items():assert sha(p)==h,p
    for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources)
    save(OUT/'source_version.json',dict(parent_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated source SHA manifest, not claiming new files belong to parent Git commit'))
    save(OUT/'asset_audit.json',dict(passed=True,source_files=len(sources),plan_hash=sha(PLAN/'asset_audit.json'),
        controls=read(PLAN/'graphs.json'),initials=read(PLAN/'models.json'),disk_free_GiB=shutil.disk_usage(ROOT).free/2**30))
    diff=[];old=ROOT/'research_branches/structure_count_unit_formal_review_20261107'
    for p in SRC.rglob('*.py'):
        rel=p.relative_to(SRC);before=(old/rel).read_text().splitlines(True) if (old/rel).exists() else []
        diff.extend(difflib.unified_diff(before,p.read_text().splitlines(True),fromfile=str(old/rel),tofile=str(p)))
    (OUT/'runtime_diff.patch').write_text(''.join(diff))
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    print(str(OUT))
if __name__=='__main__':main()

