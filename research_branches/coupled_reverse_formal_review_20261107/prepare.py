"""Authorized ticket registration and read-only asset/source validation."""
import json,hashlib,shutil,datetime,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/coupled_reverse_formal_20261107T105500'
OLD=ROOT/'results/pb_degree_formal_20261106T170600'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    OUT.mkdir(exist_ok=True)
    assert not list((OUT/'leases').glob('*.json')), 'do not re-register after GPU acquisition'
    for n in ['engineering','prepared','leases','logs']: (OUT/n).mkdir(exist_ok=True)
    start=datetime.datetime.fromisoformat('2026-11-07T10:55:00+08:00').timestamp()
    save(OUT/'ledger.json',dict(start_epoch=start,science_stop_epoch=start+150*60,deadline_epoch=start+180*60,GPU_hours_limit=8,engineering_GPU_hours_limit=.25,ordinary_recompute_tolerance_passed=False,start_note='conservatively rounded before first implementation/resource check; no reset'))
    save(OUT/'approval.json',dict(authorized=True,user_instruction='那就开始吧',scope='4 A+PB+coupled-risk models x3000 from sealed zero, 16 T1000; engineering and budget gates; 180min/8GPUh',launcher_pid=None,risk_weight=.013715673231930771,automatic_followup=False))
    tasks=read(OLD/'tasks.json')
    for task in tasks:task['arm']='A_PB_CR'
    save(OUT/'tasks.json',tasks)
    for p in (OLD/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name)
    rows=read(ROOT/'results/coupled_reverse_scale_20261107/reusable_controls.json')
    for row in rows:
        for name in ['initial','graph','readout']:
            assert sha(Path(row[name]))==row[name+'_sha256'],row[name]
    # Exact historical training input pairing must be true for PB and parent A.
    for task in tasks:
        records=[json.loads(s) for s in (OLD/'run'/task['task_id']/'training.jsonl').read_text().splitlines()]
        contract=read(OUT/'prepared'/('input_contract_%s_%s.json'%(task['dataset'],task['train_seed'])))
        assert len(records)==len(contract)==3000
        assert all(r['input_sha256']==c['sha256'] and r['t']==c['t'] and r['update']==i+1 for i,(r,c) in enumerate(zip(records,contract)))
    sources=read(OLD/'source_binding.json')
    for p,h in sources.items():assert sha(Path(p))==h,p
    for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources)
    for n in ['pb_degree.py']:
        assert sha(SRC/n)==sha(ROOT/'research_branches/pb_degree_output_review_20261106'/n)
    for n in ['triangle_risk.py','global_scale.py','sensitivity_scale.py']:
        assert sha(SRC/n)==sha(ROOT/'research_branches/coupled_reverse_review_20261107'/n)
    save(OUT/'asset_audit.json',dict(passed=True,controls=rows,source_files=len(sources),PB_input_hashes_verified=12000,disk_free_GiB=shutil.disk_usage(ROOT).free/2**30))
    save(OUT/'source_version.json',dict(parent_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated source manifest; not claiming uncommitted files belong to parent commit'))
    print(str(OUT))
if __name__=='__main__':main()
