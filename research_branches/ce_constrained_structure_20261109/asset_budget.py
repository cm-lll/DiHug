"""Read-only parent/hash and historical timing audit, CPU only."""
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ce_constrained_structure_20261109'

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

assets=[]
old=ROOT/'results/joint_output_formal_projected_20261108T043000'
for r in rows(old/'checkpoint_inventory.csv'):
    if r['arm']!='A_CE' or r['updates']!='3000' or not r['file'].endswith('/final.pt'):continue
    p=Path(r['file']);actual=sha(p) if p.is_file() else None
    assets.append(dict(**r,exists=p.is_file(),actual_sha256=actual,matches=actual==r['sha256'],
        scope='File identity checked; internal optimizer/RNG/EMA load gate not rerun here.'))
assert len(assets)==4 and all(r['matches'] for r in assets)
(OUT/'parents.json').write_text(json.dumps(assets,indent=2))

geom=rows(ROOT/'results/direct_one_step_20261108T212756/gradient_update_geometry.csv')
curves=rows(ROOT/'results/direct_medium_run_20261108T230502/curves.csv')
scale=[]
for g in geom:
    if g['group']!='all':continue
    r=next(x for x in curves if x['condition']==g['condition'] and x['arm']=='start')
    ce,s=float(r['CE']),float(r['DIRECT'])
    # Old geometry contains lambda=.001. New normalized objective at eta=1.
    ratio=(float(g['weighted_structure_gradient_L2'])/.001/s)/(float(g['CE_gradient_L2'])/ce)
    scale.append(dict(condition=g['condition'],CE_reference=ce,structure_reference=s,
        normalized_structure_to_CE_gradient_ratio_at_eta1=ratio,
        scope='Saved first-step stochastic gradient, NOT production calibration or all-step ratio.'))
(OUT/'initial_scale_risk.json').write_text(json.dumps(scale,indent=2))

timing=rows(old/'runtime_per_model.csv')
sample={ds:[float(r['mean_T1000_chain_seconds']) for r in timing
        if r['dataset']==ds and r['arm']=='A_CE' and r['mean_T1000_chain_seconds']]
        for ds in ['IMDB','PubMed']}
smi=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used','--format=csv,noheader'],capture_output=True,text=True)
resource=dict(hostname=subprocess.check_output(['hostname'],text=True).strip(),
    nvidia_device_nodes=[str(p) for p in Path('/dev').glob('nvidia*')],
    nvidia_smi_returncode=smi.returncode,nvidia_smi_stdout=smi.stdout,nvidia_smi_stderr=smi.stderr,
    GPU_hours=0,any_GPU_process_started=False)
(OUT/'resource_status.json').write_text(json.dumps(resource,indent=2))
budget=dict(status='planning_only_not_a_running_job',parent_updates=3000,end_updates=4000,
    arms=['matched_CE_continuation','CE_constrained_DIRECT_continuation'],
    seeds={'IMDB':[2026110411,2026110412],'PubMed':[2026110421,2026110422]},
    models=8,new_updates=8000,samples_per_model=4,new_full_chains=32,
    sampling_seeds=[2026102821,2026102822,2026110231,2026110232],
    note='Prior 3000-step graphs are context only, not matched 4000-step controls. OA/OAB first; OB is not a prerequisite.',
    historical_A_CE_chain_seconds=sample,
    per_candidate_update_seconds_bound=3.,per_CE_update_seconds_bound=2.,
    per_chain_seconds_bound=180.,estimated_update_GPU_hours=(4000*3+4000*2)/3600,
    estimated_chain_GPU_hours=32*180/3600,
    proposed_GPU_hours_cap=10.,proposed_wall_minutes_cap=180,proposed_science_minutes=150,
    remaining_GPU_allowance_for_load_save_checks=10-(4000*3+4000*2+32*180)/3600,
    assumptions=['Up to 8 verified free 2080Ti, no reservation while waiting for CPU.',
        'No extra model forward in stochastic dual step; implementation not yet GPU tested.',
        'Historical chain timings include different execution policies; 180s is extrapolation, not a guarantee.',
        'Parent references/new independent evaluation/GPU recovery and numerical scales must pass before training.',
        'Production choice of dual timescale and CE-reference estimation remains unresolved; no launch permitted yet.'],
    launch=False,blockers=['GPU devices unavailable in current execution context.','Production stochastic controller/integration gate incomplete.'])
(OUT/'prospective_budget.json').write_text(json.dumps(budget,indent=2))
print(json.dumps(dict(parents=assets,scales=scale,budget=budget,resource=resource),indent=2))
