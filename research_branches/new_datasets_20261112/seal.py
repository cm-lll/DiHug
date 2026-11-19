"""Final prelaunch source closure and removal of inactive inherited metadata."""
import sys,json,hashlib,zipfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=Path(sys.argv[1]);assert not (out/'launch.lock').exists();p=json.loads((out/'protocol.json').read_text());assert not (out/'unsealed_protocol.json').exists()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 (out/'unsealed_protocol.json').write_bytes((out/'protocol.json').read_bytes());(out/'frozen_source.zip').rename(out/'unsealed_source.zip')
 extra=['research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py','research_branches/one_step_vs_terminal_plan_20261111/gpu_preflight.py','research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py','research_branches/terminal_structure_real_20261109/worker.py','research_branches/typed_heat_audit_20261111/run.py','research_branches/typed_heat_degree_audit_20261111/run.py','research_branches/cn_multiseed_20261104/experiments/late_final_single_input_v1/single_input.py']
 vendor=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval/vendor'
 files=[ROOT/x for x in extra]+[vendor/x for x in ['frozen_metrics.py','aligned_evaluate.py','hgen_utils.py']]+list(Path(__file__).parent.glob('*.py'))
 for f in files:p['bindings'][str(f.resolve())]=sha(f)
 for job in p['jobs']:
  old=job['config'];cfg={k:v for k,v in old.items() if k not in ('datasets','feature_spec','arms','source_note')};cfg['arms']=['A'];cfg['source_note']='Original A scientific settings; new data from bound small_192.npz; no roles/dictionary. Inactive historical E and dataset metadata removed before training.'
  assert cfg['model']==old['model'] and cfg['optimizer']==old['optimizer'] and cfg['T']==old['T'] and cfg['seed']==old['seed'];job['config']=cfg
 p['git_HEAD']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();p['working_tree']='pre-existing dirty tree; runtime source identified by file hashes and frozen_source.zip, not HEAD alone'
 p['seal_note']='Added explicit hashes for importlib/AST-loaded sources; removed inactive inherited E/dataset metadata, no tensor/optimizer/kernel/CE changes.'
 with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in p['bindings']:
   if f.endswith('.py'):z.write(f,arcname=str(Path(f).relative_to(ROOT)))
 (out/'protocol.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(bindings=len(p['bindings']),protocol_sha256=sha(out/'protocol.json'),git_HEAD=p['git_HEAD'])))
if __name__=='__main__':main()
