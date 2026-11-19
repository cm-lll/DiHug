"""Freeze before formal results; preserves source and evidence, never replaces a run."""
from r2_common import *
import argparse,shutil,subprocess
def freeze(out):
 pre=json.loads((ROOT/'results/r2_preflight_v1/preflight.json').read_text());gg=json.loads((ROOT/'results/r2_generation_gate_v2/gate.json').read_text());entry=json.loads((ROOT/'results/r2_entry_gate_v1/gate.json').read_text())
 assert pre['passed'] and gg['passed'] and entry['passed']
 # Do not silently claim full relabel invariance.
 assert pre['inherited_same_type_endpoint_ordering']['equal'] is False
 out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
 paths=list((ROOT/'src/dihug').rglob('*.py'))+list((ROOT/'scripts').glob('r2_*.py'))+[ROOT/'scripts/review_endpoints.py',ROOT/'docs/R2_EXECUTION_PROTOCOL.md']
 files={}
 for p in paths:
  q=source/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q);files[str(q)]=sha(q)
 # Raw data and old weights remain read-only; model workers import only frozen new-project source.
 history=json.loads((HISTORY/'manifest.json').read_text());sampling_seconds=0;training_seconds=0;suffix_attempts=0
 for ds,a in ASSETS.items():
  for name in ('mixed_edit_diffusion.pt','mixed_edit_dataset.npz','train_report.json'):
   p=OLD/a['directory']/name;files[str(p)]=sha(p)
  p=OLD/history['references'][ds]['assets']['role'];files[str(p)]=sha(p)
  p=Path(history['references'][ds]['clean_reference_path']);files[str(p)]=sha(p)
  for prefix in range(100,108):
   p=HISTORY/ds/f'seed{prefix}/prefix/transition.json';files[str(p)]=sha(p);snap=json.loads(p.read_text())['payload'];remaining=10000-snap['completed'];suffix_attempts+=9*remaining
   sampling_seconds+=9*remaining*gg['records'][ds]['per_attempt_seconds']
   for n in ('final_graph.json','final_report.json'):
    p=HISTORY/ds/f'seed{prefix}/r0/R'/n;files[str(p)]=sha(p)
  updates=a['epochs']*int(np.ceil((3100 if ds=='imdb' else 1500)/64))
  training_seconds+=sum(3*updates*v['median_update_seconds'] for v in pre[ds]['gpu'].values())
 for p in (HISTORY/'manifest.json',HISTORY/'public_motif_counts.json',ROOT/'results/r1_full_journal/acceptance.json',ROOT/'results/r2_preflight_v1/preflight.json',ROOT/'results/r2_generation_gate_v2/gate.json',ROOT/'results/r2_entry_gate_v1/gate.json'):
  files[str(p)]=sha(p)
 manifest={'task':'R2_ROLE_INFORMATION_PAIRED_PILOT','files':files,'datasets':list(ASSETS),'training_seeds':[210,211,212],'arms':ARMS,'fits':18,'updates':48420,'outputs':144,'prefix_seeds':list(range(100,108)),'reuse_random_outputs':16,'suffix_attempts':suffix_attempts,'engineering_gate_passed':True,'known_non_equivariance':'inherited same-type endpoint ordering, fixed-ID paired comparison only','training_seconds_estimated_gpu_sum':training_seconds,'generation_seconds_estimated_cpu_sum':sampling_seconds,'parallelism':{'GPU':[0,1],'CPU_workers':4},'estimated_generation_wall_hours_at_four_workers':sampling_seconds/4/3600,'scientific_conclusion':'pending','full_pipeline_no_learning_control':'not part of this conditional stage2 pilot; still required'}
 atomic(out/'manifest.json',manifest);atomic(out/'status.json',{'stage':'frozen_not_started'});print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();freeze(a.out)
