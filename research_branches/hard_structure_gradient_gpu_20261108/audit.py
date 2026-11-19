"""Read-only post-run binding audit; never imports a model or CUDA."""
import json,hashlib,time,csv,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_gradient_gpu_20261108T180629'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks={}
for source in [O/'source_binding.json',R/'results/hard_structure_incremental_cpu_20261108/source_binding.json',R/'results/joint_output_formal_projected_20261108T043000/source_binding.json']:
 values=read(source)
 checks[str(source)]={p:sha(p)==v for p,v in values.items()}
assets=read(O/'assets.json') if (O/'assets.json').exists() else []
for a in assets:checks[a['condition']]={a['parent']:sha(a['parent'])==a['sha256'],a['reference']:sha(a['reference'])==a['reference_sha256']}
rows=read(O/'results.json') if (O/'results.json').exists() else []
with (O/'gradient_summary.csv').open('w') as f:
 fields=['condition','seed','parent_updates','training_t','group','CE_L2','structure_L2','ratio','cosine'];wr=csv.DictWriter(f,fields);wr.writeheader()
 for x in rows:
  for group,v in x['geometry'].items():wr.writerow(dict(**{k:x[k] for k in fields[:4]},group=group,**v))
audit=dict(at_epoch=time.time(),all_bindings_unchanged=all(all(v.values()) for v in checks.values()),checks=checks,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),local_tensors={str(p):sha(p) for p in O.glob('*.pt')},output_gradients={str(p):sha(p) for p in O.glob('*.npz')})
(O/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k not in ['checks','local_tensors','output_gradients']}))
assert audit['all_bindings_unchanged']
