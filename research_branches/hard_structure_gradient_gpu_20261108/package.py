import json,time,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_gradient_gpu_20261108T180629';H=Path(__file__).parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lease=read(O/'lease.json');p=read(O/'protocol.json')
delivery=dict(at_epoch=time.time(),wall_seconds=time.time()-p['start_epoch'],gpu_hours=lease['gpu_hours'],gpu_released=True,release_observation='GPU0 0 MiB, 0%, no compute processes',completed=True,formal_training_authorized=False,remaining='No parameter response, independent structural validation or generated quality evidence; no lambda chosen',source_hashes={str(x):sha(x) for x in H.glob('*.py')})
(O/'delivery.json').write_text(json.dumps(delivery,indent=2))
z=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for x in O.iterdir():
  if x.suffix in ['.json','.csv','.md','.log']:f.write(x,'results/'+x.name)
 for x in H.glob('*.py'):f.write(x,'code/'+x.name)
 for x in [R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py']:f.write(x,'frozen_reference/'+x.name)
print(json.dumps(dict(package=str(z),sha256=sha(z),bytes=z.stat().st_size,delivery=delivery),indent=2))
