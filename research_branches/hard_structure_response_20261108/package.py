import json,hashlib,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/hard_structure_response_20261108T182154';H=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(O/'protocol.json');lease=read(O/'lease.json');audit=read(O/'audit.json');receipt=read(O/'receipt.json')
assert receipt['status']=='complete' and audit['all_bindings_unchanged']
for path,digest in read(O/'source_binding.json').items():assert sha(Path(path))==digest
delivery=dict(at_epoch=time.time(),wall_seconds=time.time()-p['start_epoch'],science_worker_seconds=receipt['seconds'],gpu_seconds=lease['end_epoch']-lease['start_epoch'],gpu_hours=lease['gpu_hours'],gpu_released=True,release_observation='GPU0 0 MiB, 0%, no compute processes',scope=dict(temporary_updates=receipt['updates'],forwards=receipt['forwards'],backwards=receipt['backwards'],training_auxiliary_hard_draws=32,evaluation_auxiliary_hard_draws=1152,CPU_audit_reconstructed_same_draws=768,new_full_T1000_chains=0,formal_models=0),exceptions=[],limitations=['Only two historical training seeds, four local parents from PS-trained history','One auxiliary sample/update, eight updates/arm','Raw evaluation only; EMA saved not evaluated','Conditional p0 auxiliary graphs are not full generated graphs','New auxiliary RNG persisted; no new interrupted/resumed-next-update test performed','Post-hoc hard-graph metric audit marked descriptive; no candidate selection'],formal_training_authorized=False,source_hashes={str(x):sha(x) for x in H.glob('*.py')},result_hashes={str(x):sha(x) for x in O.iterdir() if x.suffix in ['.json','.csv','.md','.log'] and x.name!='delivery.json'})
assert delivery['wall_seconds']<1200 and delivery['gpu_seconds']<720
(O/'delivery.json').write_text(json.dumps(delivery,indent=2))
z=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for x in O.iterdir():
  if x.suffix in ['.json','.csv','.md','.log']:f.write(x,'results/'+x.name)
 for x in H.glob('*.py'):f.write(x,'code/'+x.name)
 for x in [R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py',R/'research_branches/hard_structure_incremental_cpu_20261108/incremental.py']:f.write(x,'frozen_reference/'+x.name)
print(json.dumps(dict(package=str(z),sha256=sha(z),bytes=z.stat().st_size,wall_seconds=delivery['wall_seconds'],gpu_hours=delivery['gpu_hours']),indent=2))
