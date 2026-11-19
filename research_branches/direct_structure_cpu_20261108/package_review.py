import json,hashlib,time,zipfile,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/direct_structure_cpu_20261108'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(O/'protocol.json');receipt=read(O/'receipt.json');execution=read(O/'execution.json');now=time.time()
assert receipt['status']=='complete' and execution['returncode']==0
assert now<p['delivery_stop_epoch']
checks={}
for name in ['source_binding.json','asset_binding.json']:
    manifest=read(O/name);changed=[f for f,h in manifest.items() if sha(Path(f))!=h];assert not changed
    checks[name]=dict(files=len(manifest),changed=changed)
production=read(R/'results/joint_output_formal_projected_20261108T043000/source_binding.json');assert all(sha(Path(f))==h for f,h in production.items());checks['production']=dict(files=len(production),changed=[])
(O/'final_audit.json').write_text(json.dumps(checks,indent=2))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
d=dict(status='complete_stopped',start_epoch=p['start_epoch'],report_end_epoch=now,wall_minutes=(now-p['start_epoch'])/60,worker_seconds=receipt['seconds'],supervisor_seconds=execution['seconds'],GPU_hours=0,model_calls=0,updates=0,T1000_chains=0,CPU_threads=1,peak_RSS_MiB=receipt['peak_RSS_MiB'],failed=[],missing=[],formal_training_authorized=False,git_head=head,new_files_are_uncommitted=True,final_source_hashes={str(f.relative_to(R)):sha(f) for f in H.glob('*.py')})
(O/'delivery.json').write_text(json.dumps(d,indent=2))
files=sorted(f for f in O.iterdir() if f.is_file())+sorted(H.glob('*.py'))+[R/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py']
(O/'review_hashes.json').write_text(json.dumps({str(f.relative_to(R)):sha(f) for f in files},indent=2));files.append(O/'review_hashes.json')
zpath=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,str(f.relative_to(R)))
with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
archive=dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),wall_minutes=d['wall_minutes']);(O/'archive_receipt.json').write_text(json.dumps(archive,indent=2));print(json.dumps(archive))
