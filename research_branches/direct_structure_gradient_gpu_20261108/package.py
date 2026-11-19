import json,time,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/direct_structure_gradient_gpu_20261108T205133';H=Path(__file__).parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lease=read(O/'lease.json');p=read(O/'protocol.json');receipt=read(O/'receipt.json');audit=read(O/'audit.json')
assert receipt['status']=='complete' and audit['all_bindings_unchanged'] and lease['returncode']==0
assert lease['gpu_hours']<=.1 and time.time()<p['delivery_stop_epoch']
release=dict(gpu_index=0,uuid=p['gpu_uuid'],memory_used_MiB=0,utilization_percent=0,compute_processes=[],observation='host nvidia-smi after worker exit; read-only verification')
(O/'release.json').write_text(json.dumps(release,indent=2))
delivery=dict(at_epoch=time.time(),wall_seconds=time.time()-p['start_epoch'],gpu_hours=lease['gpu_hours'],gpu_released=True,completed=True,formal_training_authorized=False,lambda_selected=False,remaining='No parameter update, independent structural validation, EMA prediction or generated quality evidence',source_hashes={str(x):sha(x) for x in H.glob('*.py')})
(O/'delivery.json').write_text(json.dumps(delivery,indent=2))
z=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
    for x in O.iterdir():
        if x.suffix in ['.json','.csv','.md','.log']:f.write(x,'results/'+x.name)
    for x in H.glob('*.py'):f.write(x,'code/'+x.name)
    for x in [R/'research_branches/direct_structure_cpu_20261108/score.py',R/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py']:f.write(x,'frozen_reference/'+x.name)
with zipfile.ZipFile(z) as f:assert f.testzip() is None
record=dict(package=str(z),sha256=sha(z),bytes=z.stat().st_size,delivery=delivery);(O/'archive_receipt.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
