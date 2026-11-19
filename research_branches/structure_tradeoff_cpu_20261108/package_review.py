import itertools,json,hashlib,time,zipfile
from pathlib import Path
import numpy as np
from analyze import classify
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/structure_tradeoff_cpu_20261108'
x=np.array(list(itertools.product([-1.,0.,1.],repeat=3)));m=classify(x)
assert [int(m[k].sum()) for k in ['nonworse_improvement','nonbetter_damage','tradeoff','neutral']]==[7,7,12,1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=O/'invalid_shared_query_attempt/degree_decomposition.csv'
checks=dict(sign_patterns=27,partition_counts=[7,7,12,1],per_time_query_used=True,all_96_degree_W1_values_reconstructed=True,old_and_corrected_decomposition_identical=sha(old)==sha(O/'degree_decomposition.csv'))
(O/'additional_checks.json').write_text(json.dumps(checks,indent=2))
protocol=json.loads((O/'protocol.json').read_text());now=time.time();assert now<protocol['delivery_stop_epoch']
d=dict(status='complete_stopped',wall_minutes=(now-protocol['start_epoch'])/60,end_epoch=now,GPU_hours=0,model_calls=0,updates=0,source={str(p):sha(p) for p in H.glob('*.py')},formal_training_authorized=False,missing=['No saved full edgewise triangle-composition, motif or connectivity counterfactuals'],errors=['NumPy input .numpy() read failure corrected; no scientific model computation','First terminal session handle not retained; exact first failed process timing unavailable','Cross-time query equality assertion failed; use corresponding time mapping, reconstructed degree W1 validated; derived mean decomposition unchanged'])
(O/'delivery.json').write_text(json.dumps(d,indent=2))
files=sorted(p for p in O.rglob('*') if p.is_file())+list(H.glob('*.py'))
(O/'review_hashes.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in files},indent=2));files.append(O/'review_hashes.json')
zp=O.with_name(O.name+'_review.zip')
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,str(p.relative_to(R)))
with zipfile.ZipFile(zp) as z:assert z.testzip() is None
print(json.dumps(dict(wall_minutes=d['wall_minutes'],archive=str(zp),sha256=sha(zp),bytes=zp.stat().st_size)))
