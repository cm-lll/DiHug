"""Preserve failed attempt; carry both deadlines and prior lease time forward."""
import json,hashlib,os
from pathlib import Path
from exact_count import tests
ROOT=Path(__file__).resolve().parents[2];old=ROOT/'results/exact_count_step_20261110';out=ROOT/'results/exact_count_step_20261110_attempt2'
out.mkdir(exist_ok=False)
p=json.loads((old/'protocol.json').read_text());budget=json.loads((old/'budget.json').read_text());failure=json.loads((old/'receipt.json').read_text())
assert failure['forwards']==failure['backwards']==failure['optimizer_updates']==0
budget['prior_occupied_seconds']=json.loads((old/'supervisor.json').read_text())['occupied_seconds']
for path in Path(__file__).parent.glob('*.py'):p['bindings'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
p['retry_reason']='CPU entry import cleared CUDA_VISIBLE_DEVICES; isolated pure NumPy helper now used; no scientific rule changed'
p['failed_attempt']=str(old)
for path,h in p['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert tests()['passed'];assert os.environ.get('CUDA_VISIBLE_DEVICES')==visible
(out/'cpu_tests.json').write_text(json.dumps(dict(pure_math_tests=tests(),helper_does_not_change_device_environment=True)))
(out/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2));(out/'budget.json').write_text(json.dumps(budget,indent=2))
print(json.dumps(budget))
