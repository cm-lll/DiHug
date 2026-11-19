import ast, hashlib, json, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/real192_update_audit_20261111'
OLD=ROOT/'results/single_step_formal_resume_20261111'
OUT.mkdir(exist_ok=False)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))
previous=json.loads((OLD/'protocol.json').read_text())
tasks=[t for t in previous['tasks'] if t['B']==4]
assert len(tasks)==4
for i,t in enumerate(tasks): t['gpu']=i
save('protocol.json',dict(tasks=tasks, updates_each=50, B=4,K=16,
  derivative_updates=[2001,2008,2010,2012,2017,2046,2050],
  evaluation_updates=[2000,2012,2020,2050], evaluation_times=[1,75,250,550,800,1000],
  readouts=['raw','EMA'], extra_component_vjps=4*7*4*5,
  main_backwards=4*50*4, evaluation_forwards=4*4*2*6,
  objective='unchanged five hard-graph risks arithmetic mean, CE gradient zero',
  scope='temporary diagnostics only; no formal training or reverse generation',
  numeric_check='VJP component sum relative L2 error <= 0.005; observation only, no training tolerance altered',
  note='Saved historical scalar trajectory may differ under ordinary CUDA; input/RNG identities must match exactly. Risk and gradient dot displacement are local observations, not unique causes.'))
# Use the start already announced for this ticket; do not reset for coding time.
start=1794375800.0  # 2026-11-11 13:43:20 Asia/Shanghai
assert abs(time.time()-start)<1800
save('window.json',dict(start_epoch=start,science_stop_epoch=start+35*60,
                       delivery_stop_epoch=start+45*60,GPU_seconds_cap=3600))
bindings=json.loads((OLD/'bindings.json').read_text())
for p,h in bindings.items():
    assert sha(p)==h, 'prior source/asset changed '+p
for p in HERE.glob('*.py'):
    ast.parse(p.read_text());bindings[str(p)]=sha(p)
for t in tasks:
    p=OLD/t['id']/'training.jsonl';bindings[str(p)]=sha(p)
bindings[str(OUT/'protocol.json')]=sha(OUT/'protocol.json')
save('bindings.json',bindings)
save('preparation.json',dict(prior_bindings_checked=True,source_syntax_pass=True,
                            source_files=len(list(HERE.glob('*.py'))),models=4,
                            conservative_GPU_seconds_estimate=3000,
                            explanation='Old four B4 first50 cost ~970 GPU seconds. Allow >3x for VJPs, saves, validation and loading; hard stop independently enforced.'))
print(json.dumps(dict(out=str(OUT),window=json.loads((OUT/'window.json').read_text()),bindings=len(bindings))))
