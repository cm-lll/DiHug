"""Bind retained states and estimate remaining work; never launch GPU jobs."""
import hashlib,json,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/single_step_formal_20261111'
OUT=ROOT/'results/single_step_formal_resume_20261111'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False))
p=read(OLD/'protocol.json');cal=read(OLD/'budget_calibration.json');used=read(OLD/'supervisor_receipt.json')['GPU_hours'];bindings={};cost=[]
for task,initial in zip(p['tasks'],cal['per_task']):
    d=OLD/task['id'];f=d/'stopped_complete.pt';expected=read(d/'stopped_complete.sha256.json');digest=sha(f)
    assert digest==expected['sha256'] and expected['updates']==2500
    log=d/'training.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
    assert [r['update'] for r in rows]==list(range(2001,2501))
    bindings[task['id']]=dict(checkpoint=str(f),checkpoint_sha256=digest,training_sha256=sha(log),updates=2500)
    mean=statistics.mean(r['seconds'] for r in rows[-100:]);sample=initial['reverse_step_seconds']
    estimate=1.4*(500*mean+4000*sample+180)
    cost.append(dict(task=task['id'],update_mean_last100_seconds=mean,remaining_training_seconds=500*mean,sampling_seconds_extrapolated=4000*sample,load_save_validation_allowance_seconds=180,margin_factor=1.4,estimated_occupied_seconds=estimate))
for name in ('protocol.json','bindings.json'):
    target=OUT/name
    if target.exists():assert target.read_bytes()==(OLD/name).read_bytes()
    else:target.write_bytes((OLD/name).read_bytes())
save('resume_bindings.json',bindings)
preflight_allowance=240;new_gpu=(sum(r['estimated_occupied_seconds'] for r in cost)+preflight_allowance)/3600
budget=dict(at_epoch=time.time(),per_task=cost,prior_GPU_hours=used,preflight_allowance_GPU_seconds=preflight_allowance,
    estimated_new_GPU_hours=new_gpu,estimated_cumulative_GPU_hours=used+new_gpu,cumulative_cap_GPU_hours=16,
    estimated_parallel_science_minutes=max(r['estimated_occupied_seconds'] for r in cost)/60+preflight_allowance/60,
    proposed_science_minutes=120,proposed_delivery_minutes=150,
    window_started=False,original_window_not_reset=True,formal_resume_launched=False,
    limitations=['Full T1000 chain still extrapolated from five reverse steps','GPU resume of retained u2500 not yet executed','CPU evaluation may vary with generated graph density'])
save('remaining_budget.json',budget)
summary=OUT/'previous_failure_summary';summary.mkdir(exist_ok=True)
(summary/'protocol.json').write_bytes((OLD/'protocol.json').read_bytes())
for t in p['tasks']:
    src=OLD/t['id'];dst=summary/t['id'];dst.mkdir(exist_ok=True)
    for n in ('receipt.json','training.jsonl','per_graph.json'):
        if (src/n).exists():(dst/n).write_bytes((src/n).read_bytes())
subprocess.run([sys.executable,str(Path(__file__).with_name('finalize.py')),'--out',str(summary)],check=True,timeout=40)
save('source_bindings.json',{str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')})
print(json.dumps(budget,indent=2))
