"""Bind full panel before GPU preflight; does not launch scientific work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_20261111'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
OUT.mkdir(exist_ok=False);old=json.loads((ROOT/'results/exact_feedback_updates_20261110/protocol.json').read_text());bind=json.loads((ROOT/'results/exact_feedback_updates_20261110/bindings.json').read_text());tasks=[]
for row in old['tasks']:
    for B,K in [(1,64),(4,16)]:
        tasks.append(dict(id=f"{row['dataset']}_{row['seed']}_B{B}K{K}",parent=row,B=B,K=K,gpu=len(tasks),architecture='A late CN',parent_updates=2000,final_updates=3000,new_updates=1000))
for p,h in bind.items():assert sha(p)==h,'previous binding changed '+p
for p in Path(__file__).parent.glob('*.py'):bind[str(p)]=sha(p)
contracts=ROOT/'results/cn_multiseed_20261104T042546/prepared/sampling_contracts.json';bind[str(contracts)]=sha(contracts)
sample_seeds=[2026102821,2026102822,2026110231,2026110232]
for row in old['tasks']:
    for key in ('checkpoint','data','history'):bind[row[key]]=sha(row[key])
    for ss in sample_seeds:assert row['dataset']+'_'+str(ss) in json.loads(contracts.read_text())
protocol=dict(track='CORE matched single-step structural continuation',tasks=tasks,sample_seeds=sample_seeds,sampling_contracts=str(contracts),updates=1000,parent_updates=2000,final_updates=3000,loss='equal mean original five HARD structural risks; CE coefficient zero; exact count and present/absent triangle squared-risk expectations, LOO joint degree/components',optimizer='parent AdamW including CE-trained moments/LR retained; EMA.99 once per optimizer update; no projection, rejection, clipping, or LR changes',sampling='unchanged R1 T1000 posterior and frozen paired contracts, final EMA only',pairing='same full parent; first input matches historical 2001..3000 at each update for both arms; B4 adds 3 independent stateless inputs, not identical exposure; hard first-input 16 uniform prefix paired',exposure=dict(B1=1000,B4=4000,hard_samples_per_update=64,optimizer_updates=1000),seeds=dict(extra_input=2026111131,hard=2026111132,evaluation=2026111133),primary_metrics=['incident4 degree W1','joint degree TV','typed triangle count error and composition'],other_metrics='edge error, relations, low degree/hub, MP2/MP3, connectivity; raw graph primary LCC separate; per-model sample mean then across-seed statistics',no_claims=['not from-scratch pure structural training','two training seeds each dataset are screening','more inputs not equal compute/exposure','ordinary CUDA rerun tolerance remains uncertified','no terminal critic or full-chain gradient'],gates='exact serialized full-state restore, same next input and independent RNG, finite unweighted five-risk accumulation divided by B, unchanged sampling contracts, measured full-panel fits budget; no short-term efficacy gate',budget=dict(wall_minutes=240,science_minutes=210,GPU_hours=16,max_GPUs=8,preflight_included=True),stop='on scientific/numerical/pairing failure stop affected task; all required tasks kept, report partial no silent seed/chain reduction',code_sha256=sha(Path(__file__).with_name('worker.py')))
start=time.time();window=dict(start_epoch=start,science_stop_epoch=start+210*60,delivery_stop_epoch=start+240*60,GPU_seconds_cap=16*3600)
for name,x in [('protocol.json',protocol),('bindings.json',bind),('window.json',window),('freeze.json',dict(epoch=start,protocol=hashlib.sha256(json.dumps(protocol,indent=2).encode()).hexdigest()))]:(OUT/name).write_text(json.dumps(x,indent=2))
print(json.dumps(dict(tasks=8,updates=8000,chains=32,window=window,output=str(OUT))))
