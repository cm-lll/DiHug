"""Asset/hash and count planning only. No model import, GPU, or graph generation."""
import hashlib,json,time,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'results/direct_medium_plan_20261108';O.mkdir(exist_ok=True)
old=R/'results/direct_structure_response_20261108T205816'
trace=R/'results/direct_step_trace_20261108T213749'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def seed(label):return int.from_bytes(hashlib.sha256(('direct_medium_evaluation_v1:'+label).encode()).digest()[:4],'little')
assets=read(old/'assets.json');historical=read(trace/'asset_binding.json');checks=read(trace/'checks.json')
tasks=[];bindings={}
for a in assets:
    parent=Path(a['parent']);assert sha(parent)==a['sha256'];bindings[str(parent)]=a['sha256']
    for arm in ['CE','CE_STRUCT']:
        check=next(x for x in checks if x['condition']==a['condition'] and x['arm']==arm)
        assert all(check['complete_endpoint_exact'].values())
        p=Path(check['last']);actual=sha(p);assert actual==historical[str(p)];bindings[str(p)]=actual
        tasks.append(dict(condition=a['condition'],dataset=a['condition'].split('_')[0],training_seed=a['seed'],arm=arm,
            common_origin=str(parent),common_origin_sha256=a['sha256'],resume_offset=8,resume_path=str(p),resume_sha256=actual,
            total_local_steps=64,new_steps=56,final_absolute_update=1096 if a['condition'].startswith('IMDB') else 2096))
sources={}
for file in [old/'source_binding.json',trace/'source_binding.json',R/'results/hard_structure_gradient_gpu_20261108T180629/source_binding.json']:
    for p,h in read(file).items():
        assert sha(p)==h, p;sources[p]=h
panels={}
for ds in ['IMDB','PubMed']:
    trng=np.random.default_rng(seed(ds+':time'))
    items=[]
    for layer in range(10):
        for replicate in range(2):
            key=f'{ds}:layer{layer}:rep{replicate}'
            items.append(dict(layer=layer+1,replicate=replicate,t=int(trng.integers(100*layer+1,100*layer+101)),
                noise_seed=seed(key+':noise'),conditional_graph_seed=seed(key+':graphs'),conditional_draws=32))
    panels[ds]=dict(time_seed=seed(ds+':time'),inputs=items)
plan=dict(status='prepared_not_executed',prepared_epoch=time.time(),scope='bounded diagnostic extension, no formal models or T1000 chains',
    tasks=tasks,evaluation=panels,coefficient=.001,objective='same frozen DIRECT; no loss/optimizer/runtime changes',
    milestones=[0,8,32,64],readout='raw only; EMA saved but not scored',
    counts=dict(new_updates=448,new_backwards=448,training_forwards=448,
        evaluation_forwards_new_panel=560,evaluation_forwards_old_t500=16,total_forwards=1024,
        new_conditional_graph_draws=17920,full_reverse_chains=0),
    count_details='4 shared origins*20 + 8 arms*3 later milestones*20 = 560; old t500:8 arms*2 new milestones=16; 560*32 conditional draws',
    budget=dict(proposed_wall_minutes=75,proposed_science_minutes=60,proposed_delivery_minutes=15,
        proposed_GPU_hours=.75,max_GPUs=1,absolute_start=None,absolute_deadline=None,
        update_reserved_seconds=448*3,evaluation_reserved_seconds=576*1.2,loading_save_reserved_seconds=300,
        runtime_margin_seconds=330,CPU_score_estimate_seconds=240,preparation_reserved_seconds=600),
    limitations=['one training seed per dataset, two histories per seed, not four independent seeds',
        'all are previously selected development trajectories; not clean from-zero comparison',
        'new stratified evaluation approximates uniform-time risk; finite 20 inputs per dataset',
        'sampling 32 independent-edge p0 graphs per input is NOT T1000 generation',
        '64 local updates cannot establish final generation quality or long-term convergence'],
    rules=['two arms continue their own fully recovered AdamW/EMA/RNG; no reset',
        'main inputs/hash/RNG matched to historical flow; auxiliary RNG continues after offset8, not reseeded',
        'new inputs constructed and hash-frozen before any model evaluation; no redraw by outcome',
        'retain old t500 separately, no extra weighting in uniform-time panel',
        'stop affected branch on state/support/nonfinite mismatch; stop at first budget cap',
        'first ready task calibrates cost within counted updates; no additional trial updates',
        'quality outcomes are not runtime elimination criteria; keep every history and time'],
    storage_free_bytes=shutil.disk_usage(R).free,storage_new_estimate_GiB=4)
for n,v in [('plan.json',plan),('assets.json',bindings),('source_binding.json',sources)]:
    (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
print(json.dumps(dict(tasks=len(tasks),assets=len(bindings),sources=len(sources),times={ds:[x['t'] for x in p['inputs']] for ds,p in panels.items()},budget=plan['budget']),ensure_ascii=False,indent=2))
