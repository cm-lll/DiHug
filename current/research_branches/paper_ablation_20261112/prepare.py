"""Read-only asset binding, fixed panel and compute gates. No model forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,datetime,hashlib,zipfile
from pathlib import Path
from scoring import *
HERE=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):base.write(p,x)
def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False);(out/'logs').mkdir()
    now=time.time();samples=[2026102821,2026102822,2026110231,2026110232]
    old=read(ROOT/'results/terminal_continuation_20261112T003000/protocol.json')
    first=read(ROOT/'results/three_arm_learning_pilot_20261111T211641/protocol.json')
    assets=read(ROOT/'results/oa_ce_progress_audit_20261104/assets.json')
    graphs=read(ROOT/'results/oa_ce_progress_audit_20261104/generated_inventory.json')
    seeds={'IMDB':[2026110411,2026110412],'PubMed':[2026110421,2026110422]}
    bindings={};parents={};reused=[];jobs=[];rows=[]
    def bind(path,expected=None):
        path=str(Path(path).resolve());h=base.sha(path)
        if expected is not None:assert h==expected,(path,h,expected)
        bindings[path]=h;return path
    for ds,ss in seeds.items():
        data=bind(old['parents'][ds]['data'],old['bindings'][old['parents'][ds]['data']])
        for seed in ss:
            plans={str(u):(first['terminal_plan'][ds][str(u)] if u<3 else old['terminal_plan'][ds][str(u)]) for u in range(1,5)}
            if seed==ss[1]:
                rng=np.random.default_rng(np.random.SeedSequence([2026111219,seed]))
                plans={str(u):[dict(seed=int(rng.integers(1,2**31-1)),times=sorted(map(int,rng.choice(np.arange(1,1001),8,replace=False)))) for _ in range(2)] for u in range(1,5)}
            for arch in ('O','A'):
                a=next(x for x in assets if x['dataset']==ds and x['train_seed']==seed and x['arm']==arch)
                folder=Path(a['folder']);r=folder/'readout_ema_0.99.pt';z=torch.load(r,map_location='cpu',weights_only=False)
                assert z['updates']==3000 and z['readout']=='ema_0.99' and z['train_seed']==seed
                cfg=z['binding']['config'];assert cfg['inputs_per_update']==1 and cfg['T']==1000 and cfg['rounds']==1
                for c in a['checkpoints']:
                    if c['update'] in (0,3000):bind(c['path'],c['sha256'])
                bind(r);bind(folder/'binding.json');bind(folder/'environment.json')
                pid=f'{ds}_{seed}_{arch}';parents[pid]=dict(dataset=ds,train_seed=seed,architecture=arch,
                    readout=str(r),data=data,config=cfg,type_names=a['data_binding']['type_names'])
                for sample in samples:
                    g=next(x for x in graphs if x['task_id']==a['task_id'] and x['sample_seed']==sample and x['updates']==3000)
                    gp=bind(g['graph_file'],g['graph_sha256']);root=Path(gp).parent
                    ident=read(root/'identity.json');summary=read(root/'summary.json');assert summary.get('steps',summary.get('executed_steps'))==1000
                    for f in ('identity.json','binding.json','summary.json','steps.jsonl'):bind(root/f)
                    reused.append(dict(dataset=ds,train_seed=seed,arm=arch,sample_seed=sample,path=gp,parent=pid))
                for reward in (('typed',) if arch=='O' else ('typed','generic')):
                    jid=f'{pid}_{reward}';cached=None
                    if arch=='A' and reward=='typed' and seed==ss[0]:
                        cached=ROOT/f'results/terminal_continuation_20261112T003000/{ds}_T_eager/stage_4/final_complete.pt'
                        zz=torch.load(cached,map_location='cpu',weights_only=False);assert zz['updates']==4 and zz['recovery']['exact_resume_allowed']
                        # Parent identity and actual first-four input plans are retained, not guessed.
                        b=read(cached.parent/'binding.json');assert b['parent_readout_sha256']==bindings[str(r)]
                        for root in (ROOT/f'results/three_arm_learning_pilot_20261111T211641/{ds}_T_eager',cached.parent.parent):
                            for f in ('binding.json','updates.jsonl'):bind(root/f)
                        bind(cached);cached=str(cached)
                    chain_s={'IMDB':480.,'PubMed':650.}[ds]
                    seconds=((4 if cached else 12)*chain_s+300)*1.20
                    jobs.append(dict(id=jid,parent=pid,dataset=ds,train_seed=seed,architecture=arch,reward=reward,
                        terminal_plan=plans,cached_complete=cached,planning_seconds=seconds,new_updates=0 if cached else 4))
                rows.append(dict(parent=pid,parameters=sum(v.numel() for v in z['model'].values()),readout_sha256=bindings[str(r)],data_sha256=bindings[data]))
                del z
    # Bind the actual imported production dependencies as well as isolated wrappers.
    for key,h in old['bindings'].items():
        if key.endswith('.py') or key.endswith('reference_scaling.json'):bind(key,h)
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(ROOT) in str(f) and str(f).endswith('.py'):bind(f)
    for f in HERE.glob('*.py'):bind(f)
    bind(ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json')
    p=dict(start_epoch=now,science_stop_epoch=now+6*3600,delivery_epoch=now+6.5*3600,
        max_GPU_hours=28.,max_GPUs=6,allowed_GPU_indices=[0,1,2,4,5,6,7],preflight_GPU_hours=0.,
        jobs=jobs,parents=parents,reused_graphs=reused,sample_seeds=samples,terminal_updates=4,bindings=bindings,
        scope='Two datasets, two historical regular training seeds: O,A,O+typed,A+typed,A+generic. No new CE pretraining.',
        generic='Type-blind counterpart of frozen score, NOT an official GDPO reproduction. Model schema/support unchanged.',
        frozen=dict(lr=1e-5,clip_gradient=1.,loss_outer_scale=.01,trajectories_per_update=2,feedback_times_per_trajectory=8,
            baseline='leave-one-out terminal costs; biased eager clean-head surrogate',EMA=.99,readout='parent EMA3000 / fine-tuned final raw',T=1000,R=1),
        gate='All parents/schema/readout/support bound; CPU score identity and invariance; actual O/A finite update, complete state reload, input RNG and sampling integration; full queue within remaining wall/GPU allocation. No quality gate.',
        primary_metrics=['relation degree W1','joint relation-degree TV','typed triangle count/composition','MP2/MP3 count/composition','LCC/isolates'],
        secondary_metrics=['edge count','low-degree/hub per type','four cycles','pairwise overlap and exact typed isomorphism with timeout'],
        conclusions='No scalar winner threshold or posthoc quality tolerance. Per-seed differences and interaction; quality and diversity reported separately with all graphs. Two training seeds/data and four draws/model remain screening.',
        limitations=['historical development seeds, not independent confirmation','feedback includes extra compute; no compute-matched CE continuation control in this ticket',
            'generic vs typed changes grouping and reference scales as a complete reward design, not only a type label toggle',
            'CUDA recomputed update bitwise equality remains uncertified','all trajectories on-policy; same random draws can pair but graph states differ',
            'four terminal updates is a fixed minimal comparison, not convergence','O/A architectures and heads differ; shared initial parameters only'],
        counts=dict(base_models_reused=8,base_graphs_reused=32,feedback_states_reused=2,new_feedback_models=10,
            new_feedback_updates=40,new_training_chains=80,new_evaluation_chains=48,total_evaluation_graphs=80),
        budget=dict(historical_chain_seconds={'IMDB':480,'PubMed':650},margin=1.20,
            estimated_GPU_hours=sum(j['planning_seconds'] for j in jobs)/3600,queue='longest remaining first; finish-release; no GPU3',
            accounting='reservation through process exit incl loading/saving/wait; preflight included; CPU evaluation after release'))
    assert p['budget']['estimated_GPU_hours']+.15<28
    # Simulate six-card ready queue. Real availability checked again at launch.
    loads=[0.]*6
    for j in sorted(jobs,key=lambda x:-x['planning_seconds']):loads[loads.index(min(loads))]+=j['planning_seconds']
    p['budget']['predicted_six_card_minutes']=max(loads)/60
    assert max(loads)+900<6*3600
    write(out/'protocol.json',p);write(out/'asset_summary.json',rows)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as f:
        for path in bindings:
            if path.endswith('.py'):f.write(path,str(Path(path).relative_to(ROOT)))
    print(json.dumps(dict(output=str(out),counts=p['counts'],budget=p['budget']),indent=2))
if __name__=='__main__':main()
