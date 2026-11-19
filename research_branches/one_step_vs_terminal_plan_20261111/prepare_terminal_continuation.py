"""Read-only asset audit and fixed terminal-feedback continuation plan.

CPU deserialization only: no model construction, forward, update or sampling.
New compute budget must be bound before a production launcher consumes plan.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/three_arm_learning_pilot_20261111T211641'
OUT=ROOT/'results/terminal_continuation_plan_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    tick=time.time();OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    old=read(OLD/'protocol.json');assets=[];schedule={};cost=[];bindings={}
    for ds in ('IMDB','PubMed'):
        folder=OLD/(ds+'_T_eager');checkpoint=folder/'final_complete.pt'
        z=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert z['updates']==z['ema']['updates']==2 and z['recovery']['exact_resume_allowed']
        assert {float(v['step']) for v in z['optimizer']['state'].values()}=={2.}
        assert z['binding']['protocol_sha256']==sha(OLD/'protocol.json')
        for key in ('model','optimizer','ema','rng','binding','updates'):assert key in z
        param_groups=z['optimizer']['param_groups'];assert all(g['lr']==1e-5 for g in param_groups)
        row=old['parents'][ds]
        retained=[]
        for arm in ('baseline','T_eager'):
            for ss in old['sample_seeds']:
                d=(ROOT/f'results/three_arm_update_preflight_20261111/{ds}/baseline_sample_{ss}'
                   if arm=='baseline' and ss==2026111341 else OLD/(ds+'_'+arm)/f'final_sample_{ss}')
                receipt=read(d/'summary.json');assert receipt['status']=='complete' and receipt['steps']==1000
                assert sha(d/'graph.npz')==receipt['graph_sha256']
                retained.append(dict(arm=arm,sample_seed=ss,path=str(d/'graph.npz'),sha256=receipt['graph_sha256']))
                for name in ('graph.npz','identity.json','summary.json','steps.jsonl'):bindings[str(d/name)]=sha(d/name)
        bindings[str(checkpoint)]=sha(checkpoint)
        for key in ('data','readout'):bindings[row[key]]=sha(row[key])
        assets.append(dict(dataset=ds,train_seed=row['train_seed'],complete_checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),updates=2,optimizer_steps=[2],EMA_updates=2,retained_graphs=retained,
            complete_state_present=True,GPU_restore_check_pending=True))
        schedule[ds]={}
        for u in range(3,9):
            schedule[ds][str(u)]=[]
            for k in range(2):
                rng=np.random.default_rng(np.random.SeedSequence([2026111505,row['train_seed'],u,k]))
                times=sorted(map(int,rng.choice(np.arange(1,1001),8,replace=False)))
                seed=2026111501+(u-1)*2+k
                assert seed not in old['sample_seeds'] and seed>2026111504
                schedule[ds][str(u)].append(dict(seed=seed,times=times))
        observed=[read(f)['seconds'] for f in folder.glob('*/summary.json')]
        train=read(folder/'training_summary.json')
        training_chains=[read(f)['seconds'] for f in folder.glob('train_round*/summary.json')]
        overhead=max(0.,train['training_seconds']-sum(training_chains))/2
        chain_seconds=480 if ds=='IMDB' else 650
        nominal=16*chain_seconds+6*overhead+90
        cost.append(dict(dataset=ds,observed_chain_seconds=observed,planning_seconds_per_chain=chain_seconds,
            observed_nonchain_training_overhead_seconds_per_update=overhead,additional_updates=6,
            new_training_chains=12,new_evaluation_chains=4,gradient_forwards=96,backwards=96,
            endpoint_validation_forwards=24,nominal_occupied_seconds=nominal,
            historical_peak_allocated_MiB=read(folder/'receipt.json')['peak_allocated_MiB']))
        del z
    for name in ('pilot_worker.py','heat_solver_candidate.py','prepare_terminal_continuation.py'):
        f=Path(__file__).with_name(name);bindings[str(f)]=sha(f)
    bindings[str(OLD/'protocol.json')]=sha(OLD/'protocol.json')
    nominal_gpu=sum(c['nominal_occupied_seconds'] for c in cost)/3600
    plan=dict(status='prepared_budget_not_yet_bound_no_compute_started',track='CORE',
        hypothesis='Whether additional current-policy terminal-feedback updates improve independent complete generation rather than merely stay near parent',
        method='T_eager terminal-cost clean-head surrogate; biased, not exact terminal transition policy gradient or full GDPO reproduction',
        datasets=['IMDB','PubMed'],assets=assets,bindings=bindings,terminal_plan=schedule,
        resume_updates=2,final_updates=8,evaluate_at_updates=[4,8],evaluation_seeds=old['sample_seeds'],
        reused_evaluation_graphs=8,new_training_chains=24,new_evaluation_chains=8,
        full_chain_T=1000,R=1,raw_primary=True,EMA_decay=.99,optimizer_lr=1e-5,
        loss_scale=.01,gradient_clip=1,reward_standardization=False,reward_clipping=False,
        terminal_draws_per_round=2,selected_uniform_times_per_trajectory=8,
        surrogate='L=.01/16 * sum_{k=1,2} (C_k-C_other) * sum_{t in selected8_k} sum_e log p_theta(G0_k,e | Xt_k,t,t)',
        no_target_gradients=True,no_CE_auxiliary=True,no_sampler_change=True,
        score='unchanged typed joint heat-degree MMD plus per-relation count risk; fixed reference scales; evd numerical repair',
        checkpoints='restore model/AdamW/EMA/RNG/counts; save complete state each round; no optimizer reset',
        collection='new on-policy trajectories each round at frozen current raw weights; never reuse old trajectories after an update',
        evaluation='fixed seeds at both stages; never used in gradients/reward calibration; test history already seen, development panel only; no best-checkpoint selection',
        comparison='baseline, old update2, new update4, new update8; S1/S16 separate historical arms; not equal-update estimator comparison',
        engineering_gate='binding/finite state/full restore and RNG unchanged; same sampler/coverage contract; solver residuals retain original limits; no short quality gate',
        interpretation='report all mean/per-sample structure changes; continuing improvement at4/8 only preliminary; one training seed per dataset cannot establish cross-seed stability',
        cost=cost,nominal_GPU_hours=nominal_gpu,nominal_parallel_wall_minutes=max(c['nominal_occupied_seconds'] for c in cost)/60,
        proposed_budget=dict(max_GPUs=2,science_minutes=200,total_minutes=220,GPU_hours=6.5,
            accounting='allocation through release includes loading/checkpoints/waiting; CPU finalization after release',
            stop='first applicable science/card-time limit; no reduced chains/rounds presented as complete'),
        no_actual_start_or_deadline_bound=True,no_long_training_started=True,
        source_cpu_seconds=time.time()-tick)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(assets=len(assets),new_chains=32,reused_graphs=8,nominal_GPU_hours=nominal_gpu,
        nominal_wall_minutes=plan['nominal_parallel_wall_minutes'],cost=cost,seconds=time.time()-tick),indent=2))
if __name__=='__main__':main()
