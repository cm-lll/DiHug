"""CPU/read-only asset and budget audit for a complete generated-graph trial.

Does not instantiate a network, create new noise inputs or launch GPU workers.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,statistics
from pathlib import Path
import numpy as np
from gradient_scale import self_test
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero_micro_macro_effect_plan_20261116'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def main():
    start=time.time();OUT.mkdir(exist_ok=False);bindings={}
    def bind(p,expected=None):
        p=Path(p).resolve();h=sha(p)
        if expected is not None:assert h==expected,str(p)
        bindings[str(p)]=h;return p
    def read(p):return json.loads(bind(p).read_text())
    def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
    bind(__file__);bind(Path(__file__).with_name('gradient_scale.py'));bind(Path(__file__).with_name('objective.py'))
    save('CPU_rule_tests.json',self_test())
    check=read(ROOT/'results/hetero_micro_macro_model_check_20261116/protocol.json')
    for p,h in check['bindings'].items():bind(p,h)
    prior=read(ROOT/'results/analytic_joint_generation_plan_20261113/protocol.json')
    panel=read(ROOT/'results/joint_transport_ablation_20261113T011928/protocol.json')
    parents=[];reuse=[];tasks=[];update_costs=[]
    for row in prior['parents']:
        job=row['id'];ds=row['dataset'];seed=row['train_seed']
        assert job in check['jobs']
        initial=ROOT/'results/joint_transport_ablation_20261113T011928'/job/(ds+'_T_eager')/'initial_complete.pt'
        bind(initial);bind(row['parent_readout']);z=np.load(bind(row['data']))
        assert len(z['types'])==192
        rec=read(ROOT/'results/hetero_micro_macro_model_check_20261116'/(job+'.json'))
        update_costs.append(dict(model=job,CE_seconds=rec['arms']['CE']['seconds'],candidate_seconds=rec['arms']['micro_macro']['seconds']))
        samples=[r for r in prior['reused'] if r['model']==job];assert len(samples)==4
        for sample in samples:
            folder=Path(sample['path']);summary=read(folder/'summary.json');identity=read(folder/'identity.json')
            bind(sample['graph'],summary['graph_sha256']);bind(sample['evaluation']);bind(sample['steps'])
            assert summary['status']=='complete' and summary['steps']==1000
            assert identity['model_hash']==row['parent_model_hash'] and identity['sample_seed']==sample['sample_seed']
            assert identity['train_seed']==seed and identity['dataset']==ds and identity['R']==1 and identity['T']==1000
            g=np.load(sample['graph']);assert np.array_equal(g['types'],z['types']) and np.array_equal(g['schema'],z['schema'])
            logs=[json.loads(x) for x in Path(sample['steps']).read_text().splitlines()]
            assert [x['time'] for x in logs]==list(range(1000,0,-1))
            assert all(x['exact_coverage'] and x['uniform_hash'] and x['query_hash'] for x in logs)
            reuse.append(sample)
            for arm in ['CE','calibrated_micro_macro']:
                tasks.append(dict(model=job,dataset=ds,train_seed=seed,arm=arm,feedback_updates=128,readout='EMA0.99',
                    sample_seed=sample['sample_seed'],paired_parent=str(folder),T=1000,R=1))
        parents.append(dict(**row,initial_complete=str(initial),initial_complete_sha256=sha(initial),nodes=192))
    assert len(parents)==4 and len(reuse)==16 and len(tasks)==32
    old_costs=prior['budgets']['history']
    chain_hours=sum(8*x['seconds_mean']/3600 for x in old_costs)
    train_hours=sum(128*(x['CE_seconds']+x['candidate_seconds'])/3600 for x in update_costs)
    large=[]
    inventory=read(ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_final_screen_v1/evidence/full_graph_inventory.json')
    for ds,row in inventory['datasets'].items():
        path=Path(row['path']);item=dict(dataset=ds,path=str(path),historical_nodes=row['nodes'],historical_edges=row['edges'],source_inventory_only=True)
        if path.exists():
            bind(path,row['sha256']);z=np.load(path);n=len(z['types']);e=z['edges'];e=e if e.shape[0]==2 else e.T
            item.update(source_inventory_only=False,nodes=n,edges=int(e.shape[1]),relations=len(z['schema']),
                dense_probability_tensor_GiB=len(z['schema'])*n*n*8/2**30,
                N_squared_ratio_to192=(n/192)**2,N_cubed_ratio_to192=(n/192)**3,
                warning='one tensor only, excludes intermediates/autograd/model; not a peak-memory measurement')
        large.append(item)
    protocol=dict(status='prepared_not_executed',GPU_or_training_started=False,requires_new_unified_budget=True,
        purpose='Fix a scale rule, then test matched complete generation, not another short-effect gate',
        parents=parents,reused=reuse,tasks=tasks,counts=dict(parents=4,arms=2,models=8,updates_per_model=128,total_updates=1024,new_chains=32,reused_graphs=16,unique_graphs=48),
        calibration=dict(inputs=24,network_forwards=24,network_VJPs=96,times=[1,75,250,550,800,1000],
            states='same sealed four parents; old update inputs, equal dataset/seed/time weighting',
            blocks=['degree','closure','propagation'],rule='c_b=RMS_panel(||grad CE||)/(3*RMS_panel(||grad L_b||)); L=CE+sum_b c_b L_b',
            block_definition='mean_type e/(4*frozen_sigma_squared), no old lambda',
            frozen=True,no_quality_selection=True,guarantee='calibration-panel macro gradient RMS <= CE gradient RMS by triangle inequality; not per-state or future-state guarantee'),
        training=dict(time_rule='original single IID uniform integer t1..1000 each update',input_root=2026111611,
            validation_root=2026111612,validation_times=[1,75,250,550,800,1000],validation_repeats=2,
            checkpoint_updates=[0,32,64,128],optimizer='same fresh feedback AdamW lr1e-5, clip1, EMA.99 for both arms',
            evaluation_readouts='raw and EMA conditional validation; only fixed final EMA for full generation',
            no_best_checkpoint=True,no_stepwise_rollback_or_extra_forward_guard=True),
        engineering_gates=['finite nonzero calibrated gradients and fixed shared coefficients',
            'input/query/schema/RNG bindings match across arms', 'saved complete state restore including Adam/EMA/RNG/count',
            'full queue fits remaining time/GPU hours after initial work; no shortened panel',
            'no short-effect threshold, no weight search'],
        readout_counts=dict(shared_parent_validation_forwards=48,arm_checkpoint_raw_EMA_validation_forwards=576,
            training_forwards=1024,training_backwards=1024,calibration_forwards=24,calibration_VJPs=96,
            planned_before_recovery_forwards=1672,planned_before_recovery_backwards=1120,
            recovery_reserve_forwards=16,recovery_reserve_backwards=16),
        budget=dict(historical_chain_costs=old_costs,measured_update_costs=update_costs,
            generation_body_GPU_hours=chain_hours,training_body_GPU_hours=train_hours,
            requested_GPU_hours=8,wall_minutes=120,science_minutes=100,delivery_minutes=20,max_GPU=8,
            calibration_and_wiring_GPU_hours_reserve=.5,validation_and_recovery_GPU_hours_reserve=.5,
            loading_saving_concurrency_GPU_hours_reserve=.5,
            remaining_margin_GPU_hours=8-chain_hours-train_hours-1.5,
            qualification='body extrapolation; not guaranteed concurrent throughput, allocation through release counts'),
        decision=dict(primary=['joint relation degree L1','typed closure distribution','MP3 composition'],
            mandatory_costs=['degree W1','joint TV','triangle counts/composition','MP2/MP3 counts','low/hub','edges by relation','components/LCC','duplicates and edge overlap'],
            interpretation='seed-level paired results and sample variability separately; two seeds/data support screening only',
            continuation='only consider larger/new-seed confirmation if improvements repeat within each dataset and no substantial repeated structure/density/diversity cost; do not infer significance or pool datasets to hide reversals'),
        large_graphs=large,large_graph_execution_included=False,
        large_graph_limit='Current reference descriptors use dense N^2 probability tensors and cubic matrix products; query blocks alone do not bound descriptor backward memory. Recompute/blocked VJP design needs equivalence and capacity validation before full-graph runs.',
        future='If complete small-graph evidence is promising, implement memory-bounded exact/recomputed statistic path and assess both full graphs; do not auto launch large graphs',
        failures='Stop affected task on nonfinite/state/support error; preserve all results. No shorter chains or changed seeds/weights/deadlines.')
    save('protocol.json',protocol);save('asset_bindings.json',bindings)
    for p,h in bindings.items():assert sha(p)==h,p
    save('audit.json',dict(status='CPU_asset_plan_complete',seconds=time.time()-start,bindings=len(bindings),new_model_calls=0,GPU_hours=0,
        source_rule_tests='passed',generation_adapter_status='existing step16 guard must be parameterized with strict manifest step128 check; not yet GPU-wired',
        remaining='GPU calibration, training loop and generation adapter implementation and acceptance under new budget'))
    print(json.dumps(dict(output=str(OUT),generation_GPU_hours=chain_hours,training_GPU_hours=train_hours,remaining_margin=protocol['budget']['remaining_margin_GPU_hours'],large=large),indent=2))
if __name__=='__main__':main()
