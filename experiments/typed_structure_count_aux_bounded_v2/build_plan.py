"""Build a non-executable, review-only queue and historical cost extrapolation."""
from pathlib import Path
import json,math,hashlib
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'typed_structure_count_aux_v1'
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((PARENT/'protocol.json').read_text());cost=json.loads((PARENT/'cost.json').read_text())
    tasks=[]
    for t in old['tasks']:
        t=dict(t);t.update(id=f"{t['dataset']}_s{t['train_seed']}_{t['arm']}",updates=500,initial_source='initial_bindings.json',optimizer='fresh AdamW',ema='fresh theta0 EMA0.99',generation_chains=0,paired_initial_id=f"{t['dataset']}_s{t['train_seed']}")
        tasks.append(t)
    plan=dict(status='proposal_only_not_executable',formal_authorized=False,formula_sha256=digest(ROOT/'formula.json'),tasks=tasks,
        checkpoint_updates=[0,100,250,500],fixed_validation_updates=[0,100,250,500],fixed_validation_times=[1,500,1000],validation_readouts=['raw','ema_0.99'],
        gradient_observation_updates=[1,250,500],
        gradient_observation=dict(inputs='the actual two scheduled normal2 inputs, unchanged, with their times and hashes',scope='all trainable parameters; also disjoint backbone/CN-context-shared, early-edge-CN projection, ReFeX projection, role projection and joint-head groups using exact named parameters in integration receipt',procedure='accumulate half-CE and half-weighted-aux parameter gradients separately over both inputs before optimizer.step; preserve model.grad and training RNG; then perform original combined backward/update; CE control records auxiliary shadow gradient but excludes it from update',fields=['CE_norm','weighted_aux_norm','dot','cosine_or_null_with_reason','sum_norm','nonfinite','zero_gradient_count','module_norms','per_relation_loss_share'],extra_backward='two additional reverse-mode passes per input only on three specified updates; no per-step parameter-gradient decomposition',limitations='cosine/norms are gradients, not Adam update attribution; no forcing module gradients to be nonzero'),
        per_update_logs=['CE','unweighted auxiliary','weighted auxiliary','weighted total','t1,t2 and input hashes','all relations mu,m,delta,scale,rho,weighted contribution','actual total gradient finite/norm','optimizer update count','EMA count'],
        stop_conditions=['nonfinite loss/gradient/parameter/optimizer state','support or full-query slot identity error','data/binding error','save/load failure','explicit STOP or approved deadline'],
        analysis_not_automatic_failures=['rare-family dominance','positive/negative CE cost','gradient opposition or imbalance','one parameter unchanged','one seed temporarily worse'],
        finish='all eight stop at500; no generation; preserve failed/missing cells, review whole panel; never select successful seeds to continue',
        continuation='only after explicit new authorization and unchanged formula: restore raw/optimizer/EMA/counter/Torch CPU CUDA/Python NumPy RNG, continue updates501..3000; no reseed, reinitialize or LR reset; then final EMA two fixed seeds per model,16chains; later6000/9000 separately authorized',
        stage1_generation_tasks=[],resume_authorized=False,approval_file_created=False,
        integration_pending=['replace only parent package/count_aux.py with this standalone file and its lambda; unchanged normal2 generic count logging accepts renamed normalized_penalty','adapt old9000 launcher/worker to explicit500 stop without chains; old launcher is NOT this stage entry','add sparse parameter-gradient observations and review-only resume boundary','perform future minimal real GPU update/restore check, not passed by CPU tests'])
    save('stage500_plan.json',plan)
    rows=[];follow=[]
    for t in tasks:
        c=cost['components'][t['dataset']];hist=c['historical_update_with_validation_seconds']
        rate=hist*1.05 # New formula/extra records not measured; explicit planning multiplier.
        parts=dict(training_seconds=500*rate,initialization_and_binding_seconds=60.,extra_validation_seconds=60.,three_gradient_observations_seconds=90.,checkpoint_save_seconds=45.)
        row=dict(id=t['id'],slot=t['slot'],dataset=t['dataset'],**parts);row['serial_seconds']=sum(parts.values());rows.append(row)
        parts2=dict(training_2500_seconds=2500*rate,two_full_chains_seconds=2*c['planned_chain_seconds'],resume_load_binding_seconds=60.,extra_validation_seconds=180.,checkpoint_readout_save_seconds=90.)
        f=dict(id=t['id'],slot=t['slot'],dataset=t['dataset'],**parts2);f['serial_seconds']=sum(parts2.values());follow.append(f)
    stage_compute=max(r['serial_seconds'] for r in rows)
    after_compute=max(r['serial_seconds'] for r in follow)
    # Exact allocation rule when fewer than 8 cards, cost only; no task deletion.
    alternatives={}
    for cards in [8,7,4]:
        loads=[0.]*cards;queue=[[] for _ in range(cards)]
        schedule=rows if cards==8 else sorted(rows,key=lambda r:(-r['serial_seconds'],r['id']))
        for r in schedule:
            card=r['slot'] if cards==8 else min(range(cards),key=lambda i:(loads[i],i))
            queue[card].append(r['id']);loads[card]+=r['serial_seconds']
        compute=max(loads);estimate=compute*1.25+600+900
        alternatives[str(cards)]=dict(queue=queue,load_seconds=loads,compute_wall_seconds=compute,planned_total_seconds=estimate,recommended_budget_seconds=math.ceil(estimate/300)*300)
    budget=dict(status='estimate_only_not_authorization',actual_gpu_availability_checked=False,gpu_model='RTX2080Ti',assumption='eight permitted free cards, one trajectory/card; CPU thread1 per worker; no DDP',
        parent_cost_sha256=digest(PARENT/'cost.json'),measured_history=cost['components'],
        planning=dict(new_loss_logging_multiplier=1.05,new_loss_logging_measured=False,stage_concurrency_multiplier=1.25,concurrency_measured=False,validation_save_gradient_allowances_are_estimates=True),
        stage500=dict(tasks=8,total_normal2_updates=4000,full_chains=0,queue=rows,compute_wall_seconds=stage_compute,concurrency_seconds=.25*stage_compute,fault_reserve_seconds=600,delivery_seconds=900,planned_total_seconds=stage_compute*1.25+1500,recommended_budget_seconds=alternatives['8']['recommended_budget_seconds']),
        alternatives_by_card_count=alternatives,
        future_gpu_integration=dict(authorized_now=False,max_gpus=2,max_temporary_optimizer_steps=6,per_dataset_steps=3,scope='one old-seed theta0 per dataset: update1/save; update2; fresh load after1 and repeat update2. Include optimizer, EMA and all RNG; final formula only. No full chain, no old full验收.',compute_estimate_seconds=2*max(c['historical_update_with_validation_seconds'] for c in cost['components'].values())*3,static_runtime_wiring_allowance_seconds=900,setup_check_and_delivery_budget_seconds=900,total_engineering_proposal_seconds=1800,explanation='15min real GPU integration/check proposal plus15min unimplemented500-stop/gradient-observer wiring; not just GPU kernels; all states engineering-only'),
        followup_500_to_3000=dict(authorized_now=False,tasks=8,additional_updates=20000,full_chains=16,queue=follow,compute_wall_seconds=after_compute,concurrency_planning_multiplier=1.20,fault_reserve_seconds=600,delivery_and_evaluation_seconds=1200,planned_total_seconds=after_compute*1.20+1800,recommended_budget_seconds=math.ceil((after_compute*1.20+1800)/300)*300),
        disk=dict(new_stage_outputs='~5GiB checkpoints plus logs/receipts; suggest >=16GiB free output volume',large_weights_not_copied=True),
        limits=['No new GPU throughput measured','historical rates include old validation overhead; explicit extra allowances conservatively overlap some cost','new formula can alter generation-state CN costs; full-chain times are historical, not a promise','no activated start or deadline; independent review and user budget confirmation pending'])
    save('budget.json',budget)
    # Frozen source input bindings, not a self-issued formal approval.
    source=json.loads((ROOT/'initial_bindings.json').read_text())
    save('provenance.json',dict(parent_commit='1665ff3acc2fa3d76e116e498cbd2b9475453927',parent_directory=str(PARENT),old_loss_sha256=digest(PARENT/'package/count_aux.py'),new_loss_sha256=digest(ROOT/'package/count_aux.py'),formula_sha256=digest(ROOT/'formula.json'),initial_bindings_sha256=digest(ROOT/'initial_bindings.json'),old_scientific_files={k:digest(PARENT/k) for k in json.loads((ROOT/'frozen_science_bindings.json').read_text())},initial_files_not_loaded=True,source_tables={n:digest(ROOT/'evidence'/n) for n in ['source_relation_bounds.csv','source_panel_summary.csv','source_inventory.json']},original_files_unchanged=True))
    print(json.dumps({'stage500':budget['stage500'],'followup_seconds':budget['followup_500_to_3000']['recommended_budget_seconds'],'gpu_preflight_seconds':900},indent=2))
if __name__=='__main__':main()
