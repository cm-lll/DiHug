import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
PREV=ROOT/'results/exact_displacement_updates_20261110'
def main():
    OUT.mkdir(exist_ok=False);p=json.loads((PREV/'protocol.json').read_text());bindings=json.loads((PREV/'bindings.json').read_text())
    for path,h in bindings.items():assert rt.sha(path)==h,path
    for name in ('audit.json','cpu_gate.json'):assert json.loads((PREV/name).read_text())['passed'];bindings[str(PREV/name)]=rt.sha(PREV/name)
    for row in p['tasks']:
        d=OLD/f"{row['dataset']}_{row['seed']}";record=json.loads((d/'updates.json').read_text());record=next(x for x in record if x['arm']=='hybrid' and x['update']==2001)
        row['diagnostic_update']=2001;row['diagnostic_t']=record['t'];row['diagnostic_input_hash']=record['input_hash']
    # Arithmetic audit only; no model or fitting. Remainder != curvature proof.
    a=np.array([1.,2.,3.]);b=np.array([1.2,1.9,3.]);s=np.array([-.1,-.2,0.]);assert np.allclose((b-a)-s,[.3,.1,0.])
    p.update(purpose='Separate same-input finite-step response from other-input response after one frozen displacement-projected AdamW update',arms=['single_displacement_update'],planned_calls=dict(training_forwards=4,same_input_after_forwards=4,other_input_after_forwards=24,VJPs=20,temporary_updates=4,reverse_chains=0),same_input_evaluation='Fresh before probabilities from training forward, fresh after forward in identical train mode; no extra feedback/acceptance. Three exact risks and two independent paired K256 risks.',same_input_evaluation_seed_offset=1104063,other_input_evaluation='Six old fixed Xt/paired K256 readout in eval mode; cached pre-update probabilities/risks reused. Development evidence, not independent confirmation.',interpretation='First-order slopes versus finite differences on same Xt; analytic remainder includes nonlinear/active-region/numerical effects, not proof of oversize LR. Other-time response is localized to six states; no claims about previous four-step bad endpoint root cause.',risk_difference_reporting_tolerance=1e-7,reporting_tolerance_note='Fixed descriptive numerical floor, not statistical significance or acceptance criterion; MC SE separately reported.',budgets=dict(wall_minutes_from_GPU_start=10,science_seconds_from_GPU_start=420,GPU_hours=.15,GPUs=1,CPU_threads=1,RSS_stop=None),training_updates_per_model=1,readout='raw; maintain EMA but do not evaluate EMA')
    for path in Path(__file__).parent.glob('*.py'):bindings[str(path)]=rt.sha(path)
    rt.save(OUT/'protocol.json',p);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'cpu_gate.json',dict(passed=True,projection_gate_reused=str(PREV/'cpu_gate.json'),previous_actual_written_constraints_reused=True,arithmetic_difference_check=True,new_model_calls=0));rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json'),bindings_sha256=rt.sha(OUT/'bindings.json')))
    print(json.dumps(dict(tasks=[(r['dataset'],r['seed'],r['diagnostic_t']) for r in p['tasks']],output=str(OUT),updates=4,forwards=32,VJPs=20)))
if __name__=='__main__':main()
