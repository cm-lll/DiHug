import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
def main():
    OUT.mkdir(exist_ok=False);tasks=[];bindings={}
    old=ROOT/'results/hard_clean_pilot_v2_20261110'
    for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]:
        for seed in seeds:
            row=json.loads((old/f'{ds}_{seed}_HARD_MGDA/identity.json').read_text())['row'];tasks.append(row)
            for p in (row['checkpoint'],row['data'],row['history']):bindings[p]=rt.sha(p)
            assert bindings[row['checkpoint']]==row['checkpoint_sha256'];assert bindings[row['data']]==row['data_sha256']
    for p,h in json.loads((old/'bindings.json').read_text()).items():
        if p.endswith('.py'):assert rt.sha(p)==h;bindings[p]=h
    for directory in ('exact_feedback_updates_20261110','exact_hard_moments_20261110','hard_clean_zero_safe_20261110'):
        for p in (ROOT/'research_branches'/directory).glob('*.py'):bindings[str(p)]=rt.sha(p)
    for name in ('finite_gate.json','audit.json','legacy_absent_checks.json'):
        p=ROOT/'results/exact_hard_moments_20261110'/name;assert json.loads(p.read_text())['passed'];bindings[str(p)]=rt.sha(p)
    protocol=dict(tasks=tasks,arms=['sampled','hybrid'],updates_each=4,parent_updates=2000,
        train_hard_draws=64,evaluation_hard_draws=256,evaluation_times=[1,75,250,550,800,1000],
        training_hard_seed_offset=1101061,evaluation_seed_offset=1102061,evaluation_hard_seed_offset=1103061,
        loss='equal arithmetic mean of same five hard risks; CE coefficient zero in both arms',
        estimator='sampled: all five K64 LOO; hybrid: exact count/present triangle/absent triangle plus same K64 LOO joint degree/components',
        optimizer='unchanged parent AdamW including CE-trained moments, EMA0.99; no clipping/projection/LR change/step matching',
        initialization='restore same raw model/AdamW/EMA/RNG u2000 per arm; original historical four main inputs hashes',
        pairing='same main inputs and all RNG; same auxiliary uniforms, graphs differ when probabilities differ',
        evaluation='new fixed noisy inputs disjoint from training and previous eval; raw only; six times; exact3 + K256 remaining2 shared uniforms; conditional clean predictions NOT full generation',
        planned_calls=dict(training_forwards=32,backwards=32,evaluation_forwards=72,temporary_updates=32,reverse_chains=0),
        no_claims=['not testing common-direction projection','not from-scratch training','not CE comparison','not full-chain quality','CUDA recomputation equality remains uncertified'],
        budgets=dict(wall_minutes_from_GPU_start=20,science_seconds_from_GPU_start=840,GPU_hours=.25,GPUs=1,CPU_threads=1,RSS_stop=None),
        interpretation='all per-seed outcomes and costs retained; improvement on average without hiding triangle/joint-degree/component/positive-edge tradeoffs; no efficacy gate or automatic follow-up')
    rt.save(OUT/'protocol.json',protocol);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json'),bindings_sha256=rt.sha(OUT/'bindings.json')))
    print(json.dumps(dict(tasks=len(tasks),updates=32,forwards=104,backwards=32,output=str(OUT))))
if __name__=='__main__':main()
