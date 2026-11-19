"""Saved scalar diagnostics and variance planning; no model or graph generation."""
import json,statistics,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/typed_edit_revision_20261116'
OUT=ROOT/'results/typed_edit_complexity_20261116'
def read(p):return json.loads(p.read_text())
def main():
    records=[];sources={}
    for p in sorted((OLD/'delicious/outputs').rglob('report.json')):
        r=read(p);arm=p.parts[-4];seed=int(p.parts[-3][5:]);prefix=int(p.parts[-2][6:])
        tr={int(x['step']):x for x in r['trace']};e=[tr[k]['relation_degree_error_per_node'] for k in [0,5000,8000,9000,10000]]
        records.append(dict(arm=arm,seed=None if arm=='N' else seed,prefix=prefix,
            residual_0=e[0],residual_5000=e[1],residual_8000=e[2],residual_9000=e[3],residual_10000=e[4],
            reduction_last2000=e[2]-e[4],reduction_last1000=e[3]-e[4],
            noop_fraction=r['action_counts']['noop']/10000,actions=r['action_counts'],
            applied=r['applied'],effective_role_weight=tr[0]['effective_role_weight']))
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    groups={}
    for arm in ['M','F','N']:
        rr=[x for x in records if x['arm']==arm]
        keys=['residual_0','residual_5000','residual_8000','residual_10000','reduction_last2000','reduction_last1000','noop_fraction','effective_role_weight']
        groups[arm]={k:{'mean':statistics.mean(x[k] for x in rr),'min':min(x[k] for x in rr),'max':max(x[k] for x in rr)} for k in keys}
        groups[arm]['improving_last1000']=sum(x['reduction_last1000']>0 for x in rr)
    a=read(OLD/'r2f/generation_acceptance.json');b=read(OLD/'delicious/independent_acceptance.json')
    contrasts={**{'r2f_'+ds:list(r['per_seed'].values()) for ds,r in a['contrasts'].items()},
        **{'delicious_'+k:r['per_seed'] for k,r in b['contrasts'].items()}}
    planning={}
    for name,x in contrasts.items():
        sd=statistics.stdev(x)
        planning[name]={'pilot_seeds':3,'sd_of_paired_seed_means':sd,
            'normal_95_halfwidth_ignoring_start_uncertainty':{str(n):1.96*sd/math.sqrt(n) for n in [3,5,8,12,20]},
            'warning':'three-seed pilot; descriptive precision only, not power guarantee or estimated new-domain variance'}
    result={'saved_reports':len(records),'groups':groups,'records':records,'variance_planning':planning,
        'source_sha256':sources,'training':0,'generation':0,'model_calls':0,
        'interpretation':['complete fixed-budget outputs remain comparable','no Delicious stage-two effect is estimable',
        'periodic residuals and total NOOP cannot isolate candidate coverage, score conflict or unique failure cause',
        'extrapolation from final slope does not certify eventual arrival']}
    (OUT/'existing_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'groups':groups,'variance_planning':planning},indent=2))
if __name__=='__main__':main()
