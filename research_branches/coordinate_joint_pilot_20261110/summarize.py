import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/coordinate_joint_pilot_20261110'
def main():
    scores=list(csv.DictReader((OUT/'scores.csv').open()));decisions=list(csv.DictReader((OUT/'decisions.csv').open()));groups={}
    for x in scores:groups.setdefault((x['dataset'],x['seed'],x['panel'],x['alpha']),{})[x['arm']]=x
    near=[x for x in decisions if float(x['same_norm_beta'])<1e-8];compare={}
    for baseline in ('original','unchanged','same_norm_scale'):
        delta=[float(v['coordinate_projected']['mean_five'])-float(v[baseline]['mean_five']) for v in groups.values()];compare[baseline]=dict(lower=sum(x < -1e-7 for x in delta),within_reporting_floor=sum(abs(x)<=1e-7 for x in delta),higher=sum(x>1e-7 for x in delta))
    summary=dict(conditions=len(groups),exact_zero_count=sum(x['zero']=='True' for x in decisions),numerically_near_zero_count=len(near),near_zero_norm_ratio_max=max(float(x['same_norm_beta']) for x in near),near_zero_reporting_threshold=1e-8,warning='Near-zero designation is a reporting aid, not a changed solver/acceptance threshold; no rerun.',mean_comparisons=compare,mean_reporting_floor=1e-7,statistical_significance_test=False,model_summaries=[])
    for ds,seed in sorted(set((x['dataset'],x['seed']) for x in decisions)):
        for a in (.25,.5,1.):
            r=[x for x in scores if x['dataset']==ds and x['seed']==seed and float(x['alpha'])==a and x['panel']!='same'];summary['model_summaries'].append(dict(dataset=ds,seed=int(seed),alpha=a,six_time_mean={arm:sum(float(x['mean_five']) for x in r if x['arm']==arm)/6 for arm in ('unchanged','original','coordinate_projected','same_norm_scale')}))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));paths=list(OUT.glob('*'))+list((ROOT/'research_branches/coordinate_joint_pilot_20261110').glob('*.py'))+list((ROOT/'research_branches/coordinate_joint_pilot_20261110').glob('*.md'));manifest={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths if p.is_file() and p.name!='artifact_manifest.json'};(OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(conditions=len(groups),near_zero=len(near),comparison=compare)))
if __name__=='__main__':main()
