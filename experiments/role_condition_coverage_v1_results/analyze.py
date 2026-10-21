"""Read-only results tables/curves. No fitting, sampling or checkpoint selection."""
import csv,json,os,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'experiments/role_condition_coverage_v1'))
from common import RESULT,read,save,sha,settings,stat
import numpy as np

def main():
    start=time.monotonic();cfg=settings();controller=read(RESULT/'controller.json')
    if controller['status']!='completed':
        actual=dict(status=controller['status'],controller=controller,
            actual_updates=sum(sum(1 for _ in p.open()) for p in RESULT.glob('jobs/*/updates.jsonl')),
            actual_graphs=len(list(RESULT.glob('jobs/*/generated/*_result.json'))),
            planned_updates=36000,planned_graphs=528,
            valid_checkpoints=[json.loads(x) for p in RESULT.glob('jobs/*/checkpoint_index.jsonl') for x in p.read_text().splitlines()])
        save(RESULT/'partial_results.json',actual);print(json.dumps(actual));return
    assert read(RESULT/'audit.json')['status']=='passed'
    per=read(RESULT/'per_seed.json');summaries=per['summaries'];conditions=read(RESULT/'conditions.json')
    rows=[];macros=[];paired=[]
    fields=['G_W1','G_joint_TV','G_max','G_second_max','G_high_ge8','G_high_ge12','G_multiple_ge8','G_low_le1','G_zero',
            'family1_signed_error','family2_signed_error','family5_signed_error','family8_signed_error',
            'Director_joint_TV','Actor_joint_TV','Movie_joint_TV','Movie_relation8_W1',
            'edges','connected','lcc','isolates','triangles','closure_path','closure_pair','MP2_total','MP3_total']
    for s in summaries:
        rows.append(dict(arm=s['arm'],seed=s['seed'],condition=s['condition'],category=s['category'],
              **{k:s['statistics'][k]['mean'] for k in fields},
              topology_classes=s['topology_classes'],topology_frequencies=json.dumps(s['topology_frequencies']),
              G_sorted_degree_mean=json.dumps(s['G_sorted_degree_mean']),G_signed_rank_bias=json.dumps(s['G_signed_rank_bias'])))
    with (RESULT/'per_seed_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in conditions['final_targets']:
        for arm in cfg['arms']:
            group=[s for s in summaries if s['arm']==arm and s['condition']==name]
            assert len(group)==3
            r=dict(arm=arm,condition=name,category=group[0]['category'],target=conditions['vectors'][name],
                **{k:stat([s['statistics'][k]['mean'] for s in group]) for k in fields},
                topology_classes=[s['topology_classes'] for s in group],
                sorted_G_mean=np.mean([s['G_sorted_degree_mean'] for s in group],axis=0).tolist())
            macros.append(r)
        diff=[r for r in per['paired_differences'] if r['condition']==name]
        paired.append(dict(condition=name,category=diff[0]['category'],
            differences={k:stat([r['mean_broad_minus_narrow'][k] for r in diff]) for k in fields}))
    save(RESULT/'macro_summary.json',dict(unit='equal1/3 training-seed means;8 draws per model/condition',
         conditions=macros,paired_broad_minus_narrow=paired,no_global_quality_average=True))
    training=[];validation=read(RESULT/'validation_curves.json')
    for arm in cfg['arms']:
        for seed in cfg['train_seeds']:
            p=RESULT/'jobs'/f'{arm}_{seed}'
            updates=[json.loads(x) for x in (p/'updates.jsonl').read_text().splitlines()]
            for lo in range(0,6000,100):
                block=updates[lo:lo+100];training.append(dict(arm=arm,seed=seed,end_update=lo+100,
                    mean_training_CE=float(np.mean([r['loss'] for r in block])),
                    update_seconds=stat([r['seconds'] for r in block]),gradient_norm=stat([r['gradient_norm'] for r in block])))
    save(RESULT/'training_curves.json',dict(rows=training,note='CE means average different condition distributions across arms; fixed validation is matched'))
    vr=[]
    for v in validation:
        by_condition=defaultdict(list);by_time=defaultdict(list)
        for r in v['rows']:by_condition[r['condition']].append(r);by_time[r['t']].append(r)
        groups={**{n:g for n,g in by_condition.items()},**{'t'+str(t):g for t,g in by_time.items()}}
        for name,g in groups.items():
            count=np.sum([r['counts_by_family_label'] for r in g],axis=0)
            loss=np.sum([r['ce_sum_by_family_label'] for r in g],axis=0)
            vr.append(dict(arm=v['arm'],seed=v['seed'],update=v['update'],group=name,
                mean_full_schema_CE=float(np.mean([r['ce'] for r in g])),counts=count.tolist(),
                CE_by_family_label=np.divide(loss,count,out=np.full(loss.shape,np.nan),where=count>0).tolist(),
                expected_edge_bias_mean=np.mean([r['expected_edge_bias'] for r in g],axis=0).tolist()))
    # Explicit NA rather than JSON NaN for unobserved relation-label combinations.
    def finite(x):
        if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
        if isinstance(x,list):return [finite(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    save(RESULT/'validation_summary.json',finite(vr))
    resources=[]
    for p in RESULT.glob('jobs/*/*_receipt.json'):resources.append(dict(job=p.parent.name,**read(p)))
    save(RESULT/'resource_summary.json',dict(jobs=resources,
        CPU_seconds_sum=sum(r['cpu_seconds'] for r in resources),
        max_job_peak_rss_mib=max(r['peak_rss_mib'] for r in resources),
        simultaneous_total_peak_measured=False,preparation=read(RESULT/'preparation_receipt.json'),
        cost_gate=read(RESULT/'cost_gate.json'),controller=controller,
        elapsed_from_activation=time.time()-read(RESULT/'activation.json')['activation_epoch'],
        note='job CPU includes setup/train/validation/IO or sample/evaluation; manager/audit and prechecks additional'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,seed in zip(axs,cfg['train_seeds']):
        for arm in cfg['arms']:
            tr=[r for r in training if r['arm']==arm and r['seed']==seed]
            ax.plot([r['end_update'] for r in tr],[r['mean_training_CE'] for r in tr],label=arm)
        ax.set_title('seed'+str(seed));ax.set_xlabel('effective updates');ax.set_ylabel('100update train CE (different condition mixtures)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RESULT/'training_curves.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,seed in zip(axs,cfg['train_seeds']):
        for arm in cfg['arms']:
            g=[v for v in validation if v['arm']==arm and v['seed']==seed]
            ax.plot([v['update'] for v in g],[v['mean_ce'] for v in g],marker='.',label=arm)
        ax.set_title('seed'+str(seed));ax.set_xlabel('effective updates');ax.set_ylabel('matched fixed validation CE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RESULT/'validation_curves.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,4,figsize=(16,11),sharey=True)
    for ax,name in zip(axs.flat,conditions['final_targets']):
        ax.plot(range(1,7),sorted(conditions['vectors'][name],reverse=True),'k--',label='target')
        for arm,color in [('narrow','tab:blue'),('broad','tab:orange')]:
            s=next(r for r in macros if r['arm']==arm and r['condition']==name)
            ax.plot(range(1,7),s['sorted_G_mean'],marker='.',color=color,label=arm)
        ax.set_title(name+' '+str(conditions['vectors'][name]),fontsize=9);ax.set_ylim(-.5,19);ax.grid(alpha=.2)
    axs.flat[-1].axis('off');axs.flat[0].legend();fig.suptitle('Actual discrete sorted Genre degrees; equal weight3 training-seed means');fig.tight_layout();fig.savefig(RESULT/'genre_degree_ranks.png',dpi=150);plt.close(fig)
    save(RESULT/'analysis_receipt.json',dict(status='completed',seconds=time.monotonic()-start,model_calls=0,extra_chains=0,
          elapsed_from_original_activation=time.time()-read(RESULT/'activation.json')['activation_epoch']))
    print('ANALYSIS completed',len(rows),'cells')

if __name__=='__main__':main()
