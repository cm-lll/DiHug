"""Post-run, read-only evidence checks and descriptive tables. No model calls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run01'
AUDIT=ROOT/'audit_repair_v1'
WORK=Path(os.environ.get('ROLE_REVIEW_SOURCE_ROOT','/tmp/dihug-role-unseen-target-validation-v1'))

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def check():
    if time.time()>=read(RUN/'activation.json')['deadline_epoch'] or (ROOT/'AUDIT_STOP').exists():
        raise TimeoutError('original unique45min ledger; no reset')

def main():
    started=time.monotonic();check()
    import numpy as np
    import sys
    sys.path.insert(0,str(WORK/'experiments/role_unseen_target_validation_v1'))
    from execute import lazy,target_objects,metadata_case
    c=lazy()
    assert read(RUN/'controller.json')['status']=='failed'  # retained initial TV-equality failure
    assert read(AUDIT/'receipt.json')['status']=='completed'
    assert read(AUDIT/'audit.json')['status']=='passed'
    rows=read(AUDIT/'generation.json');per=read(AUDIT/'per_seed.json')['summaries']
    case=metadata_case(c,Path('/data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01'),target_objects(c))
    expected=read(WORK/'experiments/role_robust_training_6000_v1_results/evidence/learning_run01/generation.json')
    historical=[r for r in expected if r['arm']=='normal2' and r['checkpoint_update']==6000]
    matched=[];linked=[]
    oldroot=Path('/data2/lyh/gnn_project/DiHug/results/role_robust_training_6000_20261021_v1/learning_run01')
    for r in rows:
        check()
        with np.load(RUN/r['graph_file'],allow_pickle=False) as z:
            f,local=case['space'].encode(z['edges'].T)
            assert np.array_equal(f,z['relations']-1)
            state=np.zeros(case['space'].total,dtype=np.uint8)
            state[case['space'].offsets[f]+local]=1
        steps=read(RUN/r['steps_file'])
        assert c.sha(state.tolist())==steps[-1]['state_sha256']
        linked.append(dict(graph_file=r['graph_file'],final_state_verified=True))
        if r['condition'] in ('balanced','extreme'):
            old=next(x for x in historical if (x['seed'],x['condition'],x['sampling_seed'])==(r['seed'],r['condition'],r['sampling_seed']))
            with np.load(oldroot/old['graph_file'],allow_pickle=False) as z:
                oldedges=z['edges'].copy()
            with np.load(RUN/r['graph_file'],allow_pickle=False) as z:
                same=np.array_equal(oldedges,z['edges'])
            assert same
            assert old['graph_file']!=r['graph_file']
            matched.append(dict(seed=r['seed'],condition=r['condition'],sampling_seed=r['sampling_seed'],
                                same_labelled_graph=True,old_graph=old['graph_file'],new_graph=r['graph_file']))
    assert len(matched)==48
    write(ROOT/'post_audit.json',dict(status='passed',complete_graph_to_final_step_links=linked,
        fresh_controls_reproduce_history=matched,controls_reused_as_inputs=False,
        historical_generation_sha256=sha(WORK/'experiments/role_robust_training_6000_v1_results/evidence/learning_run01/generation.json'),
        model_calls=0,seconds=time.monotonic()-started))
    names=read(WORK/'experiments/role_unseen_target_validation_v1/config.json')['target_order']
    fields=['G_W1','G_joint_TV','G_max','G_second_max','G_high_ge8','G_high_ge12','G_low_le1','G_zero',
            'family1_edges','family2_edges','family5_edges','family8_edges','edges','Director_joint_TV','Actor_joint_TV','Movie_joint_TV',
            'connected','lcc','isolates','triangles','closure_path','closure_pair','MP2_total','MP3_total']
    table=[];macro=[]
    for s in per:
        table.append(dict(seed=s['seed'],condition=s['condition'],**{k:s['statistics'][k]['mean'] for k in fields},
                          topology_classes=s['topology_classes'],topology_frequencies=json.dumps(s['topology_frequencies'])))
    with (ROOT/'per_seed_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for name in names:
        group=[s for s in per if s['condition']==name]
        macro.append(dict(condition=name,**{k:float(np.mean([s['statistics'][k]['mean'] for s in group])) for k in fields},
                          topology_classes=[s['topology_classes'] for s in group],
                          sorted_G_degrees=np.mean([r['target_errors']['Genre']['degrees_descending'] for r in rows if r['condition']==name],axis=0).tolist()))
    write(ROOT/'macro_summary.json',dict(unit='equal weight1/3 per training seed; descriptive only',conditions=macro))
    seeds=sorted({r['seed'] for r in rows})
    resources=[read(RUN/f'seed{s}/receipt.json') for s in seeds]
    manager=read(RUN/'worker_receipt.json');controller=read(RUN/'controller.json')
    resource=dict(formal_parent_wall_seconds=controller['wall_seconds'],seed_resources=resources,manager=manager,
        CPU_seconds_sum=sum(r['cpu_seconds'] for r in resources)+manager['cpu_seconds']+controller['parent_cpu_seconds']+read(AUDIT/'receipt.json')['cpu_seconds'],
        peak_single_process_rss_mib=max(r['peak_rss_mib'] for r in resources+[manager]),
        sum_process_peaks_upper_mib=sum(r['peak_rss_mib'] for r in resources+[manager]),
        simultaneous_total_peak_measured=False,training_updates=0,scientific_graphs=120,reverse_steps=7680,
        network_forwards=30720,GPU_used=False,original_controller_status='failed',separate_evaluation_repair_status='completed',elapsed_from_original_activation_seconds=time.time()-read(RUN/'activation.json')['activation_epoch'])
    write(ROOT/'resources_summary.json',resource)
    # Exported figure describes sorted actual degrees, not assigned node roles.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets={r['name']:r['Genre_degrees'] for r in read(RUN/'targets.json')['targets']}
    fig,axes=plt.subplots(1,5,figsize=(17,3.8),sharey=True)
    for ax,name in zip(axes,names):
        ax.plot(range(1,7),sorted(targets[name],reverse=True),'k--',label='target')
        for seed in seeds:
            values=[r['target_errors']['Genre']['degrees_descending'] for r in rows if r['seed']==seed and r['condition']==name]
            ax.plot(range(1,7),np.mean(values,axis=0),marker='.',label='train seed '+str(seed)[-2:])
        ax.set_title(name);ax.set_xlabel('sorted Genre rank');ax.set_ylim(-.3,19);ax.grid(alpha=.2)
    axes[0].set_ylabel('actual discrete degree (mean of 8 graphs/model)')
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'genre_degree_ranks.png',dpi=160);plt.close(fig)
    check()
    print(json.dumps(dict(status='passed',macro=macro,resources=resource),indent=2))

if __name__=='__main__':main()
