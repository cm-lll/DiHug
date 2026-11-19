"""CPU audit, baseline evaluation and budget from finished owned workers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,hashlib,importlib.util,csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/three_arm_update_preflight_20261111'
spec=importlib.util.spec_from_file_location('baseline_eval',ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def save(name,x):ev.save(OUT/name,x)
def main():
    torch.set_num_threads(1);started=time.time();p=json.loads((OUT/'protocol.json').read_text())
    old=json.loads((ROOT/'results/terminal_structure_real_20261109T233600/protocol.json').read_text())
    verified=[];costs={};metric_rows=[];validation_rows=[];issues=[]
    def check():
        if time.time()>p['delivery_epoch']-5:raise TimeoutError('CPU delivery deadline')
    for ds in p['datasets']:
        check();folder=OUT/ds;receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='complete':issues.append(dict(dataset=ds,receipt=receipt));continue
        rows=json.loads((folder/'updates.json').read_text());assert len(rows)==6
        initial=torch.load(folder/'initial_complete.pt',map_location='cpu',weights_only=False)
        assert initial['updates']==initial['ema']['updates']==0 and not initial['optimizer']['state']
        for arm in ('S1','S16','T_eager'):
            state=torch.load(folder/(arm+'_u1_complete.pt'),map_location='cpu',weights_only=False)
            assert state['updates']==state['ema']['updates']==1 and state['optimizer']['state']
            steps={float(v['step']) for v in state['optimizer']['state'].values()};assert steps=={1.}
            assert state['recovery']['exact_resume_allowed'] and state['recovery']['phase']=='committed'
            ar=[r for r in rows if r['arm']==arm];assert [r['update'] for r in ar]==[1,2]
            for r in ar:
                assert np.isfinite(r['gradient_norm_before_clip']) and r['displacement_norm']>0
                if arm=='S16':
                    cc=np.array(r['records'][0]['costs']);a=np.array(r['records'][0]['advantages'])
                    np.testing.assert_allclose(a,[cc[i]-np.delete(cc,i).mean() for i in range(16)],atol=1e-12)
            checks=json.loads((folder/(arm+'_checks.json')).read_text())
            assert all(checks.values())
            verified.append(dict(dataset=ds,arm=arm,Adam_steps=list(steps),saved_state_complete=True,
                first_gradient_norm=ar[0]['gradient_norm_before_clip'],first_clip_scale=ar[0]['clip_scale'],
                first_displacement_norm=ar[0]['displacement_norm'],median_update_seconds=float(np.median([r['seconds'] for r in ar]))))
        s1=[r for r in rows if r['arm']=='S1'];s16=[r for r in rows if r['arm']=='S16']
        for a,b in zip(s1,s16):
            assert a['main_rng_hash']==b['main_rng_hash']
            assert a['records'][0]['input_hash']==b['records'][0]['input_hash']
            assert a['records'][0]['prefix_uniform_hash']==b['records'][0]['prefix_uniform_hash']
        directory=folder/'baseline_sample_2026111341';sr=json.loads((directory/'summary.json').read_text())
        assert sr['status']=='complete' and sr['steps']==1000 and sr['model_unchanged']
        assert hashlib.sha256((directory/'graph.npz').read_bytes()).hexdigest()==sr['graph_sha256']
        logs=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['time'] for r in logs]==list(range(1000,0,-1)) and all(r['exact_coverage'] for r in logs)
        assert len({r['uniform_hash'] for r in logs})==1000
        state=torch.load(directory/'latest_complete.pt',map_location='cpu',weights_only=False)
        assert state['next_t']==0 and state['recovery']['exact_resume_allowed'] and state['draw_calls']==1001
        ti=[r['seconds'] for r in rows if r['arm']=='T_eager']
        costs[ds]=dict(new_chain_seconds=sr['seconds'],new_chain_minutes=sr['seconds']/60,
            S1_update_seconds=float(np.median([r['seconds'] for r in s1])),S16_update_seconds=float(np.median([r['seconds'] for r in s16])),
            eager_update_seconds=float(np.median(ti)),
            eager_K4_round_seconds=4*sr['seconds']+float(np.median(ti)),
            warning='One full chain only; dependence on state density, concurrency and checkpointing remains. No quality winner.')
        data=np.load(old['models'][ds]['data']);types=data['types'];schema=data['schema'];edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        names=['Director','Actor','Movie','Genre'] if ds=='IMDB' else ['Gene','Disease','Chemical','Species']
        ref=ev.M.make_graph(types,edges,schema);rs=ev.M.stats(ref,ev.F,names,check)
        z=np.load(directory/'graph.npz');g=ev.M.make_graph(z['types'],z['edges'],z['schema'])
        assert np.array_equal(g['types'],types) and np.array_equal(g['schema'],schema)
        st=ev.M.stats(g,ev.F,names,check);cmp=ev.M.compare(g,st,ref,rs,ev.F,names)
        lg,mapping=ev.M.lcc(g);ls=ev.M.stats(lg,ev.F,names,check);lc=ev.M.compare(lg,ls,ref,rs,ev.F,names)
        ev.save(directory/'full_metrics.json',dict(reference=rs,raw=dict(statistics=st,comparison=cmp),LCC=dict(statistics=ls,comparison=lc,mapping=mapping),
            note='Unchanged baseline one sample per dataset; not a candidate quality result.'))
        for view,s,c in [('raw',st,cmp),('LCC',ls,lc)]:
            metric_rows.append(dict(dataset=ds,view=view,edges=s['edges'],reference_edges=rs['edges'],degree_W1=c['main_incident4_W1'],
                joint_TV=c['main_joint_degree_TV'],triangles=s['triangles'],reference_triangles=rs['triangles'],
                triangle_composition_TV=c['triangle_composition_TV'],MP2=s['mp'][1]['total'],reference_MP2=rs['mp'][1]['total'],
                MP3=s['mp'][2]['total'],reference_MP3=rs['mp'][2]['total'],LCC=s['lcc'],components=s['components']))
        before=json.loads((folder/'validation_parent.json').read_text())
        for arm in ('S1','S16','T_eager'):
            after=json.loads((folder/f'validation_{arm}_temporary_u2.json').read_text())
            for x,y in zip(before,after):
                assert x['t']==y['t'] and x['input_hash']==y['input_hash']
                validation_rows.append(dict(dataset=ds,arm=arm,t=x['t'],before_CE=x['CE'],after_CE=y['CE'],delta=y['CE']-x['CE'],
                    status='temporary two-step engineering check, not efficacy evidence'))
    for path,h in p['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    release=json.loads((OUT/'release.json').read_text())
    assert release['GPU_hours']<=1 and release['wall_seconds']<=1800
    save('verification.json',dict(complete=len(verified)==6,arms=verified,issues=issues,all_bound_assets_unchanged=True,new_chains=len(costs),
        original_recomputed_update_bitwise_tolerance_not_redefined=True))
    save('measured_costs.json',costs);ev.csvsave(OUT/'baseline_metrics.csv',metric_rows);save('baseline_metrics.json',metric_rows)
    ev.csvsave(OUT/'temporary_validation.csv',validation_rows)
    # Small proposed pilot, not the later four-training-seed confirmation.
    # T: two rounds of two NEW trajectories (same total collection as K4 x1).
    # Each arm gets the same dataset-specific target training occupancy.
    oldscale=1.35
    if len(costs)==2:
        terminal_per_seed={ds:x['eager_K4_round_seconds'] for ds,x in costs.items()}
        train_h=3*sum(terminal_per_seed.values())/3600
        #6 models x2 endpoint samples; one additional untouched-parent baseline
        #per dataset, the first baseline seed is provided by this ticket.
        sample_h=sum(7*x['new_chain_seconds'] for x in costs.values())/3600
        budget=dict(status='proposal_not_started',training_seeds_per_dataset=1,arms=3,models=6,
            train_seeds={'IMDB':2026110411,'PubMed':2026110421},
            terminal_rounds=2,terminal_trajectories_per_round=2,final_samples_per_model=2,final_candidate_chains=12,
            additional_baseline_chains=2,reusable_baseline_chains=2,final_sample_seeds=[2026111341,2026111342],
            proposed_primary_readout='fixed final raw parameters; retain EMA states and validate separately',
            readout_reason='With unequal update counts, EMA0.99 after2 terminal updates remains near parent while thousands of one-step updates adapt EMA. Do not confuse EMA lag with feedback quality.',
            terminal_training_chains=8,matched_training_GPU_seconds_per_seed=terminal_per_seed,
            estimated_training_GPU_hours=train_h,estimated_final_generation_GPU_hours=sample_h,
            envelope_GPU_hours=(train_h+sample_h)*oldscale,
            six_worker_nominal_wall_minutes=max(terminal_per_seed[ds]+2*x['new_chain_seconds'] for ds,x in costs.items())/60,
            recommended_wall_minutes=90,recommended_GPU_hours=8.5,
            limitations='One existing development training seed and two endpoint samples per dataset: engineering/learning pilot only, not cross-seed proof. T uses K2 x2 rounds rather than K4 in the prior cost probe, fixed before effects. Same GPU target is approximate and actual occupancy must be reported. New trajectories after each T update; do not use repeated old trajectories as on-policy.35% reserve covers unmeasured concurrency/IO/validation; recheck prior to launch.')
        save('pilot_budget_proposal.json',budget)
    lines=['# 三臂真实更新、恢复与整链成本验收','',
        f"已完成{len(verified)}臂临时检查、{len(costs)}条未改父模型完整基线链。占卡{release['GPU_hours']:.5f}小时、运行墙钟{release['wall_seconds']/60:.2f}分钟；临时模型不用于正式效果结论。",
        '', '共同起点是两份A/EMA3000，新建AdamW与EMA，LR1e-5、原其他Adam配置、全参数、同评分尺度.01、clip1；只用结构反馈，CE用于观察、不加入本次梯度；无奖励标准差归一化或奖励截断。不是承接原CE Adam历史，也不是完整GDPO复现。',
        '', 'S1无基线；S16逐图留一相对奖励，全部保留；T-eager四旧轨迹×8保存时点，按时间平均。T仅第一步处于旧轨迹采样策略；第二步重复旧批只测保存恢复，不当on-policy学习。独立评价在两步后不支持方法优劣结论。',
        '', '完整状态重载及下一输入一致，EMA验证恢复raw，有限梯度和实际更新检查通过。未改原GPU更新重算容差，未声称逐位重算轨迹认证。新链前全部临时状态丢弃，重载原父模型，原R1/T1000/固定支持与随机合同保持。',
        '', '| 数据集 | 单次S1更新秒 | S16更新秒 | T梯度/更新秒 | 新完整链分钟 | K4完整链一轮分钟 |','|---|---:|---:|---:|---:|---:|']
    for ds,c in costs.items():lines.append(f"| {ds} | {c['S1_update_seconds']:.3f} | {c['S16_update_seconds']:.3f} | {c['eager_update_seconds']:.2f} | {c['new_chain_minutes']:.2f} | {c['eager_K4_round_seconds']/60:.2f} |")
    lines+=['','两条新图仅是未改基线的计时样本，全部指标见baseline_metrics.csv及各full_metrics.json；不写成新训练收益。',
        '', '下一唯一方向是按新链实测冻结相近算力三臂小规模学习：先两图各一个既定开发seed，6模型；完整链臂预定K2×2轮，每次更新后重新收集；单步两臂匹配数据集级目标卡时而非更新数。每模型2个独立终点采样，共12候选图，复用本票2基线图并另补2基线图。建议90分钟/8.5GPU小时上限，详细外推见pilot_budget_proposal.json（主体估计5.93GPU小时，35%余量后8.003，向上取8.5而不向下冒充8）。该规模不能证明跨seed稳定性，正式多seed结果另需预算；本票30分钟/1GPU小时不自动扩张。',
        '', '为避免不等更新数造成EMA适应时长差异，建议试点固定终点raw为主采样，保留EMA完整状态并独立验证；不能用T仅2次EMA更新与单步数千次EMA更新的差别代替反馈能力结论。此为下一票设计，不改本轮或历史readout。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    save('completion.json',dict(status='complete' if len(verified)==6 else 'partial',temporary_updates=2*len(verified),new_baseline_chains=len(costs),
        formal_candidate_training=False,CPU_evaluation_seconds=time.time()-started,GPU_hours=release['GPU_hours'],wall_seconds=release['wall_seconds'],issues=issues))
    print(json.dumps(dict(costs=costs,release=release,issues=issues),ensure_ascii=False))
if __name__=='__main__':main()
