"""Aggregate existing receipts; never invokes network, optimizer or sampler."""
import json,pathlib,sys,time,platform,importlib.metadata
from prepare import ROOT,save

def main(folder):
    p=pathlib.Path(folder);data=json.loads((ROOT/'data_audit.json').read_text())
    gpu=json.loads((p/'gpu_forward_only/summary.json').read_text());cpu=json.loads((p/'cpu_remaining/summary.json').read_text());old=json.loads((p/'gpu_probe_corrected/summary.json').read_text())
    rows=[]
    for key,value in gpu['cases'].items():
        training=cpu['cases'].get(key)
        gate=value['gate']
        if isinstance(gate,str):gate=json.loads((p/'gpu_probe_corrected/IMDB_A/cpu_gpu_gate.json').read_text())
        row=dict(case=key,parameter_count=training['parameter_count'] if training else 6647441,
                 gpu_sweep_cold_seconds=value['timings'][0]['wall_seconds'],gpu_sweep_repeat_seconds=value['timings'][1]['wall_seconds'],
                 gpu_forward_peak_allocated_mib=value['peak_allocated_mib'],gpu_forward_peak_reserved_mib=value['peak_reserved_mib'],
                 gpu_forward_process_peak_rss_mib=value['peak_rss_mib'],device_total_occupancy_snapshot_mib=float(value['device_total_occupancy_snapshot']),
                 feature_seconds=value['timings'][1]['feature_seconds'],nnls_seconds=value['timings'][1]['encoding_seconds'],
                 transfer_seconds=value['timings'][1]['transfer_seconds'],cpu_gpu_gate={k:{a:b for a,b in v.items() if a!='per_parameter'} for k,v in gate.items()},
                 cpu_update_seconds=training['stable_update_median_seconds'] if training else None,
                 cpu_sweep_repeat_seconds=training['sweeps'][1]['wall_seconds'] if training else None,
                 restoration=training['restore_continuation'] if training else old['cases'][key]['restore_continuation'],
                 restoration_device='cpu' if training else 'cuda',updates_executed=5)
        rows.append(row)
    env={n:importlib.metadata.version(n) for n in ['torch','torch-geometric','numpy','scipy','scikit-learn','networkx']}
    result=dict(status='engineering_delivered_with_numerical_blockers',rows=rows,python=sys.version,dependencies=env,
                cpu_tests=dict(passed=17,failed=0,skipped=0),formal_updates=0,generated_graphs=0,
                total_engineering_updates=30,gpu_optimizer_updates=5,cpu_optimizer_updates=25,
                blockers=['CPU/GPU gradients exceed unchanged tolerance in all six cases',
                          'IMDB A GPU continuation has one parameter element beyond tolerance; load and RNG exact',
                          'PubMed reference NMF reached fixed500 iteration cap; no convergence claim'],
                cost_scope='strict deterministic; one-input engineering updates; t500 full20 forward+posterior, no sampled state commit',
                budget_examples=dict(cpu_6_cases_1000_updates_one_seed_upper_seconds=6*1000*.72*1.3,
                  cpu_update_bound_note='0.72s includes observed worst0.711; IMDB_A CPU update not measured; 30% planning allowance, not guarantee',
                  gpu_t1000_six_forward_posterior_only_seconds=sum(x['gpu_sweep_repeat_seconds'] for x in rows)*1000,
                  full_generation_not_measured=True),
                all_experiments_stopped=True)
    save(p/'RESULTS.json',result)
    lines=['192节点 CN / ReFeX / RolX 准备与工程短测结果','',
           '结论：实现/数据/CPU回归可审查，正式GPU训练未获数值认证。未训练正式模型，未生成新图。',
           '首次GPU尝试在0次更新时因缺cuBLAS确定性工作区退出；保留日志。之后未放宽任何数值门。',
           'IMDB A GPU续算超差后停止GPU更新；其余五组在CPU完成。累计每格5次，六格30次，含恢复重复计算。',
           '17项CPU回归零失败零跳过；独立数据/原后验/全覆盖/实际更新上限审计通过。',
           '原BASE当前Xt度分支三组均保留；A实测与旧PairDenoiser在非零度投影下输出和梯度逐位一致。',
           'A/B比较更换结构输入路径，参数量不同；B/C比较固定ReFeX基础上的额外非负角色表示。','',
           '数据：来自用户ZIP的根目录attempt02，未用attempt01/baseline。精确诱导与无改边核验通过。']
    for name,d in data['datasets'].items():
        lines.append(f"{name}:192节点,{d['edges']}物理边,类型{d['type_counts']},分量{d['components']};槽{d['query']['all_slots']},可动{d['query']['active_slots']};族边{d['query']['reference_family_edges']}")
        lines.append(f"{name} prior={d['query']['prior']};NMF迭代{d['basis']['iterations']},warning={d['basis']['warnings']};NNLS KKT={d['encoding']['kkt_max']:.3g}")
    lines+=['','配置：8层256/64/128,8头,FF256/64/256,联合头128;T1000/20块;稳定sumCE×R/S;AdamW2e-4,wd1e-12,betas.9/.999,eps1e-8,amsgradFalse;无裁剪/调度。',
            'ReFeX:30基列+60+120递归列=210;RolX式冻结rank8;单参考非负基不是逐节点角色表。',
            'NMF固定迭代数/角色数不因结果调参；PubMed基只能称冻结工程产物，尚未达到声明收敛。','',
            '逐格成本（秒；GPU列仅20块前向+后验，无采样；CPU更新是单输入，并非normal2）：']
    for x in rows:
        lines.append(f"{x['case']}:参数{x['parameter_count']};CPU更新={x['cpu_update_seconds']};CPU20块={x['cpu_sweep_repeat_seconds']};GPU20块冷/重复={x['gpu_sweep_cold_seconds']:.4f}/{x['gpu_sweep_repeat_seconds']:.4f};GPU forward allocated/reserved={x['gpu_forward_peak_allocated_mib']:.2f}/{x['gpu_forward_peak_reserved_mib']:.2f}MiB;恢复{ x['restoration_device']}={x['restoration']}")
        lines.append(f"  ReFeX/NNLS/H2D={x['feature_seconds']:.6f}/{x['nnls_seconds']:.6f}/{x['transfer_seconds']:.6f};CPU/GPU={x['cpu_gpu_gate']}")
    lines+=['','IMDB A实际GPU单输入更新中位约0.347秒，训练allocated316.40MiB/reserved350MiB；不代表数值门通过。',
            'GPU其他五格未做优化器更新，因此没有可靠的六组GPU训练吞吐对照。',
            'GPU前向进程RSS约3.76–4.33GiB；CPU工程进程峰约1.61GiB。设备总占用快照约1.05–1.21GiB，非连续峰值。',
            '每格前向只有冷/重复两次；更新仅5次调用、3个不同有效位置，样本小且t变化，不宣称稳态充分统计或B/C速度差显著。',
            '特征+NNLS约0.010–0.012秒/完整Xt，一步20块共享，仅占本次GPU查询扫的一小部分；未作大图复杂度承诺。','',
            '预算建议：先处理/审查数值阻塞，再确定正式规模。CPU六格各1000更新单seed的计算规划上界约72分钟，加30%验证/保存余量约94分钟。IMDB_A CPU更新未单测，此数用当前最慢实测作代理，并非保证。',
            'GPU六格各一条T1000的前向+后验等成本外推约'+str(round(result['budget_examples']['gpu_t1000_six_forward_posterior_only_seconds']/3600,2))+'小时；未包含抽样提交、图密度变化、评价与保存，不是已批准/已测完整链预算。',
            '本票保持T1000/20块作为源合同；不为降低预算擅自缩短科学日程。',
            '当前没有正式训练/新图/后续自动任务授权，资源已释放，停止等待审阅。']
    (p/'REPORT_zh.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main(sys.argv[1])
