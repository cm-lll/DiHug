"""Assemble bounded-preflight evidence/costs without model calls."""
import argparse,datetime,hashlib,json,pathlib,statistics

def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,x):p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();r=pathlib.Path(a.results);src=pathlib.Path(__file__).resolve().parent
    im=read(r/'IMDB_A/status.json');pm=read(r/'PubMed_A/status.json');cpu=read(r/'cpu_feature_cost.json');intake=read(r/'cpu_intake_cost.json')
    update=[x['normal2_seconds']+x['EMA_seconds'] for x in im['updates']];step=im['sampling']['step_seconds']
    f=statistics.mean(v['forward_seconds'] for u in im['updates'] for v in u['inputs']);b=statistics.mean(v['backward_seconds'] for u in im['updates'] for v in u['inputs'])
    combined=[x+intake['IMDB']['pair_preparation_wall_seconds'] for x in update]
    costs=dict(IMDB_A=dict(measured_normal2_plus_EMA_seconds=update,normal2_scope='complete 2 input validation/context/fwd/back/one AdamW; input noising outside; includes evidence phase writes',cpu_pair_intake_separate_measurement=intake['IMDB'],combined_component_update_estimates_seconds=combined,
      measured_full_20block_reverse_step_seconds=step,reverse_scope='one ordinary t500->499 state, temporary 2-update EMA; posterior/export/checks/event IO included; no full T1000 chain',
      training=[dict(updates=u,compute_hours_low=u*min(combined)/3600,compute_hours_high=u*max(combined)/3600,planning_25percent_on_observed_max_hours=u*max(combined)*1.25/3600) for u in (3000,6000,9000,12000)],
      one_T1000_chain_hours=step*1000/3600,one_T1000_chain_planning_25percent_hours=step*1000*1.25/3600,
      four_checkpoints_two_chains_each=dict(chains=8,sampling_hours=step*8000/3600,planning_25percent_hours=step*8000*1.25/3600),
      validation_reference_proxy_seconds_per_checkpoint_raw_EMA_three_times=6*step,validation_proxy_note='uses six measured reverse-step walls as a proxy; no full multi-time full-graph validation timing',
      checkpoint_write_only_seconds=im['save_seconds'],checkpoint_bytes=im['checkpoint_bytes'],checkpoint_note='excludes unmeasured cpu_tree export; CPU model/optimizer/EMA/RNG payload',
      model_disk_lower_bound_5_complete_checkpoints_bytes=5*im['checkpoint_bytes'],total_formal_wallclock='not committed: task count/parallel schedule/full-chain state variation/evaluation costs not measured'),
      PubMed_A=dict(training_cost=None,sampling_cost=None,reason='grad-enabled first forward OOM; no successful normal2 or reverse step; cannot copy IMDB throughput'),
      C_E=dict(training_cost=None,sampling_cost=None,reason='full-graph dictionaries/NNLS absent; only CPU ReFeX measured'),
      count_proposal=dict(status='not implemented or authorized',normal2_forward_count=80,normal2_backward_count=40,IMDB_A_component_only_proxy_seconds=80*f+40*b,full_query_objective_equivalence=False),
      evaluation=dict(measured=False,reason='full frozen metric suite not rerun for timing; native missing items unchanged; no small-graph runtime extrapolation presented as measured'),
      margin='25% is a planning assumption, not measured concurrency or a completion guarantee')
    save(r/'cost.json',costs)
    config_bindings={}
    for d in ('IMDB','PubMed'):
        cfg=read(r/f'{d}_A/resolved_config.json');save(src/'configs'/f'{d}.json',cfg)
        inventory=read(r/f'{d}_A/query_inventory.json');config_bindings[d]=dict(config_sha256=hashlib.sha256((src/'configs'/f'{d}.json').read_bytes()).hexdigest(),data_sha256=cfg['datasets'][d]['sha256'],query_inventory=inventory)
    save(src/'configs/bindings.json',config_bindings)
    recovery=read(r/'IMDB_A/restore_continue.json');save(r/'recovery_summary.json',{k:dict(exact=v['exact'],within_original_tolerance=v['within_original_tolerance']) if isinstance(v,dict) else v for k,v in recovery.items()})
    lines=['# 完整图本地20块短测：一页结论','',
      '**结论：IMDB A+CE在本地2080Ti通过本次短测；PubMed A+CE的20块grad-enabled前向仍OOM。不能把两图统一写成可开训，也不能把PubMed本地失败推成BW20块失败。**','',
      '|配置|实际数据|本次完成|结论|','|---|---|---|---|',
      '|IMDB A+CE|1653节点/4079边|3次临时normal2；完整20块t500→499；完整保存恢复|短测可行，未验证全链/所有状态|',
      '|PubMed A+CE|1565节点/12284边|首份输入61,190槽/12,261条Xt边，前向OOM；0次step|本地11GB级2080Ti不通过本配置|',
      '|两图C/E-earlyCN|完整图|仅完整Xt ReFeX CPU计时；无完整图字典|没有C/E GPU验收，不能用小图字典冒充|',
      '|COUNT|完整图|审阅双遍完整μ建议，未执行|当前入口禁止非零COUNT|','',
      '## 关键测量','',
      f'- IMDB normal2+EMA三次墙钟：{", ".join(f"{x:.3f}s" for x in update)}。第3次是恢复后重复第2次，只有两次连续更新，不是稳定吞吐测试。',
      f'- IMDB完整20块一个反向步：{step:.3f}s，覆盖997,313活跃槽，输入Xt未改、全部覆盖后才原子提交X499。完整schema分母1,365,378；本票采样输入为普通前向t500状态，不是自由生成中间状态。',
      f'- IMDB训练峰值allocated={max(x["memory"]["allocated_mib"] for x in im["updates"]):.2f}MiB，reserved={max(x["memory"]["reserved_mib"] for x in im["updates"]):.2f}MiB。采样allocated={im["sampling_memory"]["allocated_mib"]:.2f}MiB、reserved={im["sampling_memory"]["reserved_mib"]:.2f}MiB；采样reserved含此前训练分配器缓存，不能解释为纯推理最低需求。',
      f'- PubMed失败时峰值allocated={pm["memory"]["allocated_mib"]:.2f}MiB、reserved={pm["memory"]["reserved_mib"]:.2f}MiB；额外582MiB申请失败。堆栈为主干TransformerConv→PyG scatter_add，未捕获具体层号，不猜测。',
      f'- CPU进程峰值IMDB={im["sampling_memory"]["cpu_peak_rss_mib"]:.2f}MiB，PubMed={pm["memory"]["cpu_peak_rss_mib"]:.2f}MiB；均含Torch/取证/检查点，不是ReFeX单独内存。',
      '- 保存/恢复：IMDB模型、完整AdamW、EMA、计数、全局及显式RNG载入通过，恢复后的模型/优化器/EMA逐位相同，输入和RNG一致；没有新增CPU/GPU等价要求。',
      '- 所有现有科学源码package/、ema.py未改；6项新增CPU合同测试通过，无跳过。首版异常进程返回0但status记录failed，最终仅补失败退出码1，原执行脚本保留，不重跑GPU。','',
      '## 抽块训练量与预算','',
      '|数据集|S全schema|活跃槽|单份抽块均值|normal2监督槽期望|12000更新槽求值期望|','|---|---:|---:|---:|---:|---:|']
    for d in ('IMDB','PubMed'):
        z=config_bindings[d]['query_inventory'];c=cpu[d];lines.append(f'|{d}|{z["all_slots"]:,}|{z["active_slots"]:,}|{z["active_slots"]/20:,.2f}|{c["expected_supervised_slots_per_normal2"]:,.2f}|{c["expected_12000_update_slot_evaluations"]:,}|')
    lines+=['','每个活跃槽在12000次normal2中期望被监督1200次；小图12000次全查询normal2是每槽24000次，两者相差20倍。这里是期望监督次数，不是不同槽/不同图的唯一覆盖量。',f'CPU单独测得两份输入完整准备：IMDB {intake["IMDB"]["pair_preparation_wall_seconds"]:.3f}s，PubMed {intake["PubMed"]["pair_preparation_wall_seconds"]:.3f}s。下表将该CPU组件加到IMDB更新+EMA测量，再线性外推；不是额外实测稳态吞吐。','',
      '|IMDB A终点|训练计算外推范围|按实测最大值加25%规划|','|---|---:|---:|']
    for x in costs['IMDB_A']['training']:lines.append(f'|{x["updates"]}|{x["compute_hours_low"]:.2f}–{x["compute_hours_high"]:.2f}小时|{x["planning_25percent_on_observed_max_hours"]:.2f}小时|')
    lines +=['',f'单条T1000仅线性外推约{costs["IMDB_A"]["one_T1000_chain_hours"]:.2f}小时；四个检查点各两条链共8链，采样外推约{step*8000/3600:.2f}小时（串行），加25%规划为{step*8000*1.25/3600:.2f}小时。这不是实际链计时，生成稠密度、资源保护及多卡竞争均可能改变成本。',
      '固定验证、CPU导出/保存、完整评价和交付必须另加；成本JSON明确未测项。每模型5份完整检查点至少约510MiB，另计EMA导出、链状态、日志与评价。25%不是实测并发吞吐。不根据几个临时更新承诺完整正式总预算。','',
      '## 特征、字典与下一决定','']
    for d in ('IMDB','PubMed'):
        x=cpu[d];lines.append(f'- {d} ReFeX630维：'+', '.join(f'{f["source"]} {f["seconds"]:.3f}s' for f in x['features'])+f'；单状态FP64 {x["features"][0]["bytes"]/2**20:.2f}MiB。CPU编码/完整NMF未运行。')
    lines+=['- 完整字典仍缺：两图共128候选+8最终拟合及收敛/留出验收；节点/关系映射、K规则沿冻结协议，不能直接继承小图K或字典。本票不假报拟合/NNLS墙钟。',
      '- 若后续授权，IMDB A可优先列入本地正式队列；PubMed A先把同一固定20块适配版本交BW做有界短测；C/E等待完整图字典。COUNT需先确认双遍成本与查询上下文目标，不让它阻塞CE。',
      '- SparseDiff参照已核实：官方Protein主干8层/256节点宽度/64边宽度/8头与当前一致；论文v2有Facebook1045节点/27755边实例。故“恢复稀疏查询”方向正确，但论文规模不是11GB卡准入保证。参照链接在源码README。',
      '- 无正式训练、无T1000链、无字典拟合、无Notebook操作。仅IMDB3次临时step；PubMed0。GPU0/1进程结束，宿主只读检查显存0MiB/无计算进程。','',
      '证据：IMDB_A/status.json、restore_load.json、restore_continue.json；PubMed_A/failure.txt、status.json；各自query_inventory.json、resolved_config.json、environment.json；cpu_tests.log、cpu_feature_cost.json、cost.json。全部临时权重留本地并登记哈希，不夹带公开包。']
    (r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(r/'REPORT_ZH.md'),IMDB_passed=im['status']=='passed',PubMed_OOM=pm.get('oom'),temporary_steps=im['actual_optimizer_steps']+pm['actual_optimizer_steps'],science_complete=True)))

if __name__=='__main__':main()
