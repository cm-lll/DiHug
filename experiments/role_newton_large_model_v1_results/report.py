import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_large_model_20261021_v1')
def read(p):return json.loads(p.read_text())
def main():
    a=read(OUT/'audit.json');assert a['status']=='passed_for_available_evidence';paths=read(OUT/'audited_paths.json');meta=read(OUT/'input_manifest.json');f=read(OUT/'freeze.json')
    lines=['# Newton完整大图模型CPU工程探针','',
      '随机初始化资源探针，不是已训练大图模型，更不能评价生成质量。没有训练、完整反向链、GPU/BW或notebook。',
      f"实际完整项目{a['complete_items']}/6；完整冷/重复路径{a['complete_paths']}/12。审计状态{a['status']}只认证已存在证据，不把未完成部分算通过。",'',
      '## 配置与权重绑定','既有CapacityDenoiser结构来自8855；维度超参逐字段等于已封存的大图配置，未设计新架构。',
      '8层/8头，节点256/边64/全局128，FF256/64/256，联合头128，CN32，T1000/20块；FP32网络、float64似然/Newton/Bayes。',
      '节点one-hot4，边输入21=11状态+10族，时间输入1；度编码25、角色编码22、角色融合64、联合读出776；参数6,658,481。',
      'dropout配置.1，绑定的vendor实际无生效dropout；eval/no_grad，CPU单线程。所有真实实例化线性层维度见dataset_binding.json。',
      '三套广覆盖final6000权重的字节、原来源/配置/环境均核验。示范严格加载形状通过，但T64/大图T1000及先验值不兼容。',
      '因此没有转移、补零、截断任何参数；使用原类按seed2026102147全新初始化。随机状态摘要复现，未创建优化器。','',
      '## 完整数据与角色库存','|数据|节点/物理边|各类型节点数|各类型完整联合角色种类|全部槽/查询槽|每块槽数|','|---|---|---|---|---|---|']
    for name in ('IMDB','PubMed'):
        b=read(OUT/f'{name}_binding.json');m=next(x for x in meta if x['dataset']==name);t=b['anonymous_target'];q=m['query']
        lines.append(f"|{name}|{sum(t['type_counts'])}/{sum(q['reference_family_edges'])}|{t['type_counts']}|{[len(x) for x in t['roles']]}|{q['all_slots']}/{q['active_slots']}|{min(q['query_sizes'])}–{max(q['query_sizes'])}|")
    lines+=['','IMDB类型Director/Actor/Movie/Genre；PubMed类型Gene/Disease/Chemical/Species。全部关系容量、先验、角色向量与整数频数均保存，未删角色。',
      '活跃族按原参考边数>0定义；确定性空族不等于独立schema禁边规则，保持原query合同。',
      '六个Xt按原核独立前向加噪，seed[2026102148,index,t]；不是自由生成状态，也未串接。单步uniform独立seed[2026102149,index,t]。','',
      '## 完成与成本','|数据/t|状态|完整路径|冷/重复单步秒|峰RSS MiB|Newton验收|','|---|---|---:|---|---:|---|']
    for item in a['items']:
        pp=[x for x in paths if x['dataset']==item['dataset'] and x['t']==item['t']];r=item['worker_receipt'] or {}
        lines.append(f"|{item['dataset']}/{item['t']}|{item['status']}|{item['completed_paths']}/2|{' / '.join(x['phase']+':'+format(x['full_step_seconds'],'.3f') for x in pp) or '未完成'}|{r.get('peak_rss_mib','NA')}|{'；'.join(','.join(x['read_statuses']) for x in pp) or '见失败记录'}|")
    lines+=['','## 冷与复用的成本分解（每项秒，计时为包含关系，不可将父子计时相加）',
      '|数据/t/路径|主干|CN聚合|联合头|度似然建表|似然查表/装配/角色循环残差|Newton|query计划/装配|后验|独立审计|证书/输出IO|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in paths:
        m=p['module_timings'];r=p['runtime_timings'];query=sum(r.get(k,0) for k in ('query_plan_seconds','query_validation_seconds','query_transfer_seconds','graph_assembly_seconds'))
        values=[m.get('transformer_backbone_seconds',0),m.get('CN_full_aggregation_seconds',0),m.get('joint_readout_seconds',0),m.get('degree_law_cache_construction_seconds',0),p['likelihood_lookup_assembly_and_role_loop_residual_seconds'],m.get('Newton_solver_seconds',0),query,r.get('posterior_seconds',0),p['audit_seconds'],p['IO_seconds']]
        lines.append('|'+f"{p['dataset']}/{p['t']}/{p['phase']}|"+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','冷路径建造该时刻需要的精确度分布表；repeat只复用这些表，新ton/角色编码/查询/整网络/后验/抽样均重算。',
      '原FrozenTypedContext每次新建，全部20块共享同一Xt；内部既有CN节点表示和条件输入只在该次完整步内缓存，不跨冷/重复路径。',
      'role_loop_residual包含似然查询/LL装配、上下文矩阵乘法和Python记账，不伪称全部为纯似然时间。',
      'trace=True按原实现每块额外计算诊断q，实际后验counter只计真实转移调用；额外诊断、状态复制、采样和逐块进度IO处于整步残差内，未单独计纯内核。',
      '独立全槽后验复核与采样提交重放单独计时；不是重用旧预测冒充fresh网络。证书保存IO与求解器速度分开。',
      '完整工程提交/无向schema/自环/重边/全部槽覆盖及cold-repeat RNG/结果均独立审核；单步产物仅工程输出。','',
      '## 数值验收与资源门',f"每次完成读取计数{a['Newton_reads']}，分状态{a['read_statuses']}；独立最大指标{a['maxima']}。",
      '原合同与显式尾部合同分开；有限logP<-700仅在普通输出舍入，不改数学支持/目标。每行/总遗漏门1e-280/1e-276，实际角色上下文尾误差门1e-250。',
      '原Newton初始化/200迭代/4000评估/30秒单矩阵/线搜索/阻尼与资源门完全保留，失败无DP回退。',
      'CPU过程虚拟地址硬限16GiB；RSS8GiB由父进程轮询观察，非瞬时内核RSS硬限；预估门6GiB，item420秒含启动/清理，计算410秒。',
      '预估上界包含参数/全槽状态/最多计算图边/层消息临时数组/CN wedge/似然表/Newton；它是工程估算，不是形式化最坏内存证明。',
      'baseline_current_RSS与此前峰、step后峰、最终峰分别保存；峰值包括全槽后验审计及IO。不相加多个进程峰值冒充同时峰。',
      'IMDB/PubMed按顺序单进程运行，单项前检查剩余>=420+720秒。超限/未运行/部分结果在controller和item receipt中原样保留。','',
      '## 结论边界与下一步',
      '大图模型能否完成只逐项依据上表；不将少量时刻前向成功解释为训练反向可承受、T1000整链速度保证或跨规模质量成功。',
      '若粗估1000步，可将实测单步×1000仅作静态输入假设；自由链边数/度数/CN wedge/似然难度会变化，可能触发不同资源门，不能当完成时间承诺。',
      '本票停止后只提出下一阶段预算，未自动换GPU/修数值门/训练/长链/结构优化。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));print('report generated')
if __name__=='__main__':main()
