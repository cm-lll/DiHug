"""Finish this ticket's CPU report after fixed jobs exit; never schedules science."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,hashlib,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve();p=json.loads((out/'protocol.json').read_text())
def read(name):return json.loads((out/name).read_text())
def write(name,x):
    q=out/name;tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(q)
def fmt(x):return '未定义/缺失' if x is None else f'{x:.4f}'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
try:
    while time.time()<p['delivery_epoch']-30:
        if (out/'release.json').exists() and ((out/'analysis_receipt.json').exists() or (out/'analysis_failure.json').exists()):break
        time.sleep(5)
    release=read('release.json') if (out/'release.json').exists() else {}
    analysis=read('analysis_receipt.json') if (out/'analysis_receipt.json').exists() else {}
    receipts=[json.loads(f.read_text()) for f in out.glob('*/receipt_*.json')]
    vectors=len(list(out.glob('*/vector_*.npy')));chains=len(list(out.glob('*/chain_*/summary.json')))
    counts={k:sum(r.get(k,0) for r in receipts) for k in ('sampling_steps','gradient_forwards','backwards')}
    changed=[]
    for path,h in read('asset_recheck.json')['bindings'].items():
        if not Path(path).exists() or sha(path)!=h:changed.append(path)
    result=dict(status='complete' if chains==40 and vectors==64 and analysis.get('status')=='complete' and not changed
                and all(r['status']=='complete' for r in receipts) and counts==dict(sampling_steps=40000,gradient_forwards=512,backwards=512)
                and release.get('all_owned_processes_exited') else 'partial_or_failed',
                start_epoch=p['start_epoch'],report_epoch=time.time(),wall_minutes=(time.time()-p['start_epoch'])/60,
                GPU_hours=release.get('GPU_hours'),new_complete_chains=chains,reused_chains=24,gradient_vectors=vectors,
                counts=counts,optimizer_updates=0,model_optimizer_EMA_RNG_unchanged=all(r.get('model_optimizer_EMA_RNG_unchanged',False) for r in receipts),
                original_assets_changed=changed,all_owned_GPU_processes_exited=release.get('all_owned_processes_exited'),
                no_additional_scientific_tasks_launched=True)
    write('completion.json',result)
    lines=['# 完整链反馈可靠性：固定模型4／8／16轨迹比较',
           f"执行状态：{result['status']}。本票新增完整链 {chains}/40，复用24链；逐轨迹梯度 {vectors}/64。没有模型参数、AdamW或EMA更新。",
           '目的：检查同一冻结策略的完整链结构反馈在不同轨迹小组间是否一致。新图是同一模型的新增采样，不是经过训练后质量改善的证据。',
           '四个父模型均为普通单输入CE、A晚CN的3000步EMA参数。保留固定热增量联合评分、R1/T1000、冻结采样seed及每链8个求导时点。',
           '## 反馈方向',
           '余弦接近1表示方向相近，接近0表示接近正交，负值表示方向相反。4链项是四个不交叠小组的六对比较；8链项只有两个不交叠小组的一次比较。',
           '|数据/训练seed|4链余弦均值|4链最小|4链最大|负向对数/6|8链余弦|16链梯度范数|',
           '|---|---:|---:|---:|---:|---:|---:|']
    reliability=read('gradient_reliability.json') if (out/'gradient_reliability.json').exists() else []
    for r in reliability:
        lines.append(f"|{r['dataset']} {r['train_seed']}|{fmt(r['disjoint4_cosine_mean'])}|{fmt(r['disjoint4_cosine_min'])}|{fmt(r['disjoint4_cosine_max'])}|{r['disjoint4_negative_pairs']}|{fmt(r['disjoint8_cosine'])}|{fmt(r['full16_gradient_norm'])}|")
    lines+=['16链只有一个总方向，不能声称16链已经重复稳定。逐轨迹剔除影响、旧6/新10对照、骨干/CN/度/联合头，以及度项和热项分量方向均另表保留；不删除高影响轨迹。',
            '求导量为既有eager clean-head代理：v_k=八个时点的平均∇log pθ(G0_k|Xt_k,t)，g_K=0.01×样本协方差(C_k,v_k)。它不是最终结构奖励的无偏梯度；本票比较原始梯度，未执行裁剪、AdamW映射或参数更新。',
            '## 冻结模型的完整生成图',
            '下表每个模型16次采样的均值±样本标准差；它们不是16个独立训练重复。原图为主，LCC另表。',
            '|数据/训练seed|边数|度W1|联合度TV|三角数量|三角绝对误差|MP3绝对误差|LCC|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    if (out/'model_summary.json').exists():
        for r in read('model_summary.json'):
            if r['view']!='raw' or r['source']!='all16':continue
            cells=[r['dataset']+' '+str(r['train_seed'])]
            for k in ('edges','degree_W1','joint_TV','triangles','triangle_abs_error','MP3_abs_error','LCC'):
                cells.append(fmt(r[k+'_mean'])+' ± '+fmt(r[k+'_sd']))
            lines.append('|'+ '|'.join(cells)+'|')
    lines+=['完整逐图数量/组成、低度/hub、逐类型/关系、原图/LCC见per_graph.csv、model_summary.csv和evaluation/*.json。旧6条与新10条单列，不把两组采样差别写成学习收益。',
            '## 判断边界与下一决定',
            '较一致的方向只说明抽样建议可能更稳定，不证明这个方向正确。score_metric_associations.csv保留低评分与结构误差的逐模型描述性关系；相关性也不是更新效果或因果证据。',
            '每链选8个时点，反馈波动同时包含轨迹及求导时点差异。本票未隔离两者。每数据集两个既有开发训练seed，不作统计显著性或跨数据集普遍性结论。',
            '本票结束后停止。是否值得进入一次更新/生成匹配验证，需要一起审阅不交叠组方向、分量代价和真实成本；不由单个余弦阈值自动触发，不自动增加K、调评分或启动正式训练。',
            '## 成本、失败与释放',
            json.dumps(result,ensure_ascii=False,indent=2),
            '工程检查的32次前后向及40个采样步骤均计入既定512/40000，不是额外试跑。最初四条新链暂停在t=990，保存边界重新载入后接续原链，未重复生成。',
            '完整状态保存/载入摘要相同不等于CUDA重新计算逐位确定；本票没有放宽历史更新重算容差。',
            '源码/环境/设备/校准与累计占卡分别保存在protocol.json、各模型environment文件、cost_gate.json、allocations.json和release.json。独立退出看门狗为并发停止预留时间，未热替换模型代码。']
    (out/'REPORT_ZH.md').write_text('\n\n'.join(lines)+'\n')
    # Repository handoff uses the required patch mechanism; no history rewriting.
    note=(f"2026-11-13 完整链可靠性票已退出，`results/{out.name}`，状态{result['status']}：新增{chains}/40链、复用24、"
          f"{vectors}/64梯度向量、{counts['gradient_forwards']}前向/{counts['backwards']}反向、0优化更新。"
          f"占卡{result['GPU_hours']}GPUh；释放回执见release.json。方向及全部图指标见REPORT_ZH.md，冻结父模型采样不是训练收益。"
          "下一仅审阅完整不交叠4/8组方向与结构代价后决定，不自动训练、增加K或改评分。")
    patches=[]
    for filename,heading in [('docs/PROJECT_STATUS.md','# DiHug current research status'),('docs/NEXT_EXECUTION_TASK.md','# Current execution task')]:
        patches.append(f'*** Update File: {ROOT/filename}\n@@\n {heading}\n+\n+{note}\n')
    patch='*** Begin Patch\n'+''.join(patches)+'*** End Patch\n'
    try:
        done=subprocess.run(['apply_patch'],input=patch,text=True,capture_output=True,timeout=20)
        write('handoff_patch.json',dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr))
    except Exception:write('handoff_patch.json',dict(error=traceback.format_exc(),confirmed_note=note))
except BaseException:write('delivery_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
