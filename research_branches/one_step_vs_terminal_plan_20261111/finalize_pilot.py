"""CPU-only completion/partial-result delivery. No successor experiment.

This finalizer waits for the owned supervisor, runs the frozen evaluation,
and produces a report even when a branch failed. It cannot start GPU work.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/three_arm_learning_pilot_20261111T211641'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    p=read(OUT/'protocol.json')
    while not (OUT/'release.json').exists():
        if time.time()>p['delivery_epoch']-120:
            write(OUT/'finalizer_failure.json',dict(reason='supervisor release unavailable before delivery reserve'));return
        time.sleep(5)
    log=(OUT/'evaluation.log').open('w')
    proc=subprocess.run([PY,str(Path(__file__).with_name('evaluate_pilot.py'))],stdout=log,stderr=subprocess.STDOUT,
                        timeout=max(1,p['delivery_epoch']-time.time()-90))
    log.close();release=read(OUT/'release.json')
    if proc.returncode:
        write(OUT/'completion.json',dict(status='evaluation_incomplete',evaluation_returncode=proc.returncode,release=release));return
    summaries=read(OUT/'model_summary.json');training=read(OUT/'training_costs.json');ver=read(OUT/'verification.json');costs=read(OUT/'objective_scores.json')
    failures=[]
    for f in OUT.glob('*/failure.json'):failures.append(dict(path=str(f),detail=read(f)))
    complete=ver['completed_graphs']==16 and not failures and all(x['returncode']==0 for x in release['processes'])
    def fmt(r,k):
        a,b=r.get(k+'_mean'),r.get(k+'_sample_sd')
        return '缺' if a is None else f'{a:.4g} ± {b:.3g}' if b is not None else f'{a:.4g}'
    lines=['# 三臂学习与完整生成试点','',
        f"状态：{'完整完成' if complete else '部分完成'}。共{ver['completed_graphs']}张最终评价图（含2张复用基线），完整计划16张。实际占卡{release['GPU_hours']:.4f}小时，GPU任务结束时总墙钟{release['wall_seconds']/60:.2f}分钟；GPU进程释放明细见release.json。",
        '', '这次是实际学习后的完整T1000生成比较。每个数据集只有一个既定开发训练seed，每模型两个生成seed；表中标准差是同权重两次采样的样本标准差，不能解释为跨训练seed稳定性。',
        '', 'S1：每个正常前向加噪输入抽一张干净硬图，以结构代价做score-function反馈。S16：抽16张，每张使用自己的留一相对代价，再汇总梯度；没有选最好一张，没有给所有图同一个平均奖励。T-eager：每轮当前raw模型生成两条新完整链，以终点图代价和8个保存时点的clean-head作加权更新，两轮均重新采样；这是有偏eager代理，不是原版GDPO完整复现。',
        '', '共同父模型为原A晚CN/3000步EMA；各臂新建AdamW/EMA，学习率1e-5、同结构目标及尺度0.01、梯度裁剪1、全部原参数、无CE辅助、无奖励标准化或奖励截断。未改模型或后验采样核。主生成读取预定终点raw，EMA仅保存和单独验证。S1/S16按照预定卡时停止，非按结果选终点；T固定两次更新。同卡时比较不等于同更新数或同输入曝光。',
        '', '目标为类型内同节点联合关系度与多尺度热描述的核分布距离，加逐关系边数量风险；热特征不完整保留沿途关系次序。目标下降和其他结构指标改善分别报告。']
    names={'baseline':'原A基线','S1':'一步1图','S16':'一步16图逐图加权','T_eager':'完整链反馈'}
    for ds in ('IMDB','PubMed'):
        ref=read(OUT/'reference_metrics.json')[ds]
        lines += ['',f'## {ds}，训练seed {p["parents"][ds]["train_seed"]}',
            f"参考图：{ref['nodes']}节点，{ref['edges']}边，{ref['triangles']}三角，MP2={ref['mp'][1]['total']}，MP3={ref['mp'][2]['total']}，LCC={ref['lcc']}。",
            '', '| 配置 | 样本 | 边数 | 边数绝对误差 | 度W1 | 联合度TV | 三角数 | 三角绝对误差 | LCC |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        rr=[r for r in summaries if r['dataset']==ds and r['view']=='raw']
        for r in rr:lines.append('| '+names[r['arm']]+' | '+str(r['samples'])+' | '+' | '.join(fmt(r,k) for k in ['edges','edge_abs_error','degree_W1','joint_TV','triangles','triangle_abs_error','LCC'])+' |')
        lines += ['', '| 配置 | 三角组成TV | MP2数 | MP2绝对误差 | MP2组成TV | MP3数 | MP3绝对误差 | MP3组成TV | 分量数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rr:lines.append('| '+names[r['arm']]+' | '+' | '.join(fmt(r,k) for k in ['triangle_composition_TV','MP2','MP2_abs_error','MP2_TV','MP3','MP3_abs_error','MP3_TV','components'])+' |')
        base=next((r for r in rr if r['arm']=='baseline'),None)
        if base:
            for r in rr:
                if r['arm']=='baseline':continue
                gains=[];losses=[]
                for k,label in [('degree_W1','度W1'),('joint_TV','联合度TV'),('triangle_abs_error','三角数量误差'),('triangle_composition_TV','三角组成'),('MP2_abs_error','MP2数量误差'),('MP3_abs_error','MP3数量误差'),('MP2_TV','MP2组成'),('MP3_TV','MP3组成')]:
                    a,b=r[k+'_mean'],base[k+'_mean']
                    if a is None or b is None:continue
                    if a<b:gains.append(label)
                    elif a>b:losses.append(label)
                lines.append(f"\n{names[r['arm']]}相对原A：这两张图均值改善项为{'、'.join(gains) or '无'}；恶化项为{'、'.join(losses) or '无'}。这是描述性比较，不是显著性或普遍优势。")
    lines += ['', '## 实际学习与资源', '', '| 数据集 | 配置 | 新更新数 | 学习阶段秒 | 最终采样前占卡秒 |', '|---|---|---:|---:|---:|']
    for r in training:lines.append(f"| {r['dataset']} | {r['arm']} | {r['updates']} | {r['training_seconds']:.2f} | {r['allocation_before_final_evaluation_seconds']:.2f} |")
    lines += ['', '保存重载验证的是完整状态与下一输入合同；未重定义历史CUDA重复更新的数值容差，也不声称所有GPU重算逐位相同。S1/S16重叠更新核对主输入与RNG；生成逐步核对公共初态、查询和全槽随机数。完整LCC、逐关系、节点类型低度/hub、类型三角、元路径及原图逐样本明细见per_graph.csv、all_metrics.csv、evaluations/。LCC连通性来自后处理，不作为模型自动连通能力。',
        '', f'缺项：{json.dumps(ver["missing"],ensure_ascii=False)}。失败记录：{len(failures)}项。',
        '', '结束后不自动续训、搜索采样数、重开CE或新增模型。最终方法选择需要同时看结构代价、随机波动和计算成本；本轮不能证明跨seed稳定性。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'completion.json',dict(status='complete' if complete else 'partial',graphs=ver['completed_graphs'],missing=ver['missing'],failures=failures,
        GPU_hours=release['GPU_hours'],GPU_wall_seconds=release['wall_seconds'],delivery_epoch=time.time(),wall_seconds=time.time()-p['start_epoch'],
        all_owned_GPU_processes_exited=all(x['returncode'] is not None for x in release['processes']),no_successor_authorized=True))
    print((OUT/'completion.json').read_text(),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'finalizer_failure.json',dict(traceback=traceback.format_exc(),at_epoch=time.time()));raise
