"""CPU-only finalization after owned workers exit; no successor launch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,subprocess,traceback
from pathlib import Path
OUT=Path(sys.argv[1]).resolve()
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    p=read(OUT/'protocol.json')
    while not (OUT/'release.json').exists():
        if time.time()>p['delivery_epoch']-120:raise TimeoutError('release unavailable at delivery reserve')
        time.sleep(5)
    with (OUT/'evaluation.log').open('w') as f:
        done=subprocess.run([PY,str(Path(__file__).with_name('evaluate_terminal_continuation.py')),str(OUT)],stdout=f,stderr=subprocess.STDOUT,
            timeout=max(1,p['delivery_epoch']-time.time()-90))
    release=read(OUT/'release.json')
    if done.returncode:
        save('completion.json',dict(status='evaluation_incomplete',returncode=done.returncode,release=release));return
    ver=read(OUT/'verification.json');summaries=read(OUT/'model_summary.json')
    checks=[];missing_training=[]
    for ds in ('IMDB','PubMed'):
        folder=OUT/(ds+'_T_eager')
        for u in range(3,9):
            bp=folder/f'round{u}_batch.json'
            if not bp.exists():missing_training.append(dict(dataset=ds,round=u,reason='no complete training batch'));continue
            batch=read(bp)
            assert batch['new_on_policy_trajectories'] and batch['clean_head_estimator_is_biased']
            c=batch['costs'];assert abs(batch['advantages'][0]-(c[0]-c[1]))<1e-12
            for plan in p['terminal_plan'][ds][str(u)]:
                d=folder/f'train_round{u}_seed{plan["seed"]}'
                s=read(d/'summary.json');i=read(d/'identity.json')
                assert s['steps']==1000 and s['status']=='complete'
                assert i['model_hash']==batch['policy_hash'] and i['updates']==u-1
                assert len(list(d.glob('Xt_*.npz')))==8
            checks.append(dict(dataset=ds,round=u,two_fresh_complete_trajectories=True,policy_binding=True,LOO_checked=True))
    complete=ver['completed_graphs']==16 and not missing_training and len(checks)==12 and all(x['returncode']==0 for x in release['processes'])
    save('training_chain_verification.json',dict(checks=checks,missing=missing_training))
    def fmt(r,k):
        v=r.get(k+'_mean');sd=r.get(k+'_sample_sd')
        return '缺' if v is None else f'{v:.5g} ± {sd:.3g}' if sd is not None else f'{v:.5g}'
    lines=['# 完整链终点反馈：固定第2→8次更新试点','',
        f"状态：{'完整完成' if complete else '部分完成'}；评价图{ver['completed_graphs']}/16，8旧+8计划新增。本票GPU占用{release['GPU_hours']:.4f}小时。",
        '', '每数据集一个开发训练seed，每模型阶段两个配对采样seed。标准差仅为同权重采样波动；不作跨训练seed稳定性结论。',
        '', '恢复第2次完整模型/AdamW/EMA/RNG，固定续到8，4/8分别独立评价。每轮两条当前raw策略的新T1000链，终点代价作LOO相对反馈，选8时间/链的clean-head对终点图log概率求导；有偏代理，不是精确终点转移策略梯度，也不声称原版GDPO复现。没有最佳图/最佳checkpoint选择。',
        '', '原结构评分/学习率/裁剪/模型/采样核保持，evd修复与原evr的数学算子/容差相同。正式微调轨迹不是逐位重算认证。raw主生成；EMA仅固定前向验证。原图为主，LCC子表单列。']
    names={'baseline':'原A','T2':'第2次（复用）','T4':'第4次','T8':'第8次'}
    for ds in ('IMDB','PubMed'):
        ref=read(OUT/'reference_metrics.json')[ds]
        lines += ['',f'## {ds} / 训练seed {p["parents"][ds]["train_seed"]}',f"参考：{ref['edges']}边、{ref['triangles']}三角、LCC={ref['lcc']}。",'',
            '|阶段|样本|边数|度W1↓|联合度TV↓|三角数量|三角绝对误差↓|LCC|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        rr=[r for r in summaries if r['dataset']==ds and r['view']=='raw']
        for r in rr:lines.append('|'+names[r['arm']]+'|'+str(r['samples'])+'|'+'|'.join(fmt(r,k) for k in ['edges','degree_W1','joint_TV','triangles','triangle_abs_error','LCC'])+'|')
        lines += ['','|阶段|三角组成TV↓|MP2数量误差↓|MP2组成TV↓|MP3数量误差↓|MP3组成TV↓|分量数|','|---|---:|---:|---:|---:|---:|---:|']
        for r in rr:lines.append('|'+names[r['arm']]+'|'+'|'.join(fmt(r,k) for k in ['triangle_composition_TV','MP2_abs_error','MP2_TV','MP3_abs_error','MP3_TV','components'])+'|')
        b=next(r for r in rr if r['arm']=='baseline')
        for r in rr:
            if r['arm']=='baseline':continue
            gains=[];costs=[]
            for k in ('degree_W1','joint_TV','triangle_abs_error','triangle_composition_TV','MP2_abs_error','MP2_TV','MP3_abs_error','MP3_TV'):
                if r[k+'_mean']<b[k+'_mean']:gains.append(k)
                elif r[k+'_mean']>b[k+'_mean']:costs.append(k)
            lines.append(f"\n{names[r['arm']]}相对原A：改善{gains}；代价{costs}。逐图/类型/关系/LCC见全表，均值不代表两图均同向。")
    lines += ['',f'未完成训练轮：{missing_training}；未完成评价：{ver["missing"]}。',
        '', '部分结果不能冒充整轮成功；奖励分数下降不等于全部结构改善。完成后不自动追加更新、筛seed或启动其他方法。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    save('completion.json',dict(status='complete' if complete else 'partial',graphs=ver['completed_graphs'],completed_training_rounds=len(checks),
        missing_training=missing_training,missing_evaluation=ver['missing'],GPU_hours=release['GPU_hours'],
        wall_seconds=time.time()-p['start_epoch'],delivery_epoch=time.time(),all_owned_GPU_processes_exited=all(x['returncode'] is not None for x in release['processes']),no_successor_started=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        save('finalizer_failure.json',dict(traceback=traceback.format_exc(),at_epoch=time.time()));raise
