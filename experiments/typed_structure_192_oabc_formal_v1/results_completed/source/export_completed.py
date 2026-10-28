"""Export the frozen completed run; no scientific task is launched."""
import csv,datetime,hashlib,json,pathlib,shutil,sys,time,zipfile

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    base=pathlib.Path(__file__).resolve().parent;root=pathlib.Path(sys.argv[1]).resolve();out=root/'delivery';out.mkdir(exist_ok=False)
    x=json.loads((root/'run/unified_evaluation.json').read_text());audit=json.loads((root/'completion_audit.json').read_text());ledger=json.loads((root/'ledger.json').read_text());launch=json.loads((root/'run/launcher_summary.json').read_text())
    assert launch['execution_complete'] and audit['status']=='passed' and x['evaluated_graphs']==16
    shutil.copytree(base/'package',out/'source/package',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(base/'evidence',out/'source/evidence',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(base/'formal_preparation',out/'source/formal_preparation')
    for name in ['summarize.py','protocol.json','activate_once.py','FORMAL_EXECUTION.md','test_resource_gate.py','LICENSE','audit_completed.py','export_completed.py']:
        shutil.copy2(base/name,out/'source'/name)
    for f in root.iterdir():
        if f.is_file():shutil.copy2(f,out/f.name)
    # Keep all original graphs, logs, curves, bindings, metrics and receipts.
    # Binary model/optimizer/RNG states remain in the original local run and are indexed by hash.
    shutil.copytree(root/'run',out/'run',ignore=shutil.ignore_patterns('*.pt'))
    excluded=[r for r in audit['payloads'] if r['path'].endswith('.pt')]
    (out/'LOCAL_BINARY_ASSETS.json').write_text(json.dumps(dict(root=str(root/'run'),files=excluded,scope='Retained local checkpoint/optimizer/RNG/trajectory states; no weight bytes in public ZIP'),indent=2)+'\n')
    with (out/'graph_metrics.csv').open('w',newline='') as f:
        cols=['dataset','arm','seed','edges','edge_abs_error','triangles','triangle_abs_error','lcc','isolates','maximum_degree','W1_incident4','W1_all10_compat','joint_degree_TV','typed_triangle_TV','MP2_composition_TV','MP3_composition_TV']
        w=csv.DictWriter(f,cols);w.writeheader()
        for label,c in x['cells'].items():
            ds,arm=label.split('_')
            for seed,g in c['graphs'].items():
                s=g['summary'];d=s['distribution_error'];r=s['raw'];e=s['absolute_error']
                w.writerow(dict(dataset=ds,arm=arm,seed=seed,**{k:r[k] for k in ['edges','triangles','lcc','isolates','maximum_degree']},edge_abs_error=e['edges'],triangle_abs_error=e['triangles'],W1_incident4=d['degree_W1_incident4'],W1_all10_compat=d['degree_W1_all10_compat'],joint_degree_TV=d['joint_degree_TV'],typed_triangle_TV=d['typed_triangle_composition_TV'],MP2_composition_TV=d['MP2_composition_TV'],MP3_composition_TV=d['MP3_composition_TV']))
    with (out/'training_curves.csv').open('w',newline='') as f:
        cols=['dataset','arm','update','loss','gradient_norm','normal2_seconds','t1','t2','input_sha256'];w=csv.DictWriter(f,cols);w.writeheader()
        for label in x['cells']:
            ds,arm=label.split('_')
            for line in (root/'run'/label/'training.jsonl').read_text().splitlines():
                r=json.loads(line);w.writerow(dict(dataset=ds,arm=arm,**{k:r[k] for k in ['update','loss','gradient_norm','normal2_seconds','input_sha256']},t1=r['t'][0],t2=r['t'][1]))
    with (out/'paired_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','contrast','sample','group','metric','difference'])
        for c in x['contrasts']:
            for seed,m in [(str(r['seed']),r['difference']) for r in c['pairs']]+[('two_seed_mean',c['mean'])]:
                for group,values in m.items():
                    if group=='undefined':continue
                    for k,v in values.items():w.writerow([c['dataset'],c['contrast'],seed,group,k,v])
    tz=datetime.timezone(datetime.timedelta(hours=8));finish=datetime.datetime.fromtimestamp(ledger['start_epoch']+launch['elapsed_since_activation'],tz).isoformat()
    lines=['# O/A/B/C两数据集192节点实验结果','',
      '八次正式训练各1000次normal2更新全部完成，最终权重共生成16张T1000原始图；八个子任务退出码均为0，无重启、补抽或未完成项。',
      f"正式起点：{ledger['start_beijing']}；自动汇总完成：{finish}；执行及自动汇总{launch['elapsed_since_activation']/60:.3f}分钟。人工审计、发布耗时另见delivery_receipt，不以79分钟代替最终全票耗时。",'',
      'O为异质适配基础；A为O+CN；B为O+一轮630维ReFeX（无CN）；C为B+冻结分类型RolX式非负系数（无CN）。两臂/多臂参数量不完全相等。',
      'IMDB与PubMed各一个训练seed、每格两个采样seed；以下都是两张原始图均值。W1主口径为incident4，越小越好；all10另列兼容值，等于主口径的0.4倍。', '',
      '|数据|组|边数|边数绝对误差均值|度W1|联合度TV|三角组成TV|三角数量|三角绝对误差均值|LCC|孤立点|最大度|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,c in x['cells'].items():
        ds,a=label.split('_');m=c['mean'];r,d,e=m['raw'],m['distribution_error'],m['absolute_error']
        vals=[ds,a,r['edges'],e['edges'],d['degree_W1_incident4'],d['joint_degree_TV'],d['typed_triangle_composition_TV'],r['triangles'],e['triangles'],r['lcc'],r['isolates'],r['maximum_degree']]
        lines.append('|'+ '|'.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in vals)+'|')
    lines += ['',
      '参考IMDB为323边、45三角、LCC188、最大度65；PubMed为926边、1937三角、LCC192、最大度92。数量增加/减少本身不代表改善，所有数量误差先逐图取绝对值再平均。', '',
      '## 已完成的探索性比较',
      '- A−O：CN在两数据集降低了度W1和三角数量误差。IMDB的连通代价明显：平均LCC从188降到166.5、孤立点从2增到11.5；联合度TV和其他指标须逐项看。',
      '- B−A：ReFeX替代CN没有统一收益。IMDB度W1降低，但三角数量从48降至3.5（参考45）；PubMed边数和度W1均退化。',
      '- C−B：RolX式系数在本次两个采样seed中均降低度W1，平均联合度TV也降低。PubMed边数绝对误差均值从262.5降至14，三角数量误差均值从1714降至229.5；IMDB三角不足仍在，三角组成TV反而从0.0222升至0.25。',
      '- C−A：C的度W1在两数据集均较低；IMDB闭合/三角结构明显较差。PubMed联合度TV与三角组成TV不及A，不能称所有结构指标胜出。',
      'IMDB B/C的三角组成比例看似接近时，总三角数仍可严重不足，必须同时阅读数量误差。16图都不同仅指带编号邻接不同，不能证明类型保持非同构或总体多样性充分。', '',
      '## 正确性与成本',
      '独立CPU脚本audit_completed.py读取保存图，复核schema、简单无向边、关系边数、逐节点关系度、incident4/all10 W1、联合度TV、三角数量/类型、聚类、连通与MP1/2/3数量；16图全部通过。核对1000个训练输入配对、训练随机流、共同初始参数及CN头共有列，全部通过。所有已发出checkpoint/state的SHA256核验通过。',
      '本轮不重跑已通过的GPU工程门；旧CPU/GPU等价失败保持历史事实。首次新审计脚本错误地比较CN头不同宽度矩阵的整张哈希而失败；已改为核对实际共有列，未修改任何实验源码、权重、结果或判据。', '',
      '|格|参数量|预热后normal2秒/步|两条采样耗时秒|allocated MiB|reserved MiB|RSS MiB|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for label,c in audit['cells'].items():
        lines.append(f"|{label}|{c['parameters']}|{c['measured_postwarmup_mean_seconds']:.4f}|{sum(c['sampling_seconds']):.2f}|{c['peak_allocated_mib']:.2f}|{c['peak_reserved_mib']:.2f}|{c['peak_rss_mib']:.2f}|")
    lines += ['',
      '八卡并发值来自本次实际执行；每进程CPU线程1。不同进程RSS峰值不相加作同步总峰值。采样耗时是每条完整1000步的实际回执，覆盖原有采样路径，不是单块估算。',
      '已只读确认本任务launcher/worker退出、无剩余GPU计算进程；没有杀死或清理他人任务。', '',
      '## 保留与复算',
      'run/unified_evaluation.json包含16图全部绝对指标、逐图参考误差、null原因及五种配对差；graph_metrics.csv为主要逐图值，paired_metrics.csv为全部配对与两图均值。training_curves.csv包含8000个实际更新，原training.jsonl及validation.json保留。',
      'SOURCE位于source/，正式执行包哈希与源码清单保存在formal_preparation/final_source.json及各格binding.json。audit_completed.py/export_completed.py仅为事后CPU审计与导出工具，不属于启动时模型源码。',
      '公开包不含私有模型/优化器/RNG二进制；它们完整留在原本地结果目录，LOCAL_BINARY_ASSETS.json登记路径、大小和哈希。抽样状态、初态、500/1000步checkpoint均未删除。',
      '复核命令（原本地完整结果目录）：',
      '```bash',
      'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/lyh/miniconda3/envs/sparse_block/bin/python source/audit_completed.py --run "$EXISTING_COMPLETE_RUN" --output "$NEW_AUDIT_JSON"',
      '```',
      '## 限制与停止',
      '这是单训练seed/数据集、两生成seed/模型的探索性结果，没有充分收敛、稳定显著收益、跨图/大图泛化或总体多样性结论。PubMed四类型K均命中候选上界的既有限制保留。',
      '未追加训练、生成、拟合、参数搜索或下一候选。本轮完成后停止，等待独立结果审查。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest=[]
    for f in sorted(out.rglob('*')):
        if f.is_file():manifest.append(dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)))
    (out/'PAYLOAD_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in manifest)+sha(out/'PAYLOAD_MANIFEST.json')+'  PAYLOAD_MANIFEST.json\n')
    zp=root/'typed_structure_192_oabc_formal_v1_results.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,str(pathlib.Path('typed_structure_192_oabc_formal_v1_results')/f.relative_to(out)))
    (zp.with_suffix('.zip.sha256')).write_text(sha(zp)+'  '+zp.name+'\n')
    print(json.dumps(dict(zip=str(zp),bytes=zp.stat().st_size,sha256=sha(zp),files=len(manifest))))

if __name__=='__main__':main()
