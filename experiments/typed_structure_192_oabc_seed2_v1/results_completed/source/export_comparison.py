"""Read completed artifacts only. Export both training seeds without pooling them."""
import csv,datetime,hashlib,json,pathlib,shutil,sys,time,zipfile

SEEDS=[{'IMDB':2026102811,'PubMed':2026102812},{'IMDB':2026102834,'PubMed':2026102833}]
EPS=1e-12  # Only a floating-point zero guard; not a scientific effect threshold.

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_csv(path,cols,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)

def classify(first,second,first_pairs,second_pairs):
    """Descriptive signs; no significance claim and no arbitrary effect cutoff."""
    if any(v is None for v in [first,second]+first_pairs+second_pairs):return '证据不足'
    sign=lambda v:0 if abs(v)<=EPS else 1 if v>0 else -1
    a,b=sign(first),sign(second)
    if a==0 or b==0:return '证据不足'
    if a!=b:return '翻转'
    if all(sign(v)==a for v in first_pairs+second_pairs):return '复现'
    return '部分复现'

def main():
    src=pathlib.Path(__file__).resolve().parent
    old,new=map(lambda p:pathlib.Path(p).resolve(),sys.argv[1:3]);out=new/'delivery';out.mkdir(exist_ok=False)
    roots=[old,new];xs=[json.loads((r/'run/unified_evaluation.json').read_text()) for r in roots]
    audit=json.loads((new/'completion_audit.json').read_text());assert audit['status']=='passed'
    assert all(x['status']=='complete' and x['evaluated_graphs']==16 for x in xs)
    # Verify the first-round aggregate against its sealed launcher receipt.
    for root in roots:
        l=json.loads((root/'run/launcher_summary.json').read_text())
        assert l['execution_complete'] and not l['primary_failures'] and all(j['exit_code']==0 for j in l['jobs'])
        assert sha(root/'run/unified_evaluation.json')==l['evaluation']['output_sha256']
    for ds in ['IMDB','PubMed']:
        for arm in 'OABC':
            bindings=[json.loads((r/'run'/f'{ds}_{arm}'/'binding.json').read_text()) for r in roots]
            assert bindings[0]['source_sha256']==bindings[1]['source_sha256']
            assert bindings[0]['dictionary_sha256']==bindings[1]['dictionary_sha256'] and bindings[0]['dataset']==bindings[1]['dataset']
            for i,b in enumerate(bindings):
                assert b['config']['seed']==SEEDS[i][ds] and b['updates']==1000
            ca,cb=[dict(b['config']) for b in bindings];ca.pop('seed');cb.pop('seed');assert ca==cb
    shutil.copytree(src/'package',out/'source/package',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(src/'evidence',out/'source/evidence',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(src/'formal_preparation',out/'source/formal_preparation')
    shutil.copytree(src/'preparation',out/'source/preparation',ignore=shutil.ignore_patterns('*.zip'))
    for name in ['summarize.py','protocol.json','activate_once.py','LICENSE','EXECUTION.md','audit_completed.py','export_comparison.py']:
        shutil.copy2(src/name,out/'source'/name)
    # All second-round raw graphs, logs and numerical receipts; binary checkpoints stay local.
    for f in new.iterdir():
        if f.is_file():shutil.copy2(f,out/f.name)
    shutil.copytree(new/'run',out/'round2/run',ignore=shutil.ignore_patterns('*.pt'))
    (out/'round1/run').mkdir(parents=True)
    for name in ['unified_evaluation.json','launcher_summary.json','schedule.json']:
        shutil.copy2(old/'run'/name,out/'round1/run'/name)
    for cell in xs[0]['cells']:
        d=out/'round1/run'/cell;d.mkdir()
        for name in ['summary.json','binding.json','initialization.json','training_rng_reset.json','validation.json','training.jsonl','initial.sha256.json','midpoint.sha256.json','final.sha256.json']:
            shutil.copy2(old/'run'/cell/name,d/name)
        for seed in [2026102821,2026102822]:
            folder=d/f'sample_{seed}';folder.mkdir()
            for name in ['graph.npz','binding.json','metrics.json']:
                source=old/'run'/cell/f'sample_{seed}'/name
                if name=='graph.npz':assert sha(source)==xs[0]['cells'][cell]['graphs'][str(seed)]['sha256']
                shutil.copy2(source,folder/name)
    rows=[];flat=[];paired=[];meanrows=[];costs=[]
    for rnd,(root,x) in enumerate(zip(roots,xs),1):
        for cell,c in x['cells'].items():
            ds,arm=cell.split('_');train_seed=SEEDS[rnd-1][ds]
            for seed,g in c['graphs'].items():
                s=g['summary'];r,e,d=s['raw'],s['absolute_error'],s['distribution_error']
                base=dict(round=rnd,dataset=ds,train_seed=train_seed,sample_seed=int(seed),arm=arm)
                rows.append(dict(**base,**{k:r[k] for k in ['edges','triangles','lcc','isolates','maximum_degree']},edge_abs_error=e['edges'],triangle_abs_error=e['triangles'],lcc_abs_error=e['lcc'],max_degree_abs_error=e['maximum_degree'],W1_incident4=d['degree_W1_incident4'],W1_all10_compat=d['degree_W1_all10_compat'],joint_degree_TV=d['joint_degree_TV'],typed_triangle_TV=d['typed_triangle_composition_TV'],MP2_composition_TV=d['MP2_composition_TV'],MP3_composition_TV=d['MP3_composition_TV']))
                for group in ['raw','reference','absolute_error','distribution_error']:
                    for k,v in s[group].items():flat.append(dict(**base,group=group,metric=k,value=v,undefined_reason=s['undefined'].get(group,{}).get(k)))
            m=c['mean'];meanrows.append(dict(round=rnd,dataset=ds,train_seed=train_seed,arm=arm,raw=m['raw'],absolute_error=m['absolute_error'],distribution_error=m['distribution_error'],diversity=c['diversity']))
            tr=[json.loads(line) for line in (root/'run'/cell/'training.jsonl').read_text().splitlines()]
            val=json.loads((root/'run'/cell/'validation.json').read_text());rs=c['worker_status']
            cost=dict(round=rnd,cell=cell,train_seed=train_seed,normal2_total_seconds=sum(r['normal2_seconds'] for r in tr),normal2_postwarmup_mean_seconds=sum(r['normal2_seconds'] for r in tr if not r['warmup'])/sum(not r['warmup'] for r in tr),validation_seconds=sum(v['seconds'] for v in val),sample_1_seconds=json.loads((root/'run'/cell/'sample_2026102821/binding.json').read_text())['seconds'],sample_2_seconds=json.loads((root/'run'/cell/'sample_2026102822/binding.json').read_text())['seconds'],peak_allocated_mib=rs['peak_allocated_mib'],peak_reserved_mib=rs['peak_reserved_mib'],peak_rss_mib=rs['peak_rss_mib']);costs.append(cost)
        for c in x['contrasts']:
            for sample,m in [(str(p['seed']),p['difference']) for p in c['pairs']]+[('two_sample_mean',c['mean'])]:
                for group,values in m.items():
                    if group=='undefined':continue
                    for k,v in values.items():paired.append(dict(round=rnd,dataset=c['dataset'],train_seed=SEEDS[rnd-1][c['dataset']],contrast=c['contrast'],sample=sample,group=group,metric=k,difference=v,undefined_reason=m['undefined'].get(group,{}).get(k)))
    directions=[]
    for a,b in zip(xs[0]['contrasts'],xs[1]['contrasts']):
        assert (a['dataset'],a['contrast'])==(b['dataset'],b['contrast'])
        for group in ['absolute_error_difference','distribution_error_difference']:
            assert set(a['mean'][group])==set(b['mean'][group])
            for k in a['mean'][group]:
                av,bv=a['mean'][group][k],b['mean'][group][k]
                ap=[p['difference'][group][k] for p in a['pairs']];bp=[p['difference'][group][k] for p in b['pairs']]
                directions.append(dict(dataset=a['dataset'],contrast=a['contrast'],group=group,metric=k,round1_mean=av,round2_mean=bv,round1_pairs=ap,round2_pairs=bp,result=classify(av,bv,ap,bp)))
    (out/'comparison.json').write_text(json.dumps(dict(status='complete',rounds=[dict(round=i+1,train_seeds=SEEDS[i],source_run=str(r),aggregate_sha256=sha(r/'run/unified_evaluation.json')) for i,r in enumerate(roots)],cells=meanrows,directions=directions,costs=costs,interpretation='Signs are descriptive: mean sign reversal=翻转; same sign and all four within-round paired values same sign=复现; same mean sign but mixed/tied pairs=部分复现; undefined/tied mean=证据不足. 1e-12 is floating-point zero guard, not effect threshold. No cross-training-seed pooling.'),indent=2,ensure_ascii=False)+'\n')
    write_csv(out/'graphs_both_rounds.csv',list(rows[0]),rows);write_csv(out/'all_metrics_both_rounds.csv',list(flat[0]),flat);write_csv(out/'paired_both_rounds.csv',list(paired[0]),paired);write_csv(out/'costs_both_rounds.csv',list(costs[0]),costs)
    with (out/'training_curves_both_rounds.csv').open('w',newline='') as f:
        cols=['round','dataset','train_seed','arm','update','loss','gradient_norm','normal2_seconds','t1','t2','input_sha256'];w=csv.DictWriter(f,cols);w.writeheader()
        for rnd,(root,x) in enumerate(zip(roots,xs),1):
            for cell in x['cells']:
                ds,arm=cell.split('_')
                for line in (root/'run'/cell/'training.jsonl').read_text().splitlines():
                    r=json.loads(line);w.writerow(dict(round=rnd,dataset=ds,train_seed=SEEDS[rnd-1][ds],arm=arm,**{k:r[k] for k in ['update','loss','gradient_norm','normal2_seconds','input_sha256']},t1=r['t'][0],t2=r['t'][1]))
    oldassets=json.loads((old/'completion_audit.json').read_text())['payloads'];local=[]
    for rnd,(root,payload) in enumerate([(old,oldassets),(new,audit['payloads'])],1):
        local.append(dict(round=rnd,root=str(root/'run'),files=[r for r in payload if r['path'].endswith('.pt')]))
    (out/'LOCAL_BINARY_ASSETS.json').write_text(json.dumps(local,indent=2)+'\n')
    # All headline claims below reference separately reported training-seed means.
    lines=['# 第二训练seed收益复核：O/A/B/C两数据集192节点','',
      '本轮8次新训练各1000次normal2有效更新、16条新T1000完整链和统一评价全部成功；八子任务exit0，没有重启、补抽或未完成项。运行与自动汇总4741.700秒（79.03分钟）。实际启动11:05:34.571895，固定截止13:25:34.571895 Asia/Shanghai；核验和发布继续计入原窗口。',
      '四组：O异质适配基础、A加CN、B以一轮630维ReFeX替代CN、C在B上加冻结RolX式系数。package与首轮绑定相同，仅训练seed改变。两轮IMDB seed为2026102811/2026102834，PubMed为2026102812/2026102833；每轮每格采样seed均为2026102821/2026102822。', '',
      '## 逐项结论',
      '- PubMed C−B：度W1、边数绝对误差、三角数绝对误差收益复现，四个配对图差均同向。两轮度W1差为−0.4010/−0.3203，边数误差差−248.5/−125.5，三角误差差−1484.5/−1394.5。',
      '- PubMed C−A：边数及三角数量误差收益复现。度W1平均收益仍在（−0.1510/−0.1680），但第二轮一个生成seed变差、另一个改善，只记部分复现。',
      '- PubMed C−A的联合度TV代价复现：两轮平均差+0.03646/+0.03125，四个配对差均为正。改善各关系边际度，未带来相对CN的联合关系度优势。',
      '- PubMed连通收益部分复现：C对B的LCC误差均值下降，但C对A第二轮平均LCC误差相等；C第二轮孤立点比A/B更多（9.5对8.5），不再重复首轮的孤立点优势。MP2/MP3组成逐项保留，C−B的MP3组成误差由首轮较差变为本轮较好，不能合并为统一趋势。',
      '- IMDB A−O：三角数量与闭合收益、连通代价再次出现。A三角两轮均值48/60（参考45），闭合率约0.926/0.932（参考1）；LCC166.5/179（参考188），相对O均下降。',
      '- IMDB C−B：首轮度W1收益在第二轮均值上翻转，差−0.02083变为+0.00391；第二轮两个生成seed方向不同。C−A由−0.05469变为+0.08724，第二轮两个生成seed均不及A。',
      '- IMDB C的三角不足复现：两轮2.5/1.5，参考45。最大度误差仍明显，但偏差方向翻转：首轮75/77偏高，第二轮21/23偏低，参考65。不能只说相同hub偏差稳定复现。',
      '- PubMed O对训练seed较敏感：边数均值640.5变1854.5、三角640变9203.5，参考926边/1937三角。因此B−O等比较会随基础臂变化而翻转，必须保留每个训练seed的绝对结果。', '',
      '## 两轮绝对结果（每格两图均值，不跨训练seed平均）',
      '|数据|训练seed|组|边数|边绝对误差|incident4 W1|联合度TV|三角组成TV|三角数|三角绝对误差|LCC|孤立|最大度|',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in meanrows:
        r,e,d=m['raw'],m['absolute_error'],m['distribution_error'];v=[m['dataset'],m['train_seed'],m['arm'],r['edges'],e['edges'],d['degree_W1_incident4'],d['joint_degree_TV'],d['typed_triangle_composition_TV'],r['triangles'],e['triangles'],r['lcc'],r['isolates'],r['maximum_degree']]
        lines.append('|'+ '|'.join(f'{q:.4f}' if isinstance(q,float) else str(q) for q in v)+'|')
    lines+=['','参考IMDB：323边、45三角、LCC188、最大度65；PubMed：926边、1937三角、LCC192、最大度92。数量误差先逐图取绝对值再求均值；组成TV和数量误差必须同时看。all10=0.4×incident4另列，不混用。', '',
      '## 五类配对差与方向复核',
      '下表均为左组误差减右组误差，负值表示该指标误差较小；不代表整体更优。复现为两轮均值和四个生成seed配对差全部同向；部分复现为均值同向但配对差有混合/零；翻转为两轮均值异号；缺失或均值数值同分为证据不足。仅1e-12作计算零保护，不设效果显著性阈值。',
      '|数据|比较|指标|首轮差|第二轮差|描述性结论|','|---|---|---|---:|---:|---|']
    selected=[('distribution_error_difference','degree_W1_incident4'),('distribution_error_difference','joint_degree_TV'),('absolute_error_difference','edges'),('absolute_error_difference','triangles'),('absolute_error_difference','lcc'),('distribution_error_difference','MP2_composition_TV'),('distribution_error_difference','MP3_composition_TV')]
    for d in directions:
        if (d['group'],d['metric']) in selected:
            fmt=lambda v:'null' if v is None else f'{v:+.6f}'
            lines.append(f"|{d['dataset']}|{d['contrast']}|{d['metric']}|{fmt(d['round1_mean'])}|{fmt(d['round2_mean'])}|{d['result']}|")
    lines+=['','## 核验、成本与资料',
      '复用首轮已发布的独立CPU审计脚本，对本轮16图重算关键度W1、联合度TV、schema、三角类型/数量、连通、聚类和MP1/2/3数量，通过；1000步序列与16条全链完整覆盖、初态公共参数及CN公共头切片、训练输入配对、随机流、checkpoint/state哈希均通过。审计不调用网络，不执行optimizer.step。原GPU/字典工程验收直接复用。',
      '原package、数据、字典与两轮所有模型config除训练seed外逐项核对相同；两轮汇总哈希核对原launcher回执。源审计代码与发布载荷均保留。八个任务自然退出，已只读确认GPU计算进程为空。',
      'round2/run保留本轮16张graph.npz、逐图metrics、binding、全部training/steps日志、验证、初始化/中点/最终哈希及退出记录。round1/run保留原逐图和汇总用于并列；旧原始目录未改。comparison.json保留全部指标方向复核，不只挑收益。',
      'graphs_both_rounds.csv为32图主要指标；all_metrics_both_rounds.csv为原值/参考/误差/null原因；paired_both_rounds.csv为每训练seed各自的五类逐生成seed差与两图均值；training_curves_both_rounds.csv为两轮16000次实际更新；costs_both_rounds.csv为实际单格耗时和峰值。',
      '公开包不含私有权重/优化器/RNG二进制；所有初态/500/1000checkpoint和逐链状态保留本地，LOCAL_BINARY_ASSETS.json登记两轮路径与SHA256。没有删除文件、覆盖旧实验或生成替代输入。',
      '复算使用原完整本地目录：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/lyh/miniconda3/envs/sparse_block/bin/python source/audit_completed.py --run "$COMPLETE_RUN" --output "$NEW_AUDIT_JSON"`。跨轮导出脚本仅读取已完成数据，输出目录须不存在。', '',
      '## 解释边界与停止',
      '本轮确认问题由首轮结果提出，不能称为首轮预注册发现。两训练seed和每模型两图仍不足以证明统计显著性、稳定整体增益、跨图/原始大图泛化或总体多样性。PubMed四类型K均命中上界，未改变；CE仅辅助。',
      '每格两个图均非完全相同的带编号邻接；该差异不证明类型保持非同构，更不证明多样性充分。全部原始图保留，不筛图或补抽。',
      '本轮完成后停止，等待独立审查；未追加模型、闭合修复、联合度修正、训练seed、图或超参数搜索。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'PAYLOAD_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in manifest)+sha(out/'PAYLOAD_MANIFEST.json')+'  PAYLOAD_MANIFEST.json\n')
    zpath=new/'typed_structure_192_oabc_seed2_results.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,str(pathlib.Path('typed_structure_192_oabc_seed2_results')/f.relative_to(out)))
    zpath.with_suffix('.zip.sha256').write_text(sha(zpath)+'  '+zpath.name+'\n')
    print(json.dumps(dict(zip=str(zpath),sha256=sha(zpath),bytes=zpath.stat().st_size,files=len(manifest))))

if __name__=='__main__':main()
