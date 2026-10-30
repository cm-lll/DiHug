"""Presentation summary and full comparisons from the saved merged evaluation."""
import argparse, collections, csv, datetime, pathlib, shutil, time
from build import read, save, csvwrite, sha, SEEDS, STAGES, MAIN, EARLY, IMDB, SOURCE

FIELDS=[('absolute_error','edges','边数绝对误差'),('distribution_error','main_incident4_W1','incident4度W1'),('distribution_error','main_joint_degree_TV','联合度TV'),('absolute_error','triangles','三角数绝对误差'),('distribution_error','triangle_composition_TV','三角组成TV'),('absolute_error','lcc','LCC节点数误差'),('distribution_error','MP2_composition_TV','MP2组成TV'),('distribution_error','MP3_composition_TV','MP3组成TV')]
def fmt(v):return 'null' if v is None else f'{v:.6g}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);out=pathlib.Path(ap.parse_args().output);doc=read(out/'evaluation.json')
    # Also supports finishing an already-built report after adding path columns;
    # hashes and values are checked, never inferred from a common dataset name.
    inventory=read(out/'material_inventory.json')
    for item in inventory:
        graph=pathlib.Path(item['original_graph']);cell=graph.parent.parents[2];side=cell/f"readouts_u{item['checkpoint']}.json";rr=read(side)
        if (rr['dataset'],rr['train_seed'],rr['arm'],rr['updates'])!=(item['dataset'],item['train_seed'],item['arm'],item['checkpoint']):raise ValueError('readout identity')
        if rr['files']['ema_0.99']['sha256']!=item['weight_sha256'] or rr['checkpoint_sha256']!=item['checkpoint_sha256']:raise ValueError('readout binding')
        parent=MAIN if item['checkpoint'] in (1000,2000) else cell.parents[1]
        item.update(weight_file=str(cell/rr['files']['ema_0.99']['file']),checkpoint_file=str(parent/'run'/cell.name/f"checkpoint_u{item['checkpoint']}.pt"),readout_receipt=str(side),readout_receipt_sha256=sha(side))
        dst=out/'run'/cell.name/side.name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(side,dst)
    save(out/'material_inventory.json',inventory);csvwrite(out/'tables/material_inventory.csv',inventory)
    idx={(x['dataset'],x['train_seed'],x['arm'],x['checkpoint'],x['version'],x['group'],x['metric']):x['value'] for x in doc['cell_means']}
    def value(d,s,a,u,g,m,version='original'):return idx.get((d,s,a,u,version,g,m))
    core=[];lines=['# 训练长度与分关系数量辅助：完整结果','', '本报告只整理已保存结果。原图为主，LCC另列；所有生成均为对应检查点EMA0.99。',
        '同阶段增量=CE_COUNT−CE；跨阶段变化=同一轨迹后期−早期。raw指标数量本身无单调优劣；absolute_error和TV/W1通常越低越好。先逐图求误差再平均两图；null不填0。','']
    for d in SEEDS:
        fields=FIELDS+([('absolute_error','family1_edges','GD误差'),('absolute_error','family5_edges','DC误差'),('raw','type_Chemical_low_le1_fraction','Chemical低度比例（参考0.075758）'),('absolute_error','type_Chemical_maximum','Chemical最大度误差'),('absolute_error','type_Chemical_second','Chemical次大度误差')] if d=='PubMed' else [('absolute_error','type_Genre_maximum','Genre最大度误差'),('absolute_error','type_Genre_second','Genre次大度误差'),('absolute_error','type_Genre_low_le1_fraction','Genre低度比例误差'),('absolute_error','triangle_(0, 1, 2)','DAM误差'),('absolute_error','triangle_(0, 1, 1)','DAA误差'),('absolute_error','triangle_(1, 1, 2)','AAM误差')])
        for s in SEEDS[d]:
            lines += [f'## {d}，训练seed {s}','','|指标|CE3000|CE6000|COUNT3000|COUNT6000|6000 COUNT−CE|','|---|---:|---:|---:|---:|---:|']
            for g,m,label in fields:
                vs=[value(d,s,a,u,g,m) for a,u in [('CE',3000),('CE',6000),('CE_COUNT',3000),('CE_COUNT',6000)]]
                delta=vs[3]-vs[1] if vs[1] is not None and vs[3] is not None else None
                lines.append('|'+label+'|'+'|'.join(map(fmt,vs))+'|'+fmt(delta)+'|')
                core.append(dict(dataset=d,train_seed=s,metric=m,group=g,label=label,CE3000=vs[0],CE6000=vs[1],COUNT3000=vs[2],COUNT6000=vs[3],same_stage_COUNT_minus_CE=delta,CE_6000_minus_3000=vs[1]-vs[0] if vs[0] is not None and vs[1] is not None else None,COUNT_6000_minus_3000=vs[3]-vs[2] if vs[2] is not None and vs[3] is not None else None))
            lines+=['']
    csvwrite(out/'tables/presentation_core.csv',core)
    direction=collections.defaultdict(list)
    for r in doc['paired_means']:
        if r['group'] not in ('absolute_error','distribution_error'):continue
        key=tuple(r[k] for k in ('dataset','checkpoint','version','group','metric'));direction[key].append(r)
    csvwrite(out/'tables/all_metric_seed_directions.csv',[dict(zip(('dataset','checkpoint','version','group','metric'),k),decreased=sum(r['difference'] is not None and r['difference']<0 for r in rows),tied=sum(r['difference']==0 for r in rows),increased=sum(r['difference'] is not None and r['difference']>0 for r in rows),undefined=sum(r['difference'] is None for r in rows),interpretation='numeric direction only; raw MMD U-stat decrease does not prove better quality' if 'MMD' in k[-1] else 'lower reference error is preferable; no aggregate quality claim',per_seed={r['train_seed']:r['difference'] for r in rows}) for k,rows in direction.items()])
    lines+=['## PubMed全部预定阶段（原图）','', '1000/2000为事后补充探索，不用于替换预定3000/6000节点。','']
    for s in SEEDS['PubMed']:
        lines += [f'### 训练seed {s}','','|指标/组别|1000|2000|3000|6000|','|---|---:|---:|---:|---:|']
        for g,m,label in FIELDS:
            for a in ('CE','CE_COUNT'):lines.append('|'+label+'/'+a+'|'+'|'.join(fmt(value('PubMed',s,a,u,g,m)) for u in STAGES['PubMed'])+'|')
    lines += ['','## 结论与证据边界','',
        '1. 数量项有阶段性、可重复的局部收益，尚无统一优胜者。6000步两数据集、四个seed的W1均优于同期CE，但联合度、类别三角、元路径和尾部并不全改善。',
        '2. PubMed6000：边数绝对误差176→125、131→21.5；W1 0.74740→0.55469、0.55208→0.52474；DC误差50.5→44.5、25→14。GD第一seed24.5→30恶化，第二27→21.5改善。Chemical低度比例0.23485→0.22727、0.27273→0.18182，仍高于参考0.07576；最大度误差3→6.5、4.5→9.5，均恶化。三角误差924→305.5与226→821.5，方向翻转。',
        '3. PubMed数量项自身3000→6000：第一seed边数误差101.5→125、W1 0.51563→0.55469变差；第二74.5→21.5、0.53906→0.52474改善。同期胜过CE不能写成持续进步。CE两seed的边数误差和W1均恶化，但三角数量误差有所降低、部分MP组成变好，不能笼统说所有结构退化。',
        '4. PubMed1000/2000时数量项边数误差在两seed均低于CE；3000时仅第一seed更好，6000时又两seed更好。此结果显示阶段依赖，不以最好阶段替代原定终点。',
        '5. IMDB6000：数量项的W1、三角数量误差、LCC误差两seed均优于CE；边数误差第一12.5→5、第二7.5→7.5。联合度TV、三角组成TV、MP3组成TV均变差，Genre尾部一seed变差、一seed部分改善。DAM/DAA/AAM和闭合机会/比率保留明细，不能以总三角接近替代类型化检查。',
        '6. 纯CE延长3000→6000的边数误差在四个seed全部增大；其他结构存在改善或反转。不把这些点认证为收敛，也不凭退化直接认定过拟合。',
        '7. 完整图阶段建议保留纯CE作为匹配基线、数量辅助作为候选，先单独核对完整查询分块的数量估计、特征/字典成本；本报告不授权该实验，不宣称超过HGEN。两个训练seed、每格两图仅为本面板探索证据。','',
        '所有原值、误差、分布距离与null逐项在tables；变长模式/闭合/度谱/联合度在长表及evaluation.json。原图与LCC各有独立趋势图；LCC连通/零孤立点是后处理定义，不是模型收益。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary='''# 下午汇报摘要：数量辅助有局部收益，更长训练不是统一修复

**完成材料。** 同批E-earlyCN，IMDB两个训练seed、PubMed两个训练seed，各CE/CE_COUNT；数量公式为平方根目标尺度的有界斜率项，λ=0.001。3000、6000各16张原图；PubMed1000、2000补充16张全部完成，总计48张原图及48份LCC派生评价。未新增训练、前向或生成。

|6000步，原图两样本平均误差|PubMed2912 CE→COUNT|PubMed2932 CE→COUNT|IMDB2911 CE→COUNT|IMDB2931 CE→COUNT|
|---|---:|---:|---:|---:|
|边数绝对误差↓|176→125|131→21.5|12.5→5|7.5→7.5|
|incident4度W1↓|0.7474→0.5547|0.5521→0.5247|0.08594→0.08203|0.06901→0.05990|
|联合度TV↓|0.68229→0.68490|0.72396→0.70052|0.15365→0.16927|0.15104→0.15625|
|三角数绝对误差↓|924→305.5|226→821.5|5.5→3.5|2→0.5|

**主要成果。** 数量项在6000步四个seed的度W1均优于同期CE；PubMed两seed的总边数误差、DC误差、Chemical低度堆积及连通性也均优于CE。说明它确实可以带来生成层面的增量，而不只是去噪CE变化。

**明确代价。** PubMed的GD误差方向不一，Chemical最大度误差两seed均更大；2932的数量更准伴随三角明显过量。IMDB的联合度、三角组成和MP3组成TV两seed均比CE差，Genre尾部没有一致优势。数量正确不能替代结构正确。

**训练长度。** 从3000到6000，纯CE的边数误差在四个seed全部增大；PubMed的数量项自身第一seed数量/W1变差、第二seed改善。因此“数量项优于6000步CE”与“数量项随训练持续进步”必须区分。早期1000/2000结果也不单调，不挑最好节点、不认定已收敛或过拟合。

**建议。** 目前没有跨指标、跨seed的统一赢家。进入完整图验证时保留原CE匹配对照，并将数量辅助作为待验证候选；完整图分块数量估计、特征与字典成本须另行授权核对。不能丢弃纯CE，也不能把本面板写成稳定显著收益或超过HGEN。

**交付和限制。** 全指标、逐图误差、配对差、两个seed分列及原图/LCC全表随附；HGEN原生powerlaw依赖、社区标注和通用orbit缺项仍披露。原PubMed收尾脚本错误为读取嵌套job身份时错误查外层dataset；原失败记录保留，评价值未修改。科学完成、修复与发布时间分别记录；旧截止不倒改。
'''
    (out/'SUMMARY_ZH.md').write_text(summary)
    # Reference/definitions are split by dataset, not inferred from metric names.
    (out/'METRIC_DIRECTORY.md').write_text('''# 全指标目录与口径

完整逐字段目录为`tables/metric_catalog.csv`，包含dataset、group、定义、单位、参考值、优劣方向与实现来源。原图与LCC全部字段分别评价；raw组名指指标原值，不是raw权重读出。

各数据集635个原值字段、对应635个逐图绝对误差、73个分布字段；同名字段也保留dataset避免参考值混淆。未定义值及原因见`tables/undefined_values.csv`，不填0、不删除。

详细定义见`by_dataset/IMDB/METRIC_DIRECTORY.md`与`by_dataset/PubMed/METRIC_DIRECTORY.md`。W1主口径固定参考类型权重incident4，all10兼容=0.4×incident4；TV为归一频数半L1。精确整数联合度TV不能直接解释成网络学会了角色机制。

MP1/2/3对应1/2/3条边的简单路径，顶点互异、反向去重；walk允许重复节点/回退，另列。类别三角按排序类型计一次；参考为0不是schema禁止。闭合path与unique_pair分开，分母0为null。真正四环与HGEN legacy square(K4)分开；标准聚类与legacy 3T/claws分开。

度MMD采用冻结核exp(-4*(x-y)^2)，组内去对角除n(n-1)、交叉除nm的U-stat形式；raw允许负值，另列max(raw,0)，不宣称节点独立。旧源码中若有“biased”初稿文字，以本目录及vendor实现为准，数值从未改变。

HGEN原生powerlaw：缺依赖；原生社区/NMI：无可比标签；通用graphlet/orbit：未实现。项目fixed-xmin1估计不是原生powerlaw替代认证，现有四环并非完整orbit谱。对这些缺项不造数字。无机会闭合、无类型、非连通全局直径等按数学定义保留null。

同条件两图样本差异完整见`tables/sample_diversity.json`；编号对齐边差/Jaccard不是同构不变的结构多样性证据。LCC按原存储节点编号规则选最大分量，映射保留；统计分母用实际节点集合，固定参考权重不变。IMDB固定参考自身有188/2/2三分量，PubMed参考连通；不能偷偷更换参考。

数量/闭合/度谱明细为all_pattern_counts_compositions、all_typed_closure_details、complete_degree_spectra、all_degree_histograms、complete_joint_degree_histograms长表。完整嵌套原值见evaluation.json；报告并不局限摘要选中的字段。
''')
    (out/'README.md').write_text('''# 结果收尾包

先读SUMMARY_ZH.md（一页汇报）、REPORT.md（分seed/分阶段对照），再读METRIC_DIRECTORY.md。
全部实际生成使用EMA0.99，不含raw新生成。48张原图+48派生LCC；IMDB无1000/2000图，未列为失败或用p0替代。
`tables/paired*`为同阶段数量项增量；`length*`为同一训练轨迹相邻阶段变化；`total_length_paired`明确3000→6000。raw数量差与absolute_error差分列。
`tables/validation*`为普通固定前向噪声状态的raw/EMA验证；`training*`为训练记录。它们与真实生成指标独立，不画成同一类曲线。
`saved_logs`保留原日志副本和原来的时间/错误；`evidence`保留原失败与旧审计。权重仅登记来源和SHA256，不在公开包夹带大型.pt。
本次只用CPU整理，训练/前向/生成新增均为0。未新增科学审批，无自动继续入口。
''')
    repaired=out/'pubmed_repaired';repaired.mkdir(exist_ok=True)
    pubmed=['# PubMed已修复的完整结果报告','', '完整IMDB报告关联见根目录REPORT.md及旧IMDB独立归档；本子包只含PubMed实际图。',''];active=False
    for line in lines:
        if line.startswith('## '):active=line.startswith(('## PubMed','## 结论'))
        if active:pubmed.append(line)
    (repaired/'REPORT.md').write_text('\n'.join(pubmed)+'\n')
    save(repaired/'evaluation.json',dict(records=[r for r in doc['records'] if r['dataset']=='PubMed'],**{k:[r for r in doc[k] if r.get('dataset',r.get('job',{}).get('dataset'))=='PubMed'] for k in ('cell_means','paired','paired_means','length_paired','length_means','total_length_paired','lcc_minus_original','diversity')},missing_identities=[x for x in doc['missing_identities'] if x['dataset']=='PubMed']))
    save(out/'report_repair_receipt.json',dict(repaired_epoch=time.time(),root_cause="finalize.py:35 accesses taskrows[i]['dataset']; actual identity is taskrows[i]['job']['dataset']",field_was_not_lost=True,unknown_records_not_imputed=True,original_failure=str(MAIN/'delivery_failure.json'),original_failure_sha256=sha(MAIN/'delivery_failure.json'),metric_values_changed=False,early_complete_graphs=16,generated_originals=48,derived_lcc=48,
        source_scientific_completion=dict(main=read(MAIN/'run/launcher_summary.json')['finished_epoch'] if 'finished_epoch' in read(MAIN/'run/launcher_summary.json') else read(MAIN/'run/launcher_summary.json').get('finished'),imdb=read(IMDB/'scientific_completion.json'),early=read(EARLY/'final_delivery.json')['scientific_finished_epoch']),old_imdb_archive=dict(path=str(IMDB/'IMDB_3000_to_6000_full_evaluation.zip'),sha256_file=(IMDB/'ZIP_SHA256.txt').read_text()),old_delivery_deadlines_not_changed=True))
    print('Chinese summary and complete report ready:',out)
if __name__=='__main__':main()
